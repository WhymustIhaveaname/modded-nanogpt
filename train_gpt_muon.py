"""
Reproduces Muon optimizer record from 2024-10-11.
Based on commit b356a1f.

Muon uses Newton-Schulz iteration to orthogonalize the gradient updates
for transformer block parameters, while using AdamW for embedding/lm_head.

Expected first few losses (from records/track_1_short/2024-10-10_Muon, 8 GPU):
  step:0  val_loss:10.9264
  step:1  train_loss:10.9184
  step:2  train_loss:8.6834
  step:3  train_loss:7.7596
  step:4  train_loss:7.5281
  step:5  train_loss:7.2838

Actual results (4 GPU + grad_accum=2, 2025-12-18, avg of 2 runs):
  step 0 | val loss 10.98645 | train loss 10.98884
  step 1 | train loss 9.08150 ± 0.00002
  step 2 | train loss 7.77348 ± 0.00010
  step 3 | train loss 7.39948 ± 0.00020
  step 4 | train loss 7.35630 ± 0.00019
  step 5 | train loss 7.26357 ± 0.00033

Note: Differences from original due to different random seed and grad_accum.

Run with:
  torchrun --standalone --nproc_per_node=8 train_gpt_muon.py --run_name "muon_baseline"
"""
import os
import uuid
import argparse
import time

import torch
import wandb
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

from train_gpt_adam import GPT, GPTConfig, DistributedDataLoader, print0

# -----------------------------------------------------------------------------
# Hyperparameters (matching the 2024-10-11 Muon record)
# -----------------------------------------------------------------------------
B = 64                    # batch size per GPU
T = 1024                  # sequence length
TOTAL_BATCH_SIZE = 8 * 64 * 1024  # total tokens per step
NUM_ITERATIONS = 6200     # total training steps
LEARNING_RATE = 0.0036    # AdamW learning rate
MUON_LR = 0.1 * LEARNING_RATE  # Muon learning rate
WARMUP_ITERS = 0
WARMDOWN_ITERS = 1800
WEIGHT_DECAY = 0
VAL_LOSS_EVERY = 125
VAL_MAX_STEPS = 20

# Newton-Schulz coefficients
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5

# Data paths
TRAIN_FILES = "data/fineweb10B/fineweb_train_*.bin"
VAL_FILES = "data/fineweb10B/fineweb_val_*.bin"

# Reproducibility
SEED = 42

# -----------------------------------------------------------------------------
# Muon Optimizer
# -----------------------------------------------------------------------------

@torch.compile
def zeropower_via_newtonschulz5(G, steps=NS_STEPS, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = NS_COEFFS
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend_steps=NS_STEPS):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                # Handle grouped QKV parameters
                if g.size(0) == 3 * g.size(1):
                    g = torch.cat([zeropower_via_newtonschulz5(g1, steps=group['backend_steps'])
                                   for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5
                p.data.add_(g, alpha=-lr * scale)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="wandb run name")
    args = parser.parse_args()

    # DDP setup
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)

    print0(f"Running pytorch {torch.version.__version__}")
    print0(f"World size: {ddp_world_size}")

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Calculate gradient accumulation steps to match total batch size
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    # Mixed precision context
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # Data loaders
    train_loader = DistributedDataLoader(TRAIN_FILES, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(VAL_FILES, B, T, ddp_rank, ddp_world_size)

    # Model
    model_config = GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
    model = GPT(model_config)
    model = model.train().cuda()

    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True
    print0("Compiling model...")
    model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

    # Optimizers: AdamW for lm_head, Muon for transformer blocks
    optimizer_adamw = torch.optim.AdamW(
        raw_model.lm_head.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        fused=True
    )
    optimizer_muon = Muon(
        raw_model.transformer.h.parameters(),
        lr=MUON_LR,
        momentum=0.95
    )
    optimizers = [optimizer_adamw, optimizer_muon]
    base_lrs = [LEARNING_RATE, MUON_LR]

    # LR schedule: warmup -> constant -> warmdown
    def get_lr_scale(step):
        if step < WARMUP_ITERS:
            return (step + 1) / WARMUP_ITERS
        elif step < NUM_ITERATIONS - WARMDOWN_ITERS:
            return 1.0
        else:
            return (NUM_ITERATIONS - step) / WARMDOWN_ITERS

    # Logging
    run_id = str(uuid.uuid4())
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.log"
        print0(f"Logging to {logfile}")
        wandb.init(
            project="muon",
            name=args.run_name,
            config={
                "batch_size": B,
                "sequence_length": T,
                "total_batch_size": TOTAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "muon_lr": MUON_LR,
                "warmup_iters": WARMUP_ITERS,
                "warmdown_iters": WARMDOWN_ITERS,
                "weight_decay": WEIGHT_DECAY,
                "num_iterations": NUM_ITERATIONS,
                "grad_accum_steps": grad_accum_steps,
                "world_size": ddp_world_size,
                "seed": SEED,
            }
        )

    # Pre-fetch first batch
    x, y = train_loader.next_batch()

    # Training loop
    for step in range(NUM_ITERATIONS + 1):
        t0 = time.time()
        last_step = (step == NUM_ITERATIONS)

        # Validation
        if (VAL_LOSS_EVERY > 0 and step % VAL_LOSS_EVERY == 0) or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(VAL_MAX_STEPS):
                    x_val, y_val = val_loader.next_batch()
                    with ctx:
                        _, loss = model(x_val, y_val)
                    val_loss += loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss = val_loss.item() / VAL_MAX_STEPS
            print0(f"step {step} | val loss {val_loss:.6f}")
            if master_process:
                if logfile:
                    with open(logfile, "a") as f:
                        f.write(f"s:{step} tel:{val_loss:.6f}\n")
                wandb.log({"eval/loss": val_loss}, step=step)

        if last_step:
            break

        # Training step with gradient accumulation
        model.train()
        train_loss = 0.0
        for micro_step in range(grad_accum_steps):
            with ctx:
                _, loss = model(x, y)
            train_loss += loss.detach()
            x, y = train_loader.next_batch()
            if micro_step < grad_accum_steps - 1:
                with model.no_sync():
                    (loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()
        train_loss /= grad_accum_steps

        # Update learning rates and step
        lr_scale = get_lr_scale(step)
        for opt, base_lr in zip(optimizers, base_lrs):
            opt.param_groups[0]['lr'] = base_lr * lr_scale
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.time()

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()

        current_lr = LEARNING_RATE * lr_scale
        print0(f"step {step:4d}/{NUM_ITERATIONS} | train loss {lossf:.6f} | lr {current_lr:.2e} | {(t1-t0)*1000:.1f}ms")
        if master_process:
            if logfile:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} trl:{lossf:.6f}\n")
            wandb.log({
                "train/loss": lossf,
                "train/lr": current_lr,
            }, step=step)

    print0(f"Peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    if master_process:
        wandb.finish()
    dist.destroy_process_group()
