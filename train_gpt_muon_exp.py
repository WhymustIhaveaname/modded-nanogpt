"""
Experimental Muon training script with configurable Newton-Schulz coefficients.

Differences from train_gpt_muon.py:
1. abc (Newton-Schulz coefficients) configurable via --ns_a, --ns_b, --ns_c or NS_A, NS_B, NS_C env vars
2. LR schedule: constant + linear decay to 0, with configurable --decay_start
3. run_name auto-generated to reflect abc and lr

Run with:
  torchrun --standalone --nproc_per_node=8 train_gpt_muon_exp.py --ns_a 3.4445 --ns_b -4.7750 --ns_c 2.0315
"""

import os
import uuid
import argparse
import time
import math

import torch
import wandb
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

from nanogpt.model.gpt import GPT, GPTConfig
from nanogpt.data.distributed_loader import DistributedDataLoader
from nanogpt.optimizer.muon import Muon
from nanogpt.utils import print0, set_seed

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
B = 64  # batch size per GPU
T = 1024  # sequence length
TOTAL_BATCH_SIZE = 8 * 64 * 1024  # total tokens per step
NUM_ITERATIONS = 6200  # total training steps
VAL_LOSS_EVERY = 125
VAL_MAX_STEPS = 20

# Data paths
TRAIN_FILES = "data/fineweb10B/fineweb_train_*.bin"
VAL_FILES = "data/fineweb10B/fineweb_val_*.bin"

# Reproducibility
SEED = 42

# my Note -- possible ab to try, c=0.701-a-b
# Jordan: 3.4445 -4.7750
# uniform spectrum: 3.577 -4.912
# power spectrum: 3.565 -5.225
# power spectrum: 3.15  -4.5
# MP spectrum (1280): 3.55 -5.34
# MP spectrum (768): 3.55 -5.35
# MP spectrum v2: 3.32 -5.37

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="run name prefix")
    parser.add_argument(
        "--ns_a", type=float, default=float(os.environ.get("NS_A", 3.4445))
    )
    parser.add_argument(
        "--ns_b", type=float, default=float(os.environ.get("NS_B", -4.7750))
    )
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", 0.0036)))
    # parser.add_argument("--decay_start", type=int, default=int(os.environ.get("DECAY_START", 4400)))
    parser.add_argument("--L0", type=float, default=float(os.environ.get("L0", 4.0)))
    args = parser.parse_args()

    ns_a, ns_b = args.ns_a, args.ns_b
    ns_c = 0.701 - ns_a - ns_b
    learning_rate = args.lr
    # decay_start = args.decay_start
    L0 = args.L0
    muon_lr = 0.1 * learning_rate

    # Auto-generate run_name
    run_name = (
        f"{args.run_name}_a{ns_a:.3f}_b{ns_b:.3f}_lr{learning_rate:.4f}_L0{L0:.1f}"
    )

    # DDP setup
    assert torch.cuda.is_available()
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    print0(f"Running pytorch {torch.version.__version__}")
    print0(f"World size: {ddp_world_size}")
    print0(f"NS coefficients: a={ns_a}, b={ns_b}, c={ns_c}")
    print0(f"Learning rate: {learning_rate}, Muon LR: {muon_lr}")
    print0(f"Run name: {run_name}")

    # Set random seeds for reproducibility
    set_seed(SEED)

    # Calculate gradient accumulation steps to match total batch size
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    # Mixed precision context
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

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

    # Optimizers: AdamW for lm_head (wd=0), Muon for transformer blocks (wd=1.2)
    optimizer_adamw = torch.optim.AdamW(
        raw_model.lm_head.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    optimizer_muon = Muon(
        raw_model.transformer.h.parameters(),
        lr=muon_lr,
        momentum=0.95,
        ns_coeffs=(ns_a, ns_b, ns_c),
        weight_decay=1.2,
    )
    optimizers = [optimizer_adamw, optimizer_muon]
    base_lrs = [learning_rate, muon_lr]

    # # LR schedule: constant + linear decay to 0 (ORIGINAL - commented out for experiment)
    # def get_lr_scale(step):
    #     if step < decay_start:
    #         return 1.0
    #     else:
    #         return (NUM_ITERATIONS - step) / (NUM_ITERATIONS - decay_start)

    # LR schedule: loss-based (EXPERIMENTAL)
    # lr ∝ L^(1/0.21), capped at lr0 when L >= L0
    def get_lr_from_loss(current_loss, lr0, L0):
        if current_loss is None or current_loss >= L0:
            return lr0
        else:
            return lr0 * (current_loss / L0) ** (1 / 0.21)

    # Logging
    run_id = str(uuid.uuid4())
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.log"
        print0(f"Logging to {logfile}")
        wandb.init(
            project="muon",
            name=run_name,
            config={
                "batch_size": B,
                "sequence_length": T,
                "total_batch_size": TOTAL_BATCH_SIZE,
                "learning_rate": learning_rate,
                "muon_lr": muon_lr,
                # "decay_start": decay_start,
                "L0": L0,
                "muon_weight_decay": 1.2,
                "num_iterations": NUM_ITERATIONS,
                "grad_accum_steps": grad_accum_steps,
                "world_size": ddp_world_size,
                "seed": SEED,
                "ns_a": ns_a,
                "ns_b": ns_b,
                "ns_c": ns_c,
            },
        )

    # Pre-fetch first batch
    x, y = train_loader.next_batch()

    # Training loop
    start_time = time.time()
    prev_loss = None  # For loss-based LR scheduling
    for step in range(NUM_ITERATIONS + 1):
        t0 = time.time()
        last_step = step == NUM_ITERATIONS

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

        # Compute gradient norm before optimizer step
        grad_norm = (
            sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
            ** 0.5
        )

        # Update learning rates and step
        # # ORIGINAL: step-based lr schedule
        # lr_scale = get_lr_scale(step)
        # for opt, base_lr in zip(optimizers, base_lrs):
        #     opt.param_groups[0]['lr'] = base_lr * lr_scale
        #     opt.step()

        # EXPERIMENTAL: loss-based lr schedule
        current_lr = get_lr_from_loss(prev_loss, learning_rate, L0)
        current_muon_lr = 0.1 * current_lr
        optimizer_adamw.param_groups[0]["lr"] = current_lr
        optimizer_muon.param_groups[0]["lr"] = current_muon_lr
        for opt in optimizers:
            opt.step()

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.time()

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()
        if math.isnan(lossf):
            print0(f"step {step} | loss is NaN, stopping training")
            break
        prev_loss = lossf  # Update prev_loss for next step
        print0(
            f"step {step:4d}/{NUM_ITERATIONS} | train loss {lossf:.6f} | lr {current_lr:.2e} | {(t1 - t0) * 1000:.1f}ms"
        )
        if master_process:
            if logfile:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} trl:{lossf:.6f}\n")
            elapsed_min = (time.time() - start_time) / 60
            wandb.log(
                {
                    "train/loss": lossf,
                    "train/lr": current_lr,
                    "train/grad_norm": grad_norm.item(),
                    "train/elapsed_time_min": elapsed_min,
                },
                step=step,
            )

    print0(f"Peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    if master_process:
        wandb.finish()
    dist.destroy_process_group()
