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
# Helper functions
# -----------------------------------------------------------------------------


def build_data(args, ddp_rank, ddp_world_size):
    """Build train and validation data loaders."""
    train_loader = DistributedDataLoader(TRAIN_FILES, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(VAL_FILES, B, T, ddp_rank, ddp_world_size)
    return train_loader, val_loader


def build_model(args, ddp_local_rank):
    model_config = GPTConfig(
        vocab_size=50257, n_layer=12, n_head=12, n_embd=768, tie_word_embeddings=True
    )
    model = GPT(model_config)

    if not model_config.tie_word_embeddings:
        with torch.no_grad():
            model.lm_head.weight.zero_()

        for name, param in model.named_parameters():
            print0(f"{name}: var={param.var().item():.6e}")

    model = model.train().cuda()

    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True
    print0("Compiling model...")
    model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_local_rank])

    return model


def build_optimizer_and_scheduler(args, model):
    """Build optimizers and LR scheduler."""
    raw_model = model.module
    ns_coeffs = (args.ns_a, args.ns_b, 0.701 - args.ns_a - args.ns_b)
    learning_rate = args.lr
    muon_lr = 0.1 * learning_rate
    decay_start = args.decay_start

    adamw_params = list(raw_model.lm_head.parameters())
    if not raw_model.config.tie_word_embeddings:
        adamw_params += list(raw_model.transformer.wte.parameters())
    optimizer_adamw = torch.optim.AdamW(
        adamw_params,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    optimizer_muon = Muon(
        raw_model.transformer.h.parameters(),
        lr=muon_lr,
        momentum=0.95,
        ns_coeffs=ns_coeffs,
        weight_decay=1.2,
    )
    optimizers = [optimizer_adamw, optimizer_muon]
    base_lrs = [learning_rate, muon_lr]

    # LR schedule: constant + linear decay to 0
    def get_lr_scale(step):
        if step < decay_start:
            return 1.0
        else:
            return (NUM_ITERATIONS - step) / (NUM_ITERATIONS - decay_start)

    return optimizers, base_lrs, get_lr_scale


def validate(model, val_loader, ctx):
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
    return val_loss


def train_one_step(
    model, train_loader, optimizers, base_lrs, lr_scale, grad_accum_steps, ctx
):
    """Execute one training step with gradient accumulation."""
    model.train()
    train_loss = 0.0
    x, y = train_loader.next_batch()

    for micro_step in range(grad_accum_steps):
        with ctx:
            _, loss = model(x, y)
        train_loss += loss.detach()
        if micro_step < grad_accum_steps - 1:
            x, y = train_loader.next_batch()
            with model.no_sync():
                (loss / grad_accum_steps).backward()
        else:
            (loss / grad_accum_steps).backward()
    train_loss /= grad_accum_steps

    # Compute gradient norm before optimizer step
    grad_norm = (
        sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    )

    # Update learning rates and step
    for opt, base_lr in zip(optimizers, base_lrs):
        opt.param_groups[0]["lr"] = base_lr * lr_scale
        opt.step()

    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    return train_loss, grad_norm


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="run name prefix")
    parser.add_argument("--ns_a", type=float, default=3.4445)
    parser.add_argument("--ns_b", type=float, default=-4.7750)
    parser.add_argument("--lr", type=float, default=0.0036)
    parser.add_argument("--decay_start", type=int, default=4000)  # was 4400
    args = parser.parse_args()

    # Derived values
    args.ns_c = 0.701 - args.ns_a - args.ns_b
    args.run_name_full = (
        f"{args.run_name}_a{args.ns_a:.3f}_b{args.ns_b:.3f}_lr{args.lr:.4f}"
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
    print0(f"NS coefficients: a={args.ns_a}, b={args.ns_b}, c={args.ns_c}")
    print0(f"Learning rate: {args.lr}")
    print0(f"Run name: {args.run_name_full}")

    # Set random seeds for reproducibility
    set_seed(SEED)

    # Calculate gradient accumulation steps to match total batch size
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    # Mixed precision context
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    train_loader, val_loader = build_data(args, ddp_rank, ddp_world_size)
    model = build_model(args, ddp_local_rank)
    optimizers, base_lrs, get_lr_scale = build_optimizer_and_scheduler(args, model)

    # Verify optimizer parameter coverage (all ranks)
    opt0_params = sum(
        p.numel() for g in optimizers[0].param_groups for p in g["params"]
    )
    opt1_params = sum(
        p.numel() for g in optimizers[1].param_groups for p in g["params"]
    )
    model_params = sum(p.numel() for p in model.module.parameters())
    assert opt0_params + opt1_params == model_params, (
        f"optimizer params {opt0_params + opt1_params} != model params {model_params}"
    )

    # Logging
    run_id = str(uuid.uuid4())
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.log"
        print0(f"Logging to {logfile}")
        # Get actual values from optimizers for logging
        adamw_lr = optimizers[0].param_groups[0]["lr"]
        muon_lr = optimizers[1].param_groups[0]["lr"]
        muon_wd = optimizers[1].param_groups[0]["weight_decay"]
        wandb.init(
            project="muon",
            name=args.run_name_full,
            config={
                "batch_size": B,
                "sequence_length": T,
                "total_batch_size": TOTAL_BATCH_SIZE,
                "learning_rate": adamw_lr,
                "muon_lr": muon_lr,
                "decay_start": args.decay_start,
                "muon_weight_decay": muon_wd,
                "num_iterations": NUM_ITERATIONS,
                "grad_accum_steps": grad_accum_steps,
                "world_size": ddp_world_size,
                "seed": SEED,
                "ns_a": args.ns_a,
                "ns_b": args.ns_b,
                "ns_c": args.ns_c,
                "opt0_params": opt0_params,
                "opt1_params": opt1_params,
                "model_params": model_params,
            },
        )

    # Training loop
    start_time = time.time()
    for step in range(NUM_ITERATIONS + 1):
        t0 = time.time()
        last_step = step == NUM_ITERATIONS

        if (VAL_LOSS_EVERY > 0 and step % VAL_LOSS_EVERY == 0) or last_step:
            val_loss = validate(model, val_loader, ctx)
            print0(f"step {step} | val loss {val_loss:.6f}")
            if master_process:
                if logfile:
                    with open(logfile, "a") as f:
                        f.write(f"s:{step} tel:{val_loss:.6f}\n")
                wandb.log({"eval/loss": val_loss}, step=step)

        if last_step:
            break

        # Training step
        lr_scale = get_lr_scale(step)
        current_lr = args.lr * lr_scale
        train_loss, grad_norm = train_one_step(
            model, train_loader, optimizers, base_lrs, lr_scale, grad_accum_steps, ctx
        )

        torch.cuda.synchronize()
        t1 = time.time()

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()
        if math.isnan(lossf):
            print0(f"step {step} | loss is NaN, stopping training")
            break
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
