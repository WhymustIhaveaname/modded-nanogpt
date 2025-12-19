"""
Reproduces records/track_1_short/2024-06-06_AdamW baseline.
Based on commit cc2e08a (2024-06-14).

Expected first few training losses (from the log):
  s:0 trl:10.965596
  s:1 trl:10.814037
  s:2 trl:10.543272
  s:3 trl:10.260427
  s:4 trl:10.041298

Actual results (4 GPU + grad_accum=2, 2025-12-18, avg of 2 runs):
  step 0 | val loss 10.98645 ± 0.000001 | train loss 10.98884 ± 0.000001
  step 1 | train loss 10.83568 ± 0.000003
  step 2 | train loss 10.57335 ± 0.000005
  step 3 | train loss 10.27135 ± 0.000006
  step 4 | train loss 10.04594 ± 0.000003

Run with:
  torchrun --standalone --nproc_per_node=8 train_gpt_adam.py
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

from nanogpt.model.gpt import GPT, GPTConfig
from nanogpt.data.distributed_loader import DistributedDataLoader
from nanogpt.utils import print0, set_seed

# -----------------------------------------------------------------------------
# Hyperparameters (matching the 2024-06-06_AdamW baseline)
# -----------------------------------------------------------------------------
B = 64  # batch size per GPU
T = 1024  # sequence length
TOTAL_BATCH_SIZE = 524288  # total tokens per step (must match original: 64 * 1024 * 8)
NUM_ITERATIONS = 9537  # total training steps (from log: last step is 9536)
LEARNING_RATE = 0.0018
WARMUP_ITERS = 256
WARMDOWN_ITERS = 2048
WEIGHT_DECAY = 0.1
VAL_LOSS_EVERY = 128
VAL_MAX_STEPS = 20

# Data paths
TRAIN_FILES = "data/fineweb10B/fineweb_train_*.bin"
VAL_FILES = "data/fineweb10B/fineweb_val_*.bin"

# Reproducibility
SEED = 42

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="wandb run name")
    args = parser.parse_args()

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

    # Set random seeds for reproducibility
    set_seed(SEED)

    # Calculate gradient accumulation steps to match total batch size
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, (
        f"TOTAL_BATCH_SIZE must be divisible by {tokens_per_fwdbwd}"
    )
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

    # Optimizer
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    # LR schedule: warmup -> constant -> warmdown
    def get_lr(step):
        if step < WARMUP_ITERS:
            return LEARNING_RATE * (step + 1) / WARMUP_ITERS
        elif step < NUM_ITERATIONS - WARMDOWN_ITERS:
            return LEARNING_RATE
        else:
            decay_ratio = (NUM_ITERATIONS - step) / WARMDOWN_ITERS
            return LEARNING_RATE * decay_ratio

    # Logging
    run_id = str(uuid.uuid4())
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.log"
        print0(f"Logging to {logfile}")
        # Initialize wandb
        wandb.init(
            project="muon",
            name=args.run_name,
            config={
                "batch_size": B,
                "sequence_length": T,
                "total_batch_size": TOTAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "warmup_iters": WARMUP_ITERS,
                "warmdown_iters": WARMDOWN_ITERS,
                "weight_decay": WEIGHT_DECAY,
                "num_iterations": NUM_ITERATIONS,
                "grad_accum_steps": grad_accum_steps,
                "world_size": ddp_world_size,
                "seed": SEED,
            },
        )

    # Pre-fetch first batch
    x, y = train_loader.next_batch()

    # Training loop
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
            # Scale loss for gradient accumulation
            (loss / grad_accum_steps).backward()
        train_loss /= grad_accum_steps

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.time()

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()

        print0(
            f"step {step:4d}/{NUM_ITERATIONS} | train loss {lossf:.6f} | lr {lr:.2e} | {(t1 - t0) * 1000:.1f}ms"
        )
        if master_process:
            if logfile:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} trl:{lossf:.6f}\n")
            wandb.log(
                {
                    "train/loss": lossf,
                    "train/lr": lr,
                },
                step=step,
            )

    print0(f"Peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    if master_process:
        wandb.finish()
    dist.destroy_process_group()
