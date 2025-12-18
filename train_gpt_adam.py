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
import sys
import uuid
import math
import glob
from dataclasses import dataclass

import argparse
import time

import numpy as np
import torch
import wandb
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# -----------------------------------------------------------------------------
# Hyperparameters (matching the 2024-06-06_AdamW baseline)
# -----------------------------------------------------------------------------
B = 64                    # batch size per GPU
T = 1024                  # sequence length
TOTAL_BATCH_SIZE = 524288 # total tokens per step (must match original: 64 * 1024 * 8)
NUM_ITERATIONS = 9537     # total training steps (from log: last step is 9536)
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
# PyTorch nn.Module definitions for the GPT-2 model
# -----------------------------------------------------------------------------

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    assert header[0] == 20240520, "magic number mismatch"
    assert header[1] == 1, "unsupported version"
    return header[2]

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520
        assert header[1] == 1
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"no files found: {filename_pattern}"
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="wandb run name")
    args = parser.parse_args()

    # DDP setup
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
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
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, f"TOTAL_BATCH_SIZE must be divisible by {tokens_per_fwdbwd}"
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

    # Optimizer
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
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
            # Scale loss for gradient accumulation
            (loss / grad_accum_steps).backward()
        train_loss /= grad_accum_steps

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.time()

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()

        print0(f"step {step:4d}/{NUM_ITERATIONS} | train loss {lossf:.6f} | lr {lr:.2e} | {(t1-t0)*1000:.1f}ms")
        if master_process:
            if logfile:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} trl:{lossf:.6f}\n")
            wandb.log({
                "train/loss": lossf,
                "train/lr": lr,
            }, step=step)

    print0(f"Peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    if master_process:
        wandb.finish()
    destroy_process_group()
