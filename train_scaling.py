"""
Scaling Law Experiments with Hydra Configuration.

Sweeps over different model configurations (n_layer, n_head) and trains each one.
Uses AdamW optimizer.

Run with:
  torchrun --standalone --nproc_per_node=8 train_scaling.py
  torchrun --standalone --nproc_per_node=8 train_scaling.py 'model.configs=[[4,4],[8,8]]'
"""

import os
import time
from dataclasses import dataclass
from typing import List

import torch
import wandb
import torch.distributed as dist
import torch._inductor.config as inductor_config
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf
import hydra

from nanogpt.model.gpt import GPT, GPTConfig
from nanogpt.data.distributed_loader import DistributedDataLoader
from nanogpt.flops import get_device_flops, estimate_gpt_flops, flops_to_pf_days
from nanogpt.optimizer.muon import Muon
from nanogpt.utils import print0, set_seed


@dataclass
class ModelConfig:
    """Single model configuration for training."""

    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int


def get_model_configs(cfg: DictConfig) -> List[ModelConfig]:
    """Generate model configurations from the config list."""
    configs = []
    for n_layer, n_head in cfg.model.configs:
        n_embd = n_head * cfg.model.head_dim
        configs.append(
            ModelConfig(
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                vocab_size=cfg.model.vocab_size,
            )
        )
    return configs


def get_lr_scale(step: int, cfg: DictConfig) -> float:
    """Learning rate schedule: warmup -> constant -> warmdown. Returns scale [0, 1]."""
    warmup = cfg.optimizer.warmup_iters
    warmdown = cfg.optimizer.warmdown_iters
    total = cfg.optimizer.num_iterations

    if step < warmup:
        return (step + 1) / warmup
    elif step < total - warmdown:
        return 1.0
    else:
        return (total - step) / warmdown


def train_single_model(
    model_cfg: ModelConfig,
    cfg: DictConfig,
    ddp_rank: int,
    ddp_local_rank: int,
    ddp_world_size: int,
    device: str,
    master_process: bool,
) -> dict:
    """Train a single model configuration and return final metrics."""

    run_name = f"{cfg.wandb.run_name}_l{model_cfg.n_layer}h{model_cfg.n_head}"
    print0(f"\n{'=' * 60}")
    print0(f"Training: {run_name}")
    print0(
        f"  n_layer={model_cfg.n_layer}, n_head={model_cfg.n_head}, n_embd={model_cfg.n_embd}"
    )
    print0(f"{'=' * 60}")

    # Reset seed for each model
    set_seed(cfg.seed)

    # Calculate gradient accumulation steps
    B = cfg.training.batch_size
    T = cfg.training.sequence_length
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert cfg.training.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = cfg.training.total_batch_size // tokens_per_fwdbwd

    # Data loaders
    train_loader = DistributedDataLoader(
        cfg.data.train_files, B, T, ddp_rank, ddp_world_size
    )
    val_loader = DistributedDataLoader(
        cfg.data.val_files, B, T, ddp_rank, ddp_world_size
    )

    # Model
    gpt_config = GPTConfig(
        vocab_size=model_cfg.vocab_size,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_embd=model_cfg.n_embd,
        tie_word_embeddings=cfg.model.tie_word_embeddings,
    )
    model = GPT(gpt_config)

    model = model.train().cuda()

    # Compile and wrap with DDP
    if hasattr(inductor_config, "coordinate_descent_tuning"):
        inductor_config.coordinate_descent_tuning = True
    print0("Compiling model...")
    model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

    # Optimizer
    if cfg.optimizer.type == "AdamW":
        optimizers = [
            torch.optim.AdamW(
                raw_model.parameters(),
                lr=cfg.optimizer.learning_rate,
                betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
                weight_decay=cfg.training.weight_decay,
            )
        ]
        base_lrs = [cfg.optimizer.learning_rate]
    elif cfg.optimizer.type == "Muon":
        # AdamW for lm_head (and wte if not tied)
        adamw_params = list(raw_model.lm_head.parameters())
        if not cfg.model.tie_word_embeddings:
            adamw_params += list(raw_model.transformer.wte.parameters())
        adamw_lr = cfg.optimizer.learning_rate
        muon_lr = cfg.optimizer.learning_rate * cfg.optimizer.muon_lr_scale
        optimizer_adamw = torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            weight_decay=0.0,
            fused=True,
        )
        optimizer_muon = Muon(
            raw_model.transformer.h.parameters(),
            lr=muon_lr,
            momentum=cfg.optimizer.muon_momentum,
        )
        optimizers = [optimizer_adamw, optimizer_muon]
        base_lrs = [adamw_lr, muon_lr]
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.optimizer.type}")

    # Verify all parameters are covered by optimizers
    opt_params = sum(
        p.numel() for opt in optimizers for g in opt.param_groups for p in g["params"]
    )
    model_params = raw_model.num_params
    assert opt_params == model_params, (
        f"Optimizer params ({opt_params}) != model params ({model_params})"
    )

    # Mixed precision context
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Logging
    if master_process and cfg.wandb.enabled:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config.update(
            {
                "n_layer": model_cfg.n_layer,
                "n_head": model_cfg.n_head,
                "n_embd": model_cfg.n_embd,
                "num_params": raw_model.num_params,
                "num_params_no_embd": raw_model.num_params_no_embd,
                "grad_accum_steps": grad_accum_steps,
                "world_size": ddp_world_size,
            }
        )
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config=wandb_config,
            reinit=True,
        )

    # Pre-fetch first batch
    x, y = train_loader.next_batch()

    # FLOPS tracking
    flops_promised = get_device_flops() * ddp_world_size  # PF
    total_flops = 0  # Accumulate total FLOPS

    # Training loop
    num_iterations = cfg.optimizer.num_iterations
    val_loss_every = cfg.training.val_loss_every
    val_max_steps = cfg.training.val_max_steps
    final_val_loss = None
    final_train_loss = None
    start_time = time.time()

    for step in range(num_iterations + 1):
        t0 = time.time()
        last_step = step == num_iterations

        # Validation
        if (val_loss_every > 0 and step % val_loss_every == 0) or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    with ctx:
                        _, loss = model(x_val, y_val)
                    val_loss += loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss = val_loss.item() / val_max_steps

            print0(f"[{run_name}] step {step} | val loss {val_loss:.6f}")
            final_val_loss = val_loss

            if master_process and cfg.wandb.enabled:
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
            (loss / grad_accum_steps).backward()
        train_loss /= grad_accum_steps

        # Compute gradient norm before optimizer step
        grad_norm = (
            sum(
                p.grad.norm() ** 2 for p in raw_model.parameters() if p.grad is not None
            )
            ** 0.5
        )

        lr_scale = get_lr_scale(step, cfg)
        for opt, base_lr in zip(optimizers, base_lrs):
            opt.param_groups[0]["lr"] = base_lr * lr_scale
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        lr = cfg.optimizer.learning_rate * lr_scale  # for logging

        torch.cuda.synchronize()
        t1 = time.time()
        delta_time = t1 - t0

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()
        final_train_loss = lossf

        if step % 100 == 0:  # Print less frequently for scaling experiments
            print0(
                f"[{run_name}] step {step:4d}/{num_iterations} | "
                f"loss {lossf:.6f} | lr {lr:.2e} | {(t1 - t0) * 1000:.1f}ms"
            )

        if master_process and cfg.wandb.enabled:
            # Calculate FLOPS for this step (use non-embedding params for scaling law)
            step_flops = estimate_gpt_flops(
                num_params=raw_model.num_params_no_embd,
                n_layer=model_cfg.n_layer,
                n_head=model_cfg.n_head,
                n_embd=model_cfg.n_embd,
                seq_len=T,
                tokens=cfg.training.total_batch_size,
            )
            total_flops += step_flops
            flops_achieved_pf_days = flops_to_pf_days(total_flops)
            step_pflops = step_flops / delta_time / 1e15  # PFLOPS for this step
            elapsed_min = (time.time() - start_time) / 60
            wandb.log(
                {
                    "train/loss": lossf,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm.item(),
                    "train/flops(PF-days)": flops_achieved_pf_days,
                    "timing/elapsed_time_min": elapsed_min,
                    "timing/flops_GPUs(PF)": flops_promised,
                    "timing/flops(PF)": step_pflops,
                },
                step=step,
            )

    # Cleanup
    if master_process and cfg.wandb.enabled:
        wandb.finish()

    # Free memory
    del model, optimizers, train_loader, val_loader
    torch.cuda.empty_cache()

    return {
        "run_name": run_name,
        "n_layer": model_cfg.n_layer,
        "n_head": model_cfg.n_head,
        "n_embd": model_cfg.n_embd,
        "final_val_loss": final_val_loss,
        "final_train_loss": final_train_loss,
    }


@hydra.main(version_base=None, config_path="configs", config_name="scaling")
def main(cfg: DictConfig):
    """Main entry point for scaling law experiments."""

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
    print0(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Generate all model configurations
    model_configs = get_model_configs(cfg)
    print0(f"\nWill train {len(model_configs)} model configurations:")
    for mc in model_configs:
        print0(f"  - L{mc.n_layer}_H{mc.n_head}_E{mc.n_embd}")

    # Train each model
    results = []
    for i, model_cfg in enumerate(model_configs):
        print0(f"\n[{i + 1}/{len(model_configs)}] Starting training...")
        result = train_single_model(
            model_cfg,
            cfg,
            ddp_rank,
            ddp_local_rank,
            ddp_world_size,
            device,
            master_process,
        )
        results.append(result)

        # Sync all processes before next model
        dist.barrier()

    # Print summary
    if master_process:
        print0("\n" + "=" * 60)
        print0("SCALING LAW EXPERIMENT SUMMARY")
        print0("=" * 60)
        print0(
            f"{'Model':<20} {'n_layer':>8} {'n_head':>8} {'n_embd':>8} {'Val Loss':>12}"
        )
        print0("-" * 60)
        for r in results:
            print0(
                f"{r['run_name']:<20} {r['n_layer']:>8} {r['n_head']:>8} "
                f"{r['n_embd']:>8} {r['final_val_loss']:>12.6f}"
            )
        print0("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
