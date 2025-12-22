"""
FLOPS counting utilities for GPT models.

Units:
- PF = 1e15 FLOPS (PetaFLOPS)
- PF-days = 1e15 * 86400 FLOPS-seconds
"""

import torch


def get_device_flops() -> float:
    """
    Get theoretical peak FLOPS for the current GPU in PF (PetaFLOPS).

    Returns:
        float: Peak FLOPS in PF (1e15 FLOPS)
    """
    device_name = torch.cuda.get_device_name()

    # BF16 Tensor Core FLOPS for common GPUs
    flops = float("inf")
    if "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    elif "B200" in device_name:
        flops = 2250e12
    elif "4090" in device_name:
        flops = 165e12
    elif "3090" in device_name:
        flops = 71e12

    return flops / 1e15  # Convert to PF


def estimate_gpt_flops(
    num_params: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    seq_len: int,
    tokens: int,
) -> float:
    """
    Estimate total FLOPS for GPT training (forward + backward).

    Based on the standard transformer FLOPS formula:
    - Linear layers: 6 * N * tokens (2 for fwd matmul, 2 for bwd input grad, 2 for bwd weight grad)
    - Attention: 12 * seq_len^2 * head_dim * n_head * n_layer (Q@K^T and attn@V, fwd+bwd)

    Args:
        num_params: Total number of model parameters
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        seq_len: Sequence length
        tokens: Total number of tokens processed

    Returns:
        float: Total FLOPS for training
    """
    # Linear layer FLOPS: 6 * N * tokens
    linear_flops = 6 * num_params * tokens

    # Attention FLOPS: 12 * seq_len^2 * head_dim * n_head * n_layer * num_seqs
    num_seqs = tokens // seq_len
    head_dim = n_embd // n_head
    # attn_flops = 12 * (seq_len ** 2) * head_dim * n_head * n_layer * num_seqs

    # return linear_flops + attn_flops
    return linear_flops


def flops_to_pf_days(flops: float) -> float:
    """
    Convert FLOPS to PF-days.

    Args:
        flops: Total FLOPS

    Returns:
        float: PF-days (PetaFLOPS * days)
    """
    pf = flops / 1e15
    pf_days = pf / 86400  # 86400 seconds per day
    return pf_days
