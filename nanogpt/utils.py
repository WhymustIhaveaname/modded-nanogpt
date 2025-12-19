"""Training utilities."""

import os
import torch


def print0(*args, **kwargs):
    """Print only on rank 0."""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
