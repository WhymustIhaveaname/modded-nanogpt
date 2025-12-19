"""
Distributed data loading for GPT training.
"""

import glob
import numpy as np
import torch


def _peek_data_shard(filename):
    """Read header from data shard to get token count."""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    assert header[0] == 20240520, "magic number mismatch"
    assert header[1] == 1, "unsupported version"
    return header[2]


def _load_data_shard(filename):
    """Load tokens from a data shard file."""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520
        assert header[1] == 1
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok
    return tokens


class DistributedDataLoader:
    """
    Data loader for distributed training.

    Loads sharded binary token files and distributes batches across processes.
    """

    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        """
        Args:
            filename_pattern: Glob pattern for data files (e.g., "data/train_*.bin")
            B: Batch size per GPU
            T: Sequence length
            process_rank: Rank of current process
            num_processes: Total number of processes
        """
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"no files found: {filename_pattern}"
        self.reset()

    def reset(self):
        """Reset loader to beginning of first shard."""
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        """Advance to next shard."""
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        """Get next batch of data."""
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()
