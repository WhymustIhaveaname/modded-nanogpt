"""
Calculate 2-gram conditional entropy of the Fineweb dataset.
"""

import glob
import numpy as np
from collections import Counter, defaultdict
import argparse
from tqdm import tqdm


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, f"token count mismatch: {len(tokens)} vs {ntok}"
    return tokens


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    probs = counts / total
    return float(-(probs * np.log(probs)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/fineweb10B")
    parser.add_argument("--pattern", type=str, default="fineweb_train_*.bin")
    args = parser.parse_args()

    file_pattern = f"{args.data_dir}/{args.pattern}"
    files = sorted(glob.glob(file_pattern))
    print(f"Found {len(files)} data files")

    # ctx -> Counter(next_token)
    ctx2next: dict[int, Counter] = defaultdict(Counter)

    for filename in tqdm(files, desc="Counting"):
        tokens = _load_data_shard(filename)
        seq = tokens.tolist()
        for i in range(len(seq) - 1):
            ctx = seq[i]
            nxt = seq[i + 1]
            ctx2next[ctx][nxt] += 1

    # Calculate per-context entropy and weighted mean
    total_count = 0
    weighted_entropy_sum = 0.0

    for ctx, nxt_counter in tqdm(ctx2next.items(), desc="Computing entropy"):
        counts = np.fromiter(nxt_counter.values(), dtype=np.float64)
        ctx_count = counts.sum()
        H = _entropy_from_counts(counts)
        weighted_entropy_sum += H * ctx_count
        total_count += ctx_count

    weighted_mean = weighted_entropy_sum / total_count

    print(f"\n2-gram conditional entropy: {weighted_mean:.2f}")


if __name__ == "__main__":
    main()
