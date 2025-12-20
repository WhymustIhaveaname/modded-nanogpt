#! /usr/bin/env python3

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(input_dir: str, regex: str):
    pattern = re.compile(regex)
    csv_files = [
        f
        for f in glob.glob(os.path.join(input_dir, "*.csv"))
        if pattern.search(os.path.basename(f))
    ]

    dfs = {}
    for csv_file in sorted(csv_files):
        df = (
            pd.read_csv(csv_file)
            .sort_values("config/muon_lr")
            .set_index("config/muon_lr")
        )
        label = os.path.basename(csv_file).replace(".csv", "")
        dfs[label] = df["summary/eval/loss"]

    plt.figure(figsize=(10, 6))
    for label, series in dfs.items():
        plt.plot(series.index, series, marker="o", label=label)
    plt.xlabel("muon_lr")
    plt.ylabel("eval/loss")
    plt.legend()
    plt.grid(True)
    output = os.path.join(input_dir, f"{regex}.png")
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to {output}")

    labels = list(dfs.keys())
    print("\n=== Pairwise comparison (row - col, negative = row is better) ===")
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1 :]:
            common_idx = dfs[label_a].index.intersection(dfs[label_b].index)
            diff = dfs[label_a].loc[common_idx] - dfs[label_b].loc[common_idx]
            mean, std, n = diff.mean(), np.std(diff), len(diff)
            sigma = mean / std * np.sqrt(n)
            print(
                f"{label_a} vs {label_b}: mean={mean:.6f}, std={std:.6f}, sigma={sigma:.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="wandb_histories")
    parser.add_argument("--regex", "-r", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.regex)
