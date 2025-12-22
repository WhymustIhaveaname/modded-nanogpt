#! /usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def load_runs_from_csv(csv_path: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    return {name: group for name, group in df.groupby("tab")}


def plot_scaling(csv_path: str):
    runs = load_runs_from_csv(csv_path)
    output_path = csv_path.removesuffix(".csv") + ".png"

    plt.figure(figsize=(10, 6))
    for tab_name, group in runs.items():
        group = group.dropna(subset=["eval/loss"])
        group = group[group["eval/loss"] <= 7].sort_values("train/flops(PF-days)")
        plt.plot(
            group["train/flops(PF-days)"],
            group["eval/loss"],
            label=tab_name,
            marker="o",
            markersize=3,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("train/flops (PF-days)")
    plt.ylabel("eval/loss")
    plt.legend()
    plt.grid(True, which="major", alpha=0.3)
    plt.grid(True, which="minor", alpha=0.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Input CSV file")
    args = parser.parse_args()

    plot_scaling(args.csv)


if __name__ == "__main__":
    main()
