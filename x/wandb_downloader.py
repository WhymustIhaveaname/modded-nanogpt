#! /usr/bin/env python3

import argparse
import os
import re

import pandas as pd
import wandb


class WandbClient:
    def __init__(self, host: str, entity: str):
        os.environ["WANDB_BASE_URL"] = host
        self.entity = entity
        self.api = wandb.Api()

    def list_runs(self, project: str, regex: str | None = None) -> list:
        runs = list(self.api.runs(f"{self.entity}/{project}"))
        if regex:
            pattern = re.compile(regex)
            runs = [r for r in runs if pattern.search(r.name)]
        return runs

    def export_project_to_csv(
        self, project: str, regex: str, output_dir: str
    ) -> pd.DataFrame:
        runs = self.list_runs(project, regex=regex)
        safe_stub = re.sub(r"[^A-Za-z0-9._-]+", "_", regex).strip("._-") or "runs"
        output_path = os.path.join(output_dir, f"{safe_stub}.csv")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        all_histories: list[pd.DataFrame] = []

        for run in runs:
            history = pd.DataFrame(run.scan_history())
            history.insert(0, "tab", run.name)
            all_histories.append(history)

        if all_histories:
            combined_df = pd.concat(all_histories, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        combined_df.to_csv(output_path, index=False)
        print(f"Exported {len(runs)} runs to {output_path}")
        return combined_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", required=True)
    parser.add_argument("--regex", "-r", required=True)
    parser.add_argument("--output-dir", default="wandb_histories")
    parser.add_argument("--entity", default="light-robo")
    parser.add_argument("--host", default="https://ai.lrcorp.ai")
    args = parser.parse_args()

    client = WandbClient(host=args.host, entity=args.entity)
    os.makedirs(args.output_dir, exist_ok=True)

    combined_df = client.export_project_to_csv(
        args.project, args.regex, args.output_dir
    )
    print(f"Total rows: {len(combined_df)}")


if __name__ == "__main__":
    main()
