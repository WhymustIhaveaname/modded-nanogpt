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

    def fetch_project_data(
        self, project: str, regex: str | None = None
    ) -> pd.DataFrame:
        runs = self.list_runs(project, regex=regex)
        all_data = []
        for run in runs:
            run_data = {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "created_at": run.created_at,
            }
            for key, value in run.config.items():
                run_data[f"config/{key}"] = value
            for key, value in run.summary.items():
                if not key.startswith("_"):
                    run_data[f"summary/{key}"] = value

            history = run.history(samples=run.lastHistoryStep)
            for key in history.columns:
                series = history[key].dropna()
                if not series.empty:
                    run_data[f"history/{key}"] = series.iloc[-1]

            all_data.append(run_data)
        return pd.DataFrame(all_data)

    def fetch_run_histories(
        self, project: str, regex: str | None = None
    ) -> dict[str, pd.DataFrame]:
        runs = self.list_runs(project, regex=regex)
        histories = {}
        for run in runs:
            history = pd.DataFrame(run.scan_history())
            history["run_id"] = run.id
            history["run_name"] = run.name
            histories[run.name] = history
        return histories

    def export_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} rows to {output_path}")

    def export_histories_to_csv(
        self, histories: dict[str, pd.DataFrame], output_dir: str
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for run_name, history in histories.items():
            history.to_csv(
                os.path.join(output_dir, f"{run_name}_history.csv"), index=False
            )
            print(f"Exported {len(history)} rows to {run_name}_history.csv")

        combined_df = pd.concat(histories.values(), ignore_index=True)
        combined_path = os.path.join(output_dir, "combined_history.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Exported combined history ({len(combined_df)} rows) to {combined_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", required=True)
    parser.add_argument("--regex", "-r", required=True)
    parser.add_argument("--output-dir", default="wandb_histories")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Export full training history instead of summary",
    )
    parser.add_argument("--entity", default="light-robo")
    parser.add_argument("--host", default="https://ai.lrcorp.ai")
    args = parser.parse_args()

    client = WandbClient(host=args.host, entity=args.entity)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.history:
        histories = client.fetch_run_histories(args.project, regex=args.regex)
        client.export_histories_to_csv(histories, args.output_dir)
    else:
        df = client.fetch_project_data(args.project, regex=args.regex)
        print(df.T.to_string())
        client.export_to_csv(df, os.path.join(args.output_dir, f"{args.regex}.csv"))


if __name__ == "__main__":
    main()
