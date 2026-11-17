"""Command line interface.

Usage (from the project root):
    python -m src.cli generate     # (re)create data/customers.csv
    python -m src.cli run-broken   # run the broken pipeline, print metrics
    python -m src.cli run-fixed    # run the fixed pipeline, print metrics
    python -m src.cli detect       # flaw table: broken vs fixed
    python -m src.cli evaluate     # both pipelines + ablation, writes
                                   # results/metrics.json,
                                   # results/ablation.json and
                                   # results/plots/before_after.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

from src.ablation import print_ablation, run_ablation
from src.broken_pipeline import run_broken
from src.data import load_dataset, save_dataset
from src.detect import print_report
from src.fixed_pipeline import run_fixed
from src.plots import default_plot_path, make_before_after_plot


def _results_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "results"


def _print_metrics(title: str, metrics: dict) -> None:
    print(title)
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def cmd_generate(_args) -> None:
    path = save_dataset()
    print(f"Dataset written to {path}")


def cmd_run_broken(_args) -> None:
    res = run_broken()
    print(f"Broken pipeline — REPORTED accuracy: {res['reported_accuracy']:.4f}")
    print("(Only accuracy is reported. The honest numbers for the same run:)")
    _print_metrics("Honest metrics:", res["metrics"])


def cmd_run_fixed(_args) -> None:
    res = run_fixed()
    _print_metrics("Fixed pipeline — honest metrics:", res["metrics"])


def cmd_detect(_args) -> None:
    df = load_dataset()
    print_report(run_broken(df), run_fixed(df), df)


def cmd_evaluate(_args) -> None:
    df = load_dataset()
    broken = run_broken(df)
    fixed = run_fixed(df)
    rows = run_ablation(df)

    metrics_out = {
        "broken_reported_accuracy": broken["reported_accuracy"],
        "broken_honest_metrics": broken["metrics"],
        "fixed_metrics": fixed["metrics"],
    }
    ablation_out = [
        {"variant": row["variant"], "metrics": row["metrics"]} for row in rows
    ]

    out_dir = _results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    (out_dir / "ablation.json").write_text(json.dumps(ablation_out, indent=2))

    print(f"Broken — reported accuracy: {broken['reported_accuracy']:.4f}")
    print(f"Broken — honest F1: {broken['metrics']['f1']:.4f} "
          f"(balanced acc {broken['metrics']['balanced_accuracy']:.4f})")
    print(f"Fixed  — accuracy: {fixed['metrics']['accuracy']:.4f}, "
          f"F1: {fixed['metrics']['f1']:.4f}, "
          f"ROC-AUC: {fixed['metrics']['roc_auc']:.4f}")
    print()
    print_ablation(rows)
    print()
    plot_path = make_before_after_plot(
        broken["metrics"], fixed["metrics"], default_plot_path()
    )
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(f"Wrote {out_dir / 'ablation.json'}")
    print(f"Wrote {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debugging a Broken ML Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="(re)create data/customers.csv")
    sub.add_parser("run-broken", help="run the broken pipeline")
    sub.add_parser("run-fixed", help="run the fixed pipeline")
    sub.add_parser("detect", help="flaw table: broken vs fixed")
    sub.add_parser("evaluate", help="both pipelines + ablation, write results + plot")
    args = parser.parse_args()
    {
        "generate": cmd_generate,
        "run-broken": cmd_run_broken,
        "run-fixed": cmd_run_fixed,
        "detect": cmd_detect,
        "evaluate": cmd_evaluate,
    }[args.command](args)


if __name__ == "__main__":
    main()
