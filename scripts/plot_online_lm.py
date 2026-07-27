"""Plot cumulative online language-modelling metrics from the curve CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "online_lm_curve.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "online_lm_log_ppl.png"
KV_BUDGET = 2000
CONFIG_ORDER = (
    "no_compression",
    "low_l2",
    "random",
    "high_l2",
    "snapkv",
)
REQUIRED_COLUMNS = {"config", "processed_tokens", "log_ppl"}


def load_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "Online LM curve CSV is missing required columns: " + ", ".join(missing)
        )

    frame = frame.copy()
    frame["processed_tokens"] = pd.to_numeric(
        frame["processed_tokens"],
        errors="raise",
    )
    frame["log_ppl"] = pd.to_numeric(frame["log_ppl"], errors="raise")
    if "next_token_accuracy" in frame:
        frame["next_token_accuracy"] = pd.to_numeric(
            frame["next_token_accuracy"],
            errors="raise",
        )
    return frame


def plot_curve(
    frame: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    output: Path,
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for config in CONFIG_ORDER:
        group = frame.loc[frame["config"] == config].sort_values("processed_tokens")
        if group.empty:
            continue
        ax.plot(
            group["processed_tokens"],
            group[value_column],
            marker="o",
            linewidth=2,
            label=config,
        )

    ax.axvline(
        KV_BUDGET,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="KV budget reached",
    )
    ax.set_xlabel("Processed tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Configuration")
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative metrics from the online LM benchmark."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--accuracy-output",
        type=Path,
        help="Optionally save a next-token accuracy curve.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    frame = load_results(args.input_csv)
    plot_curve(
        frame,
        value_column="log_ppl",
        ylabel="log PPL",
        output=args.output,
        title="Online language modelling",
    )
    print(f"Saved {args.output}")

    if args.accuracy_output is not None:
        if "next_token_accuracy" not in frame:
            raise ValueError(
                "Online LM curve CSV is missing required column: "
                "next_token_accuracy"
            )
        plot_curve(
            frame,
            value_column="next_token_accuracy",
            ylabel="Next-token accuracy",
            output=args.accuracy_output,
            title="Online next-token accuracy",
        )
        print(f"Saved {args.accuracy_output}")


if __name__ == "__main__":
    main()
