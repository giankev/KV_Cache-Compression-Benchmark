"""Plot exact-match passkey accuracy against context length from raw CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {"config", "context_length", "correct"}


def _correct_to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    normalized = series.astype(str).str.strip().str.lower()
    converted = normalized.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "1.0": 1.0,
            "0.0": 0.0,
        }
    )
    if converted.isna().any():
        invalid = sorted(normalized[converted.isna()].unique())
        raise ValueError(f"correct contains unsupported values: {invalid}")
    return converted


def load_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "Retrieval raw CSV is missing required columns: " + ", ".join(missing)
        )
    frame = frame.copy()
    frame["context_length"] = pd.to_numeric(
        frame["context_length"],
        errors="raise",
    )
    frame["correct"] = _correct_to_numeric(frame["correct"])
    return frame


def build_accuracy_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["config", "context_length"], sort=False, as_index=False)
        .agg(num_examples=("correct", "size"), accuracy=("correct", "mean"))
        .sort_values(["config", "context_length"])
        .reset_index(drop=True)
    )


def plot_accuracy(
    table: pd.DataFrame,
    output: Path,
    title: str | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for config, group in table.groupby("config", sort=False):
        ordered = group.sort_values("context_length")
        ax.plot(
            ordered["context_length"],
            ordered["accuracy"],
            marker="o",
            linewidth=2,
            label=str(config),
        )

    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or "Passkey retrieval accuracy by context length")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Configuration")
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot passkey accuracy against context length."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    table = build_accuracy_table(load_results(args.input_csv))
    print(table.to_string(index=False))
    plot_accuracy(table, args.output, args.title)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
