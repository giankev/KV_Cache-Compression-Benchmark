from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {"config", "depth_target", "correct"}
KNOWN_CONFIG_ORDER = [
    "no_compression",
    "low_l2_keep50",
    "low_l2_keep10",
    "random_keep50",
    "high_l2_keep50",
]


def _validate_columns(results: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(results.columns))
    if missing:
        raise ValueError(
            "Passkey raw CSV is missing required columns: " + ", ".join(missing)
        )


def _correct_as_binary(correct: pd.Series) -> pd.Series:
    text_values = correct.astype("string").str.strip().str.lower()
    text_binary = text_values.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "1.0": 1.0,
            "0.0": 0.0,
        }
    )
    numeric = pd.to_numeric(correct, errors="coerce")
    binary = text_binary.fillna(numeric)

    invalid = binary.isna() | ~binary.isin([0.0, 1.0])
    if invalid.any():
        examples = correct[invalid].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            "Column 'correct' must contain boolean or 0/1 values; "
            f"invalid values include: {examples}"
        )
    return binary.astype(float)


def load_results(input_csv: Path) -> pd.DataFrame:
    """Load passkey raw results and validate their required columns."""

    results = pd.read_csv(input_csv)
    _validate_columns(results)
    return results


def build_accuracy_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Return mean accuracy indexed by configuration and numeric depth."""

    _validate_columns(results)
    prepared = results.loc[:, ["config", "depth_target", "correct"]].copy()

    if prepared["config"].isna().any():
        raise ValueError("Column 'config' must not contain missing values")
    prepared["config"] = prepared["config"].astype(str)
    prepared["depth_target"] = pd.to_numeric(
        prepared["depth_target"], errors="coerce"
    )
    if prepared["depth_target"].isna().any():
        raise ValueError("Column 'depth_target' must contain numeric values")
    prepared["correct"] = _correct_as_binary(prepared["correct"])

    grouped = (
        prepared.groupby(["config", "depth_target"], sort=False)["correct"]
        .mean()
        .unstack("depth_target")
    )
    if grouped.empty:
        raise ValueError("The passkey raw CSV contains no result rows")

    present_configs = set(grouped.index)
    known_configs = [
        config for config in KNOWN_CONFIG_ORDER if config in present_configs
    ]
    extra_configs = sorted(present_configs - set(KNOWN_CONFIG_ORDER))
    depths = sorted(float(depth) for depth in grouped.columns)

    matrix = grouped.reindex(
        index=known_configs + extra_configs,
        columns=depths,
    )
    matrix.index.name = "config"
    matrix.columns.name = "depth_target"
    return matrix


def _format_depth(depth: float) -> str:
    return f"{depth * 100:g}%"


def plot_heatmap(
    accuracy_matrix: pd.DataFrame,
    output: Path,
    title: str | None = None,
) -> None:
    """Render and save an annotated passkey accuracy heatmap."""

    width = max(7.0, 1.6 * len(accuracy_matrix.columns) + 4.0)
    height = max(4.5, 0.65 * len(accuracy_matrix.index) + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))

    color_map = plt.colormaps["viridis"].copy()
    color_map.set_bad(color="#d9d9d9")
    image = ax.imshow(
        accuracy_matrix.to_numpy(dtype=float),
        aspect="auto",
        cmap=color_map,
        vmin=0.0,
        vmax=1.0,
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Accuracy")

    ax.set_xticks(range(len(accuracy_matrix.columns)))
    ax.set_xticklabels(
        [_format_depth(float(depth)) for depth in accuracy_matrix.columns]
    )
    ax.set_yticks(range(len(accuracy_matrix.index)))
    ax.set_yticklabels(accuracy_matrix.index)
    ax.set_xlabel("Passkey depth")
    ax.set_ylabel("Compression configuration")
    ax.set_title(title or "Passkey retrieval accuracy by depth")

    for row_idx, config in enumerate(accuracy_matrix.index):
        for column_idx, depth in enumerate(accuracy_matrix.columns):
            value = accuracy_matrix.loc[config, depth]
            if pd.isna(value):
                label = "N/A"
                text_color = "black"
            else:
                label = f"{value:.2f}"
                text_color = "white" if value < 0.55 else "black"
            ax.text(
                column_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xticks(
        [position - 0.5 for position in range(1, len(accuracy_matrix.columns))],
        minor=True,
    )
    ax.set_yticks(
        [position - 0.5 for position in range(1, len(accuracy_matrix.index))],
        minor=True,
    )
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a heatmap from raw passkey benchmark results."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = load_results(args.input_csv)
    accuracy_matrix = build_accuracy_matrix(results)

    print("Accuracy matrix:")
    print(accuracy_matrix.to_string())

    plot_heatmap(
        accuracy_matrix=accuracy_matrix,
        output=args.output,
        title=args.title,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
