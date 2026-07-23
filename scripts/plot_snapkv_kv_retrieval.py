from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CONFIG_ORDER = ("no_compression", "snapkv_1024")
REQUIRED_COLUMNS = {
    "config",
    "strategy",
    "depth_target",
    "correct",
    "skipped_due_to_baseline_failure",
    "observation_window_size",
    "pooling_kernel_size",
    "target_cache_tokens",
}


@dataclass(frozen=True)
class PlotMetadata:
    observation_window_size: int
    target_cache_tokens: int
    pooling_kernel_size: int


def _binary_value(value: Any, column: str) -> float:
    if pd.isna(value):
        return math.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and value in {0, 1}:
        return float(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return 1.0
    if normalized in {"false", "0"}:
        return 0.0
    raise ValueError(
        f"Column {column!r} contains a non-binary value: {value!r}"
    )


def load_results(input_csv: Path) -> pd.DataFrame:
    """Load one raw benchmark CSV and normalize plotting columns."""

    results = pd.read_csv(input_csv)
    missing = sorted(REQUIRED_COLUMNS - set(results.columns))
    if missing:
        raise ValueError(
            "Raw SnapKV CSV is missing required columns: " + ", ".join(missing)
        )
    normalized = results.copy()
    normalized["depth_target"] = pd.to_numeric(
        normalized["depth_target"],
        errors="raise",
    )
    normalized["correct_numeric"] = normalized["correct"].map(
        lambda value: _binary_value(value, "correct")
    )
    normalized["skipped_numeric"] = normalized[
        "skipped_due_to_baseline_failure"
    ].map(lambda value: _binary_value(value, "skipped_due_to_baseline_failure"))
    return normalized


def build_accuracy_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Average accuracy per configuration/depth, excluding gated rows."""

    working = results.copy()
    if "correct_numeric" not in working:
        working["correct_numeric"] = working["correct"].map(
            lambda value: _binary_value(value, "correct")
        )
    if "skipped_numeric" not in working:
        working["skipped_numeric"] = working[
            "skipped_due_to_baseline_failure"
        ].map(
            lambda value: _binary_value(
                value,
                "skipped_due_to_baseline_failure",
            )
        )
    working["depth_target"] = pd.to_numeric(
        working["depth_target"],
        errors="raise",
    )
    completed = working.loc[working["skipped_numeric"] == 0].copy()
    if completed["correct_numeric"].isna().any():
        raise ValueError(
            "Completed rows must contain a boolean or 0/1 'correct' value"
        )
    grouped = (
        completed.groupby(["config", "depth_target"], dropna=False)[
            "correct_numeric"
        ]
        .mean()
        .unstack("depth_target")
    )
    observed = [str(config) for config in working["config"].dropna().unique()]
    ordered = [config for config in CONFIG_ORDER if config in observed]
    ordered.extend(sorted(set(observed) - set(ordered)))
    depths = sorted(float(depth) for depth in working["depth_target"].unique())
    return grouped.reindex(index=ordered, columns=depths)


def extract_plot_metadata(results: pd.DataFrame) -> PlotMetadata:
    """Read the fixed SnapKV settings used by the raw result rows."""

    snapkv_rows = results.loc[results["strategy"].astype(str) == "snapkv"]
    if snapkv_rows.empty:
        raise ValueError("Raw CSV contains no SnapKV rows")

    values: dict[str, int] = {}
    for column in (
        "observation_window_size",
        "target_cache_tokens",
        "pooling_kernel_size",
    ):
        numeric = pd.to_numeric(snapkv_rows[column], errors="raise").dropna()
        unique = sorted({int(value) for value in numeric})
        if len(unique) != 1:
            raise ValueError(
                f"Plot expects one fixed {column}; found {unique}"
            )
        values[column] = unique[0]
    return PlotMetadata(**values)


def _depth_label(depth: float) -> str:
    return f"{depth:.0%}"


def plot_heatmap(
    matrix: pd.DataFrame,
    output: Path,
    metadata: PlotMetadata,
    title: str | None = None,
) -> None:
    """Render and save a compact report-ready accuracy heatmap."""

    if matrix.empty:
        raise ValueError("No completed benchmark rows are available to plot")
    figure_width = max(7.0, 1.2 * len(matrix.columns) + 4.0)
    figure_height = max(3.8, 0.8 * len(matrix.index) + 2.4)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#eeeeee")
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.03)
    colorbar.set_label("Accuracy")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([_depth_label(float(depth)) for depth in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Target record depth")
    ax.set_ylabel("Cache configuration")

    heading = title or "SnapKV key-value retrieval accuracy by depth"
    subtitle = (
        f"Observation window = {metadata.observation_window_size} | "
        f"Cache capacity = {metadata.target_cache_tokens} | "
        f"Kernel = {metadata.pooling_kernel_size}"
    )
    ax.set_title(f"{heading}\n{subtitle}", pad=14)

    for row_idx, config in enumerate(matrix.index):
        for column_idx, depth in enumerate(matrix.columns):
            value = matrix.loc[config, depth]
            if pd.isna(value):
                label = "N/A"
                color = "#555555"
            else:
                label = f"{float(value):.2f}"
                color = "white" if float(value) < 0.55 else "black"
            ax.text(
                column_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="semibold",
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the raw SnapKV retrieval benchmark as a heatmap."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = load_results(args.input_csv)
    matrix = build_accuracy_matrix(results)
    metadata = extract_plot_metadata(results)
    print("Accuracy matrix:")
    print(matrix.to_string())
    plot_heatmap(
        matrix=matrix,
        output=args.output,
        metadata=metadata,
        title=args.title,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
