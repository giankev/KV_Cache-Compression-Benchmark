from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "config",
    "depth_target",
    "observation_window_size",
    "keep_ratio",
    "pooling_kernel_size",
    "correct",
}
BASELINE_CONFIG = "no_compression"


def _validate_columns(results: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(results.columns))
    if missing:
        raise ValueError(
            "SnapKV ablation raw CSV is missing required columns: "
            + ", ".join(missing)
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
    """Load SnapKV ablation raw results and validate required columns."""

    results = pd.read_csv(input_csv)
    _validate_columns(results)
    return results


def build_accuracy_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Build mean accuracy by observation window and numeric target depth."""

    _validate_columns(results)
    if results.empty:
        raise ValueError("The SnapKV ablation raw CSV contains no result rows")

    prepared = results.loc[
        :,
        [
            "config",
            "depth_target",
            "observation_window_size",
            "keep_ratio",
            "pooling_kernel_size",
            "correct",
        ],
    ].copy()
    if prepared["config"].isna().any():
        raise ValueError("Column 'config' must not contain missing values")
    prepared["config"] = prepared["config"].astype(str)
    prepared["depth_target"] = pd.to_numeric(
        prepared["depth_target"], errors="coerce"
    )
    if prepared["depth_target"].isna().any():
        raise ValueError("Column 'depth_target' must contain numeric values")
    prepared["correct"] = _correct_as_binary(prepared["correct"])

    baseline_mask = prepared["config"].eq(BASELINE_CONFIG)
    snapkv_results = prepared.loc[~baseline_mask].copy()
    if not snapkv_results.empty:
        for column in (
            "observation_window_size",
            "keep_ratio",
            "pooling_kernel_size",
        ):
            snapkv_results[column] = pd.to_numeric(
                snapkv_results[column], errors="coerce"
            )
            if snapkv_results[column].isna().any():
                raise ValueError(
                    f"Column '{column}' must contain numeric values for SnapKV rows"
                )

        for column, label in (
            ("keep_ratio", "keep ratio"),
            ("pooling_kernel_size", "pooling kernel size"),
        ):
            values = sorted(snapkv_results[column].unique().tolist())
            if len(values) != 1:
                raise ValueError(
                    "The SnapKV ablation heatmap requires exactly one "
                    f"{label}; found {values}"
                )

    depths = sorted(float(depth) for depth in prepared["depth_target"].unique())
    matrix_parts: list[pd.DataFrame] = []

    baseline_results = prepared.loc[baseline_mask]
    if not baseline_results.empty:
        baseline_accuracy = baseline_results.groupby("depth_target")["correct"].mean()
        matrix_parts.append(
            pd.DataFrame(
                [baseline_accuracy],
                index=pd.Index([BASELINE_CONFIG]),
            )
        )

    if not snapkv_results.empty:
        snapkv_matrix = snapkv_results.pivot_table(
            index="observation_window_size",
            columns="depth_target",
            values="correct",
            aggfunc="mean",
        ).sort_index()
        matrix_parts.append(snapkv_matrix)

    matrix = pd.concat(matrix_parts).reindex(columns=depths)
    matrix.index.name = "observation_window_size"
    matrix.columns.name = "depth_target"
    return matrix


def _format_depth(depth: float) -> str:
    return f"{depth * 100:g}%"


def _format_observation_window(value: object) -> str:
    if value == BASELINE_CONFIG:
        return BASELINE_CONFIG
    numeric_value = float(value)
    formatted = f"{numeric_value:g}"
    return f"{formatted} tokens"


def plot_heatmap(
    accuracy_matrix: pd.DataFrame,
    output: Path,
    title: str | None = None,
) -> None:
    """Render and save the annotated SnapKV ablation heatmap."""

    width = max(7.0, 1.6 * len(accuracy_matrix.columns) + 4.0)
    height = max(4.5, 0.7 * len(accuracy_matrix.index) + 2.5)
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
    ax.set_yticklabels(
        [_format_observation_window(value) for value in accuracy_matrix.index]
    )
    ax.set_xlabel("Target record depth")
    ax.set_ylabel("Observation window")
    ax.set_title(title or "SnapKV retrieval accuracy by observation window and depth")

    for row_index in range(len(accuracy_matrix.index)):
        for column_index in range(len(accuracy_matrix.columns)):
            value = accuracy_matrix.iloc[row_index, column_index]
            if pd.isna(value):
                label = "N/A"
                text_color = "black"
            else:
                label = f"{value:.2f}"
                text_color = "white" if value < 0.55 else "black"
            ax.text(
                column_index,
                row_index,
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
        description="Generate a heatmap from raw SnapKV ablation results."
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
