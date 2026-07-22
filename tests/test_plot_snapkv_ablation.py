from __future__ import annotations

import pandas as pd
import pytest

from scripts.plot_snapkv_ablation import build_accuracy_matrix


def test_build_accuracy_matrix_orders_aggregates_and_preserves_missing() -> None:
    results = pd.DataFrame(
        [
            {
                "config": "snapkv_keep10_obs64_k5",
                "depth_target": "0.75",
                "observation_window_size": "64",
                "keep_ratio": "0.10",
                "pooling_kernel_size": "5",
                "correct": "true",
            },
            {
                "config": "snapkv_keep10_obs16_k5",
                "depth_target": 0.25,
                "observation_window_size": 16,
                "keep_ratio": 0.10,
                "pooling_kernel_size": 5,
                "correct": 1,
            },
            {
                "config": "snapkv_keep10_obs16_k5",
                "depth_target": 0.25,
                "observation_window_size": 16,
                "keep_ratio": 0.10,
                "pooling_kernel_size": 5,
                "correct": False,
            },
            {
                "config": "snapkv_keep10_obs32_k5",
                "depth_target": 0.50,
                "observation_window_size": 32,
                "keep_ratio": 0.10,
                "pooling_kernel_size": 5,
                "correct": "1",
            },
            {
                "config": "no_compression",
                "depth_target": 0.75,
                "observation_window_size": None,
                "keep_ratio": 1.0,
                "pooling_kernel_size": 0,
                "correct": 0,
            },
            {
                "config": "no_compression",
                "depth_target": 0.25,
                "observation_window_size": None,
                "keep_ratio": 1.0,
                "pooling_kernel_size": 0,
                "correct": True,
            },
        ]
    )

    matrix = build_accuracy_matrix(results)

    assert matrix.index.tolist() == ["no_compression", 16, 32, 64]
    assert matrix.columns.tolist() == [0.25, 0.50, 0.75]
    assert matrix.loc["no_compression", 0.25] == 1.0
    assert matrix.loc[16, 0.25] == 0.5
    assert matrix.loc[32, 0.50] == 1.0
    assert matrix.loc[64, 0.75] == 1.0
    assert pd.isna(matrix.loc[16, 0.75])
    assert pd.isna(matrix.loc["no_compression", 0.50])


@pytest.mark.parametrize(
    ("column", "second_value", "message"),
    [
        ("keep_ratio", 0.20, "exactly one keep ratio"),
        ("pooling_kernel_size", 7, "exactly one pooling kernel size"),
    ],
)
def test_build_accuracy_matrix_rejects_mixed_snapkv_grids(
    column: str,
    second_value: float,
    message: str,
) -> None:
    results = pd.DataFrame(
        [
            {
                "config": "snapkv_a",
                "depth_target": 0.25,
                "observation_window_size": 16,
                "keep_ratio": 0.10,
                "pooling_kernel_size": 5,
                "correct": True,
            },
            {
                "config": "snapkv_b",
                "depth_target": 0.50,
                "observation_window_size": 32,
                "keep_ratio": 0.10,
                "pooling_kernel_size": 5,
                "correct": False,
            },
        ]
    )
    results.loc[1, column] = second_value

    with pytest.raises(ValueError, match=message):
        build_accuracy_matrix(results)
