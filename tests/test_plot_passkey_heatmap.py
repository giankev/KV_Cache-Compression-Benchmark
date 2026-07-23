from __future__ import annotations

import pandas as pd

from scripts.plot_passkey_heatmap import build_accuracy_matrix


def test_build_accuracy_matrix_orders_and_aggregates_results() -> None:
    results = pd.DataFrame(
        [
            {"config": "custom_config", "depth_target": 0.75, "correct": "true"},
            {"config": "high_l2_keep50", "depth_target": 0.25, "correct": 0},
            {"config": "random_keep50", "depth_target": 0.50, "correct": False},
            {"config": "low_l2_keep10", "depth_target": 0.25, "correct": "1"},
            {"config": "low_l2_keep50", "depth_target": 0.75, "correct": True},
            {"config": "no_compression", "depth_target": 0.25, "correct": True},
            {"config": "no_compression", "depth_target": 0.25, "correct": False},
            {"config": "no_compression", "depth_target": 0.50, "correct": 1},
            {
                "config": "low_l2_keep10",
                "depth_target": 0.50,
                "correct": pd.NA,
                "skipped_due_to_baseline_failure": True,
            },
        ]
    )

    matrix = build_accuracy_matrix(results)

    assert matrix.index.tolist() == [
        "no_compression",
        "low_l2_keep50",
        "low_l2_keep10",
        "random_keep50",
        "high_l2_keep50",
        "custom_config",
    ]
    assert matrix.columns.tolist() == [0.25, 0.50, 0.75]
    assert matrix.loc["no_compression", 0.25] == 0.5
    assert matrix.loc["no_compression", 0.50] == 1.0
    assert pd.isna(matrix.loc["low_l2_keep10", 0.75])
