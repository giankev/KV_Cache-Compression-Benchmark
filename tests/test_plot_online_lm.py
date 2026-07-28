from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.plot_online_lm import get_kv_budget, load_results


@pytest.mark.parametrize("budget", [1000, 2000])
def test_get_kv_budget_derives_the_unique_budget(budget: int) -> None:
    frame = pd.DataFrame({"max_cache_tokens": [budget, budget]})

    assert get_kv_budget(frame) == budget


def test_get_kv_budget_rejects_mixed_budgets() -> None:
    frame = pd.DataFrame({"max_cache_tokens": [1000, 2000]})

    with pytest.raises(ValueError, match="exactly one unique cache budget"):
        get_kv_budget(frame)


def test_get_kv_budget_rejects_missing_budget_column() -> None:
    with pytest.raises(ValueError, match="max_cache_tokens"):
        get_kv_budget(pd.DataFrame({"processed_tokens": [512]}))


def test_load_results_requires_the_budget_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "online_lm_curve.csv"
    pd.DataFrame(
        {
            "config": ["no_compression"],
            "processed_tokens": [512],
            "log_ppl": [1.0],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="max_cache_tokens"):
        load_results(csv_path)
