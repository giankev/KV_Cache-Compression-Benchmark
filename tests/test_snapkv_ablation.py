from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from scripts import run_kv_retrieval
from scripts import run_snapkv_ablation as ablation


def test_default_run_count_is_nine_snapkv_plus_three_baselines() -> None:
    settings = ablation.parse_args([])

    counts = ablation.calculate_run_counts(settings)

    assert counts.snapkv_runs == 9
    assert counts.baseline_runs == 3
    assert counts.planned_runs == 12


def test_run_limit_is_enforced_before_model_loading(monkeypatch: Any) -> None:
    model_load_attempted = False

    def fail_if_loaded(*_: Any, **__: Any) -> None:
        nonlocal model_load_attempted
        model_load_attempted = True
        pytest.fail("model loading must happen after the max-runs check")

    monkeypatch.setattr(ablation, "load_model_and_tokenizer", fail_if_loaded)

    with pytest.raises(ValueError, match="exceed max_runs"):
        ablation.main(["--max-runs", "11"])

    assert model_load_attempted is False


def test_baseline_is_planned_once_per_prompt_and_prompts_are_reused() -> None:
    settings = replace(
        ablation.parse_args([]),
        context_lengths=(128,),
        depths=(0.25, 0.75),
        seeds=(3,),
        observation_window_sizes=(16, 32),
        keep_ratios=(0.1,),
        pooling_kernel_sizes=(5,),
        max_runs=6,
    )
    prompt_calls: list[dict[str, Any]] = []

    def prompt_factory(**kwargs: Any) -> SimpleNamespace:
        prompt_calls.append(kwargs)
        marker = len(prompt_calls)
        return SimpleNamespace(
            context_len_actual=kwargs["target_tokens"],
            prompt_ids=(marker, marker),
        )

    plan = ablation.build_execution_plan(
        tokenizer=object(),
        settings=settings,
        prompt_factory=prompt_factory,
    )

    assert len(prompt_calls) == 2
    assert all(call["observation_window_size"] == 16 for call in prompt_calls)
    baseline_runs = [
        run for run in plan if run.configuration.config == "no_compression"
    ]
    assert len(baseline_runs) == 2

    for depth in settings.depths:
        prompt_runs = [run for run in plan if run.depth == depth]
        assert len(prompt_runs) == 3
        assert len({id(run.prompt) for run in prompt_runs}) == 1
        assert len({id(run.prompt.prompt_ids) for run in prompt_runs}) == 1


def test_configuration_names_are_deterministic_and_unambiguous() -> None:
    assert ablation.snapkv_config_name(0.10, 16, 5) == "snapkv_keep10_obs16_k5"
    assert (
        ablation.snapkv_config_name(0.125, 64, 7)
        == "snapkv_keep12p5_obs64_k7"
    )
    assert ablation.snapkv_config_name(0.10, 16, 5) == ablation.snapkv_config_name(
        0.10,
        16,
        5,
    )


def test_disabling_baseline_changes_only_baseline_run_count() -> None:
    settings = replace(ablation.parse_args([]), include_baseline=False)

    counts = ablation.calculate_run_counts(settings)

    assert counts.snapkv_runs == 9
    assert counts.baseline_runs == 0
    assert counts.planned_runs == 9


def test_existing_kv_retrieval_column_contract_is_unchanged() -> None:
    assert run_kv_retrieval.RAW_COLUMNS == [
        "model_name",
        "benchmark",
        "config",
        "strategy",
        "keep_ratio",
        "context_len_target",
        "context_len_actual",
        "depth_target",
        "target_record_token_position",
        "seed",
        "target_key",
        "target_value",
        "generated_text",
        "prediction",
        "correct",
        "skip_layers",
        "observation_window_size",
        "pooling_kernel_size",
        "target_cache_tokens",
        "cache_mb_before_compression",
        "cache_mb_after_compression",
        "final_cache_mb",
        "memory_saved_percent",
        "elapsed_seconds",
    ]
    assert run_kv_retrieval.SUMMARY_COLUMNS == [
        "config",
        "strategy",
        "keep_ratio",
        "context_len",
        "num_examples",
        "accuracy",
        "mean_cache_mb_before_compression",
        "mean_cache_mb_after_compression",
        "mean_memory_saved_percent",
        "mean_elapsed_seconds",
    ]

    old_summary = run_kv_retrieval.summarize(
        pd.DataFrame(
            [
                {
                    "config": "no_compression",
                    "strategy": "none",
                    "keep_ratio": 1.0,
                    "context_len_target": 128,
                    "correct": True,
                    "cache_mb_before_compression": 2.0,
                    "cache_mb_after_compression": 2.0,
                    "memory_saved_percent": 0.0,
                    "elapsed_seconds": 0.5,
                }
            ]
        )
    )

    assert old_summary.columns.tolist() == run_kv_retrieval.SUMMARY_COLUMNS
    assert old_summary.loc[0, "context_len"] == 128
    assert old_summary.loc[0, "accuracy"] == 1.0
