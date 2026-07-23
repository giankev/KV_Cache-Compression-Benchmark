from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch

from scripts import plot_snapkv_kv_retrieval as plot_snapkv
from scripts import run_snapkv_kv_retrieval as benchmark


def test_defaults_select_one_absolute_capacity_and_six_runs() -> None:
    settings = benchmark.parse_args([])
    counts = benchmark.calculate_run_counts(settings)

    assert settings.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert settings.context_lengths == (8192,)
    assert settings.depths == (0.25, 0.50, 0.75)
    assert settings.seeds == (0,)
    assert settings.observation_window_size == 16
    assert settings.target_cache_tokens == 1024
    assert settings.keep_ratio is None
    assert settings.pooling_kernel_size == 5
    assert settings.pooling_mode == "max"
    assert settings.skip_layers == (0, 1)
    assert settings.chunk_size == 512
    assert settings.max_new_tokens == 24
    assert settings.include_baseline is True
    assert settings.max_runs == 6
    assert settings.prompt_style == "explicit_v2"
    assert settings.require_baseline_success is True
    assert counts.baseline_runs == 3
    assert counts.snapkv_runs == 3
    assert counts.planned_runs == 6


def test_run_limit_is_checked_before_model_loading(monkeypatch: Any) -> None:
    attempted_load = False

    def fail_if_loaded(*_: Any, **__: Any) -> None:
        nonlocal attempted_load
        attempted_load = True
        pytest.fail("model loading must happen after the run-count gate")

    monkeypatch.setattr(benchmark, "load_model_and_tokenizer", fail_if_loaded)

    with pytest.raises(ValueError, match="exceed max_runs"):
        benchmark.main(["--max-runs", "5"])

    assert attempted_load is False


def test_absolute_capacity_and_keep_ratio_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            ["--target-cache-tokens", "1024", "--keep-ratio", "0.125"]
        )

    ratio_settings = benchmark.parse_args(["--keep-ratio", "0.125"])
    assert ratio_settings.target_cache_tokens is None
    assert ratio_settings.keep_ratio == 0.125


def test_capacity_is_exact_and_reports_effective_ratio() -> None:
    absolute = benchmark.resolve_capacity(
        prompt_length=8192,
        observation_window_size=16,
        target_cache_tokens=1024,
        keep_ratio=None,
    )
    ratio = benchmark.resolve_capacity(
        prompt_length=8191,
        observation_window_size=16,
        target_cache_tokens=None,
        keep_ratio=0.125,
    )

    assert absolute.prefix_length == 8176
    assert absolute.selected_prefix_tokens == 1008
    assert absolute.selected_prefix_tokens + 16 == 1024
    assert absolute.effective_keep_ratio == pytest.approx(0.125)
    assert ratio.target_cache_tokens == 1023
    assert ratio.selected_prefix_tokens == 1007
    assert ratio.effective_keep_ratio == pytest.approx(1023 / 8191)

    with pytest.raises(ValueError, match="at least observation_window_size"):
        benchmark.resolve_capacity(
            prompt_length=8192,
            observation_window_size=16,
            target_cache_tokens=15,
            keep_ratio=None,
        )


def test_post_compression_contract_checks_exact_layer_capacities() -> None:
    observation_window_size = 2
    prompt_length = 10
    capacity = benchmark.resolve_capacity(
        prompt_length=prompt_length,
        observation_window_size=observation_window_size,
        target_cache_tokens=4,
        keep_ratio=None,
    )
    full_keys = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10, 1)
    full_values = full_keys + 100
    compressed_keys = torch.tensor([[[[1.0], [5.0], [8.0], [9.0]]]])
    compressed_values = compressed_keys + 100
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(keys=full_keys, values=full_values),
            SimpleNamespace(keys=full_keys.clone(), values=full_values.clone()),
            SimpleNamespace(keys=compressed_keys, values=compressed_values),
        ]
    )
    snapshots = (
        (full_keys[:, :, -2:, :].clone(), full_values[:, :, -2:, :].clone()),
        (full_keys[:, :, -2:, :].clone(), full_values[:, :, -2:, :].clone()),
        (
            compressed_keys[:, :, -2:, :].clone(),
            compressed_values[:, :, -2:, :].clone(),
        ),
    )

    lengths = benchmark.assert_snapkv_cache_contract(
        cache,
        prompt_length=prompt_length,
        capacity=capacity,
        observation_window_size=observation_window_size,
        skip_layers=(0, 1),
        observation_snapshots=snapshots,
    )

    assert lengths == (10, 10, 4)


def test_configuration_set_contains_only_baseline_and_one_snapkv_policy() -> None:
    settings = benchmark.parse_args([])
    configurations = benchmark.build_configurations(settings, 8192)

    assert [configuration.config for configuration in configurations] == [
        "no_compression",
        "snapkv_1024",
    ]
    assert [configuration.strategy for configuration in configurations] == [
        "none",
        "snapkv",
    ]
    source = inspect.getsource(benchmark).lower()
    assert "compress_cache_to_budget" not in source
    assert '"low_l2"' not in source
    assert '"high_l2"' not in source
    assert '"random"' not in source


def test_disabling_baseline_requires_disabling_the_gate() -> None:
    with pytest.raises(ValueError, match="no-include-baseline"):
        benchmark.parse_args(["--no-include-baseline"])

    settings = benchmark.parse_args(
        ["--no-include-baseline", "--no-require-baseline-success"]
    )
    counts = benchmark.calculate_run_counts(settings)
    configurations = benchmark.build_configurations(settings, 8192)

    assert counts.baseline_runs == 0
    assert counts.snapkv_runs == 3
    assert counts.planned_runs == 3
    assert [configuration.config for configuration in configurations] == [
        "snapkv_1024"
    ]


@pytest.mark.parametrize(
    ("baseline_correct", "require_success", "expected"),
    [(True, True, False), (False, True, True), (False, False, False)],
)
def test_baseline_gate(
    baseline_correct: bool,
    require_success: bool,
    expected: bool,
) -> None:
    assert (
        benchmark.should_skip_snapkv(baseline_correct, require_success)
        is expected
    )


def test_failed_baseline_checkpoints_a_skipped_snapkv_row(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    settings = replace(
        benchmark.parse_args([]),
        context_lengths=(128,),
        depths=(0.5,),
        seeds=(7,),
        target_cache_tokens=32,
        max_runs=2,
    )
    prompt = SimpleNamespace(
        prompt_ids=(1,) * 128,
        context_len_actual=128,
        target_key="gentle-acmi",
        target_value="38733",
        target_record_token_position=32,
    )
    snapkv_called = False
    checkpoint_sizes: list[int] = []

    monkeypatch.setattr(
        benchmark,
        "build_prompt_for_case",
        lambda **_: prompt,
    )

    def failed_baseline(
        *_: Any,
        configuration: Any = None,
        **__: Any,
    ) -> dict[str, Any]:
        del configuration
        return {
            column: False if column == "correct" else pd.NA
            for column in benchmark.RAW_COLUMNS
        }

    def forbidden_snapkv(*_: Any, **__: Any) -> dict[str, Any]:
        nonlocal snapkv_called
        snapkv_called = True
        pytest.fail("SnapKV must not execute after a failed gated baseline")

    monkeypatch.setattr(benchmark, "execute_baseline", failed_baseline)
    monkeypatch.setattr(benchmark, "execute_snapkv", forbidden_snapkv)
    checkpoint_raw = benchmark.checkpoint_raw

    def tracking_checkpoint(
        rows: Any,
        raw_path: Path,
    ) -> pd.DataFrame:
        checkpoint_sizes.append(len(rows))
        return checkpoint_raw(rows, raw_path)

    monkeypatch.setattr(benchmark, "checkpoint_raw", tracking_checkpoint)
    fake_model = SimpleNamespace(parameters=lambda: iter(()))
    raw_path = tmp_path / "nested" / "raw.csv"

    result = benchmark.run_benchmark(
        model=fake_model,
        tokenizer=object(),
        settings=settings,
        raw_path=raw_path,
    )

    assert snapkv_called is False
    assert checkpoint_sizes == [1, 2]
    assert raw_path.is_file()
    assert len(result) == 2
    assert result.loc[1, "config"] == "snapkv_32"
    assert bool(result.loc[1, "skipped_due_to_baseline_failure"]) is True
    assert pd.isna(result.loc[1, "correct"])
    checkpoint = pd.read_csv(raw_path)
    assert len(checkpoint) == 2
    assert checkpoint.columns.tolist() == benchmark.RAW_COLUMNS


def test_raw_and_summary_schemas_include_required_fields() -> None:
    raw_required = {
        "model_name",
        "benchmark",
        "benchmark_version",
        "prompt_style",
        "config",
        "strategy",
        "context_len_target",
        "context_len_actual",
        "depth_target",
        "seed",
        "target_key",
        "target_value",
        "target_record_token_position",
        "generated_text",
        "prediction",
        "correct",
        "baseline_correct",
        "skipped_due_to_baseline_failure",
        "skip_layers",
        "cache_mb_before_compression",
        "cache_mb_after_compression",
        "final_cache_mb",
        "memory_saved_percent",
        "elapsed_seconds",
        "observation_window_size",
        "prefix_length",
        "pooling_kernel_size",
        "pooling_mode",
        "target_cache_tokens",
        "effective_keep_ratio",
        "selected_prefix_tokens",
    }
    summary_required = {
        "config",
        "strategy",
        "target_cache_tokens",
        "effective_keep_ratio",
        "num_examples",
        "num_skipped_due_to_baseline_failure",
        "accuracy",
    }

    assert raw_required <= set(benchmark.RAW_COLUMNS)
    assert summary_required <= set(benchmark.SUMMARY_COLUMNS)


def _plot_rows() -> pd.DataFrame:
    common = {
        "strategy": "none",
        "skipped_due_to_baseline_failure": False,
        "observation_window_size": 16,
        "pooling_kernel_size": 5,
        "target_cache_tokens": 8192,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "config": "no_compression",
                "depth_target": 0.75,
                "correct": True,
            },
            {
                **common,
                "config": "no_compression",
                "depth_target": 0.25,
                "correct": "true",
            },
            {
                **common,
                "config": "no_compression",
                "depth_target": 0.25,
                "correct": "false",
            },
            {
                **common,
                "strategy": "snapkv",
                "config": "snapkv_1024",
                "depth_target": 0.25,
                "correct": "1",
                "target_cache_tokens": 1024,
            },
            {
                **common,
                "strategy": "snapkv",
                "config": "snapkv_1024",
                "depth_target": 0.75,
                "correct": pd.NA,
                "skipped_due_to_baseline_failure": True,
                "target_cache_tokens": 1024,
            },
        ]
    )


def test_plot_matrix_order_means_missing_cells_and_metadata() -> None:
    results = _plot_rows()
    matrix = plot_snapkv.build_accuracy_matrix(results)
    metadata = plot_snapkv.extract_plot_metadata(results)

    assert matrix.index.tolist() == ["no_compression", "snapkv_1024"]
    assert matrix.columns.tolist() == [0.25, 0.75]
    assert matrix.loc["no_compression", 0.25] == pytest.approx(0.5)
    assert matrix.loc["snapkv_1024", 0.25] == pytest.approx(1.0)
    assert pd.isna(matrix.loc["snapkv_1024", 0.75])
    assert metadata == plot_snapkv.PlotMetadata(
        observation_window_size=16,
        target_cache_tokens=1024,
        pooling_kernel_size=5,
    )


def test_plot_writes_png_and_creates_parent_directory(tmp_path: Path) -> None:
    results = _plot_rows()
    matrix = plot_snapkv.build_accuracy_matrix(results)
    metadata = plot_snapkv.extract_plot_metadata(results)
    output = tmp_path / "nested" / "heatmap.png"

    plot_snapkv.plot_heatmap(matrix, output, metadata)

    assert output.is_file()
    assert output.stat().st_size > 0
