from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from scripts import run_l2_kv_retrieval as runner


def test_default_count_and_cli_contract() -> None:
    settings = runner.parse_args([])
    counts = runner.calculate_run_counts(settings)

    assert settings.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert settings.context_lengths == (8192,)
    assert settings.depths == (0.25, 0.50, 0.75)
    assert settings.seeds == (0,)
    assert settings.skip_layers == (0, 1)
    assert settings.chunk_size == 512
    assert settings.max_new_tokens == 24
    assert settings.max_runs == 15
    assert settings.prompt_style == "explicit_v2"
    assert settings.require_baseline_success is True
    assert counts.baseline_runs == 3
    assert counts.compressed_runs == 12
    assert counts.planned_runs == 15


def test_configuration_contract_has_only_the_five_l2_family_policies() -> None:
    configurations = runner.build_configurations()

    assert [configuration.config for configuration in configurations] == [
        "no_compression",
        "low_l2_keep50",
        "low_l2_keep10",
        "random_keep50",
        "high_l2_keep50",
    ]
    assert [configuration.strategy for configuration in configurations] == [
        "none",
        "low_l2",
        "low_l2",
        "random",
        "high_l2",
    ]
    source = inspect.getsource(runner).lower()
    assert "l2kv.snapkv" not in source
    assert "prefill_snapkv" not in source
    assert "output_attentions" not in source


def test_run_limit_is_checked_before_model_loading(monkeypatch: Any) -> None:
    model_load_attempted = False

    def fail_if_loaded(*_: Any, **__: Any) -> None:
        nonlocal model_load_attempted
        model_load_attempted = True
        pytest.fail("model loading must happen after the max-runs gate")

    monkeypatch.setattr(runner, "load_model_and_tokenizer", fail_if_loaded)

    with pytest.raises(ValueError, match="exceed max_runs"):
        runner.main(["--max-runs", "14"])

    assert model_load_attempted is False


def _fake_prompt(context_length: int) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_ids=tuple(range(context_length)),
        context_len_actual=context_length,
        target_key="gentle-acmi",
        target_value="38733",
        target_record_token_position=12,
    )


def _fake_completed_row(
    settings: runner.BenchmarkSettings,
    prompt: Any,
    context_length: int,
    depth: float,
    seed: int,
    configuration: runner.L2Configuration,
    *,
    correct: bool,
) -> dict[str, Any]:
    row = {column: pd.NA for column in runner.RAW_COLUMNS}
    row.update(
        {
            "model_name": settings.model_name,
            "benchmark": runner.BENCHMARK_NAME,
            "benchmark_version": runner.BENCHMARK_VERSION,
            "prompt_style": settings.prompt_style,
            "config": configuration.config,
            "strategy": configuration.strategy,
            "context_len_target": context_length,
            "context_len_actual": prompt.context_len_actual,
            "depth_target": depth,
            "seed": seed,
            "target_key": prompt.target_key,
            "target_value": prompt.target_value,
            "target_record_token_position": (
                prompt.target_record_token_position
            ),
            "generated_text": "38733" if correct else "00000",
            "prediction": "38733" if correct else "00000",
            "correct": correct,
            "skipped_due_to_baseline_failure": False,
            "skip_layers": "[0, 1]",
            "cache_mb_before_compression": 10.0,
            "cache_mb_after_compression": 10.0,
            "final_cache_mb": 10.1,
            "memory_saved_percent": 0.0,
            "elapsed_seconds": 0.1,
            "keep_ratio": configuration.keep_ratio,
            "target_cache_tokens": runner.target_capacity(
                context_length,
                configuration.keep_ratio,
            ),
        }
    )
    return row


def test_baseline_gate_skips_compressed_runs_and_checkpoints_raw_csv(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    settings = replace(
        runner.parse_args([]),
        context_lengths=(128,),
        depths=(0.5,),
        seeds=(3,),
        max_runs=5,
    )
    prompt = _fake_prompt(128)
    executed: list[str] = []
    checkpoint_sizes: list[int] = []
    original_checkpoint = runner.checkpoint_raw

    monkeypatch.setattr(runner, "cuda_devices", lambda _model: ())
    monkeypatch.setattr(
        runner,
        "build_prompt_for_case",
        lambda **_kwargs: prompt,
    )

    def fake_execute(
        model: Any,
        tokenizer: Any,
        current_settings: runner.BenchmarkSettings,
        current_prompt: Any,
        context_length: int,
        depth: float,
        seed: int,
        configuration: runner.L2Configuration,
        devices: Any,
    ) -> dict[str, Any]:
        del model, tokenizer, devices
        executed.append(configuration.config)
        return _fake_completed_row(
            current_settings,
            current_prompt,
            context_length,
            depth,
            seed,
            configuration,
            correct=False,
        )

    def recording_checkpoint(
        rows: Any,
        raw_path: Any,
    ) -> pd.DataFrame:
        checkpoint_sizes.append(len(rows))
        return original_checkpoint(rows, raw_path)

    monkeypatch.setattr(runner, "execute_configuration", fake_execute)
    monkeypatch.setattr(runner, "checkpoint_raw", recording_checkpoint)
    raw_path = tmp_path / "nested" / "l2_raw.csv"

    raw_df = runner.run_benchmark(
        model=object(),
        tokenizer=object(),
        settings=settings,
        raw_path=raw_path,
    )

    assert executed == ["no_compression"]
    assert checkpoint_sizes == [1, 2, 3, 4, 5]
    assert raw_path.is_file()
    assert len(pd.read_csv(raw_path)) == 5
    assert len(raw_df) == 5
    assert raw_df.loc[0, "skipped_due_to_baseline_failure"] == False  # noqa: E712
    assert raw_df.loc[1:, "skipped_due_to_baseline_failure"].all()
    assert raw_df.loc[1:, "correct"].isna().all()
    assert not raw_df["baseline_correct"].astype(bool).any()


def test_disabling_baseline_gate_executes_all_configurations(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    settings = replace(
        runner.parse_args(["--no-require-baseline-success"]),
        context_lengths=(128,),
        depths=(0.5,),
        seeds=(0,),
        max_runs=5,
    )
    prompt = _fake_prompt(128)
    executed: list[str] = []
    checkpoint_sizes: list[int] = []
    original_checkpoint = runner.checkpoint_raw

    monkeypatch.setattr(runner, "cuda_devices", lambda _model: ())
    monkeypatch.setattr(
        runner,
        "build_prompt_for_case",
        lambda **_kwargs: prompt,
    )

    def fake_execute(
        model: Any,
        tokenizer: Any,
        current_settings: runner.BenchmarkSettings,
        current_prompt: Any,
        context_length: int,
        depth: float,
        seed: int,
        configuration: runner.L2Configuration,
        devices: Any,
    ) -> dict[str, Any]:
        del model, tokenizer, devices
        executed.append(configuration.config)
        return _fake_completed_row(
            current_settings,
            current_prompt,
            context_length,
            depth,
            seed,
            configuration,
            correct=configuration.strategy != "none",
        )

    monkeypatch.setattr(runner, "execute_configuration", fake_execute)

    def recording_checkpoint(
        rows: Any,
        raw_path: Any,
    ) -> pd.DataFrame:
        checkpoint_sizes.append(len(rows))
        return original_checkpoint(rows, raw_path)

    monkeypatch.setattr(runner, "checkpoint_raw", recording_checkpoint)

    raw_df = runner.run_benchmark(
        model=object(),
        tokenizer=object(),
        settings=settings,
        raw_path=tmp_path / "raw.csv",
    )

    assert executed == [configuration.config for configuration in runner.CONFIGURATIONS]
    assert checkpoint_sizes == [1, 2, 3, 4, 5]
    assert not raw_df["skipped_due_to_baseline_failure"].any()
    assert not raw_df["baseline_correct"].astype(bool).any()


def test_raw_and_summary_column_contracts_include_gate_and_l2_fields() -> None:
    required_raw = {
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
        "keep_ratio",
        "target_cache_tokens",
    }
    assert required_raw.issubset(runner.RAW_COLUMNS)
    assert {
        "config",
        "strategy",
        "keep_ratio",
        "target_cache_tokens",
        "num_examples",
        "num_skipped_due_to_baseline_failure",
        "accuracy",
    }.issubset(runner.SUMMARY_COLUMNS)

    empty_summary = runner.summarize(
        pd.DataFrame(columns=runner.RAW_COLUMNS)
    )
    assert empty_summary.columns.tolist() == runner.SUMMARY_COLUMNS
