from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.cache_metrics import (
    get_cache_layer,
    kv_cache_size_mb,
    num_cache_layers,
)
from l2kv.kv_retrieval import make_kv_retrieval_prompt
from l2kv.kv_retrieval_eval import (
    assert_cache_capacity,
    cuda_devices,
    generate_greedy,
    prefill_plain,
    prefill_snapkv,
    synchronize_cuda_devices,
)
from l2kv.model_utils import load_model_and_tokenizer
from l2kv.runtime_metadata import (
    make_run_metadata,
    print_run_metadata,
    save_run_metadata,
)
from l2kv.snapkv import compress_snapkv_cache, validate_snapkv_capacity


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CONTEXT_LENGTHS = (8192,)
DEPTHS = (0.25, 0.50, 0.75)
SEEDS = (0,)
OBSERVATION_WINDOW_SIZE = 16
TARGET_CACHE_TOKENS = 1024
POOLING_KERNEL_SIZE = 5
POOLING_MODE = "max"
SKIP_LAYERS = (0, 1)
CHUNK_SIZE = 512
MAX_NEW_TOKENS = 24
INCLUDE_BASELINE = True
MAX_RUNS = 6
PROMPT_STYLE = "explicit_v2"
OUTPUT_PREFIX = "snapkv_kv_retrieval"
REQUIRE_BASELINE_SUCCESS = True
DTYPE = "auto"
ATTENTION_IMPLEMENTATION = "eager"
BENCHMARK_NAME = "kv_retrieval"
BENCHMARK_VERSION = "kv_retrieval_v2"


RAW_COLUMNS = [
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
]


SUMMARY_COLUMNS = [
    "benchmark",
    "benchmark_version",
    "prompt_style",
    "config",
    "strategy",
    "observation_window_size",
    "pooling_kernel_size",
    "pooling_mode",
    "target_cache_tokens",
    "effective_keep_ratio",
    "context_len_target",
    "num_examples",
    "num_skipped_due_to_baseline_failure",
    "accuracy",
    "mean_cache_mb_before_compression",
    "mean_cache_mb_after_compression",
    "mean_memory_saved_percent",
    "mean_elapsed_seconds",
]


@dataclass(frozen=True)
class BenchmarkSettings:
    model_name: str
    context_lengths: tuple[int, ...]
    depths: tuple[float, ...]
    seeds: tuple[int, ...]
    observation_window_size: int
    target_cache_tokens: int | None
    keep_ratio: float | None
    pooling_kernel_size: int
    pooling_mode: str
    skip_layers: tuple[int, ...]
    chunk_size: int
    max_new_tokens: int
    include_baseline: bool
    max_runs: int
    prompt_style: str
    output_prefix: str
    require_baseline_success: bool


@dataclass(frozen=True)
class RunCounts:
    baseline_runs: int
    snapkv_runs: int
    planned_runs: int


@dataclass(frozen=True)
class CapacityPlan:
    prefix_length: int
    target_cache_tokens: int
    effective_keep_ratio: float
    selected_prefix_tokens: int


@dataclass(frozen=True)
class SnapKVConfiguration:
    config: str
    strategy: str
    capacity: CapacityPlan


def calculate_run_counts(settings: BenchmarkSettings) -> RunCounts:
    """Count one SnapKV run and, optionally, one baseline per prompt."""

    prompt_count = (
        len(settings.context_lengths)
        * len(settings.depths)
        * len(settings.seeds)
    )
    baseline_runs = prompt_count if settings.include_baseline else 0
    return RunCounts(
        baseline_runs=baseline_runs,
        snapkv_runs=prompt_count,
        planned_runs=baseline_runs + prompt_count,
    )


def enforce_run_limit(run_counts: RunCounts, max_runs: int) -> None:
    if run_counts.planned_runs > max_runs:
        raise ValueError(
            f"Total planned runs ({run_counts.planned_runs}) exceed "
            f"max_runs ({max_runs})"
        )


def resolve_capacity(
    *,
    prompt_length: int,
    observation_window_size: int,
    target_cache_tokens: int | None,
    keep_ratio: float | None,
) -> CapacityPlan:
    """Resolve the single requested budget and validate its exact split."""

    if target_cache_tokens is not None and keep_ratio is not None:
        raise ValueError(
            "target_cache_tokens and keep_ratio are mutually exclusive"
        )
    if prompt_length < 1:
        raise ValueError("prompt_length must be >= 1")
    if target_cache_tokens is None:
        if (
            keep_ratio is None
            or not math.isfinite(keep_ratio)
            or not 0 < keep_ratio <= 1
        ):
            raise ValueError("keep_ratio must satisfy 0 < keep_ratio <= 1")
        target_cache_tokens = math.floor(prompt_length * keep_ratio)
    selected_prefix_tokens = validate_snapkv_capacity(
        prompt_length=prompt_length,
        target_capacity=target_cache_tokens,
        observation_window_size=observation_window_size,
    )
    prefix_length = prompt_length - observation_window_size
    if selected_prefix_tokens < 0:
        raise AssertionError("selected_prefix_tokens must be non-negative")
    if selected_prefix_tokens + observation_window_size != target_cache_tokens:
        raise AssertionError("SnapKV cache capacity split is inconsistent")
    return CapacityPlan(
        prefix_length=prefix_length,
        target_cache_tokens=target_cache_tokens,
        effective_keep_ratio=target_cache_tokens / prompt_length,
        selected_prefix_tokens=selected_prefix_tokens,
    )


def snapkv_config_name(target_cache_tokens: int) -> str:
    return f"snapkv_{target_cache_tokens}"


def build_configurations(
    settings: BenchmarkSettings,
    context_length: int,
) -> tuple[SnapKVConfiguration, ...]:
    """Return only the baseline and the single requested SnapKV policy."""

    snapkv_capacity = resolve_capacity(
        prompt_length=context_length,
        observation_window_size=settings.observation_window_size,
        target_cache_tokens=settings.target_cache_tokens,
        keep_ratio=settings.keep_ratio,
    )
    configurations: list[SnapKVConfiguration] = []
    if settings.include_baseline:
        configurations.append(
            SnapKVConfiguration(
                config="no_compression",
                strategy="none",
                capacity=CapacityPlan(
                    prefix_length=(
                        context_length - settings.observation_window_size
                    ),
                    target_cache_tokens=context_length,
                    effective_keep_ratio=1.0,
                    selected_prefix_tokens=(
                        context_length - settings.observation_window_size
                    ),
                ),
            )
        )
    configurations.append(
        SnapKVConfiguration(
            config=snapkv_config_name(snapkv_capacity.target_cache_tokens),
            strategy="snapkv",
            capacity=snapkv_capacity,
        )
    )
    return tuple(configurations)


def build_prompt_for_case(
    tokenizer: Any,
    context_length: int,
    depth: float,
    seed: int,
    prompt_style: str,
    observation_window_size: int = OBSERVATION_WINDOW_SIZE,
) -> Any:
    """Build the deterministic input shared with the companion runner."""

    return make_kv_retrieval_prompt(
        tokenizer=tokenizer,
        target_tokens=context_length,
        seed=seed,
        depth=depth,
        observation_window_size=observation_window_size,
        prompt_style=prompt_style,
    )


def should_skip_snapkv(
    baseline_correct: bool | None,
    require_baseline_success: bool,
) -> bool:
    """Apply the baseline gate without silently creating a baseline run."""

    if require_baseline_success and baseline_correct is None:
        raise ValueError(
            "require_baseline_success needs an included baseline result"
        )
    return bool(require_baseline_success and not baseline_correct)


def _validate_settings(settings: BenchmarkSettings) -> None:
    if not settings.context_lengths or any(
        length < 1 for length in settings.context_lengths
    ):
        raise ValueError("context_lengths must contain positive integers")
    if not settings.depths or any(
        not math.isfinite(depth) or not 0 <= depth <= 1
        for depth in settings.depths
    ):
        raise ValueError("depths must contain finite values between 0 and 1")
    if not settings.seeds:
        raise ValueError("at least one seed is required")
    if settings.observation_window_size < 1:
        raise ValueError("observation_window_size must be >= 1")
    if (
        settings.target_cache_tokens is not None
        and settings.keep_ratio is not None
    ):
        raise ValueError(
            "target_cache_tokens and keep_ratio are mutually exclusive"
        )
    if settings.target_cache_tokens is None and settings.keep_ratio is None:
        raise ValueError("one cache capacity option is required")
    if settings.pooling_kernel_size < 1 or settings.pooling_kernel_size % 2 == 0:
        raise ValueError("pooling_kernel_size must be a positive odd integer")
    if settings.pooling_mode not in {"max", "avg", "mean"}:
        raise ValueError("pooling_mode must be 'max' or 'avg'")
    if any(layer < 0 for layer in settings.skip_layers):
        raise ValueError("skip_layers must contain non-negative integers")
    if len(set(settings.skip_layers)) != len(settings.skip_layers):
        raise ValueError("skip_layers must not contain duplicates")
    if settings.chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if settings.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if settings.max_runs < 1:
        raise ValueError("max_runs must be >= 1")
    if settings.prompt_style not in {"legacy", "explicit_v2"}:
        raise ValueError("prompt_style must be 'legacy' or 'explicit_v2'")
    if not settings.output_prefix:
        raise ValueError("output_prefix must not be empty")
    if settings.require_baseline_success and not settings.include_baseline:
        raise ValueError(
            "--require-baseline-success cannot be used with "
            "--no-include-baseline; also pass "
            "--no-require-baseline-success"
        )

    for context_length in settings.context_lengths:
        if settings.observation_window_size >= context_length:
            raise ValueError(
                "observation_window_size must be smaller than every context "
                "length"
            )
        resolve_capacity(
            prompt_length=context_length,
            observation_window_size=settings.observation_window_size,
            target_cache_tokens=settings.target_cache_tokens,
            keep_ratio=settings.keep_ratio,
        )


def _raw_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    raw_df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    for column in (
        "cache_mb_before_compression",
        "cache_mb_after_compression",
        "final_cache_mb",
        "memory_saved_percent",
        "elapsed_seconds",
    ):
        raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce").round(4)
    raw_df["effective_keep_ratio"] = pd.to_numeric(
        raw_df["effective_keep_ratio"],
        errors="coerce",
    )
    return raw_df


def checkpoint_raw(
    rows: Sequence[dict[str, Any]],
    raw_path: Path,
) -> pd.DataFrame:
    """Rewrite the raw CSV after every completed or gated run."""

    raw_df = _raw_frame(rows)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    return raw_df


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    working = raw_df.copy()
    working["correct_numeric"] = pd.to_numeric(
        working["correct"],
        errors="coerce",
    )
    working["skipped_numeric"] = (
        working["skipped_due_to_baseline_failure"].fillna(False).astype(int)
    )
    summary = (
        working.groupby(
            [
                "benchmark",
                "benchmark_version",
                "prompt_style",
                "config",
                "strategy",
                "observation_window_size",
                "pooling_kernel_size",
                "pooling_mode",
                "target_cache_tokens",
                "effective_keep_ratio",
                "context_len_target",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            num_examples=("correct_numeric", "count"),
            num_skipped_due_to_baseline_failure=("skipped_numeric", "sum"),
            accuracy=("correct_numeric", "mean"),
            mean_cache_mb_before_compression=(
                "cache_mb_before_compression",
                "mean",
            ),
            mean_cache_mb_after_compression=(
                "cache_mb_after_compression",
                "mean",
            ),
            mean_memory_saved_percent=("memory_saved_percent", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
        .loc[:, SUMMARY_COLUMNS]
    )
    for column in SUMMARY_COLUMNS[13:]:
        summary[column] = summary[column].round(4)
    return summary


def _base_result_fields(
    settings: BenchmarkSettings,
    prompt: Any,
    context_length: int,
    depth: float,
    seed: int,
    configuration: SnapKVConfiguration,
) -> dict[str, Any]:
    return {
        "model_name": settings.model_name,
        "benchmark": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "prompt_style": settings.prompt_style,
        "config": configuration.config,
        "strategy": configuration.strategy,
        "context_len_target": context_length,
        "context_len_actual": prompt.context_len_actual,
        "depth_target": depth,
        "seed": seed,
        "target_key": prompt.target_key,
        "target_value": prompt.target_value,
        "target_record_token_position": prompt.target_record_token_position,
        "skip_layers": json.dumps(list(settings.skip_layers)),
        "observation_window_size": settings.observation_window_size,
        "prefix_length": configuration.capacity.prefix_length,
        "pooling_kernel_size": settings.pooling_kernel_size,
        "pooling_mode": settings.pooling_mode,
        "target_cache_tokens": configuration.capacity.target_cache_tokens,
        "effective_keep_ratio": configuration.capacity.effective_keep_ratio,
        "selected_prefix_tokens": (
            configuration.capacity.selected_prefix_tokens
        ),
    }


def make_skipped_result(
    settings: BenchmarkSettings,
    prompt: Any,
    context_length: int,
    depth: float,
    seed: int,
    configuration: SnapKVConfiguration,
) -> dict[str, Any]:
    """Represent a SnapKV run suppressed by the baseline gate."""

    row = _base_result_fields(
        settings,
        prompt,
        context_length,
        depth,
        seed,
        configuration,
    )
    row.update(
        {
            "generated_text": "",
            "prediction": "",
            "correct": pd.NA,
            "baseline_correct": False,
            "skipped_due_to_baseline_failure": True,
            "cache_mb_before_compression": math.nan,
            "cache_mb_after_compression": math.nan,
            "final_cache_mb": math.nan,
            "memory_saved_percent": math.nan,
            "elapsed_seconds": math.nan,
        }
    )
    return row


def _snapshot_observation_window(
    cache: Any,
    observation_window_size: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(num_cache_layers(cache)):
        keys, values = get_cache_layer(cache, layer_idx)
        if keys.shape != values.shape or keys.ndim != 4:
            raise AssertionError(
                f"Layer {layer_idx} must have matching four-dimensional K/V"
            )
        snapshots.append(
            (
                keys[:, :, -observation_window_size:, :].detach().clone(),
                values[:, :, -observation_window_size:, :].detach().clone(),
            )
        )
    return tuple(snapshots)


def assert_snapkv_cache_contract(
    cache: Any,
    *,
    prompt_length: int,
    capacity: CapacityPlan,
    observation_window_size: int,
    skip_layers: Sequence[int],
    observation_snapshots: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[int, ...]:
    """Validate exact layer lengths, aligned K/V, and the untouched suffix."""

    if capacity.prefix_length != prompt_length - observation_window_size:
        raise AssertionError("prefix_length does not match the prompt split")
    if (
        capacity.selected_prefix_tokens + observation_window_size
        != capacity.target_cache_tokens
    ):
        raise AssertionError("selected prefix and observation window mismatch")
    lengths = assert_cache_capacity(
        cache=cache,
        prompt_length=prompt_length,
        target_capacity=capacity.target_cache_tokens,
        skip_layers=skip_layers,
        compressed=True,
    )
    if len(observation_snapshots) != num_cache_layers(cache):
        raise AssertionError("missing observation-window snapshots")

    for layer_idx in range(num_cache_layers(cache)):
        keys, values = get_cache_layer(cache, layer_idx)
        if keys.shape != values.shape:
            raise AssertionError(
                f"Layer {layer_idx} K/V shapes differ after compression"
            )
        expected_keys, expected_values = observation_snapshots[layer_idx]
        actual_keys = keys[:, :, -observation_window_size:, :]
        actual_values = values[:, :, -observation_window_size:, :]
        if not torch.equal(actual_keys, expected_keys):
            raise AssertionError(
                f"Layer {layer_idx} did not preserve the observation keys"
            )
        if not torch.equal(actual_values, expected_values):
            raise AssertionError(
                f"Layer {layer_idx} did not preserve the observation values"
            )
    return lengths


@torch.inference_mode()
def execute_baseline(
    model: Any,
    tokenizer: Any,
    settings: BenchmarkSettings,
    prompt: Any,
    context_length: int,
    depth: float,
    seed: int,
    configuration: SnapKVConfiguration,
    devices: Sequence[torch.device],
) -> dict[str, Any]:
    """Run the uncompressed reference without collecting attention maps."""

    print(
        f"{configuration.config} | context_len={context_length} | "
        f"depth={depth} | seed={seed}"
    )
    synchronize_cuda_devices(devices)
    started = perf_counter()
    prefill = prefill_plain(
        model=model,
        prompt_ids=prompt.prompt_ids,
        chunk_size=settings.chunk_size,
    )
    if prefill.logical_position != context_length:
        raise AssertionError(
            "Baseline logical position after prefill must equal prompt length"
        )
    cache = prefill.cache
    cache_mb_before = kv_cache_size_mb(cache)
    assert_cache_capacity(
        cache=cache,
        prompt_length=context_length,
        target_capacity=context_length,
        skip_layers=settings.skip_layers,
        compressed=False,
    )
    cache_mb_after = kv_cache_size_mb(cache)
    if cache_mb_after != cache_mb_before:
        raise AssertionError("no_compression unexpectedly changed cache memory")

    generated_text, prediction, cache = generate_greedy(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        last_logits=prefill.last_logits,
        logical_position=context_length,
        max_new_tokens=settings.max_new_tokens,
        validate_cache_growth=True,
    )
    synchronize_cuda_devices(devices)
    elapsed_seconds = perf_counter() - started
    row = _base_result_fields(
        settings,
        prompt,
        context_length,
        depth,
        seed,
        configuration,
    )
    row.update(
        {
            "generated_text": generated_text,
            "prediction": prediction,
            "correct": prediction == prompt.target_value,
            "baseline_correct": pd.NA,
            "skipped_due_to_baseline_failure": False,
            "cache_mb_before_compression": cache_mb_before,
            "cache_mb_after_compression": cache_mb_after,
            "final_cache_mb": kv_cache_size_mb(cache),
            "memory_saved_percent": 0.0,
            "elapsed_seconds": elapsed_seconds,
        }
    )
    del cache
    del prefill
    return row


@torch.inference_mode()
def execute_snapkv(
    model: Any,
    tokenizer: Any,
    settings: BenchmarkSettings,
    prompt: Any,
    context_length: int,
    depth: float,
    seed: int,
    configuration: SnapKVConfiguration,
    devices: Sequence[torch.device],
) -> dict[str, Any]:
    """Run the single fixed SnapKV policy and validate its cache contract."""

    print(
        f"{configuration.config} | context_len={context_length} | "
        f"depth={depth} | seed={seed}"
    )
    if prompt.context_len_actual != context_length:
        raise AssertionError("prompt_length must equal context_len_target")
    capacity = configuration.capacity
    if (
        capacity.prefix_length
        != context_length - settings.observation_window_size
    ):
        raise AssertionError("prefix_length does not match prompt_length - obs")
    if capacity.selected_prefix_tokens < 0:
        raise AssertionError("selected_prefix_tokens must be non-negative")

    synchronize_cuda_devices(devices)
    started = perf_counter()
    prefill = prefill_snapkv(
        model=model,
        prompt_ids=prompt.prompt_ids,
        observation_window_size=settings.observation_window_size,
        chunk_size=settings.chunk_size,
        skip_layers=settings.skip_layers,
    )
    if prefill.logical_position != context_length:
        raise AssertionError(
            "SnapKV logical position after prefill must equal prompt length"
        )
    if prefill.scores_by_layer is None:
        raise AssertionError("SnapKV prefill did not return attention scores")

    cache = prefill.cache
    last_logits = prefill.last_logits
    cache_mb_before = kv_cache_size_mb(cache)
    assert_cache_capacity(
        cache=cache,
        prompt_length=context_length,
        target_capacity=context_length,
        skip_layers=settings.skip_layers,
        compressed=False,
    )
    observation_snapshots = _snapshot_observation_window(
        cache,
        settings.observation_window_size,
    )
    cache = compress_snapkv_cache(
        cache=cache,
        scores_by_layer=prefill.scores_by_layer,
        target_capacity=capacity.target_cache_tokens,
        observation_window_size=settings.observation_window_size,
        pooling_kernel_size=settings.pooling_kernel_size,
        pooling_mode=settings.pooling_mode,
        skip_layers=settings.skip_layers,
    )
    assert_snapkv_cache_contract(
        cache,
        prompt_length=context_length,
        capacity=capacity,
        observation_window_size=settings.observation_window_size,
        skip_layers=settings.skip_layers,
        observation_snapshots=observation_snapshots,
    )
    del observation_snapshots
    del prefill
    cache_mb_after = kv_cache_size_mb(cache)
    memory_saved_percent = 100.0 * (
        1.0 - cache_mb_after / cache_mb_before
    )

    generated_text, prediction, cache = generate_greedy(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        last_logits=last_logits,
        logical_position=context_length,
        max_new_tokens=settings.max_new_tokens,
        validate_cache_growth=True,
    )
    synchronize_cuda_devices(devices)
    elapsed_seconds = perf_counter() - started
    row = _base_result_fields(
        settings,
        prompt,
        context_length,
        depth,
        seed,
        configuration,
    )
    row.update(
        {
            "generated_text": generated_text,
            "prediction": prediction,
            "correct": prediction == prompt.target_value,
            "baseline_correct": pd.NA,
            "skipped_due_to_baseline_failure": False,
            "cache_mb_before_compression": cache_mb_before,
            "cache_mb_after_compression": cache_mb_after,
            "final_cache_mb": kv_cache_size_mb(cache),
            "memory_saved_percent": memory_saved_percent,
            "elapsed_seconds": elapsed_seconds,
        }
    )
    del cache
    del last_logits
    return row


@torch.inference_mode()
def run_benchmark(
    model: Any,
    tokenizer: Any,
    settings: BenchmarkSettings,
    raw_path: Path,
) -> pd.DataFrame:
    """Run each prompt baseline-first and checkpoint every planned row."""

    rows: list[dict[str, Any]] = []
    devices = cuda_devices(model)

    for context_length in settings.context_lengths:
        configurations = build_configurations(settings, context_length)
        baseline_configuration = next(
            (config for config in configurations if config.strategy == "none"),
            None,
        )
        snapkv_configuration = next(
            config for config in configurations if config.strategy == "snapkv"
        )
        for depth in settings.depths:
            for seed in settings.seeds:
                prompt = build_prompt_for_case(
                    tokenizer=tokenizer,
                    context_length=context_length,
                    depth=depth,
                    seed=seed,
                    prompt_style=settings.prompt_style,
                    observation_window_size=settings.observation_window_size,
                )
                if prompt.context_len_actual != context_length:
                    raise AssertionError(
                        f"Prompt length {prompt.context_len_actual} != "
                        f"{context_length}"
                    )

                baseline_correct: bool | None = None
                if baseline_configuration is not None:
                    baseline_row = execute_baseline(
                        model,
                        tokenizer,
                        settings,
                        prompt,
                        context_length,
                        depth,
                        seed,
                        baseline_configuration,
                        devices,
                    )
                    baseline_correct = bool(baseline_row["correct"])
                    baseline_row["baseline_correct"] = baseline_correct
                    rows.append(baseline_row)
                    checkpoint_raw(rows, raw_path)

                if should_skip_snapkv(
                    baseline_correct,
                    settings.require_baseline_success,
                ):
                    print(
                        "Baseline failed; skipping SnapKV for "
                        f"context_len={context_length}, depth={depth}, "
                        f"seed={seed}."
                    )
                    rows.append(
                        make_skipped_result(
                            settings,
                            prompt,
                            context_length,
                            depth,
                            seed,
                            snapkv_configuration,
                        )
                    )
                    checkpoint_raw(rows, raw_path)
                    continue

                snapkv_row = execute_snapkv(
                    model,
                    tokenizer,
                    settings,
                    prompt,
                    context_length,
                    depth,
                    seed,
                    snapkv_configuration,
                    devices,
                )
                snapkv_row["baseline_correct"] = (
                    baseline_correct if baseline_correct is not None else pd.NA
                )
                rows.append(snapkv_row)
                checkpoint_raw(rows, raw_path)

    return _raw_frame(rows)


def parse_args(argv: Sequence[str] | None = None) -> BenchmarkSettings:
    parser = argparse.ArgumentParser(
        description="Run the final baseline and SnapKV retrieval benchmark."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=CONTEXT_LENGTHS,
    )
    parser.add_argument("--depths", type=float, nargs="+", default=DEPTHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--observation-window-size",
        type=int,
        default=OBSERVATION_WINDOW_SIZE,
    )
    capacity_group = parser.add_mutually_exclusive_group()
    capacity_group.add_argument("--target-cache-tokens", type=int)
    capacity_group.add_argument("--keep-ratio", type=float)
    parser.add_argument(
        "--pooling-kernel-size",
        type=int,
        default=POOLING_KERNEL_SIZE,
    )
    parser.add_argument(
        "--pooling-mode",
        choices=("max", "avg", "mean"),
        default=POOLING_MODE,
    )
    parser.add_argument(
        "--skip-layers",
        type=int,
        nargs="*",
        default=SKIP_LAYERS,
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=INCLUDE_BASELINE,
    )
    parser.add_argument("--max-runs", type=int, default=MAX_RUNS)
    parser.add_argument(
        "--prompt-style",
        choices=("legacy", "explicit_v2"),
        default=PROMPT_STYLE,
    )
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument(
        "--require-baseline-success",
        action=argparse.BooleanOptionalAction,
        default=REQUIRE_BASELINE_SUCCESS,
    )
    args = parser.parse_args(argv)

    target_cache_tokens = args.target_cache_tokens
    if target_cache_tokens is None and args.keep_ratio is None:
        target_cache_tokens = TARGET_CACHE_TOKENS
    settings = BenchmarkSettings(
        model_name=args.model_name,
        context_lengths=tuple(args.context_lengths),
        depths=tuple(args.depths),
        seeds=tuple(args.seeds),
        observation_window_size=args.observation_window_size,
        target_cache_tokens=target_cache_tokens,
        keep_ratio=args.keep_ratio,
        pooling_kernel_size=args.pooling_kernel_size,
        pooling_mode=args.pooling_mode,
        skip_layers=tuple(args.skip_layers),
        chunk_size=args.chunk_size,
        max_new_tokens=args.max_new_tokens,
        include_baseline=args.include_baseline,
        max_runs=args.max_runs,
        prompt_style=args.prompt_style,
        output_prefix=args.output_prefix,
        require_baseline_success=args.require_baseline_success,
    )
    _validate_settings(settings)
    return settings


def main(argv: Sequence[str] | None = None) -> None:
    settings = parse_args(argv)
    run_counts = calculate_run_counts(settings)
    print(f"Planned baseline runs: {run_counts.baseline_runs}")
    print(f"Planned SnapKV runs: {run_counts.snapkv_runs}")
    print(f"Total planned runs: {run_counts.planned_runs}")
    enforce_run_limit(run_counts, settings.max_runs)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    raw_path = results_dir / f"{settings.output_prefix}_raw.csv"
    summary_path = results_dir / f"{settings.output_prefix}_summary.csv"

    print(f"Loading {settings.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        settings.model_name,
        dtype=DTYPE,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    metadata_configurations = [
        {
            "context_length": context_length,
            **asdict(configuration),
        }
        for context_length in settings.context_lengths
        for configuration in build_configurations(settings, context_length)
    ]
    metadata = make_run_metadata(
        script=Path(__file__).name,
        model_name=settings.model_name,
        model=model,
        requested_dtype=DTYPE,
        attention_implementation=ATTENTION_IMPLEMENTATION,
        seed=settings.seeds[0],
        lengths=settings.context_lengths,
        depths=settings.depths,
        configurations=metadata_configurations,
        skip_layers=settings.skip_layers,
        extra={
            "benchmark": BENCHMARK_NAME,
            "benchmark_version": BENCHMARK_VERSION,
            "settings": asdict(settings),
            "planned_runs": run_counts.planned_runs,
        },
    )
    print_run_metadata(metadata)
    save_run_metadata(
        results_dir / f"{settings.output_prefix}_metadata.json",
        metadata,
    )

    raw_df = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        settings=settings,
        raw_path=raw_path,
    )
    summary_df = summarize(raw_df)
    summary_df.to_csv(summary_path, index=False)

    print("\nRaw results:")
    print(raw_df.to_string(index=False))
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
