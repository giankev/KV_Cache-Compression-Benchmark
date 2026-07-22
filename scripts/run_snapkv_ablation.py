from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.cache_metrics import kv_cache_size_mb
from l2kv.kv_retrieval import make_kv_retrieval_prompt
from l2kv.kv_retrieval_eval import (
    assert_cache_capacity,
    cuda_devices,
    generate_greedy,
    prefill_plain,
    prefill_snapkv,
    synchronize_cuda_devices,
    target_capacity,
)
from l2kv.model_utils import load_model_and_tokenizer
from l2kv.runtime_metadata import (
    make_run_metadata,
    print_run_metadata,
    save_run_metadata,
)
from l2kv.snapkv import compress_snapkv_cache


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CONTEXT_LENGTHS = (32000,)
DEPTHS = (0.25, 0.50, 0.75)
SEEDS = (0,)
OBSERVATION_WINDOW_SIZES = (16, 32, 64)
KEEP_RATIOS = (0.10,)
POOLING_KERNEL_SIZES = (5,)
POOLING_MODE = "max"
SKIP_LAYERS = (0, 1)
CHUNK_SIZE = 512
MAX_NEW_TOKENS = 24
INCLUDE_BASELINE = True
MAX_RUNS = 12
OUTPUT_PREFIX = "snapkv_ablation"
DTYPE = "auto"
ATTENTION_IMPLEMENTATION = "eager"
BENCHMARK_NAME = "snapkv_ablation_kv_retrieval"


RAW_COLUMNS = [
    "model_name",
    "benchmark",
    "config",
    "strategy",
    "context_len_target",
    "context_len_actual",
    "depth_target",
    "seed",
    "target_key",
    "target_value",
    "target_record_token_position",
    "keep_ratio",
    "observation_window_size",
    "prefix_length",
    "pooling_kernel_size",
    "pooling_mode",
    "target_cache_tokens",
    "generated_text",
    "prediction",
    "correct",
    "cache_mb_before_compression",
    "cache_mb_after_compression",
    "final_cache_mb",
    "memory_saved_percent",
    "elapsed_seconds",
]


SUMMARY_COLUMNS = [
    "config",
    "strategy",
    "keep_ratio",
    "observation_window_size",
    "pooling_kernel_size",
    "context_len_target",
    "num_examples",
    "accuracy",
    "mean_memory_saved_percent",
    "mean_elapsed_seconds",
    "mean_cache_mb_before_compression",
    "mean_cache_mb_after_compression",
]


@dataclass(frozen=True)
class AblationSettings:
    model_name: str
    context_lengths: tuple[int, ...]
    depths: tuple[float, ...]
    seeds: tuple[int, ...]
    observation_window_sizes: tuple[int, ...]
    keep_ratios: tuple[float, ...]
    pooling_kernel_sizes: tuple[int, ...]
    pooling_mode: str
    skip_layers: tuple[int, ...]
    chunk_size: int
    max_new_tokens: int
    include_baseline: bool
    max_runs: int
    output_prefix: str


@dataclass(frozen=True)
class RunCounts:
    snapkv_runs: int
    baseline_runs: int
    planned_runs: int


@dataclass(frozen=True)
class AblationConfig:
    config: str
    strategy: str
    keep_ratio: float
    observation_window_size: int
    pooling_kernel_size: int
    pooling_mode: str


@dataclass(frozen=True)
class PlannedRun:
    context_length: int
    depth: float
    seed: int
    prompt: Any
    configuration: AblationConfig


def calculate_run_counts(settings: AblationSettings) -> RunCounts:
    """Count SnapKV grid runs and one optional baseline per prompt."""

    prompt_count = (
        len(settings.context_lengths)
        * len(settings.depths)
        * len(settings.seeds)
    )
    snapkv_runs = (
        prompt_count
        * len(settings.observation_window_sizes)
        * len(settings.keep_ratios)
        * len(settings.pooling_kernel_sizes)
    )
    baseline_runs = prompt_count if settings.include_baseline else 0
    return RunCounts(
        snapkv_runs=snapkv_runs,
        baseline_runs=baseline_runs,
        planned_runs=snapkv_runs + baseline_runs,
    )


def enforce_run_limit(run_counts: RunCounts, max_runs: int) -> None:
    if run_counts.planned_runs > max_runs:
        raise ValueError(
            f"Total planned runs ({run_counts.planned_runs}) exceed "
            f"max_runs ({max_runs})"
        )


def _keep_percentage_label(keep_ratio: float) -> str:
    percentage = Decimal(str(keep_ratio)) * Decimal(100)
    normalized = format(percentage.normalize(), "f")
    return normalized.replace(".", "p")


def snapkv_config_name(
    keep_ratio: float,
    observation_window_size: int,
    pooling_kernel_size: int,
) -> str:
    """Return a deterministic, unambiguous SnapKV configuration name."""

    return (
        f"snapkv_keep{_keep_percentage_label(keep_ratio)}"
        f"_obs{observation_window_size}_k{pooling_kernel_size}"
    )


def build_configurations(settings: AblationSettings) -> tuple[AblationConfig, ...]:
    configurations: list[AblationConfig] = []
    if settings.include_baseline:
        configurations.append(
            AblationConfig(
                config="no_compression",
                strategy="none",
                keep_ratio=1.0,
                observation_window_size=0,
                pooling_kernel_size=0,
                pooling_mode="none",
            )
        )

    for observation_window_size, keep_ratio, pooling_kernel_size in product(
        settings.observation_window_sizes,
        settings.keep_ratios,
        settings.pooling_kernel_sizes,
    ):
        configurations.append(
            AblationConfig(
                config=snapkv_config_name(
                    keep_ratio,
                    observation_window_size,
                    pooling_kernel_size,
                ),
                strategy="snapkv",
                keep_ratio=keep_ratio,
                observation_window_size=observation_window_size,
                pooling_kernel_size=pooling_kernel_size,
                pooling_mode=settings.pooling_mode,
            )
        )
    return tuple(configurations)


def build_execution_plan(
    tokenizer: Any,
    settings: AblationSettings,
    prompt_factory: Callable[..., Any] = make_kv_retrieval_prompt,
) -> tuple[PlannedRun, ...]:
    """Create each base prompt once and share it across all configurations."""

    minimum_observation_window = min(settings.observation_window_sizes)
    configurations = build_configurations(settings)
    runs: list[PlannedRun] = []

    for context_length, depth, seed in product(
        settings.context_lengths,
        settings.depths,
        settings.seeds,
    ):
        prompt = prompt_factory(
            tokenizer=tokenizer,
            target_tokens=context_length,
            observation_window_size=minimum_observation_window,
            seed=seed,
            depth=depth,
        )
        if prompt.context_len_actual != context_length:
            raise AssertionError(
                f"Prompt length {prompt.context_len_actual} != {context_length}"
            )
        for configuration in configurations:
            runs.append(
                PlannedRun(
                    context_length=context_length,
                    depth=depth,
                    seed=seed,
                    prompt=prompt,
                    configuration=configuration,
                )
            )
    return tuple(runs)


def _validate_settings(settings: AblationSettings) -> None:
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
    if not settings.observation_window_sizes or any(
        window < 1 for window in settings.observation_window_sizes
    ):
        raise ValueError("observation_window_sizes must be positive integers")
    if not settings.keep_ratios or any(
        not math.isfinite(ratio) or not 0 < ratio <= 1
        for ratio in settings.keep_ratios
    ):
        raise ValueError("keep_ratios must satisfy 0 < ratio <= 1")
    if not settings.pooling_kernel_sizes or any(
        kernel < 1 or kernel % 2 == 0
        for kernel in settings.pooling_kernel_sizes
    ):
        raise ValueError("pooling_kernel_sizes must contain positive odd integers")
    if settings.pooling_mode not in {"max", "mean"}:
        raise ValueError("pooling_mode must be 'max' or 'mean'")
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
    if not settings.output_prefix:
        raise ValueError("output_prefix must not be empty")

    for context_length, observation_window_size, keep_ratio in product(
        settings.context_lengths,
        settings.observation_window_sizes,
        settings.keep_ratios,
    ):
        if observation_window_size >= context_length:
            raise ValueError(
                "every observation window must be smaller than each context length"
            )
        capacity = target_capacity(context_length, keep_ratio)
        if capacity < observation_window_size:
            raise ValueError(
                f"target capacity {capacity} is smaller than observation window "
                f"{observation_window_size}"
            )


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df.groupby(
            [
                "config",
                "strategy",
                "keep_ratio",
                "observation_window_size",
                "pooling_kernel_size",
                "context_len_target",
            ],
            as_index=False,
        )
        .agg(
            num_examples=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_memory_saved_percent=("memory_saved_percent", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
            mean_cache_mb_before_compression=(
                "cache_mb_before_compression",
                "mean",
            ),
            mean_cache_mb_after_compression=(
                "cache_mb_after_compression",
                "mean",
            ),
        )
        .loc[:, SUMMARY_COLUMNS]
    )
    for column in SUMMARY_COLUMNS[7:]:
        summary[column] = summary[column].round(4)
    return summary


@torch.inference_mode()
def run_benchmark(
    model: Any,
    tokenizer: Any,
    settings: AblationSettings,
) -> pd.DataFrame:
    execution_plan = build_execution_plan(tokenizer, settings)
    devices = cuda_devices(model)
    rows: list[dict[str, Any]] = []

    for planned_run in execution_plan:
        configuration = planned_run.configuration
        prompt = planned_run.prompt
        context_length = planned_run.context_length
        print(
            f"{configuration.config} | context_len={context_length} | "
            f"depth={planned_run.depth} | seed={planned_run.seed}"
        )
        synchronize_cuda_devices(devices)
        started = perf_counter()

        if configuration.strategy == "none":
            prefill = prefill_plain(
                model,
                prompt.prompt_ids,
                settings.chunk_size,
            )
        else:
            prefill = prefill_snapkv(
                model,
                prompt.prompt_ids,
                configuration.observation_window_size,
                settings.chunk_size,
                settings.skip_layers,
            )
        if prefill.logical_position != context_length:
            raise AssertionError(
                "Logical position after prefill does not match the prompt"
            )

        cache = prefill.cache
        last_logits = prefill.last_logits
        logical_position = prefill.logical_position
        cache_mb_before = kv_cache_size_mb(cache)

        if configuration.strategy == "snapkv":
            if prefill.scores_by_layer is None:
                raise AssertionError("SnapKV did not return layer scores")
            capacity = target_capacity(context_length, configuration.keep_ratio)
            cache = compress_snapkv_cache(
                cache=cache,
                scores_by_layer=prefill.scores_by_layer,
                target_capacity=capacity,
                observation_window_size=configuration.observation_window_size,
                pooling_kernel_size=configuration.pooling_kernel_size,
                pooling_mode=configuration.pooling_mode,
                skip_layers=settings.skip_layers,
            )
            prefix_length = (
                context_length - configuration.observation_window_size
            )
            compressed = True
        else:
            capacity = context_length
            prefix_length = context_length
            compressed = False

        cache_mb_after = kv_cache_size_mb(cache)
        assert_cache_capacity(
            cache=cache,
            prompt_length=context_length,
            target_capacity=capacity,
            skip_layers=settings.skip_layers,
            compressed=compressed,
        )
        if compressed:
            memory_saved_percent = 100.0 * (
                1.0 - cache_mb_after / cache_mb_before
            )
        else:
            memory_saved_percent = 0.0
            if cache_mb_after != cache_mb_before:
                raise AssertionError(
                    "no_compression unexpectedly changed cache memory"
                )

        del prefill
        generated_text, prediction, cache = generate_greedy(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            last_logits=last_logits,
            logical_position=logical_position,
            max_new_tokens=settings.max_new_tokens,
        )
        synchronize_cuda_devices(devices)
        elapsed_seconds = perf_counter() - started

        rows.append(
            {
                "model_name": settings.model_name,
                "benchmark": BENCHMARK_NAME,
                "config": configuration.config,
                "strategy": configuration.strategy,
                "context_len_target": context_length,
                "context_len_actual": prompt.context_len_actual,
                "depth_target": planned_run.depth,
                "seed": planned_run.seed,
                "target_key": prompt.target_key,
                "target_value": prompt.target_value,
                "target_record_token_position": (
                    prompt.target_record_token_position
                ),
                "keep_ratio": configuration.keep_ratio,
                "observation_window_size": (
                    configuration.observation_window_size
                ),
                "prefix_length": prefix_length,
                "pooling_kernel_size": configuration.pooling_kernel_size,
                "pooling_mode": configuration.pooling_mode,
                "target_cache_tokens": capacity,
                "generated_text": generated_text,
                "prediction": prediction,
                "correct": prediction == prompt.target_value,
                "cache_mb_before_compression": cache_mb_before,
                "cache_mb_after_compression": cache_mb_after,
                "final_cache_mb": kv_cache_size_mb(cache),
                "memory_saved_percent": memory_saved_percent,
                "elapsed_seconds": elapsed_seconds,
            }
        )
        del cache
        del last_logits

    raw_df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    for column in (
        "cache_mb_before_compression",
        "cache_mb_after_compression",
        "final_cache_mb",
        "memory_saved_percent",
        "elapsed_seconds",
    ):
        raw_df[column] = raw_df[column].round(4)
    return raw_df


def parse_args(argv: Sequence[str] | None = None) -> AblationSettings:
    parser = argparse.ArgumentParser(
        description="Run the controlled SnapKV key-value retrieval ablation."
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
        "--observation-window-sizes",
        type=int,
        nargs="+",
        default=OBSERVATION_WINDOW_SIZES,
    )
    parser.add_argument(
        "--keep-ratios",
        type=float,
        nargs="+",
        default=KEEP_RATIOS,
    )
    parser.add_argument(
        "--pooling-kernel-sizes",
        type=int,
        nargs="+",
        default=POOLING_KERNEL_SIZES,
    )
    parser.add_argument(
        "--pooling-mode",
        choices=("max", "mean"),
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
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    args = parser.parse_args(argv)

    settings = AblationSettings(
        model_name=args.model_name,
        context_lengths=tuple(args.context_lengths),
        depths=tuple(args.depths),
        seeds=tuple(args.seeds),
        observation_window_sizes=tuple(args.observation_window_sizes),
        keep_ratios=tuple(args.keep_ratios),
        pooling_kernel_sizes=tuple(args.pooling_kernel_sizes),
        pooling_mode=args.pooling_mode,
        skip_layers=tuple(args.skip_layers),
        chunk_size=args.chunk_size,
        max_new_tokens=args.max_new_tokens,
        include_baseline=args.include_baseline,
        max_runs=args.max_runs,
        output_prefix=args.output_prefix,
    )
    _validate_settings(settings)
    return settings


def main(argv: Sequence[str] | None = None) -> None:
    settings = parse_args(argv)
    run_counts = calculate_run_counts(settings)
    print(f"Planned SnapKV runs: {run_counts.snapkv_runs}")
    print(f"Planned baseline runs: {run_counts.baseline_runs}")
    print(f"Total planned runs: {run_counts.planned_runs}")
    enforce_run_limit(run_counts, settings.max_runs)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"Loading {settings.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        settings.model_name,
        dtype=DTYPE,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )

    configurations = build_configurations(settings)
    metadata = make_run_metadata(
        script=Path(__file__).name,
        model_name=settings.model_name,
        model=model,
        requested_dtype=DTYPE,
        attention_implementation=ATTENTION_IMPLEMENTATION,
        seed=settings.seeds[0],
        lengths=settings.context_lengths,
        depths=settings.depths,
        configurations=[asdict(configuration) for configuration in configurations],
        skip_layers=settings.skip_layers,
        extra={
            "benchmark": BENCHMARK_NAME,
            "seeds": settings.seeds,
            "observation_window_sizes": settings.observation_window_sizes,
            "keep_ratios": settings.keep_ratios,
            "pooling_kernel_sizes": settings.pooling_kernel_sizes,
            "pooling_mode": settings.pooling_mode,
            "chunk_size": settings.chunk_size,
            "max_new_tokens": settings.max_new_tokens,
            "include_baseline": settings.include_baseline,
            "max_runs": settings.max_runs,
            "planned_snapkv_runs": run_counts.snapkv_runs,
            "planned_baseline_runs": run_counts.baseline_runs,
            "planned_runs": run_counts.planned_runs,
        },
    )
    print_run_metadata(metadata)
    save_run_metadata(results_dir / "run_metadata.json", metadata)

    raw_df = run_benchmark(model=model, tokenizer=tokenizer, settings=settings)
    summary_df = summarize(raw_df)
    raw_path = results_dir / f"{settings.output_prefix}_raw.csv"
    summary_path = results_dir / f"{settings.output_prefix}_summary.csv"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nRaw results:")
    print(raw_df.to_string(index=False))
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
