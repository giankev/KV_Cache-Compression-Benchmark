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

from l2kv.cache_compression import compress_cache_to_budget
from l2kv.cache_metrics import kv_cache_size_mb
from l2kv.kv_retrieval import make_kv_retrieval_prompt
from l2kv.kv_retrieval_eval import (
    assert_cache_capacity,
    cuda_devices,
    generate_greedy,
    prefill_plain,
    synchronize_cuda_devices,
    target_capacity,
)
from l2kv.model_utils import load_model_and_tokenizer
from l2kv.runtime_metadata import (
    make_run_metadata,
    print_run_metadata,
    save_run_metadata,
)


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CONTEXT_LENGTHS = (8192,)
DEPTHS = (0.25, 0.50, 0.75)
SEEDS = (0,)
SKIP_LAYERS = (0, 1)
CHUNK_SIZE = 512
MAX_NEW_TOKENS = 24
MAX_RUNS = 15
PROMPT_STYLE = "explicit_v2"
OUTPUT_PREFIX = "l2_kv_retrieval"
REQUIRE_BASELINE_SUCCESS = True
PROMPT_OBSERVATION_WINDOW_SIZE = 16
DTYPE = "auto"
ATTENTION_IMPLEMENTATION = None
BENCHMARK_NAME = "kv_retrieval"
BENCHMARK_VERSION = "kv_retrieval_v2"


@dataclass(frozen=True)
class L2Configuration:
    config: str
    strategy: str
    keep_ratio: float


CONFIGURATIONS: tuple[L2Configuration, ...] = (
    L2Configuration("no_compression", "none", 1.0),
    L2Configuration("low_l2_keep50", "low_l2", 0.5),
    L2Configuration("low_l2_keep10", "low_l2", 0.1),
    L2Configuration("random_keep50", "random", 0.5),
    L2Configuration("high_l2_keep50", "high_l2", 0.5),
)


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
    "keep_ratio",
    "target_cache_tokens",
]


SUMMARY_COLUMNS = [
    "benchmark",
    "benchmark_version",
    "prompt_style",
    "config",
    "strategy",
    "keep_ratio",
    "target_cache_tokens",
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
    skip_layers: tuple[int, ...]
    chunk_size: int
    max_new_tokens: int
    max_runs: int
    prompt_style: str
    output_prefix: str
    require_baseline_success: bool


@dataclass(frozen=True)
class RunCounts:
    baseline_runs: int
    compressed_runs: int
    planned_runs: int


def build_configurations() -> tuple[L2Configuration, ...]:
    """Return the fixed, deliberately non-grid L2 evaluation contract."""

    return CONFIGURATIONS


def calculate_run_counts(settings: BenchmarkSettings) -> RunCounts:
    """Count one baseline and four compressed runs for every prompt."""

    prompt_count = (
        len(settings.context_lengths)
        * len(settings.depths)
        * len(settings.seeds)
    )
    baseline_runs = prompt_count
    compressed_runs = prompt_count * (len(CONFIGURATIONS) - 1)
    return RunCounts(
        baseline_runs=baseline_runs,
        compressed_runs=compressed_runs,
        planned_runs=baseline_runs + compressed_runs,
    )


def enforce_run_limit(run_counts: RunCounts, max_runs: int) -> None:
    if run_counts.planned_runs > max_runs:
        raise ValueError(
            f"Total planned runs ({run_counts.planned_runs}) exceed "
            f"max_runs ({max_runs})"
        )


def build_prompt_for_case(
    tokenizer: Any,
    context_length: int,
    depth: float,
    seed: int,
    prompt_style: str,
) -> Any:
    """Build the shared deterministic input for one evaluation case."""

    return make_kv_retrieval_prompt(
        tokenizer=tokenizer,
        target_tokens=context_length,
        seed=seed,
        depth=depth,
        observation_window_size=PROMPT_OBSERVATION_WINDOW_SIZE,
        prompt_style=prompt_style,
    )


def _validate_settings(settings: BenchmarkSettings) -> None:
    if not settings.context_lengths or any(
        length < 1 for length in settings.context_lengths
    ):
        raise ValueError("context_lengths must contain positive integers")
    if any(
        length < PROMPT_OBSERVATION_WINDOW_SIZE
        for length in settings.context_lengths
    ):
        raise ValueError(
            "every context length must be at least "
            f"{PROMPT_OBSERVATION_WINDOW_SIZE} tokens"
        )
    if not settings.depths or any(
        not math.isfinite(depth) or not 0 <= depth <= 1
        for depth in settings.depths
    ):
        raise ValueError("depths must contain finite values between 0 and 1")
    if not settings.seeds:
        raise ValueError("at least one seed is required")
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

    for context_length in settings.context_lengths:
        for configuration in CONFIGURATIONS[1:]:
            capacity = target_capacity(
                context_length,
                configuration.keep_ratio,
            )
            if capacity < 1:
                raise ValueError(
                    f"{configuration.config} produces an empty cache capacity "
                    f"for context length {context_length}"
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
    return raw_df


def checkpoint_raw(
    rows: Sequence[dict[str, Any]],
    raw_path: Path,
) -> pd.DataFrame:
    """Rewrite the raw CSV with every result accumulated so far."""

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
                "keep_ratio",
                "target_cache_tokens",
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
    for column in SUMMARY_COLUMNS[10:]:
        summary[column] = summary[column].round(4)
    return summary


def _base_result_fields(
    settings: BenchmarkSettings,
    prompt: Any,
    context_length: int,
    depth: float,
    seed: int,
    configuration: L2Configuration,
) -> dict[str, Any]:
    capacity = target_capacity(context_length, configuration.keep_ratio)
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
        "keep_ratio": configuration.keep_ratio,
        "target_cache_tokens": capacity,
    }


def make_skipped_result(
    settings: BenchmarkSettings,
    prompt: Any,
    context_length: int,
    depth: float,
    seed: int,
    configuration: L2Configuration,
) -> dict[str, Any]:
    """Represent a compressed run suppressed by the baseline gate."""

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


@torch.inference_mode()
def execute_configuration(
    model: Any,
    tokenizer: Any,
    settings: BenchmarkSettings,
    prompt: Any,
    context_length: int,
    depth: float,
    seed: int,
    configuration: L2Configuration,
    devices: Sequence[torch.device],
) -> dict[str, Any]:
    """Execute one baseline or L2-family cache policy."""

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
            "Logical position after prefill does not match the prompt"
        )

    cache = prefill.cache
    last_logits = prefill.last_logits
    logical_position = prefill.logical_position
    cache_mb_before = kv_cache_size_mb(cache)
    capacity = target_capacity(context_length, configuration.keep_ratio)
    compressed = configuration.strategy != "none"

    if compressed:
        cache = compress_cache_to_budget(
            cache,
            max_cache_tokens=capacity,
            strategy=configuration.strategy,
            skip_layers=settings.skip_layers,
            seed=seed,
        )

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
    """Run baseline-first prompts and checkpoint each resulting raw row."""

    rows: list[dict[str, Any]] = []
    devices = cuda_devices(model)
    baseline_configuration = CONFIGURATIONS[0]

    for context_length in settings.context_lengths:
        for depth in settings.depths:
            for seed in settings.seeds:
                prompt = build_prompt_for_case(
                    tokenizer=tokenizer,
                    context_length=context_length,
                    depth=depth,
                    seed=seed,
                    prompt_style=settings.prompt_style,
                )
                if prompt.context_len_actual != context_length:
                    raise AssertionError(
                        f"Prompt length {prompt.context_len_actual} != "
                        f"{context_length}"
                    )

                baseline_row = execute_configuration(
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

                if (
                    settings.require_baseline_success
                    and not baseline_correct
                ):
                    print(
                        "Baseline failed; skipping compressed configurations "
                        f"for context_len={context_length}, depth={depth}, "
                        f"seed={seed}."
                    )
                    for configuration in CONFIGURATIONS[1:]:
                        rows.append(
                            make_skipped_result(
                                settings,
                                prompt,
                                context_length,
                                depth,
                                seed,
                                configuration,
                            )
                        )
                        checkpoint_raw(rows, raw_path)
                    continue

                for configuration in CONFIGURATIONS[1:]:
                    row = execute_configuration(
                        model,
                        tokenizer,
                        settings,
                        prompt,
                        context_length,
                        depth,
                        seed,
                        configuration,
                        devices,
                    )
                    row["baseline_correct"] = baseline_correct
                    rows.append(row)
                    checkpoint_raw(rows, raw_path)

    return _raw_frame(rows)


def parse_args(argv: Sequence[str] | None = None) -> BenchmarkSettings:
    parser = argparse.ArgumentParser(
        description="Run the final L2-family key-value retrieval benchmark."
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

    settings = BenchmarkSettings(
        model_name=args.model_name,
        context_lengths=tuple(args.context_lengths),
        depths=tuple(args.depths),
        seeds=tuple(args.seeds),
        skip_layers=tuple(args.skip_layers),
        chunk_size=args.chunk_size,
        max_new_tokens=args.max_new_tokens,
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
    print(f"Planned compressed runs: {run_counts.compressed_runs}")
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
    metadata = make_run_metadata(
        script=Path(__file__).name,
        model_name=settings.model_name,
        model=model,
        requested_dtype=DTYPE,
        attention_implementation=ATTENTION_IMPLEMENTATION,
        seed=settings.seeds[0],
        lengths=settings.context_lengths,
        depths=settings.depths,
        configurations=[asdict(config) for config in CONFIGURATIONS],
        skip_layers=settings.skip_layers,
        extra={
            "benchmark": BENCHMARK_NAME,
            "benchmark_version": BENCHMARK_VERSION,
            "prompt_style": settings.prompt_style,
            "seeds": settings.seeds,
            "chunk_size": settings.chunk_size,
            "max_new_tokens": settings.max_new_tokens,
            "require_baseline_success": settings.require_baseline_success,
            "max_runs": settings.max_runs,
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
