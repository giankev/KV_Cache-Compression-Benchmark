from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
    assert_cache_capacity as _assert_cache_capacity,
    cuda_devices as _cuda_devices,
    generate_greedy as _generate_greedy,
    prefill_plain as _prefill_plain,
    prefill_snapkv as _prefill_snapkv,
    synchronize_cuda_devices as _synchronize,
    target_capacity as _target_capacity,
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
OBSERVATION_WINDOW_SIZE = 64
POOLING_KERNEL_SIZE = 5
POOLING_MODE = "max"
SKIP_LAYERS = (0, 1)
CHUNK_SIZE = 512
MAX_NEW_TOKENS = 12
DTYPE = "auto"
ATTENTION_IMPLEMENTATION = "eager"


EVAL_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "config": "no_compression",
        "strategy": "none",
        "keep_ratio": 1.0,
    },
    {
        "config": "low_l2_keep50",
        "strategy": "low_l2",
        "keep_ratio": 0.5,
    },
    {
        "config": "low_l2_keep10",
        "strategy": "low_l2",
        "keep_ratio": 0.1,
    },
    {
        "config": "random_keep50",
        "strategy": "random",
        "keep_ratio": 0.5,
    },
    {
        "config": "high_l2_keep50",
        "strategy": "high_l2",
        "keep_ratio": 0.5,
    },
    {
        "config": "snapkv_keep50",
        "strategy": "snapkv",
        "keep_ratio": 0.5,
    },
    {
        "config": "snapkv_keep10",
        "strategy": "snapkv",
        "keep_ratio": 0.1,
    },
)


RAW_COLUMNS = [
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


SUMMARY_COLUMNS = [
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


@dataclass(frozen=True)
class BenchmarkSettings:
    model_name: str
    context_lengths: tuple[int, ...]
    depths: tuple[float, ...]
    seeds: tuple[int, ...]
    observation_window_size: int
    pooling_kernel_size: int
    pooling_mode: str
    skip_layers: tuple[int, ...]
    chunk_size: int
    max_new_tokens: int
    dtype: str
    attention_implementation: str
    output_prefix: str


def _validate_settings(settings: BenchmarkSettings) -> None:
    if not settings.context_lengths or any(
        length < 1 for length in settings.context_lengths
    ):
        raise ValueError("context lengths must be positive integers")
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
        settings.pooling_kernel_size < 1
        or settings.pooling_kernel_size % 2 == 0
    ):
        raise ValueError("pooling_kernel_size must be a positive odd integer")
    if settings.pooling_mode not in {"max", "mean"}:
        raise ValueError("pooling_mode must be 'max' or 'mean'")
    if settings.chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if settings.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if any(layer < 0 for layer in settings.skip_layers):
        raise ValueError("skip_layers must contain non-negative indices")
    if len(set(settings.skip_layers)) != len(settings.skip_layers):
        raise ValueError("skip_layers must not contain duplicates")
    if settings.attention_implementation != "eager":
        raise ValueError("SnapKV requires attention_implementation='eager'")

    for context_length in settings.context_lengths:
        if settings.observation_window_size > context_length:
            raise ValueError(
                "observation_window_size must not exceed a context length"
            )
        smallest_capacity = math.floor(0.1 * context_length)
        if smallest_capacity < settings.observation_window_size:
            raise ValueError(
                "The keep10 target capacity must be at least the observation "
                f"window: floor(0.1 * {context_length})={smallest_capacity} < "
                f"{settings.observation_window_size}"
            )


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df.groupby(
            ["config", "strategy", "keep_ratio", "context_len_target"],
            as_index=False,
        )
        .agg(
            num_examples=("correct", "size"),
            accuracy=("correct", "mean"),
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
        .rename(columns={"context_len_target": "context_len"})
        .loc[:, SUMMARY_COLUMNS]
    )
    for column in SUMMARY_COLUMNS[5:]:
        summary[column] = summary[column].round(4)
    return summary


@torch.inference_mode()
def run_benchmark(
    model: Any,
    tokenizer: Any,
    settings: BenchmarkSettings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cuda_devices = _cuda_devices(model)

    for context_length in settings.context_lengths:
        for depth in settings.depths:
            for seed in settings.seeds:
                prompt = make_kv_retrieval_prompt(
                    tokenizer=tokenizer,
                    target_tokens=context_length,
                    observation_window_size=settings.observation_window_size,
                    seed=seed,
                    depth=depth,
                )
                if prompt.context_len_actual != context_length:
                    raise AssertionError(
                        f"Prompt length {prompt.context_len_actual} != {context_length}"
                    )

                lengths_for_ratio: dict[float, tuple[int, ...]] = {}
                for config in EVAL_CONFIGS:
                    config_name = str(config["config"])
                    strategy = str(config["strategy"])
                    keep_ratio = float(config["keep_ratio"])
                    compressed = strategy != "none"
                    target_capacity = _target_capacity(context_length, keep_ratio)
                    if target_capacity < settings.observation_window_size:
                        raise ValueError(
                            f"{config_name}: target capacity {target_capacity} is "
                            "smaller than the observation window"
                        )

                    print(
                        f"{config_name} | context_len={context_length} | "
                        f"depth={depth} | seed={seed}"
                    )
                    _synchronize(cuda_devices)
                    started = perf_counter()

                    if strategy == "snapkv":
                        prefill = _prefill_snapkv(
                            model,
                            prompt.prompt_ids,
                            settings.observation_window_size,
                            settings.chunk_size,
                            settings.skip_layers,
                        )
                    else:
                        prefill = _prefill_plain(
                            model,
                            prompt.prompt_ids,
                            settings.chunk_size,
                        )

                    if prefill.logical_position != context_length:
                        raise AssertionError(
                            "Logical position after prefill does not match the prompt"
                        )
                    cache = prefill.cache
                    last_logits = prefill.last_logits
                    logical_position = prefill.logical_position
                    cache_mb_before = kv_cache_size_mb(cache)

                    if strategy == "snapkv":
                        if prefill.scores_by_layer is None:
                            raise AssertionError("SnapKV did not return layer scores")
                        cache = compress_snapkv_cache(
                            cache=cache,
                            scores_by_layer=prefill.scores_by_layer,
                            target_capacity=target_capacity,
                            observation_window_size=(
                                settings.observation_window_size
                            ),
                            pooling_kernel_size=settings.pooling_kernel_size,
                            pooling_mode=settings.pooling_mode,
                            skip_layers=settings.skip_layers,
                        )
                    elif strategy != "none":
                        cache = compress_cache_to_budget(
                            cache,
                            max_cache_tokens=target_capacity,
                            strategy=strategy,
                            skip_layers=settings.skip_layers,
                            seed=seed,
                        )

                    cache_mb_after = kv_cache_size_mb(cache)
                    lengths_after = _assert_cache_capacity(
                        cache=cache,
                        prompt_length=context_length,
                        target_capacity=target_capacity,
                        skip_layers=settings.skip_layers,
                        compressed=compressed,
                    )
                    if compressed:
                        previous_lengths = lengths_for_ratio.setdefault(
                            keep_ratio,
                            lengths_after,
                        )
                        if lengths_after != previous_lengths:
                            raise AssertionError(
                                "Compressed methods with the same keep ratio have "
                                "different physical cache lengths"
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
                    generated_text, prediction, cache = _generate_greedy(
                        model=model,
                        tokenizer=tokenizer,
                        cache=cache,
                        last_logits=last_logits,
                        logical_position=logical_position,
                        max_new_tokens=settings.max_new_tokens,
                    )
                    _synchronize(cuda_devices)
                    elapsed_seconds = perf_counter() - started

                    rows.append(
                        {
                            "model_name": settings.model_name,
                            "benchmark": "kv_retrieval",
                            "config": config_name,
                            "strategy": strategy,
                            "keep_ratio": keep_ratio,
                            "context_len_target": context_length,
                            "context_len_actual": prompt.context_len_actual,
                            "depth_target": depth,
                            "target_record_token_position": (
                                prompt.target_record_token_position
                            ),
                            "seed": seed,
                            "target_key": prompt.target_key,
                            "target_value": prompt.target_value,
                            "generated_text": generated_text,
                            "prediction": prediction,
                            "correct": prediction == prompt.target_value,
                            "skip_layers": json.dumps(
                                list(settings.skip_layers)
                            ),
                            "observation_window_size": (
                                settings.observation_window_size
                            ),
                            "pooling_kernel_size": (
                                settings.pooling_kernel_size
                            ),
                            "target_cache_tokens": target_capacity,
                            "cache_mb_before_compression": cache_mb_before,
                            "cache_mb_after_compression": cache_mb_after,
                            "final_cache_mb": kv_cache_size_mb(cache),
                            "memory_saved_percent": memory_saved_percent,
                            "elapsed_seconds": elapsed_seconds,
                        }
                    )
                    # Avoid overlapping the next full 32k prefill with the
                    # final cache retained by the preceding configuration.
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


def parse_args(argv: Sequence[str] | None = None) -> BenchmarkSettings:
    parser = argparse.ArgumentParser(
        description="Run the synthetic key-value retrieval benchmark."
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
    parser.add_argument(
        "--pooling-kernel-size",
        type=int,
        default=POOLING_KERNEL_SIZE,
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
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument(
        "--attention-implementation",
        default=ATTENTION_IMPLEMENTATION,
    )
    parser.add_argument("--output-prefix", default="kv_retrieval")
    args = parser.parse_args(argv)

    settings = BenchmarkSettings(
        model_name=args.model_name,
        context_lengths=tuple(args.context_lengths),
        depths=tuple(args.depths),
        seeds=tuple(args.seeds),
        observation_window_size=args.observation_window_size,
        pooling_kernel_size=args.pooling_kernel_size,
        pooling_mode=args.pooling_mode,
        skip_layers=tuple(args.skip_layers),
        chunk_size=args.chunk_size,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        output_prefix=args.output_prefix,
    )
    _validate_settings(settings)
    return settings


def main(argv: Sequence[str] | None = None) -> None:
    settings = parse_args(argv)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"Loading {settings.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        settings.model_name,
        dtype=settings.dtype,
        attn_implementation=settings.attention_implementation,
    )

    metadata = make_run_metadata(
        script=Path(__file__).name,
        model_name=settings.model_name,
        model=model,
        requested_dtype=settings.dtype,
        attention_implementation=settings.attention_implementation,
        seed=settings.seeds[0],
        lengths=settings.context_lengths,
        depths=settings.depths,
        configurations=EVAL_CONFIGS,
        skip_layers=settings.skip_layers,
        extra={
            "benchmark": "kv_retrieval",
            "seeds": settings.seeds,
            "observation_window_size": settings.observation_window_size,
            "pooling_kernel_size": settings.pooling_kernel_size,
            "pooling_mode": settings.pooling_mode,
            "chunk_size": settings.chunk_size,
            "max_new_tokens": settings.max_new_tokens,
        },
    )
    print_run_metadata(metadata)
    save_run_metadata(results_dir / "run_metadata.json", metadata)

    raw_df = run_benchmark(model, tokenizer, settings)
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
