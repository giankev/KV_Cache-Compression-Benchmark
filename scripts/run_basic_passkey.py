from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.configs import BASIC_PASSKEY_CONFIGS, get_default_skip_layers
from l2kv.model_utils import load_model_and_tokenizer
from l2kv.passkey import (
    generate_passkey_answer,
    is_correct_prediction,
    make_passkey_prompt,
)
from l2kv.runtime_metadata import (
    make_run_metadata,
    print_run_metadata,
    save_run_metadata,
)


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CONTEXT_LENGTHS = [8192, 32768]
DEPTHS = [0.25, 0.50, 0.75]
SEEDS = [0]
MAX_NEW_TOKENS = 12
PRUNE_AFTER = 1024
CHUNK_SIZE = 512
DTYPE = "auto"
ATTN_IMPLEMENTATION = "eager"

RAW_COLUMNS = [
    "model_name",
    "config",
    "strategy",
    "keep_ratio",
    "context_len_target",
    "context_len_actual",
    "depth_target",
    "needle_token_position",
    "seed",
    "answer",
    "generated_text",
    "prediction",
    "correct",
    "skip_layers",
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
    max_new_tokens: int
    prune_after: int
    chunk_size: int
    dtype: str
    attention_implementation: str
    output_prefix: str


def make_eval_configs(skip_layers: tuple[int, ...]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for config in BASIC_PASSKEY_CONFIGS:
        copied = dict(config)
        copied["skip_layers"] = skip_layers
        configs.append(copied)
    return configs


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

    for column in (
        "accuracy",
        "mean_cache_mb_before_compression",
        "mean_cache_mb_after_compression",
        "mean_memory_saved_percent",
        "mean_elapsed_seconds",
    ):
        summary[column] = summary[column].round(4)
    return summary


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
    model: Any,
    tokenizer: Any,
    settings: BenchmarkSettings,
) -> pd.DataFrame:
    skip_layers = get_default_skip_layers()
    eval_configs = make_eval_configs(skip_layers)
    rows: list[dict[str, Any]] = []
    device = next(model.parameters()).device

    for context_len in settings.context_lengths:
        for depth in settings.depths:
            for seed in settings.seeds:
                prompt = make_passkey_prompt(
                    tokenizer=tokenizer,
                    target_tokens=context_len,
                    seed=seed,
                    depth=depth,
                )
                if prompt.context_len_actual != context_len:
                    raise AssertionError(
                        f"Context length {prompt.context_len_actual} != {context_len}"
                    )

                for config in eval_configs:
                    print(
                        f"{config['config']} | context_len={context_len} | "
                        f"depth={depth} | seed={seed}"
                    )
                    _synchronize(device)
                    started = perf_counter()
                    result = generate_passkey_answer(
                        model=model,
                        tokenizer=tokenizer,
                        context_ids=prompt.context_ids,
                        question_ids=prompt.question_ids,
                        max_new_tokens=settings.max_new_tokens,
                        expected_digits=len(prompt.answer),
                        use_compression=config["use_compression"],
                        keep_ratio=config["keep_ratio"],
                        prune_after=settings.prune_after,
                        chunk_size=settings.chunk_size,
                        strategy=config["strategy"],
                        skip_layers=config["skip_layers"],
                        compression_seed=seed,
                    )
                    _synchronize(device)
                    elapsed_seconds = perf_counter() - started
                    prediction = result["prediction"]

                    rows.append(
                        {
                            "model_name": settings.model_name,
                            "config": config["config"],
                            "strategy": config["strategy"],
                            "keep_ratio": config["keep_ratio"],
                            "context_len_target": context_len,
                            "context_len_actual": prompt.context_len_actual,
                            "depth_target": depth,
                            "needle_token_position": prompt.needle_token_position,
                            "seed": seed,
                            "answer": prompt.answer,
                            "generated_text": result["generated_text"],
                            "prediction": prediction,
                            "correct": is_correct_prediction(
                                prediction,
                                prompt.answer,
                            ),
                            "skip_layers": json.dumps(list(config["skip_layers"])),
                            "cache_mb_before_compression": result[
                                "cache_mb_before_compression"
                            ],
                            "cache_mb_after_compression": result[
                                "cache_mb_after_compression"
                            ],
                            "final_cache_mb": result["final_cache_mb"],
                            "memory_saved_percent": result[
                                "memory_saved_percent"
                            ],
                            "elapsed_seconds": elapsed_seconds,
                        }
                    )

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
    parser = argparse.ArgumentParser(description="Run the passkey KV-cache benchmark.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=CONTEXT_LENGTHS,
    )
    parser.add_argument("--depths", type=float, nargs="+", default=DEPTHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--prune-after", type=int, default=PRUNE_AFTER)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument(
        "--attention-implementation",
        default=ATTN_IMPLEMENTATION,
    )
    parser.add_argument("--output-prefix", default="basic_passkey")
    args = parser.parse_args(argv)
    return BenchmarkSettings(
        model_name=args.model_name,
        context_lengths=tuple(args.context_lengths),
        depths=tuple(args.depths),
        seeds=tuple(args.seeds),
        max_new_tokens=args.max_new_tokens,
        prune_after=args.prune_after,
        chunk_size=args.chunk_size,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        output_prefix=args.output_prefix,
    )


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

    skip_layers = get_default_skip_layers()
    eval_configs = make_eval_configs(skip_layers)
    metadata = make_run_metadata(
        script=Path(__file__).name,
        model_name=settings.model_name,
        model=model,
        requested_dtype=settings.dtype,
        attention_implementation=settings.attention_implementation,
        seed=settings.seeds[0],
        lengths=settings.context_lengths,
        depths=settings.depths,
        configurations=[
            {
                "config": config["config"],
                "strategy": config["strategy"],
                "keep_ratio": config["keep_ratio"],
            }
            for config in eval_configs
        ],
        skip_layers=skip_layers,
        extra={
            "seeds": settings.seeds,
            "prune_after": settings.prune_after,
            "chunk_size": settings.chunk_size,
            "max_new_tokens": settings.max_new_tokens,
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
