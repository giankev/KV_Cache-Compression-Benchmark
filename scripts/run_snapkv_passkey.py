"""Run baseline and SnapKV on the professor-style passkey benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.model_utils import load_model_and_tokenizer
from l2kv.passkey import make_passkey_example
from l2kv.retrieval_eval import (
    checkpoint_raw,
    evaluate_plain_or_l2,
    evaluate_snapkv,
    summarize_results,
)
from l2kv.runtime_metadata import (
    make_run_metadata,
    print_run_metadata,
    save_run_metadata,
)


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CONTEXT_LENGTHS = (8192,)
SEEDS = tuple(range(10))
OBSERVATION_WINDOW_SIZE = 16
TARGET_CACHE_TOKENS = 1024
POOLING_KERNEL_SIZE = 5
POOLING_MODE = "max"

# Keep the first two layers intact to match the project comparison protocol.
SKIP_LAYERS = (0, 1)
CHUNK_SIZE = 512
OUTPUT_PREFIX = "snapkv_passkey_3b_8k"
DTYPE = "auto"
ATTENTION_IMPLEMENTATION = "eager"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the baseline and SnapKV passkey benchmark."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=CONTEXT_LENGTHS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--observation-window-size",
        type=int,
        default=OBSERVATION_WINDOW_SIZE,
    )
    parser.add_argument(
        "--target-cache-tokens",
        type=int,
        default=TARGET_CACHE_TOKENS,
    )
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
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    return parser.parse_args(argv)


def run_benchmark(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    raw_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for context_length in args.context_lengths:
        for seed in args.seeds:
            example = make_passkey_example(
                tokenizer=tokenizer,
                context_length=context_length,
                seed=seed,
                observation_window_size=args.observation_window_size,
            )
            print(
                f"no_compression | context={context_length} | seed={seed}"
            )
            baseline = evaluate_plain_or_l2(
                model=model,
                tokenizer=tokenizer,
                model_name=args.model_name,
                example=example,
                config="no_compression",
                strategy="none",
                keep_ratio=1.0,
                skip_layers=args.skip_layers,
                chunk_size=args.chunk_size,
                method="snapkv",
            )
            rows.append(baseline)
            checkpoint_raw(rows, raw_path)
            if not baseline["correct"]:
                print("Baseline failed; SnapKV skipped.")
                continue

            print(f"snapkv | context={context_length} | seed={seed}")
            rows.append(
                evaluate_snapkv(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=args.model_name,
                    example=example,
                    target_cache_tokens=args.target_cache_tokens,
                    observation_window_size=args.observation_window_size,
                    pooling_kernel_size=args.pooling_kernel_size,
                    pooling_mode=args.pooling_mode,
                    skip_layers=args.skip_layers,
                    chunk_size=args.chunk_size,
                )
            )
            checkpoint_raw(rows, raw_path)
    return checkpoint_raw(rows, raw_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    raw_path = results_dir / f"{args.output_prefix}_raw.csv"
    summary_path = results_dir / f"{args.output_prefix}_summary.csv"
    metadata_path = results_dir / f"{args.output_prefix}_metadata.json"

    print(f"Loading {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        dtype=DTYPE,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    metadata = make_run_metadata(
        script=Path(__file__).name,
        model_name=args.model_name,
        model=model,
        requested_dtype=DTYPE,
        attention_implementation=ATTENTION_IMPLEMENTATION,
        seed=args.seeds[0],
        lengths=args.context_lengths,
        depths=None,
        configurations=[
            {
                "config": "no_compression",
                "target_cache_tokens": "context_length",
            },
            {
                "config": "snapkv",
                "target_cache_tokens": args.target_cache_tokens,
            },
        ],
        skip_layers=args.skip_layers,
        extra={
            "seeds": args.seeds,
            "chunk_size": args.chunk_size,
            "metric": "exact_token_match",
            "observation_window_size": args.observation_window_size,
            "target_cache_tokens": args.target_cache_tokens,
            "pooling_kernel_size": args.pooling_kernel_size,
            "pooling_mode": args.pooling_mode,
        },
    )
    print_run_metadata(metadata)
    save_run_metadata(metadata_path, metadata)

    raw_df = run_benchmark(model, tokenizer, args, raw_path)
    summary_df = summarize_results(raw_df)
    summary_df.to_csv(summary_path, index=False)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
