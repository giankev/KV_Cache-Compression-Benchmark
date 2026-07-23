from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.model_utils import load_model_and_tokenizer
from l2kv.passkey import PasskeyExample, make_passkey_example
from l2kv.retrieval_eval import (
    checkpoint_raw,
    evaluate_plain_or_l2,
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
KEEP_RATIO = 0.8
SKIP_LAYERS = (0, 1)
CHUNK_SIZE = 512
OUTPUT_PREFIX = "l2_passkey_3b_8k"
DTYPE = "auto"
ATTENTION_IMPLEMENTATION = None
CONFIGURATIONS = (
    ("no_compression", "none"),
    ("low_l2", "low_l2"),
    ("random", "random"),
    ("high_l2", "high_l2"),
)


def build_example(
    tokenizer: Any,
    context_length: int,
    seed: int,
) -> PasskeyExample:
    return make_passkey_example(
        tokenizer=tokenizer,
        context_length=context_length,
        seed=seed,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the L2 passkey benchmark.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=CONTEXT_LENGTHS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--keep-ratio", type=float, default=KEEP_RATIO)
    parser.add_argument(
        "--skip-layers",
        type=int,
        nargs="*",
        default=SKIP_LAYERS,
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    args = parser.parse_args(argv)

    if any(length < 1 for length in args.context_lengths):
        raise ValueError("context_lengths must contain positive integers")
    if not args.seeds:
        raise ValueError("seeds must not be empty")
    if not 0 < args.keep_ratio <= 1 or not math.isfinite(args.keep_ratio):
        raise ValueError("keep_ratio must satisfy 0 < keep_ratio <= 1")
    if any(layer < 0 for layer in args.skip_layers):
        raise ValueError("skip_layers must contain non-negative integers")
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    return args


def run_benchmark(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    raw_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for context_length in args.context_lengths:
        for seed in args.seeds:
            example = build_example(tokenizer, context_length, seed)
            for config, strategy in CONFIGURATIONS:
                print(f"{config} | context={context_length} | seed={seed}")
                row = evaluate_plain_or_l2(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=args.model_name,
                    example=example,
                    config=config,
                    strategy=strategy,
                    keep_ratio=args.keep_ratio,
                    skip_layers=args.skip_layers,
                    chunk_size=args.chunk_size,
                )
                rows.append(row)
                checkpoint_raw(rows, raw_path)
                if config == "no_compression" and not row["correct"]:
                    print("Baseline failed; compressed configurations skipped.")
                    break
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
                "config": config,
                "strategy": strategy,
                "keep_ratio": 1.0 if strategy == "none" else args.keep_ratio,
            }
            for config, strategy in CONFIGURATIONS
        ],
        skip_layers=args.skip_layers,
        extra={
            "seeds": args.seeds,
            "chunk_size": args.chunk_size,
            "metric": "exact_token_match",
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
