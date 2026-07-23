"""Run the fixed keep-10% L2 passkey experiment on three deterministic seeds."""

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
    summarize_results,
)
from l2kv.runtime_metadata import (
    make_run_metadata,
    print_run_metadata,
    save_run_metadata,
)


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CONTEXT_LENGTHS = (8192,)
SEEDS = (0, 1, 2)
KEEP_RATIO = 0.10

# Keep the first two layers intact to match the project comparison protocol.
SKIP_LAYERS = (0, 1)
CHUNK_SIZE = 512
OUTPUT_PREFIX = "l2_passkey_3b_8k_keep10"
DTYPE = "auto"
ATTENTION_IMPLEMENTATION = None
CONFIGURATIONS = (
    ("no_compression", "none"),
    ("low_l2_keep10", "low_l2"),
    ("random_keep10", "random"),
    ("high_l2_keep10", "high_l2"),
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
            example = make_passkey_example(tokenizer, context_length, seed)
            baseline_config, baseline_strategy = CONFIGURATIONS[0]
            print(
                f"{baseline_config} | context={context_length} | seed={seed}"
            )
            baseline = evaluate_plain_or_l2(
                model=model,
                tokenizer=tokenizer,
                model_name=args.model_name,
                example=example,
                config=baseline_config,
                strategy=baseline_strategy,
                keep_ratio=KEEP_RATIO,
                skip_layers=args.skip_layers,
                chunk_size=args.chunk_size,
            )
            rows.append(baseline)
            checkpoint_raw(rows, raw_path)
            if not baseline["correct"]:
                print("Baseline failed; compressed configurations skipped.")
                continue

            for config, strategy in CONFIGURATIONS[1:]:
                print(f"{config} | context={context_length} | seed={seed}")
                row = evaluate_plain_or_l2(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=args.model_name,
                    example=example,
                    config=config,
                    strategy=strategy,
                    keep_ratio=KEEP_RATIO,
                    skip_layers=args.skip_layers,
                    chunk_size=args.chunk_size,
                )
                rows.append(row)
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
                "config": config,
                "strategy": strategy,
                "keep_ratio": 1.0 if strategy == "none" else KEEP_RATIO,
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
