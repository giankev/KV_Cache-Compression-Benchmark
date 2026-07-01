from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.configs import get_light_config_group
from l2kv.model_utils import load_model_and_tokenizer
from l2kv.passkey import run_passkey_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight passkey benchmark.")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--task",
        choices=["standard", "distractor"],
        default="standard",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[8192, 32768],
    )
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--passkey-digits", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--prune-after", type=int, default=1024)
    parser.add_argument("--evaluation-mode", default="strict_context")
    parser.add_argument(
        "--configs",
        choices=["light_final", "light_standard"],
        default="light_final",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["config", "context_len"])
        .agg(
            accuracy=("correct", "mean"),
            final_cache_mb=("final_cache_mb", "mean"),
            uncompressed_cache_mb=("uncompressed_cache_mb", "mean"),
            memory_saved_percent=("memory_saved_percent", "mean"),
        )
        .reset_index()
    )

    summary["accuracy"] = (summary["accuracy"] * 100).round(2)
    summary["final_cache_mb"] = summary["final_cache_mb"].round(2)
    summary["uncompressed_cache_mb"] = summary["uncompressed_cache_mb"].round(2)
    summary["memory_saved_percent"] = summary["memory_saved_percent"].round(2)
    return summary[
        [
            "config",
            "context_len",
            "accuracy",
            "final_cache_mb",
            "uncompressed_cache_mb",
            "memory_saved_percent",
        ]
    ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    eval_configs = get_light_config_group(args.configs)

    df = run_passkey_benchmark(
        model=model,
        tokenizer=tokenizer,
        eval_configs=eval_configs,
        context_lengths=args.context_lengths,
        depths=args.depths,
        seeds=[args.seed],
        prompt_kind=args.task,
        max_new_tokens=args.max_new_tokens,
        prune_after=args.prune_after,
        chunk_size=args.chunk_size,
        passkey_digits=args.passkey_digits,
        evaluation_mode=args.evaluation_mode,
    )

    summary = summarize(df)

    raw_path = output_dir / f"light_passkey_{args.task}_{args.configs}_raw.csv"
    summary_path = output_dir / f"light_passkey_{args.task}_{args.configs}_summary.csv"

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(df)
    print(summary)
    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
