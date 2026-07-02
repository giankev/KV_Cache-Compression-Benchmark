from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.configs import BASIC_PASSKEY_CONFIGS, get_default_skip_layers
from l2kv.model_utils import load_model_and_tokenizer
from l2kv.passkey import (
    generate_passkey_answer,
    is_correct_prediction,
    make_passkey_prompt,
)


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CONTEXT_LENGTHS = [8192, 32768]
DEPTHS = [0.5]
SEEDS = [0]
MAX_NEW_TOKENS = 12
PRUNE_AFTER = 1024
CHUNK_SIZE = 512

RAW_COLUMNS = [
    "config",
    "context_len",
    "answer",
    "prediction",
    "correct",
    "keep_ratio",
    "compressed_cache_mb",
    "uncompressed_cache_mb",
    "memory_saved_percent",
]

SUMMARY_COLUMNS = [
    "config",
    "context_len",
    "accuracy",
    "compressed_cache_mb",
    "uncompressed_cache_mb",
    "memory_saved_percent",
]


def make_eval_configs(skip_layers: tuple[int, ...]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    for cfg in BASIC_PASSKEY_CONFIGS:
        cfg = dict(cfg)
        if cfg["use_compression"]:
            cfg["skip_layers"] = skip_layers
        configs.append(cfg)

    return configs


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df.groupby(["config", "context_len"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            compressed_cache_mb=("compressed_cache_mb", "mean"),
            uncompressed_cache_mb=("uncompressed_cache_mb", "mean"),
            memory_saved_percent=("memory_saved_percent", "mean"),
        )
        .loc[:, SUMMARY_COLUMNS]
    )

    summary["accuracy"] = summary["accuracy"].round(4)
    summary["compressed_cache_mb"] = summary["compressed_cache_mb"].round(2)
    summary["uncompressed_cache_mb"] = summary["uncompressed_cache_mb"].round(2)
    summary["memory_saved_percent"] = summary["memory_saved_percent"].round(2)
    return summary


def run_benchmark(model: Any, tokenizer: Any) -> pd.DataFrame:
    skip_layers = get_default_skip_layers()
    eval_configs = make_eval_configs(skip_layers)
    rows: list[dict[str, Any]] = []

    print(f"Using skip layers for compressed configs: {skip_layers}")

    for context_len in CONTEXT_LENGTHS:
        for depth in DEPTHS:
            for seed in SEEDS:
                context, question, answer = make_passkey_prompt(
                    tokenizer=tokenizer,
                    target_tokens=context_len,
                    seed=seed,
                    depth=depth,
                )

                for cfg in eval_configs:
                    print(
                        f"{cfg['config']} | context_len={context_len} | "
                        f"depth={depth} | seed={seed}"
                    )

                    result = generate_passkey_answer(
                        model=model,
                        tokenizer=tokenizer,
                        context=context,
                        question=question,
                        max_new_tokens=MAX_NEW_TOKENS,
                        expected_digits=len(answer),
                        use_compression=cfg["use_compression"],
                        keep_ratio=cfg["keep_ratio"],
                        prune_after=PRUNE_AFTER,
                        chunk_size=CHUNK_SIZE,
                        strategy=cfg["strategy"],
                        skip_layers=cfg["skip_layers"],
                    )

                    prediction = result["prediction"]
                    rows.append(
                        {
                            "config": cfg["config"],
                            "context_len": context_len,
                            "answer": answer,
                            "prediction": prediction,
                            "correct": is_correct_prediction(prediction, answer),
                            "keep_ratio": cfg["keep_ratio"],
                            "compressed_cache_mb": result["compressed_cache_mb"],
                            "uncompressed_cache_mb": result["uncompressed_cache_mb"],
                            "memory_saved_percent": result["memory_saved_percent"],
                        }
                    )

    raw_df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw_df["compressed_cache_mb"] = raw_df["compressed_cache_mb"].round(2)
    raw_df["uncompressed_cache_mb"] = raw_df["uncompressed_cache_mb"].round(2)
    raw_df["memory_saved_percent"] = raw_df["memory_saved_percent"].round(2)
    return raw_df


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    raw_df = run_benchmark(model=model, tokenizer=tokenizer)
    summary_df = summarize(raw_df)

    raw_path = results_dir / "basic_passkey_raw.csv"
    summary_path = results_dir / "basic_passkey_summary.csv"

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
