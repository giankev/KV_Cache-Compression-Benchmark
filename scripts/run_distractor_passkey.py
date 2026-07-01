from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.model_utils import load_model_and_tokenizer
from l2kv.passkey import run_passkey_benchmark


CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "dtype": "auto",
    "device_map": "auto",
    "context_lengths": [8192],
    "depths": [0.1, 0.5, 0.9],
    "seeds": [0],
    "passkey_digits": 5,
    "max_new_tokens": 12,
    "prune_after": 1024,
    "chunk_size": 512,
    "strategy": "low_l2",
    "eval_configs": [
        {
            "config": "no_compression",
            "use_compression": False,
            "keep_ratio": 1.0,
            "skip_layers": (),
        },
        {
            "config": "low_l2_no_skip_keep50",
            "use_compression": True,
            "keep_ratio": 0.5,
            "skip_layers": (),
        },
        {
            "config": "low_l2_skip01_keep50",
            "use_compression": True,
            "keep_ratio": 0.5,
            "skip_layers": (0, 1, 29),
        },
        {
            "config": "low_l2_no_skip_keep10",
            "use_compression": True,
            "keep_ratio": 0.1,
            "skip_layers": (),
        },
        {
            "config": "low_l2_skip01_keep10",
            "use_compression": True,
            "keep_ratio": 0.1,
            "skip_layers": (0, 1, 29),
        },
    ],
}


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        CONFIG["model_name"],
        dtype=CONFIG["dtype"],
        device_map=CONFIG["device_map"],
    )

    df = run_passkey_benchmark(
        model=model,
        tokenizer=tokenizer,
        eval_configs=CONFIG["eval_configs"],
        context_lengths=CONFIG["context_lengths"],
        depths=CONFIG["depths"],
        seeds=CONFIG["seeds"],
        prompt_kind="distractor",
        max_new_tokens=CONFIG["max_new_tokens"],
        prune_after=CONFIG["prune_after"],
        chunk_size=CONFIG["chunk_size"],
        strategy=CONFIG["strategy"],
        passkey_digits=CONFIG["passkey_digits"],
        output_csv=results_dir / "distractor_passkey_results.csv",
    )

    summary = (
        df.groupby(["config", "context_len", "keep_ratio", "compression_ratio"])
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
    summary.to_csv(results_dir / "distractor_passkey_summary.csv", index=False)

    print(f"Saved {results_dir / 'distractor_passkey_results.csv'}")
    print(f"Saved {results_dir / 'distractor_passkey_summary.csv'}")


if __name__ == "__main__":
    main()
