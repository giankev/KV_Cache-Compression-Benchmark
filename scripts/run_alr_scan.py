from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.alr import (
    scan_alr_qwen_decode_step,
    summarize_alr_by_layer,
    suggest_skip_layers_from_alr,
)
from l2kv.model_utils import load_model_and_tokenizer


CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "dtype": "auto",
    "device_map": "auto",
    "max_tokens": 128,
    "normalize_attention": True,
    "debug_shapes": False,
    "top_k": 2,
    "always_include_first_two": True,
}

TEXTS = [
    "An embarrassingly simple way to compress the KV cache. "
    "The model stores key and value tensors for each previous token during autoregressive decoding.",
    "The capital of France is Paris. The capital of Italy is Rome. "
    "The capital of Germany is Berlin. These facts are useful for retrieval tasks.",
    "Needle in a haystack evaluations test whether a language model can retrieve important information "
    "from a long context full of irrelevant tokens.",
    "Large language models based on decoder-only Transformers use self-attention layers, feed-forward networks, "
    "rotary positional embeddings, and key-value caches to speed up generation.",
    "In a passkey retrieval task, a secret number is inserted inside a long document, and the model must remember "
    "the number and output it correctly at the end.",
]


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        CONFIG["model_name"],
        dtype=CONFIG["dtype"],
        device_map=CONFIG["device_map"],
    )

    alr_df = scan_alr_qwen_decode_step(
        model=model,
        tokenizer=tokenizer,
        texts=TEXTS,
        max_tokens=CONFIG["max_tokens"],
        normalize_attention=CONFIG["normalize_attention"],
        debug_shapes=CONFIG["debug_shapes"],
    )
    alr_df.to_csv(results_dir / "alr_scan.csv", index=False)

    layer_summary, high_alr_threshold = summarize_alr_by_layer(alr_df)
    layer_summary.to_csv(results_dir / "alr_layer_summary.csv", index=False)

    skip_layers = suggest_skip_layers_from_alr(
        layer_summary,
        top_k=CONFIG["top_k"],
        always_include_first_two=CONFIG["always_include_first_two"],
    )
    pd.DataFrame(
        {
            "skip_layer": list(skip_layers),
            "high_alr_threshold": [high_alr_threshold] * len(skip_layers),
        }
    ).to_csv(results_dir / "alr_suggested_skip_layers.csv", index=False)

    print(f"Saved {results_dir / 'alr_scan.csv'}")
    print(f"Saved {results_dir / 'alr_layer_summary.csv'}")
    print(f"Saved {results_dir / 'alr_suggested_skip_layers.csv'}")
    print(f"Suggested skip layers: {skip_layers}")


if __name__ == "__main__":
    main()
