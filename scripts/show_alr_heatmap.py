from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.alr import scan_alr_qwen_decode_step, summarize_alr_by_layer
from l2kv.attention_viz import plot_alr_heatmap
from l2kv.model_utils import load_model_and_tokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS = 128

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a layer/head ALR heatmap.")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    alr_path = results_dir / "alr_scan.csv"
    summary_path = results_dir / "alr_layer_summary.csv"
    heatmap_path = results_dir / "alr_heatmap.png"

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    alr_df = scan_alr_qwen_decode_step(
        model=model,
        tokenizer=tokenizer,
        texts=TEXTS,
        max_tokens=MAX_TOKENS,
    )
    alr_df.to_csv(alr_path, index=False)

    layer_summary, _ = summarize_alr_by_layer(alr_df)
    layer_summary.to_csv(summary_path, index=False)

    plot_alr_heatmap(
        alr_df,
        save_path=heatmap_path,
        show=args.show,
    )

    print(f"Saved {alr_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {heatmap_path}")


if __name__ == "__main__":
    main()
