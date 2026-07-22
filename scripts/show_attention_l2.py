from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.attention_viz import plot_attention_l2_heatmap
from l2kv.configs import get_default_skip_layers
from l2kv.model_utils import load_model_and_tokenizer
from l2kv.runtime_metadata import (
    make_run_metadata,
    print_run_metadata,
    save_run_metadata,
)


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS = 64
SEED = 0
DTYPE = "auto"
ATTN_IMPLEMENTATION = "eager"
TEXT = (
    "There is an important pass key hidden in this short document. "
    "The pass key is 57291. Remember it. 57291 is the pass key. "
    "The grass is green. The sky is blue. What is the pass key?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save an attention/L2 heatmap.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "attention_l2_heatmap.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME,
        dtype=DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    metadata = make_run_metadata(
        script=Path(__file__).name,
        model_name=MODEL_NAME,
        model=model,
        requested_dtype=DTYPE,
        attention_implementation=ATTN_IMPLEMENTATION,
        seed=SEED,
        lengths=[MAX_TOKENS],
        depths=None,
        configurations=[{"name": "attention_l2_heatmap"}],
        skip_layers=get_default_skip_layers(),
    )
    print_run_metadata(metadata)
    save_run_metadata(PROJECT_ROOT / "results" / "run_metadata.json", metadata)
    plot_attention_l2_heatmap(
        model=model,
        tokenizer=tokenizer,
        text=TEXT,
        max_tokens=MAX_TOKENS,
        save_path=output_path,
        show=args.show,
    )

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
