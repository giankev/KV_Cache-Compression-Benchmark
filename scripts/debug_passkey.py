from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.configs import LIGHT_STANDARD_CONFIGS, get_light_config
from l2kv.model_utils import load_model_and_tokenizer
from l2kv.passkey import (
    generate_passkey_answer,
    is_correct_prediction,
    make_distractor_passkey_prompt,
    make_passkey_prompt,
)


def parse_args() -> argparse.Namespace:
    config_names = [cfg["config"] for cfg in LIGHT_STANDARD_CONFIGS]

    parser = argparse.ArgumentParser(description="Run one passkey debug example.")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--task",
        choices=["standard", "distractor"],
        default="distractor",
    )
    parser.add_argument("--context-len", type=int, default=8192)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--passkey-digits", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--config", choices=config_names, default="no_compression")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--prune-after", type=int, default=1024)
    parser.add_argument("--evaluation-mode", default="strict_context")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_light_config(args.config)

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    if args.task == "standard":
        context, question, answer = make_passkey_prompt(
            tokenizer,
            target_tokens=args.context_len,
            seed=args.seed,
            depth=args.depth,
        )
    else:
        context, question, answer = make_distractor_passkey_prompt(
            tokenizer,
            target_tokens=args.context_len,
            seed=args.seed,
            depth=args.depth,
            passkey_digits=args.passkey_digits,
        )

    res = generate_passkey_answer(
        model=model,
        tokenizer=tokenizer,
        context=context,
        question=question,
        max_new_tokens=args.max_new_tokens,
        expected_digits=len(answer),
        use_compression=cfg["use_compression"],
        keep_ratio=cfg["keep_ratio"],
        prune_after=args.prune_after,
        chunk_size=args.chunk_size,
        strategy=cfg["strategy"],
        skip_layers=cfg["skip_layers"],
    )

    correct = is_correct_prediction(
        prediction=res["prediction"],
        answer=answer,
        evaluation_mode=args.evaluation_mode,
    )

    print(f"answer: {answer}")
    print(f"generated_text: {res['generated_text']!r}")
    print(f"generated_text_raw: {res['generated_text_raw']!r}")
    print(f"prediction: {res['prediction']}")
    print(f"correct: {correct}")
    print(f"generated_ids: {res['generated_ids']}")
    print(f"generated_tokens: {res['generated_tokens']}")
    print(f"context_tokens: {res['context_tokens']}")
    print(f"question_tokens: {res['question_tokens']}")
    print(f"final_cache_len: {res['final_cache_len']}")
    print(
        "cache_lens_before_compression for first 6 layers: "
        f"{res['cache_lens_before_compression'][:6]}"
    )
    print(
        "cache_lens_after_compression for first 6 layers: "
        f"{res['cache_lens_after_compression'][:6]}"
    )
    print(f"final_cache_mb: {res['final_cache_mb']}")
    print(f"memory_saved_percent: {res['memory_saved_percent']}")


if __name__ == "__main__":
    main()
