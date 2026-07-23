from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.cache_compression import compress_cache
from l2kv.cache_metrics import cache_layer_lengths, get_cache_layer
from l2kv.configs import get_default_skip_layers
from l2kv.model_utils import load_model_and_tokenizer
from l2kv.passkey import make_passkey_example
from l2kv.position_utils import make_cache_position, make_position_ids
from l2kv.runtime_metadata import (
    make_run_metadata,
    print_run_metadata,
    save_run_metadata,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXT_TOKENS = 256
KEEP_RATIO = 0.5
SEED = 0
DTYPE = "auto"
ATTN_IMPLEMENTATION = "eager"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test Qwen with heterogeneous DynamicCache layer lengths."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--context-tokens", type=int, default=CONTEXT_TOKENS)
    parser.add_argument("--keep-ratio", type=float, default=KEEP_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument(
        "--attention-implementation",
        default=ATTN_IMPLEMENTATION,
    )
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    skip_layers = get_default_skip_layers()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"Loading {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        dtype=args.dtype,
        attn_implementation=args.attention_implementation,
    )
    device = next(model.parameters()).device

    metadata = make_run_metadata(
        script=Path(__file__).name,
        model_name=args.model_name,
        model=model,
        requested_dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        seed=args.seed,
        lengths=[args.context_tokens],
        depths=None,
        configurations=[
            {
                "strategy": "low_l2",
                "keep_ratio": args.keep_ratio,
            }
        ],
        skip_layers=skip_layers,
        extra={"batch_size": 1, "prune_after": 0},
    )
    print_run_metadata(metadata)
    save_run_metadata(results_dir / "run_metadata.json", metadata)

    prompt = make_passkey_example(
        tokenizer=tokenizer,
        context_length=args.context_tokens,
        seed=args.seed,
    )
    context_ids = torch.tensor(
        prompt.prompt_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    prefill_position_ids = make_position_ids(0, args.context_tokens, device)
    prefill_out = model(
        input_ids=context_ids,
        position_ids=prefill_position_ids,
        cache_position=make_cache_position(prefill_position_ids),
        use_cache=True,
        return_dict=True,
    )
    cache = prefill_out.past_key_values

    lengths_before = cache_layer_lengths(cache)
    if not lengths_before or any(
        length != args.context_tokens for length in lengths_before
    ):
        raise AssertionError(f"Unexpected prefill layer lengths: {lengths_before}")

    compress_cache(
        cache,
        keep_ratio=args.keep_ratio,
        prune_after=0,
        strategy="low_l2",
        skip_layers=skip_layers,
        seed=args.seed,
    )
    lengths_after = cache_layer_lengths(cache)
    expected_compressed_length = math.ceil(args.keep_ratio * args.context_tokens)

    print(f"Layer lengths before compression: {lengths_before}")
    print(f"Layer lengths after compression:  {lengths_after}")
    for layer_idx, length in enumerate(lengths_after):
        expected = (
            args.context_tokens
            if layer_idx in skip_layers
            else expected_compressed_length
        )
        if length != expected:
            raise AssertionError(
                f"Layer {layer_idx} length {length}, expected {expected}"
            )

    compressed_lengths = [
        length
        for layer_idx, length in enumerate(lengths_after)
        if layer_idx not in skip_layers
    ]
    if not compressed_lengths or not all(
        lengths_after[layer_idx] > max(compressed_lengths)
        for layer_idx in skip_layers
    ):
        raise AssertionError("Skip layers are not longer than compressed layers")

    logical_position = args.context_tokens
    question_token = prefill_out.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    ).to(device)
    position_ids = make_position_ids(logical_position, 1, device)
    cache_position = make_cache_position(position_ids)
    if position_ids.tolist() != [[args.context_tokens]]:
        raise AssertionError("Logical position did not continue after the full prefill")
    if cache_position.tolist() != [args.context_tokens]:
        raise AssertionError("cache_position does not match the logical position")

    decode_out = model(
        input_ids=question_token,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    updated_cache = decode_out.past_key_values
    lengths_after_forward = cache_layer_lengths(updated_cache)
    print(f"Layer lengths after one decode: {lengths_after_forward}")

    for layer_idx, expected_before in enumerate(lengths_after):
        keys, values = get_cache_layer(updated_cache, layer_idx)
        if keys.shape[2] != values.shape[2]:
            raise AssertionError(f"Layer {layer_idx} K/V lengths differ")
        if keys.shape[2] != expected_before + 1:
            raise AssertionError(
                f"Layer {layer_idx} did not append exactly one token"
            )

    print("PASS: heterogeneous cache lengths support a logical-position decode forward")


if __name__ == "__main__":
    main()
