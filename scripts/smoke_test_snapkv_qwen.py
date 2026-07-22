from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.cache_metrics import (
    cache_layer_lengths,
    get_cache_layer,
    num_cache_layers,
)
from l2kv.kv_retrieval import make_kv_retrieval_prompt
from l2kv.model_utils import get_model_config, load_model_and_tokenizer
from l2kv.position_utils import make_cache_position, make_position_ids
from l2kv.runtime_metadata import (
    make_run_metadata,
    print_run_metadata,
    save_run_metadata,
)
from l2kv.snapkv import compress_snapkv_cache, prefill_and_score_snapkv


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXT_TOKENS = 256
OBSERVATION_WINDOW_SIZE = 16
KEEP_RATIO = 0.5
POOLING_KERNEL_SIZE = 5
POOLING_MODE = "max"
SKIP_LAYERS = (0, 1)
CHUNK_SIZE = 64
SEED = 0
DEPTH = 0.5
DTYPE = "auto"
ATTENTION_IMPLEMENTATION = "eager"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test GQA-compatible SnapKV with Qwen2.5."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--context-tokens", type=int, default=CONTEXT_TOKENS)
    parser.add_argument(
        "--observation-window-size",
        type=int,
        default=OBSERVATION_WINDOW_SIZE,
    )
    parser.add_argument("--keep-ratio", type=float, default=KEEP_RATIO)
    parser.add_argument(
        "--pooling-kernel-size",
        type=int,
        default=POOLING_KERNEL_SIZE,
    )
    parser.add_argument(
        "--pooling-mode",
        choices=("max", "mean"),
        default=POOLING_MODE,
    )
    parser.add_argument(
        "--skip-layers",
        type=int,
        nargs="*",
        default=SKIP_LAYERS,
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument(
        "--attention-implementation",
        default=ATTENTION_IMPLEMENTATION,
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> int:
    if args.context_tokens < 1:
        raise ValueError("context_tokens must be >= 1")
    if not 0 < args.keep_ratio < 1 or not math.isfinite(args.keep_ratio):
        raise ValueError("The smoke test requires 0 < keep_ratio < 1")
    if not 0 < args.observation_window_size < args.context_tokens:
        raise ValueError(
            "observation_window_size must be smaller than context_tokens"
        )
    if args.pooling_kernel_size < 1 or args.pooling_kernel_size % 2 == 0:
        raise ValueError("pooling_kernel_size must be a positive odd integer")
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if any(layer < 0 for layer in args.skip_layers):
        raise ValueError("skip_layers must contain non-negative indices")
    if len(set(args.skip_layers)) != len(args.skip_layers):
        raise ValueError("skip_layers must not contain duplicates")
    if args.attention_implementation != "eager":
        raise ValueError("SnapKV requires attention_implementation='eager'")

    target_capacity = math.floor(args.context_tokens * args.keep_ratio)
    if target_capacity < args.observation_window_size:
        raise ValueError(
            f"target capacity {target_capacity} is smaller than the "
            f"observation window {args.observation_window_size}"
        )
    return target_capacity


def _last_token_argmax(last_logits: torch.Tensor) -> torch.Tensor:
    if last_logits.ndim == 3:
        last_logits = last_logits[:, -1, :]
    if last_logits.ndim != 2 or last_logits.shape[0] != 1:
        raise AssertionError(
            f"Unexpected final-logit shape: {tuple(last_logits.shape)}"
        )
    return last_logits.argmax(dim=-1, keepdim=True)


def _snapshot_observation_window(
    cache: Any,
    observation_window_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(num_cache_layers(cache)):
        keys, values = get_cache_layer(cache, layer_idx)
        snapshots.append(
            (
                keys[..., -observation_window_size:, :].clone(),
                values[..., -observation_window_size:, :].clone(),
            )
        )
    return snapshots


def _assert_scores_are_gqa_shaped(
    scores_by_layer: Sequence[torch.Tensor | None],
    num_layers: int,
    num_kv_heads: int,
    prefix_length: int,
    skip_layers: Sequence[int],
) -> None:
    if len(scores_by_layer) != num_layers:
        raise AssertionError(
            f"Received {len(scores_by_layer)} score tensors for {num_layers} layers"
        )

    skipped = set(skip_layers)
    for layer_idx, scores in enumerate(scores_by_layer):
        if layer_idx in skipped:
            if scores is not None:
                raise AssertionError(
                    f"Skipped layer {layer_idx} unexpectedly retained scores"
                )
            continue
        if scores is None:
            raise AssertionError(f"Layer {layer_idx} is missing SnapKV scores")
        expected_shape = (1, num_kv_heads, prefix_length)
        if tuple(scores.shape) != expected_shape:
            raise AssertionError(
                f"Layer {layer_idx} score shape {tuple(scores.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(scores).all().item():
            raise AssertionError(f"Layer {layer_idx} scores contain NaN or infinity")


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_capacity = _validate_args(args)
    skip_layers = tuple(args.skip_layers)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"Loading {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        dtype=args.dtype,
        attn_implementation=args.attention_implementation,
    )
    device = next(model.parameters()).device
    config = get_model_config(model)
    num_query_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    if num_query_heads % num_kv_heads != 0:
        raise AssertionError(
            f"Invalid GQA shape: {num_query_heads} query heads and "
            f"{num_kv_heads} KV heads"
        )
    group_size = num_query_heads // num_kv_heads

    metadata = make_run_metadata(
        script=Path(__file__).name,
        model_name=args.model_name,
        model=model,
        requested_dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        seed=args.seed,
        lengths=[args.context_tokens],
        depths=[DEPTH],
        configurations=[
            {
                "strategy": "snapkv",
                "keep_ratio": args.keep_ratio,
                "observation_window_size": args.observation_window_size,
                "pooling_kernel_size": args.pooling_kernel_size,
                "pooling_mode": args.pooling_mode,
            }
        ],
        skip_layers=skip_layers,
        extra={
            "num_query_heads": num_query_heads,
            "num_key_value_heads": num_kv_heads,
            "gqa_group_size": group_size,
            "target_capacity": target_capacity,
            "chunk_size": args.chunk_size,
        },
    )
    print_run_metadata(metadata)
    save_run_metadata(results_dir / "run_metadata.json", metadata)

    prompt = make_kv_retrieval_prompt(
        tokenizer=tokenizer,
        target_tokens=args.context_tokens,
        observation_window_size=args.observation_window_size,
        seed=args.seed,
        depth=DEPTH,
    )
    result = prefill_and_score_snapkv(
        model=model,
        prompt_ids=prompt.prompt_ids,
        observation_window_size=args.observation_window_size,
        chunk_size=args.chunk_size,
        skip_layers=skip_layers,
    )
    if result.logical_position != args.context_tokens:
        raise AssertionError(
            f"Logical position {result.logical_position}, "
            f"expected {args.context_tokens}"
        )

    lengths_before = cache_layer_lengths(result.cache)
    if not lengths_before or any(
        length != args.context_tokens for length in lengths_before
    ):
        raise AssertionError(f"Unexpected prefill layer lengths: {lengths_before}")
    invalid_skips = sorted(set(skip_layers) - set(range(len(lengths_before))))
    if invalid_skips:
        raise ValueError(f"skip layer indices do not exist: {invalid_skips}")

    prefix_length = args.context_tokens - args.observation_window_size
    _assert_scores_are_gqa_shaped(
        scores_by_layer=result.scores_by_layer,
        num_layers=len(lengths_before),
        num_kv_heads=num_kv_heads,
        prefix_length=prefix_length,
        skip_layers=skip_layers,
    )
    observation_snapshots = _snapshot_observation_window(
        result.cache,
        args.observation_window_size,
    )
    skipped_tensors = {
        layer_idx: get_cache_layer(result.cache, layer_idx)
        for layer_idx in skip_layers
    }
    last_logits = result.last_logits

    cache = compress_snapkv_cache(
        cache=result.cache,
        scores_by_layer=result.scores_by_layer,
        target_capacity=target_capacity,
        observation_window_size=args.observation_window_size,
        pooling_kernel_size=args.pooling_kernel_size,
        pooling_mode=args.pooling_mode,
        skip_layers=skip_layers,
    )
    del result
    lengths_after = cache_layer_lengths(cache)
    print(f"Query heads: {num_query_heads}")
    print(f"KV heads: {num_kv_heads}")
    print(f"GQA group size: {group_size}")
    print(f"Layer lengths before compression: {lengths_before}")
    print(f"Layer lengths after compression:  {lengths_after}")

    skipped = set(skip_layers)
    compressed_lengths: list[int] = []
    for layer_idx, length in enumerate(lengths_after):
        keys, values = get_cache_layer(cache, layer_idx)
        if keys.shape != values.shape:
            raise AssertionError(f"Layer {layer_idx} K/V shapes differ")

        if layer_idx in skipped:
            expected_length = args.context_tokens
            original_keys, original_values = skipped_tensors[layer_idx]
            if keys is not original_keys or values is not original_values:
                raise AssertionError(
                    f"Skipped layer {layer_idx} was replaced during compression"
                )
        else:
            expected_length = target_capacity
            compressed_lengths.append(length)
            expected_keys, expected_values = observation_snapshots[layer_idx]
            if not torch.equal(
                keys[..., -args.observation_window_size :, :],
                expected_keys,
            ):
                raise AssertionError(
                    f"Layer {layer_idx} did not preserve observation keys"
                )
            if not torch.equal(
                values[..., -args.observation_window_size :, :],
                expected_values,
            ):
                raise AssertionError(
                    f"Layer {layer_idx} did not preserve observation values"
                )

        if length != expected_length:
            raise AssertionError(
                f"Layer {layer_idx} length {length}, expected {expected_length}"
            )

    if not compressed_lengths:
        raise AssertionError("No layer was compressed")
    if target_capacity < args.context_tokens and skip_layers and not all(
        lengths_after[layer_idx] > max(compressed_lengths)
        for layer_idx in skip_layers
    ):
        raise AssertionError("Skip layers are not longer than compressed layers")

    del observation_snapshots
    del skipped_tensors
    next_token = _last_token_argmax(last_logits).to(device)
    position_ids = make_position_ids(args.context_tokens, 1, device)
    cache_position = make_cache_position(position_ids)
    if position_ids.tolist() != [[args.context_tokens]]:
        raise AssertionError("Logical position did not continue after the full prompt")
    if cache_position.tolist() != [args.context_tokens]:
        raise AssertionError("cache_position does not match the logical position")

    decode_outputs = model(
        input_ids=next_token,
        past_key_values=cache,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
        logits_to_keep=1,
    )
    updated_cache = decode_outputs.past_key_values
    lengths_after_decode = cache_layer_lengths(updated_cache)
    print(f"Layer lengths after one decode: {lengths_after_decode}")

    for layer_idx, prior_length in enumerate(lengths_after):
        keys, values = get_cache_layer(updated_cache, layer_idx)
        if keys.shape != values.shape:
            raise AssertionError(f"Layer {layer_idx} K/V shapes differ after decode")
        if int(keys.shape[2]) != prior_length + 1:
            raise AssertionError(
                f"Layer {layer_idx} did not append exactly one token"
            )

    print("PASS: SnapKV GQA compression supports logical-position decoding")


if __name__ == "__main__":
    main()
