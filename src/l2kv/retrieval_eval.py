"""Share prefill, exact decoding, cache metrics, and CSV retrieval results.

Both L2 and SnapKV runners use these functions to keep evaluation identical.
"""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from .cache_metrics import cache_layer_lengths, kv_cache_size_mb
from .l2_compression import compress_cache_to_budget
from .passkey import PasskeyExample
from .position_utils import make_cache_position, make_position_ids
from .snapkv_compression import (
    compress_snapkv_cache,
    prefill_and_score_snapkv,
)


RAW_COLUMNS = [
    "model_name",
    "method",
    "config",
    "context_length",
    "keep_ratio",
    "target_cache_tokens",
    "observation_window_size",
    "pooling_kernel_size",
    "pooling_mode",
    "skip_layers",
    "seed",
    "actual_depth",
    "target",
    "prediction",
    "correct",
    "cache_before_mb",
    "cache_after_mb",
    "memory_saved_mb",
    "memory_saved_percent",
    "elapsed_seconds",
]
SUMMARY_COLUMNS = [
    "model_name",
    "method",
    "config",
    "context_length",
    "keep_ratio",
    "target_cache_tokens",
    "num_examples",
    "accuracy",
    "mean_cache_before_mb",
    "mean_cache_after_mb",
    "mean_memory_saved_mb",
    "mean_memory_saved_percent",
    "mean_elapsed_seconds",
]
CONSOLE_SUMMARY_COLUMNS = [
    "config",
    "keep_ratio",
    "mean_cache_after_mb",
    "mean_memory_saved_mb",
    "accuracy",
    "mean_memory_saved_percent",
    "mean_elapsed_seconds",
]


def cuda_devices(model: Any) -> tuple[torch.device, ...]:
    devices = {
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    }
    return tuple(sorted(devices, key=str))


def synchronize_cuda_devices(devices: Sequence[torch.device]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


@torch.inference_mode()
def prefill_plain(
    model: Any,
    prompt_ids: Sequence[int],
    chunk_size: int,
) -> tuple[Any, torch.Tensor, int]:
    """Prefill an exact-token prompt in chunks without attention outputs."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    device = next(model.parameters()).device
    input_ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    if input_ids.shape[1] < 1:
        raise ValueError("prompt_ids must be non-empty")

    cache = None
    logical_position = 0
    last_logits: torch.Tensor | None = None
    for start in range(0, input_ids.shape[1], chunk_size):
        chunk = input_ids[:, start : start + chunk_size]
        position_ids = make_position_ids(
            logical_position,
            int(chunk.shape[1]),
            device,
        )
        outputs = model(
            input_ids=chunk,
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=make_cache_position(position_ids),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        logical_position += int(chunk.shape[1])
        last_logits = outputs.logits[:, -1, :].detach().clone()
        del outputs

    if cache is None or last_logits is None:
        raise AssertionError("Prompt prefill did not run")
    return cache, last_logits, logical_position


def exact_token_match(
    generated_ids: Sequence[int],
    answer_ids: Sequence[int],
) -> bool:
    return tuple(int(token_id) for token_id in generated_ids) == tuple(
        int(token_id) for token_id in answer_ids
    )


@torch.inference_mode()
def generate_exact_answer(
    model: Any,
    tokenizer: Any,
    cache: Any,
    last_logits: torch.Tensor,
    logical_position: int,
    answer_ids: Sequence[int],
) -> tuple[tuple[int, ...], str, bool, Any]:
    """Greedily generate exactly the number of answer tokens."""

    expected_ids = tuple(int(token_id) for token_id in answer_ids)
    if not expected_ids:
        raise ValueError("answer_ids must be non-empty")
    if last_logits.ndim == 3:
        last_logits = last_logits[:, -1, :]
    if last_logits.ndim != 2 or last_logits.shape[0] != 1:
        raise ValueError("last_logits must have shape [1, vocabulary]")

    input_device = next(model.parameters()).device
    next_token = last_logits.argmax(dim=-1, keepdim=True).to(input_device)
    generated_ids: list[int] = []

    for answer_index in range(len(expected_ids)):
        generated_ids.append(int(next_token[0, 0].detach().cpu().item()))
        if answer_index == len(expected_ids) - 1:
            break

        lengths_before = tuple(cache_layer_lengths(cache))
        # Physical pruning never changes the next logical model position.
        position_ids = make_position_ids(logical_position, 1, next_token.device)
        outputs = model(
            input_ids=next_token,
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=make_cache_position(position_ids),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        lengths_after = tuple(cache_layer_lengths(cache))
        expected_lengths = tuple(length + 1 for length in lengths_before)
        if lengths_after != expected_lengths:
            raise AssertionError(
                "Every cache layer must add one token per decode step; "
                f"before={lengths_before}, after={lengths_after}"
            )
        logical_position += 1
        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        ).to(input_device)
        del outputs

    generated = tuple(generated_ids)
    prediction = tokenizer.decode(list(generated), skip_special_tokens=True)
    return generated, prediction, exact_token_match(generated, expected_ids), cache


def target_capacity(prompt_length: int, keep_ratio: float) -> int:
    return math.floor(prompt_length * keep_ratio)


def cache_memory_savings(
    cache_before_mb: float,
    cache_after_mb: float,
) -> tuple[float, float]:
    memory_saved_mb = cache_before_mb - cache_after_mb
    memory_saved_percent = (
        100.0 * (1.0 - cache_after_mb / cache_before_mb)
        if cache_before_mb
        else 0.0
    )
    return memory_saved_mb, memory_saved_percent


def assert_cache_capacity(
    cache: Any,
    prompt_length: int,
    target_cache_tokens: int,
    skip_layers: Sequence[int],
    compressed: bool,
) -> tuple[int, ...]:
    lengths = tuple(cache_layer_lengths(cache))
    if not lengths:
        raise AssertionError("The model returned an empty cache")
    skipped = set(skip_layers)
    for layer_idx, length in enumerate(lengths):
        expected = (
            prompt_length
            if not compressed or layer_idx in skipped
            else target_cache_tokens
        )
        if length != expected:
            raise AssertionError(
                f"Layer {layer_idx} has {length} cache tokens, expected {expected}"
            )
    return lengths


def _finish_result(
    *,
    model_name: str,
    method: str,
    config: str,
    example: PasskeyExample,
    keep_ratio: float,
    target_cache_tokens: int,
    observation_window_size: int | None,
    pooling_kernel_size: int | None,
    pooling_mode: str | None,
    skip_layers: Sequence[int],
    cache_before_mb: float,
    cache_after_mb: float,
    memory_saved_mb: float,
    memory_saved_percent: float,
    elapsed_seconds: float,
    generated_ids: Sequence[int],
    prediction: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "method": method,
        "config": config,
        "context_length": example.context_length,
        "keep_ratio": keep_ratio,
        "target_cache_tokens": target_cache_tokens,
        "observation_window_size": observation_window_size,
        "pooling_kernel_size": pooling_kernel_size,
        "pooling_mode": pooling_mode,
        "skip_layers": list(skip_layers),
        "seed": example.seed,
        "actual_depth": example.actual_depth,
        "target": example.answer_text,
        "prediction": prediction,
        "correct": exact_token_match(generated_ids, example.answer_ids),
        "cache_before_mb": cache_before_mb,
        "cache_after_mb": cache_after_mb,
        "memory_saved_mb": memory_saved_mb,
        "memory_saved_percent": memory_saved_percent,
        "elapsed_seconds": elapsed_seconds,
    }


@torch.inference_mode()
def evaluate_plain_or_l2(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    example: PasskeyExample,
    config: str,
    strategy: str,
    keep_ratio: float,
    skip_layers: Sequence[int],
    chunk_size: int,
    method: str = "l2",
    observation_window_size: int | None = None,
    pooling_kernel_size: int | None = None,
    pooling_mode: str | None = None,
) -> dict[str, Any]:
    """Evaluate the baseline or one L2 token-selection strategy."""

    devices = cuda_devices(model)
    synchronize_cuda_devices(devices)
    started = perf_counter()
    cache, last_logits, logical_position = prefill_plain(
        model,
        example.prompt_ids,
        chunk_size,
    )
    if logical_position != example.context_length:
        raise AssertionError("Logical position after prefill must equal prompt length")

    cache_before_mb = kv_cache_size_mb(cache)
    compressed = strategy != "none"
    capacity = (
        target_capacity(example.context_length, keep_ratio)
        if compressed
        else example.context_length
    )
    if compressed:
        cache = compress_cache_to_budget(
            cache,
            max_cache_tokens=capacity,
            strategy=strategy,
            skip_layers=skip_layers,
            seed=example.seed,
        )
    cache_after_mb = kv_cache_size_mb(cache)
    assert_cache_capacity(
        cache,
        example.context_length,
        capacity,
        skip_layers,
        compressed,
    )
    memory_saved_mb, memory_saved_percent = cache_memory_savings(
        cache_before_mb,
        cache_after_mb,
    )

    generated_ids, prediction, _, cache = generate_exact_answer(
        model,
        tokenizer,
        cache,
        last_logits,
        logical_position,
        example.answer_ids,
    )
    synchronize_cuda_devices(devices)
    elapsed_seconds = perf_counter() - started
    del cache
    del last_logits
    return _finish_result(
        model_name=model_name,
        method=method,
        config=config,
        example=example,
        keep_ratio=keep_ratio if compressed else 1.0,
        target_cache_tokens=capacity,
        observation_window_size=observation_window_size,
        pooling_kernel_size=pooling_kernel_size,
        pooling_mode=pooling_mode,
        skip_layers=skip_layers,
        cache_before_mb=cache_before_mb,
        cache_after_mb=cache_after_mb,
        memory_saved_mb=memory_saved_mb,
        memory_saved_percent=memory_saved_percent,
        elapsed_seconds=elapsed_seconds,
        generated_ids=generated_ids,
        prediction=prediction,
    )


@torch.inference_mode()
def evaluate_snapkv(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    example: PasskeyExample,
    target_cache_tokens: int,
    observation_window_size: int,
    pooling_kernel_size: int,
    pooling_mode: str,
    skip_layers: Sequence[int],
    chunk_size: int,
) -> dict[str, Any]:
    """Evaluate SnapKV after a full logical-position-preserving prefill."""

    devices = cuda_devices(model)
    synchronize_cuda_devices(devices)
    started = perf_counter()
    cache, last_logits, logical_position, scores_by_layer = (
        prefill_and_score_snapkv(
            model=model,
            prompt_ids=example.prompt_ids,
            observation_window_size=observation_window_size,
            chunk_size=chunk_size,
            skip_layers=skip_layers,
        )
    )
    if logical_position != example.context_length:
        raise AssertionError("Logical position after prefill must equal prompt length")
    assert_cache_capacity(
        cache,
        example.context_length,
        example.context_length,
        skip_layers,
        compressed=False,
    )

    cache_before_mb = kv_cache_size_mb(cache)
    cache = compress_snapkv_cache(
        cache=cache,
        scores_by_layer=scores_by_layer,
        target_capacity=target_cache_tokens,
        observation_window_size=observation_window_size,
        pooling_kernel_size=pooling_kernel_size,
        pooling_mode=pooling_mode,
        skip_layers=skip_layers,
    )
    cache_after_mb = kv_cache_size_mb(cache)
    assert_cache_capacity(
        cache,
        example.context_length,
        target_cache_tokens,
        skip_layers,
        compressed=True,
    )
    del scores_by_layer
    memory_saved_mb, memory_saved_percent = cache_memory_savings(
        cache_before_mb,
        cache_after_mb,
    )
    generated_ids, prediction, _, cache = generate_exact_answer(
        model,
        tokenizer,
        cache,
        last_logits,
        logical_position,
        example.answer_ids,
    )
    synchronize_cuda_devices(devices)
    elapsed_seconds = perf_counter() - started
    del cache
    del last_logits
    return _finish_result(
        model_name=model_name,
        method="snapkv",
        config="snapkv",
        example=example,
        keep_ratio=target_cache_tokens / example.context_length,
        target_cache_tokens=target_cache_tokens,
        observation_window_size=observation_window_size,
        pooling_kernel_size=pooling_kernel_size,
        pooling_mode=pooling_mode,
        skip_layers=skip_layers,
        cache_before_mb=cache_before_mb,
        cache_after_mb=cache_after_mb,
        memory_saved_mb=memory_saved_mb,
        memory_saved_percent=memory_saved_percent,
        elapsed_seconds=elapsed_seconds,
        generated_ids=generated_ids,
        prediction=prediction,
    )


def checkpoint_raw(rows: Sequence[dict[str, Any]], path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    frame.to_csv(path, index=False)
    return frame


def print_result(row: Mapping[str, Any]) -> None:
    prediction = str(row["prediction"]).replace("\r", "\\r").replace("\n", "\\n")
    print(
        f"{row['config']} | context={row['context_length']} | "
        f"seed={row['seed']} | depth={row['actual_depth']:.2f} | "
        f"target={row['target']} | prediction={prediction} | "
        f"correct={row['correct']}"
    )


def print_summary(summary_df: pd.DataFrame) -> None:
    print("\nSummary:")
    print(summary_df.loc[:, CONSOLE_SUMMARY_COLUMNS].to_string(index=False))


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary = (
        raw_df.groupby(
            SUMMARY_COLUMNS[:6],
            sort=False,
            as_index=False,
            dropna=False,
        )
        .agg(
            num_examples=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_cache_before_mb=("cache_before_mb", "mean"),
            mean_cache_after_mb=("cache_after_mb", "mean"),
            mean_memory_saved_mb=("memory_saved_mb", "mean"),
            mean_memory_saved_percent=("memory_saved_percent", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
        .loc[:, SUMMARY_COLUMNS]
    )
    return summary
