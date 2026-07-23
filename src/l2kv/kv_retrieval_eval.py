from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .cache_metrics import cache_layer_lengths
from .kv_retrieval import extract_first_five_digit_number
from .position_utils import make_cache_position, make_position_ids


@dataclass(frozen=True)
class PrefillState:
    """Cache, logits, and optional SnapKV scores produced by prompt prefill."""

    cache: Any
    last_logits: torch.Tensor
    logical_position: int
    scores_by_layer: Sequence[torch.Tensor | None] | None = None


def cuda_devices(model: Any) -> tuple[torch.device, ...]:
    """Return every CUDA device containing at least one model parameter."""

    devices = {
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    }
    return tuple(sorted(devices, key=str))


def synchronize_cuda_devices(devices: Sequence[torch.device]) -> None:
    """Synchronize the supplied CUDA devices; do nothing on CPU-only runs."""

    for device in devices:
        torch.cuda.synchronize(device)


@torch.inference_mode()
def prefill_plain(
    model: Any,
    prompt_ids: Sequence[int],
    chunk_size: int,
) -> PrefillState:
    """Prefill one exact-token prompt in chunks without collecting attention."""

    device = next(model.parameters()).device
    input_ids = torch.as_tensor(
        prompt_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
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
    return PrefillState(
        cache=cache,
        last_logits=last_logits,
        logical_position=logical_position,
    )


@torch.inference_mode()
def prefill_snapkv(
    model: Any,
    prompt_ids: Sequence[int],
    observation_window_size: int,
    chunk_size: int,
    skip_layers: Sequence[int],
) -> PrefillState:
    """Prefill one prompt while collecting per-layer SnapKV prefix scores."""

    # Keep the plain-prefill evaluation path independent from the optional
    # attention-scoring implementation until this function is actually used.
    from .snapkv import prefill_and_score_snapkv

    result = prefill_and_score_snapkv(
        model=model,
        prompt_ids=prompt_ids,
        observation_window_size=observation_window_size,
        chunk_size=chunk_size,
        skip_layers=skip_layers,
    )
    return PrefillState(
        cache=result.cache,
        last_logits=result.last_logits,
        logical_position=result.logical_position,
        scores_by_layer=result.scores_by_layer,
    )


def _eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(
            getattr(model, "generation_config", None),
            "eos_token_id",
            None,
        )
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return {int(token_id) for token_id in eos_token_id}


def _decode_generated(
    tokenizer: Any,
    generated: Sequence[torch.Tensor],
) -> str:
    if not generated:
        return ""
    token_ids = torch.cat(list(generated), dim=-1)[0].detach().cpu().tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True)


@torch.inference_mode()
def generate_greedy(
    model: Any,
    tokenizer: Any,
    cache: Any,
    last_logits: torch.Tensor,
    logical_position: int,
    max_new_tokens: int,
    *,
    validate_cache_growth: bool = False,
) -> tuple[str, str, Any]:
    """Greedily decode until EOS, a five-digit answer, or the token limit."""

    if last_logits.ndim == 3:
        last_logits = last_logits[:, -1, :]
    if last_logits.ndim != 2 or last_logits.shape[0] != 1:
        raise ValueError("last_logits must have shape [1, vocabulary]")

    input_device = next(model.parameters()).device
    next_token = last_logits.argmax(dim=-1, keepdim=True).to(input_device)
    eos_token_ids = _eos_token_ids(model, tokenizer)
    generated: list[torch.Tensor] = []

    for step in range(max_new_tokens):
        generated.append(next_token)
        generated_text = _decode_generated(tokenizer, generated)
        prediction = extract_first_five_digit_number(generated_text)
        token_id = int(next_token[0, 0].detach().cpu().item())

        if token_id in eos_token_ids or prediction or step == max_new_tokens - 1:
            break

        position_ids = make_position_ids(
            logical_position,
            1,
            next_token.device,
        )
        lengths_before = (
            tuple(cache_layer_lengths(cache))
            if validate_cache_growth
            else ()
        )
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
        if validate_cache_growth:
            lengths_after = tuple(cache_layer_lengths(cache))
            expected_lengths = tuple(length + 1 for length in lengths_before)
            if lengths_after != expected_lengths:
                raise AssertionError(
                    "Every cache layer must add exactly one token per decode "
                    f"step; before={lengths_before}, after={lengths_after}"
                )
        logical_position += 1
        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        ).to(input_device)
        del outputs

    generated_text = _decode_generated(tokenizer, generated)
    return (
        generated_text,
        extract_first_five_digit_number(generated_text),
        cache,
    )


def target_capacity(prompt_length: int, keep_ratio: float) -> int:
    """Return the total retained-token budget used by retrieval benchmarks."""

    return math.floor(prompt_length * keep_ratio)


def assert_cache_capacity(
    cache: Any,
    prompt_length: int,
    target_capacity: int,
    skip_layers: Sequence[int],
    compressed: bool,
) -> tuple[int, ...]:
    """Validate every layer's expected physical cache length."""

    lengths = tuple(cache_layer_lengths(cache))
    if not lengths:
        raise AssertionError("The model returned an empty cache")
    invalid_skips = sorted(set(skip_layers) - set(range(len(lengths))))
    if invalid_skips:
        raise ValueError(f"skip layer indices do not exist: {invalid_skips}")

    skipped = set(skip_layers)
    for layer_idx, length in enumerate(lengths):
        expected = (
            prompt_length
            if not compressed or layer_idx in skipped
            else target_capacity
        )
        if length != expected:
            raise AssertionError(
                f"Layer {layer_idx} has {length} cache tokens, expected {expected}"
            )
    return lengths
