from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Sequence

import torch

from .cache_compression import CompressionStrategy, compress_cache
from .cache_metrics import kv_cache_size_mb
from .position_utils import make_cache_position, make_position_ids


@dataclass(frozen=True)
class PasskeyPrompt:
    """Token-level passkey example with an exact context length."""

    context_ids: tuple[int, ...]
    question_ids: tuple[int, ...]
    answer: str
    needle_token_position: int

    @property
    def context_len_actual(self) -> int:
        return len(self.context_ids)


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().cpu().tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Expected a single tokenized sequence")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def make_passkey_prompt(
    tokenizer: Any,
    target_tokens: int = 2048,
    seed: int = 0,
    depth: float = 0.5,
) -> PasskeyPrompt:
    """Build a passkey prompt directly from token IDs.

    Intro, filler, needle, and question are each tokenized once with
    ``add_special_tokens=False``. Their ID sequences are concatenated directly,
    so the context is always exactly ``target_tokens`` tokens and never passes
    through a decode-then-retokenize cycle.
    """

    if (
        isinstance(target_tokens, bool)
        or not isinstance(target_tokens, Integral)
        or target_tokens < 1
    ):
        raise ValueError("target_tokens must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if (
        isinstance(depth, bool)
        or not isinstance(depth, Real)
        or not math.isfinite(float(depth))
        or not 0 <= float(depth) <= 1
    ):
        raise ValueError("depth must satisfy 0 <= depth <= 1")

    answer = str(random.Random(int(seed)).randint(10000, 99999))
    intro = (
        "There is important information hidden inside irrelevant text. "
        "Find it and remember it. I will ask about it at the end.\n\n"
    )
    filler = (
        "The grass is green. The sky is blue. The sun is yellow. "
        "Here we go. There and back again.\n"
    )
    needle = f"\nThe pass key is {answer}. Remember it. {answer} is the pass key.\n\n"
    question = "What is the pass key? The pass key is"

    intro_ids = _encode_without_special_tokens(tokenizer, intro)
    filler_unit_ids = _encode_without_special_tokens(tokenizer, filler)
    needle_ids = _encode_without_special_tokens(tokenizer, needle)
    question_ids = _encode_without_special_tokens(tokenizer, question)

    fixed_context_tokens = len(intro_ids) + len(needle_ids)
    if target_tokens < fixed_context_tokens:
        raise ValueError(
            f"target_tokens={target_tokens} is too small for intro and needle "
            f"({fixed_context_tokens} tokens)"
        )
    if not filler_unit_ids and target_tokens > fixed_context_tokens:
        raise ValueError("The filler text produced no token IDs")
    if not question_ids:
        raise ValueError("The question produced no token IDs")

    filler_budget = int(target_tokens) - fixed_context_tokens
    repeats = math.ceil(filler_budget / len(filler_unit_ids)) if filler_budget else 0
    filler_ids = (filler_unit_ids * repeats)[:filler_budget]
    prefix_length = int(filler_budget * float(depth))
    needle_token_position = len(intro_ids) + prefix_length

    context_ids = (
        intro_ids
        + filler_ids[:prefix_length]
        + needle_ids
        + filler_ids[prefix_length:]
    )
    if len(context_ids) != target_tokens:
        raise AssertionError(
            f"Passkey context length mismatch: {len(context_ids)} != {target_tokens}"
        )

    return PasskeyPrompt(
        context_ids=tuple(context_ids),
        question_ids=tuple(question_ids),
        answer=answer,
        needle_token_position=needle_token_position,
    )


def extract_first_number(text: str) -> str:
    match = re.search(r"\d+", text)
    if match is None:
        return ""
    return match.group(0)


def is_correct_prediction(prediction: str, answer: str) -> bool:
    return prediction == answer


def _eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        generation_config = getattr(model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)

    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return {int(token_id) for token_id in eos_token_id}


def _decode_generated_text(
    tokenizer: Any,
    generated: Sequence[torch.Tensor],
) -> str:
    if not generated:
        return ""

    generated_ids = torch.cat(list(generated), dim=-1)[0].detach().cpu().tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _as_single_batch_ids(
    token_ids: Sequence[int] | torch.Tensor,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError(f"{name} must contain one non-empty sequence")
    return ids


def _forward_at_logical_position(
    model: Any,
    input_ids: torch.Tensor,
    cache: Any | None,
    logical_position: int,
) -> Any:
    position_ids = make_position_ids(
        start_position=logical_position,
        length=int(input_ids.shape[1]),
        device=input_ids.device,
    )
    model_inputs: dict[str, Any] = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "cache_position": make_cache_position(position_ids),
        "use_cache": True,
        "return_dict": True,
    }
    if cache is not None:
        model_inputs["past_key_values"] = cache
    return model(**model_inputs)


@torch.no_grad()
def generate_passkey_answer(
    model: Any,
    tokenizer: Any,
    context_ids: Sequence[int] | torch.Tensor,
    question_ids: Sequence[int] | torch.Tensor,
    max_new_tokens: int = 12,
    expected_digits: int | None = None,
    use_compression: bool = False,
    keep_ratio: float = 0.6,
    prune_after: int = 1024,
    chunk_size: int = 512,
    strategy: CompressionStrategy = "low_l2",
    skip_layers: Sequence[int] = (),
    compression_seed: int | None = None,
) -> dict[str, Any]:
    """Run deterministic passkey generation from exact token-ID sequences.

    The context is prefetched in chunks, its actual cache size is measured, and
    compression happens once after prefill. Because skipped layers can remain
    physically longer than compressed layers, every question and answer token
    after pruning is processed in a separate forward without a padding mask.
    Explicit ``position_ids`` and ``cache_position`` always use the original
    logical position rather than a physical cache-layer length.
    """

    if isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    device = next(model.parameters()).device
    context_tensor = _as_single_batch_ids(context_ids, device, "context_ids")
    question_tensor = _as_single_batch_ids(question_ids, device, "question_ids")

    cache = None
    logical_position = 0

    for start in range(0, context_tensor.shape[1], chunk_size):
        chunk = context_tensor[:, start : start + chunk_size]
        out = _forward_at_logical_position(
            model=model,
            input_ids=chunk,
            cache=cache,
            logical_position=logical_position,
        )
        cache = out.past_key_values
        logical_position += int(chunk.shape[1])

    cache_mb_before_compression = kv_cache_size_mb(cache)
    if use_compression:
        cache = compress_cache(
            cache,
            keep_ratio=keep_ratio,
            prune_after=prune_after,
            strategy=strategy,
            skip_layers=skip_layers,
            seed=compression_seed,
        )
        cache_mb_after_compression = kv_cache_size_mb(cache)
        memory_saved_percent = 100 * (
            1 - cache_mb_after_compression / cache_mb_before_compression
        )
    else:
        cache_mb_after_compression = cache_mb_before_compression
        memory_saved_percent = 0.0

    out = None
    for question_index in range(question_tensor.shape[1]):
        question_token = question_tensor[:, question_index : question_index + 1]
        out = _forward_at_logical_position(
            model=model,
            input_ids=question_token,
            cache=cache,
            logical_position=logical_position,
        )
        cache = out.past_key_values
        logical_position += 1

    if out is None:
        raise AssertionError("Question forward did not run")
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated: list[torch.Tensor] = []
    eos_token_ids = _eos_token_ids(model, tokenizer)

    for step in range(max_new_tokens):
        generated.append(next_token)
        generated_text = _decode_generated_text(tokenizer, generated)
        prediction = extract_first_number(generated_text)
        token_id = int(next_token[0, -1].detach().cpu().item())

        if token_id in eos_token_ids:
            break
        if expected_digits is not None and len(prediction) >= expected_digits:
            break
        if step == max_new_tokens - 1:
            break

        out = _forward_at_logical_position(
            model=model,
            input_ids=next_token,
            cache=cache,
            logical_position=logical_position,
        )
        cache = out.past_key_values
        logical_position += 1
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated_text = _decode_generated_text(tokenizer, generated)
    prediction = extract_first_number(generated_text)

    return {
        "generated_text": generated_text,
        "prediction": prediction,
        "cache_mb_before_compression": cache_mb_before_compression,
        "cache_mb_after_compression": cache_mb_after_compression,
        "final_cache_mb": kv_cache_size_mb(cache),
        "memory_saved_percent": memory_saved_percent,
        "logical_position": logical_position,
    }
