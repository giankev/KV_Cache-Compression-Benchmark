from __future__ import annotations

import random
import re
from typing import Any, Sequence

import torch

from .cache_compression import CompressionStrategy, compress_cache
from .cache_metrics import kv_cache_size_mb, theoretical_kv_cache_size_mb


def make_passkey_prompt(
    tokenizer: Any,
    target_tokens: int = 2048,
    seed: int = 0,
    depth: float = 0.5,
) -> tuple[str, str, str]:
    """Build the standard passkey retrieval prompt."""

    random.seed(seed)
    answer = str(random.randint(10000, 99999))

    intro = (
        "There is an important information hidden inside a lot of irrelevant text. "
        "Find it and remember it. I will ask you about it at the end.\n\n"
    )
    needle = f"The pass key is {answer}. Remember it. {answer} is the pass key.\n\n"
    question = "What is the pass key? The pass key is"

    filler_sentence = (
        "The grass is green. The sky is blue. The sun is yellow. "
        "Here we go. There and back again. "
    )

    fixed_text = intro + needle + question
    fixed_tokens = len(tokenizer(fixed_text, add_special_tokens=False).input_ids)
    filler_budget = max(0, target_tokens - fixed_tokens)

    filler_text = filler_sentence * ((filler_budget // 20) + 100)
    filler_ids = tokenizer(filler_text, add_special_tokens=False).input_ids[
        :filler_budget
    ]

    split = int(len(filler_ids) * depth)
    prefix = tokenizer.decode(filler_ids[:split], skip_special_tokens=True)
    suffix = tokenizer.decode(filler_ids[split:], skip_special_tokens=True)
    context = intro + prefix + "\n\n" + needle + suffix + "\n\n"

    return context, question, answer


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


@torch.no_grad()
def generate_passkey_answer(
    model: Any,
    tokenizer: Any,
    context: str,
    question: str,
    max_new_tokens: int = 12,
    expected_digits: int | None = None,
    use_compression: bool = False,
    keep_ratio: float = 0.6,
    prune_after: int = 1024,
    chunk_size: int = 512,
    strategy: CompressionStrategy = "low_l2",
    skip_layers: Sequence[int] = (),
) -> dict[str, Any]:
    """Run strict-context passkey generation.

    The order is:
    1. prefill the context,
    2. compress the context KV cache once,
    3. process the question with the compressed cache,
    4. decode the answer manually.
    """

    device = next(model.parameters()).device

    context_inputs = tokenizer(context, return_tensors="pt").to(device)
    question_inputs = tokenizer(
        question,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    context_ids = context_inputs["input_ids"]
    question_ids = question_inputs["input_ids"]
    batch_size = context_ids.shape[0]

    cache = None
    logical_pos = 0

    for start in range(0, context_ids.shape[1], chunk_size):
        chunk = context_ids[:, start : start + chunk_size]
        chunk_len = chunk.shape[1]

        if cache is None:
            attention_mask = torch.ones(
                (batch_size, chunk_len),
                device=device,
                dtype=torch.long,
            )
            out = model(
                input_ids=chunk,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        else:
            past_len = cache.layers[0].keys.shape[2]
            attention_mask = torch.ones(
                (batch_size, past_len + chunk_len),
                device=device,
                dtype=torch.long,
            )
            cache_position = torch.arange(
                logical_pos,
                logical_pos + chunk_len,
                device=device,
                dtype=torch.long,
            )
            out = model(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )

        cache = out.past_key_values
        logical_pos += chunk_len

    if use_compression:
        cache = compress_cache(
            cache,
            keep_ratio=keep_ratio,
            prune_after=prune_after,
            strategy=strategy,
            skip_layers=skip_layers,
        )

    q_len = question_ids.shape[1]
    past_len = cache.layers[0].keys.shape[2]
    attention_mask = torch.ones(
        (batch_size, past_len + q_len),
        device=device,
        dtype=torch.long,
    )
    cache_position = torch.arange(
        logical_pos,
        logical_pos + q_len,
        device=device,
        dtype=torch.long,
    )
    out = model(
        input_ids=question_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )

    cache = out.past_key_values
    logical_pos += q_len
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

        past_len = cache.layers[0].keys.shape[2]
        attention_mask_step = torch.ones(
            (batch_size, past_len + 1),
            device=device,
            dtype=torch.long,
        )
        cache_position = torch.tensor(
            [logical_pos],
            device=device,
            dtype=torch.long,
        )
        out = model(
            input_ids=next_token,
            attention_mask=attention_mask_step,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

        cache = out.past_key_values
        logical_pos += 1
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated_text = _decode_generated_text(tokenizer, generated)
    prediction = extract_first_number(generated_text)
    compressed_cache_mb = kv_cache_size_mb(cache)
    uncompressed_cache_mb = theoretical_kv_cache_size_mb(
        model=model,
        seq_len=logical_pos,
        batch_size=batch_size,
    )
    memory_saved_percent = 100 * (
        1 - compressed_cache_mb / (uncompressed_cache_mb + 1e-8)
    )

    return {
        "generated_text": generated_text,
        "prediction": prediction,
        "compressed_cache_mb": compressed_cache_mb,
        "uncompressed_cache_mb": uncompressed_cache_mb,
        "memory_saved_percent": memory_saved_percent,
    }
