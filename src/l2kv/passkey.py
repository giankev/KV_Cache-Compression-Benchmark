"""Build deterministic exact-token prompts for the l2compress passkey task."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch


TASK_DESCRIPTION = (
    "There is an important info hidden inside a lot of irrelevant text. Find it "
    "and memorize them. I will quiz you about the important information there."
)
GARBAGE_UNIT = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again."
)
INFORMATION_TEMPLATE = (
    "The pass key is {passkey}. Remember it. {passkey} is the pass key."
)
FINAL_QUESTION = "What is the pass key? The pass key is"


@dataclass(frozen=True)
class PasskeyExample:
    """One deterministic, exact-length professor-style passkey prompt."""

    prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    answer_text: str
    context_length: int
    seed: int
    actual_depth: float
    information_token_position: int
    question_token_position: int


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().cpu().tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Expected one tokenized sequence")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _repeat_to_length(unit_ids: list[int], length: int) -> list[int]:
    if length == 0:
        return []
    if not unit_ids:
        raise ValueError("The garbage unit produced no token IDs")
    repeats = (length + len(unit_ids) - 1) // len(unit_ids)
    return (unit_ids * repeats)[:length]


def make_passkey_example(
    tokenizer: Any,
    context_length: int = 8192,
    seed: int = 0,
    observation_window_size: int | None = None,
) -> PasskeyExample:
    """Build an exact-length passkey prompt by concatenating token IDs."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if observation_window_size is not None and observation_window_size <= 0:
        raise ValueError("observation_window_size must be positive")

    rng = random.Random(seed)
    answer_text = str(rng.randint(1, 50000))
    information_text = INFORMATION_TEMPLATE.format(passkey=answer_text)

    task_ids = _encode_without_special_tokens(tokenizer, TASK_DESCRIPTION + "\n")
    garbage_ids = _encode_without_special_tokens(tokenizer, GARBAGE_UNIT + "\n")
    information_ids = _encode_without_special_tokens(
        tokenizer,
        "\n" + information_text + "\n",
    )
    question_ids = _encode_without_special_tokens(
        tokenizer,
        "\n" + FINAL_QUESTION,
    )
    answer_ids = _encode_without_special_tokens(tokenizer, answer_text)

    if not question_ids:
        raise ValueError("The final question produced no token IDs")
    if not answer_ids:
        raise ValueError("The passkey answer produced no token IDs")
    if (
        observation_window_size is not None
        and len(question_ids) > observation_window_size
    ):
        raise ValueError(
            "The complete final question must fit inside the SnapKV "
            f"observation window ({len(question_ids)} > "
            f"{observation_window_size})"
        )

    fixed_tokens = len(task_ids) + len(information_ids) + len(question_ids)
    total_garbage_tokens = context_length - fixed_tokens
    if total_garbage_tokens < 0:
        raise ValueError(
            f"context_length={context_length} is too small for the fixed "
            f"prompt components ({fixed_tokens} tokens)"
        )

    prefix_garbage_tokens = rng.randint(0, total_garbage_tokens)
    suffix_garbage_tokens = total_garbage_tokens - prefix_garbage_tokens
    information_token_position = len(task_ids) + prefix_garbage_tokens
    question_token_position = context_length - len(question_ids)

    # Appending the question last keeps it fully inside SnapKV's final window.
    prompt_ids = (
        task_ids
        + _repeat_to_length(garbage_ids, prefix_garbage_tokens)
        + information_ids
        + _repeat_to_length(garbage_ids, suffix_garbage_tokens)
        + question_ids
    )
    if len(prompt_ids) != context_length:
        raise AssertionError(
            f"Passkey prompt length mismatch: {len(prompt_ids)} != "
            f"{context_length}"
        )
    if prompt_ids[question_token_position:] != question_ids:
        raise AssertionError("The final question must be the end of the prompt")

    actual_depth = (
        prefix_garbage_tokens / total_garbage_tokens
        if total_garbage_tokens
        else 0.0
    )
    return PasskeyExample(
        prompt_ids=tuple(prompt_ids),
        answer_ids=tuple(answer_ids),
        answer_text=answer_text,
        context_length=context_length,
        seed=seed,
        actual_depth=actual_depth,
        information_token_position=information_token_position,
        question_token_position=question_token_position,
    )
