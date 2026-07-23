from __future__ import annotations

import random
from typing import Any

import pytest

from l2kv.passkey import (
    FINAL_QUESTION,
    INFORMATION_TEMPLATE,
    make_passkey_example,
)


class FakeEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class CharacterTokenizer:
    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> FakeEncoding:
        assert add_special_tokens is False
        return FakeEncoding([ord(character) for character in text])


def _count_subsequence(sequence: tuple[int, ...], needle: tuple[int, ...]) -> int:
    return sum(
        sequence[index : index + len(needle)] == needle
        for index in range(len(sequence) - len(needle) + 1)
    )


def test_prompt_is_exact_deterministic_and_does_not_change_global_rng() -> None:
    tokenizer = CharacterTokenizer()
    random.seed(1234)
    global_state = random.getstate()

    example_a = make_passkey_example(tokenizer, context_length=512, seed=7)
    example_b = make_passkey_example(tokenizer, context_length=512, seed=7)

    assert example_a == example_b
    assert len(example_a.prompt_ids) == 512
    assert example_a.context_length == 512
    assert random.getstate() == global_state


def test_seed_changes_the_passkey_and_random_information_position() -> None:
    tokenizer = CharacterTokenizer()
    example_a = make_passkey_example(tokenizer, context_length=512, seed=0)
    example_b = make_passkey_example(tokenizer, context_length=512, seed=1)

    assert example_a.answer_text != example_b.answer_text
    assert example_a.information_token_position != (
        example_b.information_token_position
    )
    assert 1 <= int(example_a.answer_text) <= 50_000
    assert 0.0 <= example_a.actual_depth <= 1.0


def test_passkey_occurs_twice_and_question_is_the_prompt_suffix() -> None:
    tokenizer = CharacterTokenizer()
    example = make_passkey_example(tokenizer, context_length=512, seed=3)
    information_text = INFORMATION_TEMPLATE.format(
        passkey=example.answer_text
    )
    information_ids = tuple(
        tokenizer("\n" + information_text + "\n").input_ids
    )
    stored_information = example.prompt_ids[
        example.information_token_position :
        example.information_token_position + len(information_ids)
    ]
    question_ids = tuple(tokenizer("\n" + FINAL_QUESTION).input_ids)

    assert _count_subsequence(stored_information, example.answer_ids) == 2
    assert example.prompt_ids[example.question_token_position :] == question_ids


def test_snapkv_requires_the_complete_question_to_fit_the_window() -> None:
    with pytest.raises(ValueError, match="complete final question"):
        make_passkey_example(
            CharacterTokenizer(),
            context_length=512,
            seed=0,
            observation_window_size=8,
        )
