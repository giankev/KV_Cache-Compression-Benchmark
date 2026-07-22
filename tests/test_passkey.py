from __future__ import annotations

import random
from typing import Any

import pytest

from l2kv.passkey import make_passkey_prompt


class FakeEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class WhitespaceTokenizer:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> FakeEncoding:
        assert add_special_tokens is False
        input_ids: list[int] = []
        for token in text.split():
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary) + 1
            input_ids.append(self.vocabulary[token])
        return FakeEncoding(input_ids)


def test_passkey_prompt_has_exact_length_and_local_random_state() -> None:
    tokenizer = WhitespaceTokenizer()
    random.seed(1234)
    global_state = random.getstate()

    prompt_a = make_passkey_prompt(
        tokenizer,
        target_tokens=96,
        seed=0,
        depth=0.5,
    )
    prompt_b = make_passkey_prompt(
        tokenizer,
        target_tokens=96,
        seed=0,
        depth=0.5,
    )

    assert prompt_a.context_len_actual == 96
    assert prompt_a == prompt_b
    assert 0 < prompt_a.needle_token_position < prompt_a.context_len_actual
    assert prompt_a.answer.isdigit() and len(prompt_a.answer) == 5
    assert prompt_a.question_ids
    assert random.getstate() == global_state


@pytest.mark.parametrize("depth", [-0.01, 1.01, float("nan")])
def test_passkey_prompt_rejects_invalid_depth(depth: float) -> None:
    with pytest.raises(ValueError):
        make_passkey_prompt(
            WhitespaceTokenizer(),
            target_tokens=96,
            seed=0,
            depth=depth,
        )
