from __future__ import annotations

import math
import random
from typing import Any

import pytest

from l2kv.kv_retrieval import (
    extract_first_five_digit_number,
    make_kv_retrieval_prompt,
)


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


def test_prompt_has_exact_length_unique_records_and_local_randomness() -> None:
    tokenizer = WhitespaceTokenizer()
    random.seed(1234)
    global_state = random.getstate()

    prompt = make_kv_retrieval_prompt(
        tokenizer,
        target_tokens=128,
        seed=7,
        depth=0.5,
        observation_window_size=16,
    )

    assert prompt.context_len_actual == 128
    assert len(prompt.prompt_ids) == 128
    assert len(prompt.records) > 2
    assert len(prompt.record_keys) == len(set(prompt.record_keys))
    assert len(prompt.record_values) == len(set(prompt.record_values))
    assert prompt.record_keys.count(prompt.target_key) == 1
    assert prompt.record_values.count(prompt.target_value) == 1
    assert all(len(value) == 5 and value.isdigit() for value in prompt.record_values)
    assert all(
        record.text
        == f"Record {record.key}: REGISTER_CONTENT is {record.value}.\n"
        for record in prompt.records
    )
    assert random.getstate() == global_state


def test_final_question_is_inside_observation_window() -> None:
    prompt = make_kv_retrieval_prompt(
        WhitespaceTokenizer(),
        target_tokens=128,
        seed=0,
        depth=0.5,
        observation_window_size=16,
    )

    assert prompt.question_token_position >= 128 - 16
    assert (
        prompt.prompt_ids[prompt.question_token_position :]
        == prompt.question_ids
    )


@pytest.mark.parametrize("depth", [0.25, 0.50, 0.75])
def test_target_position_tracks_requested_record_depth(depth: float) -> None:
    prompt = make_kv_retrieval_prompt(
        WhitespaceTokenizer(),
        target_tokens=160,
        seed=3,
        depth=depth,
        observation_window_size=16,
    )

    expected_index = int(math.floor(depth * (len(prompt.records) - 1) + 0.5))
    target_record = prompt.records[expected_index]
    expected_token_position = sum(
        len(record.token_ids) for record in prompt.records[:expected_index]
    )

    assert prompt.target_record_index == expected_index
    assert target_record.key == prompt.target_key
    assert target_record.value == prompt.target_value
    assert prompt.target_record_token_position == expected_token_position
    assert target_record.token_position == expected_token_position


def test_seed_is_deterministic_and_different_seeds_change_prompt() -> None:
    tokenizer = WhitespaceTokenizer()
    kwargs = {
        "target_tokens": 128,
        "depth": 0.5,
        "observation_window_size": 16,
    }

    prompt_a = make_kv_retrieval_prompt(tokenizer, seed=11, **kwargs)
    prompt_b = make_kv_retrieval_prompt(tokenizer, seed=11, **kwargs)
    prompt_c = make_kv_retrieval_prompt(tokenizer, seed=12, **kwargs)

    assert prompt_a == prompt_b
    assert prompt_a.prompt_ids != prompt_c.prompt_ids


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is 10536.", "10536"),
        ("First 1234, then 92741 and 48372.", "92741"),
        ("Do not accept 123456 as a five-digit answer.", ""),
        ("No number here.", ""),
    ],
)
def test_extract_first_five_digit_number(text: str, expected: str) -> None:
    assert extract_first_five_digit_number(text) == expected


@pytest.mark.parametrize("depth", [-0.01, 1.01, float("nan")])
def test_invalid_depth_raises_value_error(depth: float) -> None:
    with pytest.raises(ValueError, match="depth"):
        make_kv_retrieval_prompt(
            WhitespaceTokenizer(),
            target_tokens=128,
            seed=0,
            depth=depth,
            observation_window_size=16,
        )


def test_too_small_observation_window_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="final question"):
        make_kv_retrieval_prompt(
            WhitespaceTokenizer(),
            target_tokens=128,
            seed=0,
            depth=0.5,
            observation_window_size=2,
        )
