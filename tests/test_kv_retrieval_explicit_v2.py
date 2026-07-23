from __future__ import annotations

import math
import random
from typing import Any

import pytest

from l2kv.kv_retrieval import make_kv_retrieval_prompt
from scripts import run_l2_kv_retrieval as l2_runner
from scripts import run_snapkv_kv_retrieval as snapkv_runner


class FakeEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class InspectableWhitespaceTokenizer:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.tokens_by_id: dict[int, str] = {}

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
                token_id = len(self.vocabulary) + 1
                self.vocabulary[token] = token_id
                self.tokens_by_id[token_id] = token
            input_ids.append(self.vocabulary[token])
        return FakeEncoding(input_ids)

    def render(self, token_ids: tuple[int, ...]) -> str:
        return " ".join(self.tokens_by_id[token_id] for token_id in token_ids)


def _make_prompt(
    tokenizer: InspectableWhitespaceTokenizer,
    *,
    depth: float = 0.5,
    seed: int = 7,
):
    return make_kv_retrieval_prompt(
        tokenizer,
        target_tokens=160,
        seed=seed,
        depth=depth,
        observation_window_size=16,
        prompt_style="explicit_v2",
    )


def test_explicit_v2_has_exact_length_and_question_is_final() -> None:
    prompt = _make_prompt(InspectableWhitespaceTokenizer())

    assert prompt.context_len_actual == 160
    assert len(prompt.prompt_ids) == 160
    assert prompt.question_token_position + len(prompt.question_ids) == 160
    assert prompt.prompt_ids[prompt.question_token_position :] == (
        prompt.question_ids
    )


def test_explicit_v2_question_fits_observation_window_sixteen() -> None:
    prompt = _make_prompt(InspectableWhitespaceTokenizer())

    assert len(prompt.question_ids) <= 16
    assert prompt.question_token_position >= len(prompt.prompt_ids) - 16


def test_explicit_v2_records_and_values_are_unique() -> None:
    tokenizer = InspectableWhitespaceTokenizer()
    prompt = _make_prompt(tokenizer)

    assert len(prompt.record_keys) == len(set(prompt.record_keys))
    assert len(prompt.record_values) == len(set(prompt.record_values))
    assert all(len(value) == 5 and value.isdigit() for value in prompt.record_values)
    assert all(
        record.text == f"Record: key={record.key}; value={record.value}.\n"
        for record in prompt.records
    )

    rendered_prompt = tokenizer.render(prompt.prompt_ids)
    assert rendered_prompt.count(prompt.target_value) == 1
    assert rendered_prompt.count(prompt.target_key) == 2
    assert prompt.record_values.count(prompt.target_value) == 1
    assert prompt.record_keys.count(prompt.target_key) == 1


@pytest.mark.parametrize("depth", [0.25, 0.50, 0.75])
def test_explicit_v2_target_tracks_requested_depth(depth: float) -> None:
    prompt = _make_prompt(
        InspectableWhitespaceTokenizer(),
        depth=depth,
    )

    expected_index = int(math.floor(depth * (len(prompt.records) - 1) + 0.5))
    target_record = prompt.records[expected_index]
    start = prompt.target_record_token_position
    stop = start + len(target_record.token_ids)

    assert prompt.target_record_index == expected_index
    assert prompt.depth_actual == expected_index / (len(prompt.records) - 1)
    assert target_record.key == prompt.target_key
    assert target_record.value == prompt.target_value
    assert target_record.token_position == start
    assert prompt.prompt_ids[start:stop] == target_record.token_ids


def test_explicit_v2_is_deterministic_and_uses_local_rng() -> None:
    tokenizer = InspectableWhitespaceTokenizer()
    random.seed(1234)
    global_state = random.getstate()

    prompt_a = _make_prompt(tokenizer, seed=11)
    prompt_b = _make_prompt(tokenizer, seed=11)
    prompt_c = _make_prompt(tokenizer, seed=12)

    assert prompt_a == prompt_b
    assert prompt_a.prompt_ids != prompt_c.prompt_ids
    assert random.getstate() == global_state


def test_default_prompt_style_remains_legacy() -> None:
    tokenizer = InspectableWhitespaceTokenizer()
    kwargs = {
        "target_tokens": 160,
        "seed": 5,
        "depth": 0.5,
        "observation_window_size": 16,
    }

    default_prompt = make_kv_retrieval_prompt(tokenizer, **kwargs)
    named_legacy_prompt = make_kv_retrieval_prompt(
        tokenizer,
        prompt_style="legacy",
        **kwargs,
    )

    assert default_prompt == named_legacy_prompt


def test_l2_and_snapkv_paths_build_the_identical_prompt() -> None:
    tokenizer = InspectableWhitespaceTokenizer()
    arguments = {
        "tokenizer": tokenizer,
        "context_length": 160,
        "depth": 0.75,
        "seed": 13,
        "prompt_style": "explicit_v2",
    }

    l2_prompt = l2_runner.build_prompt_for_case(**arguments)
    snapkv_prompt = snapkv_runner.build_prompt_for_case(**arguments)

    assert l2_prompt.prompt_ids == snapkv_prompt.prompt_ids
    assert l2_prompt.target_key == snapkv_prompt.target_key
    assert l2_prompt.target_value == snapkv_prompt.target_value
    assert (
        l2_prompt.target_record_token_position
        == snapkv_prompt.target_record_token_position
    )


def test_explicit_v2_rejects_a_window_shorter_than_the_question() -> None:
    with pytest.raises(ValueError, match="final question"):
        make_kv_retrieval_prompt(
            InspectableWhitespaceTokenizer(),
            target_tokens=160,
            seed=0,
            depth=0.5,
            observation_window_size=2,
            prompt_style="explicit_v2",
        )
