from __future__ import annotations

from typing import Any

import pytest

from scripts import run_l2_passkey, run_snapkv_passkey


class _Encoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class _Tokenizer:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> _Encoding:
        assert add_special_tokens is False
        ids: list[int] = []
        for token in text.split():
            self.vocabulary.setdefault(token, len(self.vocabulary) + 1)
            ids.append(self.vocabulary[token])
        return _Encoding(ids)


def test_l2_and_snapkv_build_identical_prompts() -> None:
    tokenizer = _Tokenizer()
    l2_example = run_l2_passkey.make_passkey_example(tokenizer, 128, 4)
    snapkv_example = run_snapkv_passkey.make_passkey_example(
        tokenizer,
        128,
        4,
        observation_window_size=16,
    )

    assert l2_example.prompt_ids == snapkv_example.prompt_ids
    assert l2_example.answer_ids == snapkv_example.answer_ids
    assert l2_example.actual_depth == snapkv_example.actual_depth


def test_runner_defaults_and_configuration_sets() -> None:
    l2_args = run_l2_passkey.parse_args([])
    snapkv_args = run_snapkv_passkey.parse_args([])

    assert l2_args.context_lengths == (8192,)
    assert l2_args.seeds == (0, 1, 2)
    assert run_l2_passkey.KEEP_RATIO == 0.10
    assert [config for config, _ in run_l2_passkey.CONFIGURATIONS] == [
        "no_compression",
        "low_l2_keep10",
        "random_keep10",
        "high_l2_keep10",
    ]
    assert len(l2_args.seeds) * len(run_l2_passkey.CONFIGURATIONS) == 12
    assert snapkv_args.context_lengths == (8192,)
    assert snapkv_args.seeds == tuple(range(10))
    assert snapkv_args.observation_window_size == 16
    assert snapkv_args.target_cache_tokens == 1024


def test_l2_keep_ratio_is_not_configurable_from_the_cli() -> None:
    with pytest.raises(SystemExit):
        run_l2_passkey.parse_args(["--keep-ratio", "0.5"])
