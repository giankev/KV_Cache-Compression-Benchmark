from __future__ import annotations

from typing import Any

import pytest

from scripts import run_keydiff_passkey, run_l2_passkey, run_snapkv_passkey


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


def test_all_passkey_runners_build_identical_prompts() -> None:
    tokenizer = _Tokenizer()
    l2_example = run_l2_passkey.make_passkey_example(tokenizer, 128, 4)
    keydiff_example = run_keydiff_passkey.make_passkey_example(tokenizer, 128, 4)
    snapkv_example = run_snapkv_passkey.make_passkey_example(
        tokenizer,
        128,
        4,
        observation_window_size=32,
    )

    assert l2_example.prompt_ids == snapkv_example.prompt_ids
    assert l2_example.prompt_ids == keydiff_example.prompt_ids
    assert l2_example.answer_ids == snapkv_example.answer_ids
    assert l2_example.answer_ids == keydiff_example.answer_ids
    assert l2_example.actual_depth == snapkv_example.actual_depth
    assert l2_example.actual_depth == keydiff_example.actual_depth


def test_runner_defaults_and_configuration_sets() -> None:
    l2_args = run_l2_passkey.parse_args([])
    keydiff_args = run_keydiff_passkey.parse_args([])
    snapkv_args = run_snapkv_passkey.parse_args([])

    assert l2_args.context_lengths == (8192,)
    assert l2_args.seeds == (0, 1, 2)
    assert l2_args.keep_ratio == 0.10
    assert run_l2_passkey.KEEP_RATIO == 0.10
    assert l2_args.skip_layers == (0, 1)
    assert [config for config, _ in run_l2_passkey.CONFIGURATIONS] == [
        "no_compression",
        "low_l2",
        "random",
        "high_l2",
    ]
    assert len(l2_args.seeds) * len(run_l2_passkey.CONFIGURATIONS) == 12
    assert keydiff_args.context_lengths == (8192,)
    assert keydiff_args.seeds == (0, 1, 2)
    assert keydiff_args.keep_ratio == 0.10
    assert keydiff_args.skip_layers == (0, 1)
    assert [config for config, _ in run_keydiff_passkey.CONFIGURATIONS] == [
        "no_compression",
        "keydiff",
    ]
    assert snapkv_args.context_lengths == (8192,)
    assert snapkv_args.seeds == tuple(range(10))
    assert snapkv_args.observation_window_size == 32
    assert snapkv_args.target_cache_tokens == 1024
    assert run_snapkv_passkey.SKIP_LAYERS == ()
    assert not hasattr(snapkv_args, "skip_layers")


def test_l2_keep_ratio_is_configurable_from_the_cli() -> None:
    args = run_l2_passkey.parse_args(["--keep-ratio", "0.5"])

    assert args.keep_ratio == 0.5


@pytest.mark.parametrize("keep_ratio", ["0", "-0.1", "1.1"])
def test_l2_keep_ratio_must_be_in_range(keep_ratio: str) -> None:
    with pytest.raises(ValueError, match="0 < keep_ratio <= 1"):
        run_l2_passkey.parse_args(["--keep-ratio", keep_ratio])


def test_l2_runner_uses_requested_ratio_for_compressed_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    calls: list[dict[str, Any]] = []
    args = run_l2_passkey.parse_args(
        [
            "--context-lengths",
            "128",
            "--seeds",
            "7",
            "--keep-ratio",
            "0.25",
        ]
    )
    monkeypatch.setattr(
        run_l2_passkey,
        "make_passkey_example",
        lambda *_: object(),
    )

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"correct": True}

    monkeypatch.setattr(
        run_l2_passkey,
        "evaluate_plain_or_l2",
        fake_evaluate,
    )
    monkeypatch.setattr(
        run_l2_passkey,
        "checkpoint_raw",
        lambda rows, _: rows,
    )
    monkeypatch.setattr(run_l2_passkey, "print_result", lambda _: None)

    run_l2_passkey.run_benchmark(object(), object(), args, tmp_path / "raw.csv")

    assert [call["config"] for call in calls] == [
        "no_compression",
        "low_l2",
        "random",
        "high_l2",
    ]
    assert [call["keep_ratio"] for call in calls] == [1.0, 0.25, 0.25, 0.25]
    assert all(call["skip_layers"] == (0, 1) for call in calls)


def test_l2_baseline_failure_skips_compressed_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    calls: list[str] = []
    args = run_l2_passkey.parse_args(
        ["--context-lengths", "128", "--seeds", "7"]
    )
    monkeypatch.setattr(
        run_l2_passkey,
        "make_passkey_example",
        lambda *_: object(),
    )

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["config"])
        return {"correct": False}

    monkeypatch.setattr(
        run_l2_passkey,
        "evaluate_plain_or_l2",
        fake_evaluate,
    )
    monkeypatch.setattr(
        run_l2_passkey,
        "checkpoint_raw",
        lambda rows, _: rows,
    )
    monkeypatch.setattr(run_l2_passkey, "print_result", lambda _: None)

    rows = run_l2_passkey.run_benchmark(
        object(),
        object(),
        args,
        tmp_path / "raw.csv",
    )

    assert calls == ["no_compression"]
    assert len(rows) == 1


def test_snapkv_runner_compresses_every_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []
    args = run_snapkv_passkey.parse_args(
        ["--context-lengths", "128", "--seeds", "7"]
    )
    monkeypatch.setattr(
        run_snapkv_passkey,
        "make_passkey_example",
        lambda **_: object(),
    )

    def fake_baseline(**kwargs: Any) -> dict[str, Any]:
        calls.append(("no_compression", kwargs["skip_layers"]))
        return {"correct": True}

    def fake_snapkv(**kwargs: Any) -> dict[str, Any]:
        calls.append(("snapkv", kwargs["skip_layers"]))
        return {"correct": True}

    monkeypatch.setattr(
        run_snapkv_passkey,
        "evaluate_plain_or_l2",
        fake_baseline,
    )
    monkeypatch.setattr(run_snapkv_passkey, "evaluate_snapkv", fake_snapkv)
    monkeypatch.setattr(
        run_snapkv_passkey,
        "checkpoint_raw",
        lambda rows, _: rows,
    )
    monkeypatch.setattr(run_snapkv_passkey, "print_result", lambda _: None)

    run_snapkv_passkey.run_benchmark(
        object(),
        object(),
        args,
        tmp_path / "raw.csv",
    )

    assert calls == [("no_compression", ()), ("snapkv", ())]


@pytest.mark.parametrize(
    ("baseline_correct", "expected_configs"),
    [
        (True, ["no_compression", "keydiff"]),
        (False, ["no_compression"]),
    ],
)
def test_keydiff_runner_reuses_one_example_and_honors_baseline_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    baseline_correct: bool,
    expected_configs: list[str],
) -> None:
    example = object()
    calls: list[tuple[str, Any]] = []
    args = run_keydiff_passkey.parse_args(
        ["--context-lengths", "128", "--seeds", "7"]
    )
    monkeypatch.setattr(
        run_keydiff_passkey,
        "make_passkey_example",
        lambda *_: example,
    )

    def fake_baseline(**kwargs: Any) -> dict[str, Any]:
        calls.append(("no_compression", kwargs["example"]))
        return {"correct": baseline_correct}

    def fake_keydiff(**kwargs: Any) -> dict[str, Any]:
        calls.append(("keydiff", kwargs["example"]))
        return {"correct": True}

    monkeypatch.setattr(
        run_keydiff_passkey,
        "evaluate_plain_or_l2",
        fake_baseline,
    )
    monkeypatch.setattr(run_keydiff_passkey, "evaluate_keydiff", fake_keydiff)
    monkeypatch.setattr(
        run_keydiff_passkey,
        "checkpoint_raw",
        lambda rows, _: rows,
    )
    monkeypatch.setattr(run_keydiff_passkey, "print_result", lambda _: None)

    run_keydiff_passkey.run_benchmark(
        object(),
        object(),
        args,
        tmp_path / "raw.csv",
    )

    assert [config for config, _ in calls] == expected_configs
    assert all(call_example is example for _, call_example in calls)
