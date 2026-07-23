from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch

from l2kv.cache_metrics import cache_layer_lengths
from l2kv.retrieval_eval import (
    assert_cache_capacity,
    exact_token_match,
    generate_exact_answer,
    summarize_results,
    target_capacity,
)


class _Layer:
    def __init__(self, length: int) -> None:
        self.keys = torch.zeros(1, 1, length, 2)
        self.values = torch.ones(1, 1, length, 2)


class _Cache:
    def __init__(self, *lengths: int) -> None:
        self.layers = [_Layer(length) for length in lengths]


class _Tokenizer:
    def decode(self, token_ids: list[int], **_: Any) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.positions: list[int] = []

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        cache = kwargs["past_key_values"]
        self.positions.extend(kwargs["position_ids"].flatten().tolist())
        for layer in cache.layers:
            layer.keys = torch.cat(
                (layer.keys, torch.zeros_like(layer.keys[:, :, :1, :])),
                dim=2,
            )
            layer.values = torch.cat(
                (layer.values, torch.zeros_like(layer.values[:, :, :1, :])),
                dim=2,
            )
        logits = torch.tensor([[[0.0, 0.0, 1.0]]])
        return SimpleNamespace(past_key_values=cache, logits=logits)


def test_exact_generation_matches_ids_and_uses_logical_positions() -> None:
    model = _Model()
    cache = _Cache(4, 4)

    generated, prediction, correct, final_cache = generate_exact_answer(
        model=model,
        tokenizer=_Tokenizer(),
        cache=cache,
        last_logits=torch.tensor([[0.0, 1.0, 0.0]]),
        logical_position=8192,
        answer_ids=(1, 2, 2),
    )

    assert generated == (1, 2, 2)
    assert prediction == "1 2 2"
    assert correct
    assert model.positions == [8192, 8193]
    assert cache_layer_lengths(final_cache) == [6, 6]
    assert exact_token_match((1, 2), (1, 2))
    assert not exact_token_match((1, 2), (1, 3))


def test_summary_accuracy_uses_only_rows_that_exist() -> None:
    raw = pd.DataFrame(
        {
            "config": ["low_l2", "low_l2", "snapkv"],
            "context_length": [8192, 8192, 8192],
            "correct": [True, False, True],
            "memory_saved_percent": [20.0, 22.0, 80.0],
            "elapsed_seconds": [1.0, 3.0, 2.0],
        }
    )

    summary = summarize_results(raw).set_index("config")

    assert summary.loc["low_l2", "num_examples"] == 2
    assert summary.loc["low_l2", "accuracy"] == pytest.approx(0.5)
    assert summary.loc["low_l2", "mean_memory_saved_percent"] == pytest.approx(
        21.0
    )
    assert summary.loc["snapkv", "num_examples"] == 1


def test_l2_and_snapkv_capacity_contracts() -> None:
    assert target_capacity(8192, 0.10) == 819
    l2_cache = _Cache(8192, 8192, 819)
    snapkv_cache = _Cache(8192, 8192, 1024)

    assert_cache_capacity(l2_cache, 8192, 819, (0, 1), compressed=True)
    assert_cache_capacity(snapkv_cache, 8192, 1024, (0, 1), compressed=True)
