from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from l2kv.cache_metrics import cache_layer_lengths
from l2kv.kv_retrieval_eval import generate_greedy


class _Layer:
    def __init__(self, length: int) -> None:
        self.keys = torch.zeros(1, 1, length, 2)
        self.values = torch.ones(1, 1, length, 2)


class _Cache:
    def __init__(self, *lengths: int) -> None:
        self.layers = [_Layer(length) for length in lengths]


class _Tokenizer:
    eos_token_id = None

    def decode(self, token_ids: list[int], **_: Any) -> str:
        return "12345" if len(token_ids) >= 2 else "pending"


class _OneTokenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=None)
        self.positions: list[list[int]] = []

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        cache = kwargs["past_key_values"]
        self.positions.append(kwargs["position_ids"].flatten().tolist())
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


def test_checked_decode_uses_logical_position_and_grows_every_layer_once() -> None:
    model = _OneTokenModel()
    cache = _Cache(2, 2, 2)
    initial_logits = torch.tensor([[0.0, 1.0, 0.0]])

    generated_text, prediction, final_cache = generate_greedy(
        model=model,
        tokenizer=_Tokenizer(),
        cache=cache,
        last_logits=initial_logits,
        logical_position=8_192,
        max_new_tokens=3,
        validate_cache_growth=True,
    )

    assert generated_text == "12345"
    assert prediction == "12345"
    assert model.positions == [[8_192]]
    assert cache_layer_lengths(final_cache) == [3, 3, 3]
