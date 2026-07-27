from __future__ import annotations

import math

import pytest
import torch

from l2kv.cache_compression import compress_cache_to_budget
from l2kv.cache_metrics import cache_layer_lengths
from l2kv.position_utils import make_cache_position, make_position_ids
from l2kv.snapkv import compress_snapkv_cache
from scripts.run_online_lm import (
    compute_metrics,
    make_block,
    make_block_causal_mask,
    should_compress,
)


class _Layer:
    def __init__(self, length: int) -> None:
        positions = torch.arange(length, dtype=torch.float32)
        self.keys = positions.reshape(1, 1, length, 1)
        self.values = self.keys + 100.0


class _Cache:
    def __init__(self, *lengths: int) -> None:
        self.layers = [_Layer(length) for length in lengths]


def test_block_labels_are_shifted_by_exactly_one_token() -> None:
    token_ids = torch.arange(12).unsqueeze(0)

    input_ids, labels = make_block(token_ids, start=3, block_size=4)

    assert input_ids.tolist() == [[3, 4, 5, 6]]
    assert labels.tolist() == [[4, 5, 6, 7]]
    assert torch.equal(labels[:, :-1], input_ids[:, 1:])


def test_metrics_use_cumulative_nll_and_prediction_counts() -> None:
    first = compute_metrics(
        total_nll=2.0,
        correct_next_tokens=1,
        num_predictions=2,
    )
    cumulative = compute_metrics(
        total_nll=2.0 + 6.0,
        correct_next_tokens=1 + 2,
        num_predictions=2 + 3,
    )

    assert first == pytest.approx((1.0, math.e, 0.5))
    assert cumulative == pytest.approx((1.6, math.exp(1.6), 0.6))


@pytest.mark.parametrize(
    ("previous_cache_length", "block_length", "expected"),
    [
        (0, 32, False),
        (1968, 32, False),
        (2000, 32, True),
    ],
)
def test_cache_is_compressed_only_when_the_next_block_exceeds_budget(
    previous_cache_length: int,
    block_length: int,
    expected: bool,
) -> None:
    assert (
        should_compress(
            previous_cache_length,
            block_length,
            max_cache_tokens=2000,
        )
        is expected
    )


@pytest.mark.parametrize("previous_cache_length", [3, 5])
def test_block_causal_mask_uses_the_physical_prefix_length(
    previous_cache_length: int,
) -> None:
    mask = make_block_causal_mask(
        previous_cache_length,
        2,
        dtype=torch.float32,
        device="cpu",
    )[0, 0]

    assert mask.shape == (2, previous_cache_length + 2)
    assert torch.equal(
        mask[:, :previous_cache_length],
        torch.zeros((2, previous_cache_length)),
    )
    assert mask[0, previous_cache_length].item() == 0.0
    assert mask[0, previous_cache_length + 1].item() == torch.finfo(
        torch.float32
    ).min
    assert torch.equal(
        mask[1, previous_cache_length:],
        torch.zeros(2),
    )


def test_l2_online_budget_keeps_layers_zero_and_one_uncompressed() -> None:
    cache = _Cache(2032, 2032, 2032, 2032)

    compress_cache_to_budget(
        cache,
        max_cache_tokens=2000,
        strategy="low_l2",
        skip_layers=(0, 1),
    )

    assert cache_layer_lengths(cache) == [2032, 2032, 2000, 2000]
    for layer in cache.layers:
        assert layer.keys.shape == layer.values.shape
        assert torch.equal(
            layer.values - layer.keys,
            torch.full_like(layer.keys, 100),
        )


def test_snapkv_online_budget_preserves_the_complete_observation_window() -> None:
    cache = _Cache(2032, 2032)
    original_observation = [
        (
            layer.keys[..., -32:, :].clone(),
            layer.values[..., -32:, :].clone(),
        )
        for layer in cache.layers
    ]
    scores_by_layer = tuple(
        torch.arange(2000, dtype=torch.float32).reshape(1, 1, 2000)
        for _ in cache.layers
    )

    compress_snapkv_cache(
        cache,
        scores_by_layer=scores_by_layer,
        target_capacity=2000,
        observation_window_size=32,
        pooling_kernel_size=5,
        pooling_mode="max",
        skip_layers=(),
    )

    assert cache_layer_lengths(cache) == [2000, 2000]
    for layer, (observation_keys, observation_values) in zip(
        cache.layers,
        original_observation,
        strict=True,
    ):
        assert torch.equal(layer.keys[..., -32:, :], observation_keys)
        assert torch.equal(layer.values[..., -32:, :], observation_values)
        assert torch.equal(
            layer.values - layer.keys,
            torch.full_like(layer.keys, 100),
        )


def test_logical_positions_do_not_follow_compressed_cache_length() -> None:
    physical_cache_length = 2000
    logical_position = 4064

    position_ids = make_position_ids(
        start_position=logical_position,
        length=32,
        device="cpu",
    )

    assert physical_cache_length != logical_position
    assert position_ids[0, 0].item() == logical_position
    assert position_ids[0, -1].item() == logical_position + 31
    assert make_cache_position(position_ids).tolist() == list(range(4064, 4096))
