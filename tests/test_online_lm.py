from __future__ import annotations

import pytest
import torch

from l2kv.cache_metrics import cache_layer_lengths
from l2kv.l2_compression import compress_cache_to_budget
from l2kv.position_utils import make_cache_position, make_position_ids
from l2kv.snapkv_compression import compress_snapkv_cache
from scripts.run_online_lm import (
    BLOCK_SIZE,
    CHECKPOINT_EVERY,
    MAX_CACHE_TOKENS,
    NUM_TOKENS,
    ONLINE_LM_CONFIGS,
    make_block_causal_mask,
)


class _Layer:
    def __init__(self, length: int) -> None:
        positions = torch.arange(length, dtype=torch.float32)
        self.keys = positions.reshape(1, 1, length, 1)
        self.values = self.keys + 100.0


class _Cache:
    def __init__(self, *lengths: int) -> None:
        self.layers = [_Layer(length) for length in lengths]


def test_online_lm_protocol_constants_and_configurations_are_unchanged() -> None:
    assert (NUM_TOKENS, MAX_CACHE_TOKENS, BLOCK_SIZE, CHECKPOINT_EVERY) == (
        8192,
        2000,
        32,
        512,
    )
    assert [config["config"] for config in ONLINE_LM_CONFIGS] == [
        "no_compression",
        "low_l2",
        "random",
        "high_l2",
        "snapkv",
    ]
    assert [config["skip_layers"] for config in ONLINE_LM_CONFIGS] == [
        (),
        (0, 1),
        (0, 1),
        (0, 1),
        (),
    ]


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
