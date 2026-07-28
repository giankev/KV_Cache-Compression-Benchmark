from __future__ import annotations

import pytest
import torch

from l2kv.snapkv_compression import (
    compress_snapkv_cache,
    scores_from_block_attentions,
)


class FakeLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class FakeCache:
    def __init__(self, *layers: FakeLayer) -> None:
        self.layers = list(layers)


def _make_layer(sequence_length: int, num_heads: int = 1) -> FakeLayer:
    positions = torch.arange(sequence_length, dtype=torch.float32)
    keys = positions.reshape(1, 1, sequence_length, 1).expand(
        1,
        num_heads,
        sequence_length,
        1,
    ).clone()
    return FakeLayer(keys, keys + 100.0)


def test_block_scoring_aggregates_gqa_heads_and_excludes_observation_keys() -> None:
    attention = torch.empty((1, 4, 2, 6), dtype=torch.float32)
    attention[:, 0, :, :4] = 1.0
    attention[:, 1, :, :4] = 3.0
    attention[:, 2, :, :4] = 10.0
    attention[:, 3, :, :4] = 14.0
    attention[..., -2:] = 1_000.0

    summed = scores_from_block_attentions(
        (attention, attention),
        num_key_value_heads=2,
        observation_window_size=2,
        skip_layers=(0,),
        reduction="sum",
    )
    averaged = scores_from_block_attentions(
        (attention,),
        num_key_value_heads=2,
        observation_window_size=2,
        reduction="mean",
    )

    assert summed[0] is None
    assert summed[1] is not None
    assert summed[1].shape == (1, 2, 4)
    assert torch.equal(summed[1][0, 0], torch.full((4,), 4.0))
    assert torch.equal(summed[1][0, 1], torch.full((4,), 24.0))
    assert averaged[0] is not None
    assert torch.equal(averaged[0][0, 0], torch.full((4,), 2.0))
    assert torch.equal(averaged[0][0, 1], torch.full((4,), 12.0))


def test_compression_pools_scores_and_preserves_chronological_kv_positions() -> None:
    layer = _make_layer(sequence_length=8)
    original_keys = layer.keys.clone()
    original_values = layer.values.clone()
    scores = torch.tensor([[[0.0, 1.0, 0.0, 3.0, 0.0, 2.0]]])

    compress_snapkv_cache(
        FakeCache(layer),
        scores_by_layer=(scores,),
        target_capacity=5,
        observation_window_size=2,
        pooling_kernel_size=3,
        pooling_mode="max",
    )

    # Max pooling produces a plateau at positions 2, 3, and 4. Stable ranking
    # keeps those earliest ties, then the cache is gathered in temporal order.
    expected_positions = torch.tensor([2, 3, 4, 6, 7])
    assert torch.equal(layer.keys, original_keys.index_select(2, expected_positions))
    assert torch.equal(
        layer.values,
        original_values.index_select(2, expected_positions),
    )
    assert torch.equal(layer.values - layer.keys, torch.full_like(layer.keys, 100))


def test_target_capacity_preserves_full_observation_and_skipped_layers() -> None:
    skipped = _make_layer(sequence_length=8)
    active = _make_layer(sequence_length=8)
    skipped_keys = skipped.keys
    skipped_values = skipped.values
    observation_keys = active.keys[..., -2:, :].clone()
    observation_values = active.values[..., -2:, :].clone()
    cache = FakeCache(skipped, active)

    result = compress_snapkv_cache(
        cache,
        scores_by_layer=(None, torch.arange(6).reshape(1, 1, 6)),
        target_capacity=4,
        observation_window_size=2,
        pooling_kernel_size=1,
        skip_layers=(0,),
    )

    assert result is cache
    assert skipped.keys is skipped_keys
    assert skipped.values is skipped_values
    assert skipped.keys.shape[2] == 8
    assert active.keys.shape == active.values.shape
    assert active.keys.shape[2] == 4
    assert torch.equal(active.keys[..., -2:, :], observation_keys)
    assert torch.equal(active.values[..., -2:, :], observation_values)


def test_keep_ratio_uses_floor_for_total_capacity() -> None:
    layer = _make_layer(sequence_length=7)

    compress_snapkv_cache(
        FakeCache(layer),
        scores_by_layer=(torch.arange(5).reshape(1, 1, 5),),
        observation_window_size=2,
        pooling_kernel_size=1,
        keep_ratio=0.5,
    )

    assert layer.keys.shape[2] == 3
    assert layer.values.shape[2] == 3


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires at least two CUDA GPUs",
)
def test_compression_moves_scores_to_each_sharded_layer_device() -> None:
    score_device = torch.device("cuda:0")
    cache_device = torch.device("cuda:1")
    layer = _make_layer(sequence_length=6)
    layer.keys = layer.keys.to(cache_device)
    layer.values = layer.values.to(cache_device)
    original_keys = layer.keys.clone()
    original_values = layer.values.clone()
    scores = torch.tensor(
        [[[0.1, 0.9, 0.2, 0.8]]],
        dtype=torch.float32,
        device=score_device,
    )

    compress_snapkv_cache(
        FakeCache(layer),
        scores_by_layer=(scores,),
        target_capacity=4,
        observation_window_size=2,
        pooling_kernel_size=1,
    )

    expected_positions = torch.tensor([1, 3, 4, 5], device=cache_device)
    assert layer.keys.device == cache_device
    assert layer.values.device == cache_device
    assert torch.equal(layer.keys, original_keys.index_select(2, expected_positions))
    assert torch.equal(
        layer.values,
        original_values.index_select(2, expected_positions),
    )
    assert scores.device == score_device
