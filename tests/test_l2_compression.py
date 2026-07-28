from __future__ import annotations

from typing import Any

import pytest
import torch

from l2kv.l2_compression import compress_cache_to_budget


class FakeLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class FakeCache:
    def __init__(self, *layers: FakeLayer) -> None:
        self.layers = list(layers)


def _make_layer(key_values: list[float]) -> FakeLayer:
    keys = torch.tensor(key_values, dtype=torch.float32).reshape(1, 1, -1, 1)
    values = (100 + torch.arange(len(key_values), dtype=torch.float32)).reshape(
        1,
        1,
        -1,
        1,
    )
    return FakeLayer(keys, values)


@pytest.mark.parametrize(
    ("strategy", "expected_indices"),
    [
        ("low_l2", [1, 3]),
        ("high_l2", [0, 2]),
    ],
)
def test_l2_selection_uses_one_chronological_index_set_for_keys_and_values(
    strategy: str,
    expected_indices: list[int],
) -> None:
    layer = _make_layer([5.0, 1.0, 4.0, 2.0, 3.0])
    original_keys = layer.keys.clone()
    original_values = layer.values.clone()

    compress_cache_to_budget(
        FakeCache(layer),
        max_cache_tokens=2,
        strategy=strategy,
    )

    indices = torch.tensor(expected_indices)
    assert torch.equal(layer.keys, original_keys.index_select(2, indices))
    assert torch.equal(layer.values, original_values.index_select(2, indices))
    assert expected_indices == sorted(expected_indices)


def test_high_l2_restores_temporal_order_after_topk() -> None:
    layer = _make_layer([1.0, 7.0, 2.0, 9.0, 8.0, 3.0])

    compress_cache_to_budget(
        FakeCache(layer),
        max_cache_tokens=3,
        strategy="high_l2",
    )

    retained_positions = (layer.values.flatten() - 100).to(torch.long)
    assert retained_positions.tolist() == [1, 3, 4]


def test_fixed_budget_leaves_skipped_layers_unchanged() -> None:
    skipped = _make_layer([5.0, 1.0, 4.0, 2.0, 3.0])
    active = _make_layer([5.0, 1.0, 4.0, 2.0, 3.0])
    skipped_keys = skipped.keys
    skipped_values = skipped.values
    skipped_keys_snapshot = skipped.keys.clone()
    skipped_values_snapshot = skipped.values.clone()
    cache = FakeCache(skipped, active)

    result = compress_cache_to_budget(
        cache,
        max_cache_tokens=2,
        strategy="low_l2",
        skip_layers=(0,),
    )

    assert result is cache
    assert skipped.keys is skipped_keys
    assert skipped.values is skipped_values
    assert torch.equal(skipped.keys, skipped_keys_snapshot)
    assert torch.equal(skipped.values, skipped_values_snapshot)
    assert skipped.keys.shape[2] == 5
    assert active.keys.shape == active.values.shape
    assert active.keys.shape[2] == 2


def test_random_selection_is_deterministic_and_does_not_use_global_rng() -> None:
    values = [float(index) for index in range(32)]
    cache_a = FakeCache(_make_layer(values))
    cache_b = FakeCache(_make_layer(values))
    cache_c = FakeCache(_make_layer(values))
    global_rng_state = torch.random.get_rng_state()

    for cache in (cache_a, cache_b):
        compress_cache_to_budget(
            cache,
            max_cache_tokens=8,
            strategy="random",
            seed=17,
        )
    compress_cache_to_budget(
        cache_c,
        max_cache_tokens=8,
        strategy="random",
        seed=18,
    )

    layer_a = cache_a.layers[0]
    layer_b = cache_b.layers[0]
    assert torch.equal(layer_a.keys, layer_b.keys)
    assert torch.equal(layer_a.values, layer_b.values)
    assert not torch.equal(layer_a.keys, cache_c.layers[0].keys)
    assert torch.equal(
        layer_a.values - layer_a.keys,
        torch.full_like(layer_a.keys, 100),
    )
    retained_positions = layer_a.keys.flatten()
    assert torch.all(retained_positions[1:] > retained_positions[:-1])
    assert torch.equal(torch.random.get_rng_state(), global_rng_state)


def test_cache_at_or_below_budget_is_an_exact_no_op() -> None:
    layer = _make_layer([3.0, 1.0, 2.0])
    original_keys = layer.keys
    original_values = layer.values

    compress_cache_to_budget(FakeCache(layer), max_cache_tokens=3)

    assert layer.keys is original_keys
    assert layer.values is original_values
