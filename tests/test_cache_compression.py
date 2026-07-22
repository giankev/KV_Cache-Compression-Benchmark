from __future__ import annotations

from typing import Any

import pytest
import torch

from l2kv.cache_compression import compress_cache, compress_cache_to_budget


class FakeLayer:
    def __init__(self, keys: Any, values: Any) -> None:
        self.keys = keys
        self.values = values


class FakeCache:
    def __init__(self, *layers: FakeLayer) -> None:
        self.layers = list(layers)


def _make_layer(norms: list[float]) -> FakeLayer:
    keys = torch.tensor(norms, dtype=torch.float32).reshape(1, 1, -1, 1)
    values = (100 + torch.arange(len(norms), dtype=torch.float32)).reshape(
        1, 1, -1, 1
    )
    return FakeLayer(keys, values)


def _assert_selected(
    layer: FakeLayer,
    original_keys: torch.Tensor,
    original_values: torch.Tensor,
    expected_indices: list[int],
) -> None:
    indices = torch.tensor(expected_indices, dtype=torch.long)
    assert torch.equal(layer.keys, original_keys.index_select(2, indices))
    assert torch.equal(layer.values, original_values.index_select(2, indices))


def test_low_l2_keeps_expected_indices_and_applies_them_to_values() -> None:
    layer = _make_layer([5.0, 1.0, 4.0, 2.0, 3.0])
    original_keys = layer.keys.clone()
    original_values = layer.values.clone()

    compress_cache(
        FakeCache(layer),
        keep_ratio=0.4,
        prune_after=0,
        strategy="low_l2",
    )

    _assert_selected(layer, original_keys, original_values, [1, 3])


def test_high_l2_keeps_expected_indices() -> None:
    layer = _make_layer([5.0, 1.0, 4.0, 2.0, 3.0])
    original_keys = layer.keys.clone()
    original_values = layer.values.clone()

    compress_cache(
        FakeCache(layer),
        keep_ratio=0.4,
        prune_after=0,
        strategy="high_l2",
    )

    _assert_selected(layer, original_keys, original_values, [0, 2])


def test_retained_tokens_are_restored_to_temporal_order() -> None:
    layer = _make_layer([1.0, 7.0, 2.0, 9.0, 8.0, 3.0])

    compress_cache_to_budget(
        FakeCache(layer), max_cache_tokens=3, strategy="high_l2"
    )

    retained_positions = (layer.values.flatten() - 100).to(torch.long)
    assert retained_positions.tolist() == [1, 3, 4]
    assert torch.all(retained_positions[1:] > retained_positions[:-1])


def test_skip_layer_is_left_exactly_unchanged() -> None:
    skipped = _make_layer([5.0, 1.0, 4.0, 2.0, 3.0])
    active = _make_layer([5.0, 1.0, 4.0, 2.0, 3.0])
    skipped_keys = skipped.keys
    skipped_values = skipped.values
    keys_snapshot = skipped.keys.clone()
    values_snapshot = skipped.values.clone()

    cache = FakeCache(skipped, active)
    result = compress_cache(
        cache,
        keep_ratio=0.5,
        prune_after=0,
        strategy="low_l2",
        skip_layers=(0,),
    )

    assert result is cache
    assert skipped.keys is skipped_keys
    assert skipped.values is skipped_values
    assert torch.equal(skipped.keys, keys_snapshot)
    assert torch.equal(skipped.values, values_snapshot)
    assert active.keys.shape[2] == 3


def test_random_strategy_is_repeatable_with_a_seed() -> None:
    cache_a = FakeCache(_make_layer([float(index + 1) for index in range(32)]))
    cache_b = FakeCache(_make_layer([float(index + 1) for index in range(32)]))
    global_rng_state = torch.random.get_rng_state()

    compress_cache(
        cache_a,
        keep_ratio=0.25,
        prune_after=0,
        strategy="random",
        seed=17,
    )
    compress_cache(
        cache_b,
        keep_ratio=0.25,
        prune_after=0,
        strategy="random",
        seed=17,
    )

    assert torch.equal(cache_a.layers[0].keys, cache_b.layers[0].keys)
    assert torch.equal(cache_a.layers[0].values, cache_b.layers[0].values)
    assert torch.equal(torch.random.get_rng_state(), global_rng_state)


def test_keep_ratio_one_is_an_exact_no_op() -> None:
    layer = _make_layer([3.0, 1.0, 2.0])
    keys = layer.keys
    values = layer.values
    keys_snapshot = keys.clone()
    values_snapshot = values.clone()
    cache = FakeCache(layer)

    result = compress_cache(cache, keep_ratio=1.0, prune_after=0)

    assert result is cache
    assert layer.keys is keys
    assert layer.values is values
    assert torch.equal(layer.keys, keys_snapshot)
    assert torch.equal(layer.values, values_snapshot)


@pytest.mark.parametrize(
    "keep_ratio",
    [0.0, -0.1, 1.01, float("nan"), float("inf"), True, "0.5"],
)
def test_invalid_keep_ratio_raises_value_error(keep_ratio: Any) -> None:
    with pytest.raises(ValueError):
        compress_cache(FakeCache(_make_layer([1.0, 2.0])), keep_ratio=keep_ratio)


@pytest.mark.parametrize("prune_after", [-1, 1.5, True])
def test_invalid_prune_after_raises_value_error(prune_after: Any) -> None:
    with pytest.raises(ValueError):
        compress_cache(
            FakeCache(_make_layer([1.0, 2.0])), prune_after=prune_after
        )


@pytest.mark.parametrize("max_cache_tokens", [0, -1, 1.5, True])
def test_invalid_max_cache_tokens_raises_value_error(
    max_cache_tokens: Any,
) -> None:
    with pytest.raises(ValueError):
        compress_cache_to_budget(
            FakeCache(_make_layer([1.0, 2.0])),
            max_cache_tokens=max_cache_tokens,
        )


@pytest.mark.parametrize("entry_point", ["ratio", "budget"])
def test_unknown_strategy_raises_value_error(entry_point: str) -> None:
    cache = FakeCache(_make_layer([1.0, 2.0]))

    with pytest.raises(ValueError):
        if entry_point == "ratio":
            compress_cache(cache, strategy="unknown")
        else:
            compress_cache_to_budget(
                cache, max_cache_tokens=1, strategy="unknown"
            )


def _invalid_layer(case: str) -> FakeLayer:
    if case == "not_tensors":
        return FakeLayer([[1.0]], [[2.0]])
    if case == "not_four_dimensional":
        return FakeLayer(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
    if case == "different_shapes":
        return FakeLayer(torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 4, 2))
    raise AssertionError(f"Unknown test case: {case}")


@pytest.mark.parametrize("entry_point", ["ratio", "budget"])
@pytest.mark.parametrize(
    "case", ["not_tensors", "not_four_dimensional", "different_shapes"]
)
def test_invalid_key_value_shapes_raise_value_error(
    entry_point: str, case: str
) -> None:
    cache = FakeCache(_invalid_layer(case))

    with pytest.raises(ValueError):
        if entry_point == "ratio":
            compress_cache(cache, keep_ratio=0.5, prune_after=0)
        else:
            compress_cache_to_budget(cache, max_cache_tokens=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"seed": 0, "generator": torch.Generator()}, id="seed-and-generator"
        ),
        pytest.param({"seed": 1.5}, id="non-integer-seed"),
        pytest.param({"seed": True}, id="boolean-seed"),
        pytest.param({"generator": object()}, id="invalid-generator"),
    ],
)
def test_invalid_random_source_raises_value_error(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        compress_cache(
            FakeCache(_make_layer([1.0, 2.0])),
            keep_ratio=0.5,
            prune_after=0,
            strategy="random",
            **kwargs,
        )
