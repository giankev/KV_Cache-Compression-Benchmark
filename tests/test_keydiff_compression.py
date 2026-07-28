from __future__ import annotations

from typing import Any

import pytest
import torch

from l2kv.keydiff_compression import compress_keydiff_cache_to_budget


class FakeLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class FakeCache:
    def __init__(self, *layers: FakeLayer) -> None:
        self.layers = list(layers)


def _make_layer(keys: torch.Tensor) -> FakeLayer:
    batch_size, num_heads, sequence_length, head_dim = keys.shape
    positions = torch.arange(sequence_length, device=keys.device).reshape(
        1,
        1,
        sequence_length,
        1,
    )
    values = positions.expand(
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
    ).to(keys.dtype)
    return FakeLayer(keys.clone(), values.clone())


def test_redundant_anchor_aligned_keys_are_evicted() -> None:
    keys = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]]
    )
    layer = _make_layer(keys)

    compress_keydiff_cache_to_budget(FakeCache(layer), max_cache_tokens=2)

    assert torch.equal(layer.keys, keys[:, :, [2, 3], :])
    assert layer.values[0, 0, :, 0].tolist() == [2.0, 3.0]


def test_keys_and_values_use_the_same_chronological_indices() -> None:
    keys = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, -1.0],
                ]
            ]
        ]
    )
    layer = _make_layer(keys)
    original_keys = layer.keys.clone()
    original_values = layer.values.clone()

    compress_keydiff_cache_to_budget(FakeCache(layer), max_cache_tokens=2)

    expected_indices = torch.tensor([1, 4])
    assert torch.equal(layer.keys, original_keys.index_select(2, expected_indices))
    assert torch.equal(layer.values, original_values.index_select(2, expected_indices))
    assert layer.values[0, 0, :, 0].tolist() == [1.0, 4.0]


def test_selection_is_independent_across_kv_heads() -> None:
    keys = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, -1.0],
                    [1.0, 0.0],
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
            ]
        ]
    )
    layer = _make_layer(keys)

    compress_keydiff_cache_to_budget(FakeCache(layer), max_cache_tokens=2)

    retained_positions = layer.values[..., 0].to(torch.long)
    assert retained_positions[0, 0].tolist() == [1, 3]
    assert retained_positions[0, 1].tolist() == [0, 2]


def test_selection_is_independent_across_batch_elements() -> None:
    keys = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, -1.0],
                    [1.0, 0.0],
                ]
            ],
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            ],
        ]
    )
    layer = _make_layer(keys)

    compress_keydiff_cache_to_budget(FakeCache(layer), max_cache_tokens=2)

    retained_positions = layer.values[..., 0].to(torch.long)
    assert retained_positions[0, 0].tolist() == [1, 3]
    assert retained_positions[1, 0].tolist() == [0, 2]


def test_skip_layers_are_exact_no_ops() -> None:
    keys = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]]]
    )
    skipped = _make_layer(keys)
    active = _make_layer(keys)
    skipped_keys = skipped.keys
    skipped_values = skipped.values
    cache = FakeCache(skipped, active)

    result = compress_keydiff_cache_to_budget(
        cache,
        max_cache_tokens=2,
        skip_layers=(0,),
    )

    assert result is cache
    assert skipped.keys is skipped_keys
    assert skipped.values is skipped_values
    assert skipped.keys.shape[2] == 4
    assert active.keys.shape[2] == 2
    assert active.values.shape[2] == 2


@pytest.mark.parametrize(("sequence_length", "budget"), [(2, 2), (2, 3)])
def test_cache_at_or_below_budget_is_an_exact_no_op(
    sequence_length: int,
    budget: int,
) -> None:
    keys = torch.arange(sequence_length * 2, dtype=torch.float32).reshape(
        1,
        1,
        sequence_length,
        2,
    )
    layer = _make_layer(keys)
    original_keys = layer.keys
    original_values = layer.values

    compress_keydiff_cache_to_budget(FakeCache(layer), max_cache_tokens=budget)

    assert layer.keys is original_keys
    assert layer.values is original_values


def test_compression_preserves_shapes_dtypes_devices_and_contiguity() -> None:
    keys = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            ],
            [
                [[1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            ],
        ],
        dtype=torch.float16,
    )
    layer = _make_layer(keys)

    compress_keydiff_cache_to_budget(FakeCache(layer), max_cache_tokens=2)

    assert layer.keys.shape == (2, 2, 2, 2)
    assert layer.values.shape == (2, 2, 2, 2)
    assert layer.keys.dtype == torch.float16
    assert layer.values.dtype == torch.float16
    assert layer.keys.device == keys.device
    assert layer.values.device == keys.device
    assert layer.keys.is_contiguous()
    assert layer.values.is_contiguous()


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_invalid_budgets_raise_clear_value_errors(budget: Any) -> None:
    layer = _make_layer(torch.ones((1, 1, 2, 2)))

    with pytest.raises(ValueError, match="positive integer"):
        compress_keydiff_cache_to_budget(FakeCache(layer), budget)


@pytest.mark.parametrize("skip_layers", [(-1,), (1,), ("0",), (True,)])
def test_invalid_skip_layers_raise_clear_value_errors(
    skip_layers: tuple[Any, ...],
) -> None:
    layer = _make_layer(torch.ones((1, 1, 2, 2)))

    with pytest.raises(ValueError, match="skip"):
        compress_keydiff_cache_to_budget(
            FakeCache(layer),
            max_cache_tokens=1,
            skip_layers=skip_layers,
        )


def test_invalid_cache_and_layer_shapes_raise_clear_value_errors() -> None:
    with pytest.raises(ValueError, match="layers attribute"):
        compress_keydiff_cache_to_budget(object(), max_cache_tokens=1)

    bad_layer = FakeLayer(torch.ones((1, 2, 3)), torch.ones((1, 2, 3)))
    with pytest.raises(ValueError, match="must be 4-D"):
        compress_keydiff_cache_to_budget(FakeCache(bad_layer), max_cache_tokens=1)

    mismatched_layer = FakeLayer(
        torch.ones((1, 1, 3, 2)),
        torch.ones((1, 1, 2, 2)),
    )
    with pytest.raises(ValueError, match="shapes must match"):
        compress_keydiff_cache_to_budget(
            FakeCache(mismatched_layer),
            max_cache_tokens=1,
        )
