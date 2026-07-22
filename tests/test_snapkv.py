from __future__ import annotations

from typing import Any

import pytest
import torch

from l2kv.snapkv import (
    aggregate_gqa_attention,
    compress_snapkv_cache,
    compute_target_capacity,
    pool_attention_scores,
    rewrite_kv_cache,
    select_topk_indices,
    validate_snapkv_capacity,
)


class FakeLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class FakeCache:
    def __init__(self, *layers: FakeLayer) -> None:
        self.layers = list(layers)


def _make_layer(sequence: int, heads: int = 1) -> FakeLayer:
    positions = torch.arange(sequence, dtype=torch.float32)
    keys = positions.reshape(1, 1, sequence, 1).expand(
        1,
        heads,
        sequence,
        1,
    ).clone()
    values = keys + 100.0
    return FakeLayer(keys, values)


def test_gqa_groups_contiguous_query_heads_and_reduces_observation() -> None:
    attention = torch.empty((1, 4, 2, 3), dtype=torch.float32)
    attention[:, 0] = 1.0
    attention[:, 1] = 3.0
    attention[:, 2] = 10.0
    attention[:, 3] = 14.0

    summed = aggregate_gqa_attention(
        attention,
        num_key_value_heads=2,
        reduction="sum",
    )
    averaged = aggregate_gqa_attention(
        attention,
        num_key_value_heads=2,
        reduction="mean",
    )

    assert summed.shape == (1, 2, 3)
    assert torch.equal(summed[0, 0], torch.full((3,), 4.0))
    assert torch.equal(summed[0, 1], torch.full((3,), 24.0))
    assert torch.equal(averaged[0, 0], torch.full((3,), 2.0))
    assert torch.equal(averaged[0, 1], torch.full((3,), 12.0))


def test_max_pooling_matches_known_temporal_neighborhoods() -> None:
    scores = torch.tensor([[[0.0, 1.0, 0.0, 3.0, 0.0]]])

    pooled = pool_attention_scores(scores, kernel_size=3, mode="max")

    assert torch.equal(pooled, torch.tensor([[[1.0, 1.0, 3.0, 3.0, 3.0]]]))


def test_topk_is_correct_deterministic_and_chronological() -> None:
    scores = torch.tensor([[[0.1, 0.9, 0.3, 0.8, 0.9]]])

    indices = select_topk_indices(scores, tokens_to_keep=3)

    # Stable tie-breaking chooses index 1 before index 4; final output is time order.
    assert indices.tolist() == [[[1, 3, 4]]]


def test_rewrite_uses_same_indices_for_kv_and_preserves_full_observation() -> None:
    layer = _make_layer(sequence=6)
    original_keys = layer.keys.clone()
    original_values = layer.values.clone()
    indices = torch.tensor([[[1, 3]]], dtype=torch.long)

    keys, values = rewrite_kv_cache(
        layer.keys,
        layer.values,
        prefix_indices=indices,
        observation_window_size=2,
    )

    expected_positions = torch.tensor([1, 3, 4, 5])
    assert torch.equal(keys, original_keys.index_select(2, expected_positions))
    assert torch.equal(values, original_values.index_select(2, expected_positions))
    assert torch.equal(keys[..., -2:, :], original_keys[..., -2:, :])
    assert torch.equal(values[..., -2:, :], original_values[..., -2:, :])
    assert torch.equal(values - keys, torch.full_like(keys, 100.0))


def test_compression_hits_total_capacity_and_leaves_skip_layer_untouched() -> None:
    skipped = _make_layer(sequence=8)
    active = _make_layer(sequence=8)
    skipped_keys = skipped.keys
    skipped_values = skipped.values
    active_observation_keys = active.keys[..., -2:, :].clone()
    active_observation_values = active.values[..., -2:, :].clone()
    scores = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.7, 0.3]]])
    cache = FakeCache(skipped, active)

    result = compress_snapkv_cache(
        cache,
        scores_by_layer=(None, scores),
        target_capacity=5,
        observation_window_size=2,
        pooling_kernel_size=1,
        skip_layers=(0,),
    )

    assert result is cache
    assert skipped.keys is skipped_keys
    assert skipped.values is skipped_values
    assert skipped.keys.shape[2] == 8
    assert active.keys.shape == active.values.shape
    assert active.keys.shape[2] == 5
    assert torch.equal(active.keys[..., -2:, :], active_observation_keys)
    assert torch.equal(active.values[..., -2:, :], active_observation_values)


def test_total_capacity_uses_floor_and_validates_observation_budget() -> None:
    assert compute_target_capacity(prompt_length=7, keep_ratio=0.5) == 3
    assert validate_snapkv_capacity(7, 3, 2) == 1

    with pytest.raises(ValueError, match="at least observation_window_size"):
        validate_snapkv_capacity(7, 1, 2)


def test_keep_ratio_one_is_an_exact_cache_no_op() -> None:
    layer = _make_layer(sequence=6)
    original_keys = layer.keys
    original_values = layer.values
    cache = FakeCache(layer)

    result = compress_snapkv_cache(
        cache,
        scores_by_layer=(None,),
        observation_window_size=2,
        keep_ratio=1.0,
    )

    assert result is cache
    assert layer.keys is original_keys
    assert layer.values is original_values


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: pool_attention_scores(
                torch.ones(1, 1, 4),
                kernel_size=2,
            ),
            "odd",
        ),
        (
            lambda: validate_snapkv_capacity(4, 4, 0),
            "observation_window_size",
        ),
        (
            lambda: validate_snapkv_capacity(4, 4, 5),
            "cannot exceed",
        ),
        (
            lambda: aggregate_gqa_attention(
                torch.ones(1, 3, 2, 4),
                num_key_value_heads=2,
            ),
            "divisible",
        ),
        (
            lambda: aggregate_gqa_attention(
                torch.ones(1, 4, 5),
                num_key_value_heads=2,
            ),
            "shape",
        ),
        (
            lambda: rewrite_kv_cache(
                torch.zeros(1, 1, 4, 2),
                torch.zeros(1, 1, 5, 2),
                torch.tensor([[[0]]]),
                observation_window_size=1,
            ),
            "identical shapes",
        ),
    ],
)
def test_invalid_inputs_raise_clear_value_errors(
    call: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_non_finite_attention_and_scores_are_rejected(
    invalid_value: float,
) -> None:
    attention = torch.ones(1, 4, 2, 3)
    attention[0, 0, 0, 0] = invalid_value
    with pytest.raises(ValueError, match="finite"):
        aggregate_gqa_attention(attention, num_key_value_heads=2)

    scores = torch.ones(1, 1, 4)
    scores[0, 0, 0] = invalid_value
    with pytest.raises(ValueError, match="finite"):
        pool_attention_scores(scores)


def test_compression_rejects_score_shape_incompatible_with_cache() -> None:
    cache = FakeCache(_make_layer(sequence=6, heads=2))

    with pytest.raises(ValueError, match="scores must have shape"):
        compress_snapkv_cache(
            cache,
            scores_by_layer=(torch.ones(1, 1, 4),),
            target_capacity=4,
            observation_window_size=2,
            pooling_kernel_size=1,
        )
