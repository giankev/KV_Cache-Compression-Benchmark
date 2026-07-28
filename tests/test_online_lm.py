from __future__ import annotations

import pytest
import torch

from l2kv.cache_metrics import cache_layer_lengths
from l2kv.keydiff_compression import compress_keydiff_cache_to_budget
from l2kv.l2_compression import compress_cache_to_budget
from l2kv.position_utils import make_cache_position, make_position_ids
from l2kv.snapkv_compression import compress_snapkv_cache
from scripts.run_online_lm import (
    BLOCK_SIZE,
    CHECKPOINT_EVERY,
    MAX_CACHE_TOKENS,
    NUM_TOKENS,
    ONLINE_LM_CONFIGS,
    make_online_lm_configs,
    make_block_causal_mask,
    parse_args,
)


class _Layer:
    def __init__(self, length: int) -> None:
        positions = torch.arange(length, dtype=torch.float32)
        self.keys = positions.reshape(1, 1, length, 1)
        self.values = self.keys + 100.0


class _Cache:
    def __init__(self, *lengths: int) -> None:
        self.layers = [_Layer(length) for length in lengths]


def test_online_lm_cli_defaults_to_compressing_every_layer() -> None:
    default_args = parse_args([])

    assert default_args.max_cache_tokens == 2000
    assert default_args.skip_layers == ()
    assert parse_args(["--skip-layers"]).skip_layers == []
    assert tuple(parse_args(["--skip-layers", "0", "1"]).skip_layers) == (0, 1)
    assert (
        parse_args(["--max-cache-tokens", "1000"]).max_cache_tokens == 1000
    )
    combined_args = parse_args(
        ["--max-cache-tokens", "1000", "--skip-layers", "0", "1"]
    )
    assert combined_args.max_cache_tokens == 1000
    assert tuple(combined_args.skip_layers) == (0, 1)


@pytest.mark.parametrize("max_cache_tokens", [0, -1])
def test_online_lm_cli_rejects_invalid_cache_budgets(
    max_cache_tokens: int,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        parse_args(["--max-cache-tokens", str(max_cache_tokens)])


def test_online_lm_protocol_constants_and_default_configurations() -> None:
    configs = make_online_lm_configs(())

    assert (NUM_TOKENS, MAX_CACHE_TOKENS, BLOCK_SIZE, CHECKPOINT_EVERY) == (
        8192,
        2000,
        32,
        512,
    )
    assert configs == ONLINE_LM_CONFIGS
    assert [config["config"] for config in configs] == [
        "no_compression",
        "low_l2",
        "keydiff",
        "random",
        "high_l2",
        "snapkv",
    ]
    assert [config["skip_layers"] for config in configs] == [
        (),
        (),
        (),
        (),
        (),
        (),
    ]


def test_online_lm_explicit_skip_layers_apply_to_every_compressed_method() -> None:
    configs = make_online_lm_configs((0, 1))

    assert [config["skip_layers"] for config in configs] == [
        (),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
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


def test_l2_online_budget_compresses_every_layer_without_skips() -> None:
    cache = _Cache(2032, 2032, 2032, 2032)

    compress_cache_to_budget(
        cache,
        max_cache_tokens=2000,
        strategy="low_l2",
        skip_layers=(),
    )

    assert cache_layer_lengths(cache) == [2000, 2000, 2000, 2000]


def test_l2_online_custom_budget_compresses_every_layer() -> None:
    cache = _Cache(1032, 1032, 1032, 1032)

    compress_cache_to_budget(
        cache,
        max_cache_tokens=1000,
        strategy="low_l2",
        skip_layers=(),
    )

    assert cache_layer_lengths(cache) == [1000, 1000, 1000, 1000]


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


def test_l2_online_custom_budget_composes_with_skip_layers() -> None:
    cache = _Cache(1032, 1032, 1032, 1032)

    compress_cache_to_budget(
        cache,
        max_cache_tokens=1000,
        strategy="low_l2",
        skip_layers=(0, 1),
    )

    assert cache_layer_lengths(cache) == [1032, 1032, 1000, 1000]


def test_keydiff_online_budget_compresses_every_layer_without_skips() -> None:
    cache = _Cache(2032, 2032, 2032, 2032)

    compress_keydiff_cache_to_budget(
        cache,
        max_cache_tokens=2000,
        skip_layers=(),
    )

    assert cache_layer_lengths(cache) == [2000, 2000, 2000, 2000]


def test_keydiff_online_custom_budget_compresses_every_layer() -> None:
    cache = _Cache(1032, 1032, 1032, 1032)

    compress_keydiff_cache_to_budget(
        cache,
        max_cache_tokens=1000,
        skip_layers=(),
    )

    assert cache_layer_lengths(cache) == [1000, 1000, 1000, 1000]


def test_keydiff_online_budget_keeps_layers_zero_and_one_uncompressed() -> None:
    cache = _Cache(2032, 2032, 2032, 2032)

    compress_keydiff_cache_to_budget(
        cache,
        max_cache_tokens=2000,
        skip_layers=(0, 1),
    )

    assert cache_layer_lengths(cache) == [2032, 2032, 2000, 2000]
    for layer in cache.layers:
        assert layer.keys.shape == layer.values.shape
        assert torch.equal(
            layer.values - layer.keys,
            torch.full_like(layer.keys, 100),
        )


def test_keydiff_online_custom_budget_composes_with_skip_layers() -> None:
    cache = _Cache(1032, 1032, 1032, 1032)

    compress_keydiff_cache_to_budget(
        cache,
        max_cache_tokens=1000,
        skip_layers=(0, 1),
    )

    assert cache_layer_lengths(cache) == [1032, 1032, 1000, 1000]


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


def test_snapkv_online_custom_budget_preserves_observation_window() -> None:
    cache = _Cache(1032, 1032)
    original_observation = [
        (
            layer.keys[..., -32:, :].clone(),
            layer.values[..., -32:, :].clone(),
        )
        for layer in cache.layers
    ]
    scores_by_layer = tuple(
        torch.arange(1000, dtype=torch.float32).reshape(1, 1, 1000)
        for _ in cache.layers
    )

    compress_snapkv_cache(
        cache,
        scores_by_layer=scores_by_layer,
        target_capacity=1000,
        observation_window_size=32,
        pooling_kernel_size=5,
        pooling_mode="max",
        skip_layers=(),
    )

    assert cache_layer_lengths(cache) == [1000, 1000]
    for layer, (observation_keys, observation_values) in zip(
        cache.layers,
        original_observation,
        strict=True,
    ):
        assert torch.equal(layer.keys[..., -32:, :], observation_keys)
        assert torch.equal(layer.values[..., -32:, :], observation_values)


def test_snapkv_online_custom_budget_composes_with_skip_layers() -> None:
    cache = _Cache(1032, 1032, 1032, 1032)
    active_scores = torch.arange(1000, dtype=torch.float32).reshape(
        1, 1, 1000
    )

    compress_snapkv_cache(
        cache,
        scores_by_layer=(None, None, active_scores, active_scores),
        target_capacity=1000,
        observation_window_size=32,
        pooling_kernel_size=5,
        pooling_mode="max",
        skip_layers=(0, 1),
    )

    assert cache_layer_lengths(cache) == [1032, 1032, 1000, 1000]


def test_snapkv_online_budget_keeps_skipped_layers_uncompressed() -> None:
    cache = _Cache(2032, 2032, 2032, 2032)
    skipped_tensors = [
        (cache.layers[layer_idx].keys, cache.layers[layer_idx].values)
        for layer_idx in (0, 1)
    ]
    active_observations = [
        (
            cache.layers[layer_idx].keys[..., -32:, :].clone(),
            cache.layers[layer_idx].values[..., -32:, :].clone(),
        )
        for layer_idx in (2, 3)
    ]
    active_scores = torch.arange(2000, dtype=torch.float32).reshape(
        1, 1, 2000
    )

    compress_snapkv_cache(
        cache,
        scores_by_layer=(None, None, active_scores, active_scores),
        target_capacity=2000,
        observation_window_size=32,
        pooling_kernel_size=5,
        pooling_mode="max",
        skip_layers=(0, 1),
    )

    assert cache_layer_lengths(cache) == [2032, 2032, 2000, 2000]
    for layer_idx, (original_keys, original_values) in zip(
        (0, 1), skipped_tensors, strict=True
    ):
        assert cache.layers[layer_idx].keys is original_keys
        assert cache.layers[layer_idx].values is original_values
    for layer_idx, (observation_keys, observation_values) in zip(
        (2, 3), active_observations, strict=True
    ):
        layer = cache.layers[layer_idx]
        assert layer.keys.shape == layer.values.shape
        assert torch.equal(layer.keys[..., -32:, :], observation_keys)
        assert torch.equal(layer.values[..., -32:, :], observation_values)


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
