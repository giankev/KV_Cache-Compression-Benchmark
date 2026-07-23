"""Compress a post-prefill KV cache with aligned L2 or random token selection.

Low-L2, high-L2, and random policies always gather matching K/V positions.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import torch

CompressionStrategy = Literal["low_l2", "high_l2", "random"]
_VALID_STRATEGIES = {"low_l2", "high_l2", "random"}


def _cache_layers(cache: Any) -> Sequence[Any]:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise ValueError("cache must expose a DynamicCache-compatible layers attribute")
    return layers


def _validate_layer_tensors(
    keys: Any,
    values: Any,
    layer_idx: int,
) -> tuple[int, int, int, int]:
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise ValueError(f"Layer {layer_idx} keys and values must be tensors")
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError(
            f"Layer {layer_idx} K and V must be 4-D [batch, heads, sequence, head_dim]; "
            f"got K={tuple(keys.shape)}, V={tuple(values.shape)}"
        )
    if keys.shape != values.shape:
        raise ValueError(
            f"Layer {layer_idx} K and V shapes must match; "
            f"got K={tuple(keys.shape)}, V={tuple(values.shape)}"
        )
    if keys.device != values.device:
        raise ValueError(
            f"Layer {layer_idx} K and V must be on the same device; "
            f"got K={keys.device}, V={values.device}"
        )
    return tuple(int(size) for size in keys.shape)


def _select_indices(
    keys: torch.Tensor,
    tokens_to_keep: int,
    strategy: CompressionStrategy,
    seed: int | None,
    generator: torch.Generator | None,
    generators_by_device: dict[str, torch.Generator],
) -> torch.Tensor:
    batch, heads, sequence, _ = keys.shape

    if strategy == "low_l2":
        scores = keys.float().square().sum(dim=-1)
        largest = False
    elif strategy == "high_l2":
        scores = keys.float().square().sum(dim=-1)
        largest = True
    else:
        local_generator = generator
        if local_generator is not None:
            if torch.device(local_generator.device) != keys.device:
                raise ValueError(
                    "The generator and cache tensor must be on the same device"
                )
        elif seed is not None:
            device_key = str(keys.device)
            local_generator = generators_by_device.get(device_key)
            if local_generator is None:
                local_generator = torch.Generator(device=keys.device)
                local_generator.manual_seed(seed)
                generators_by_device[device_key] = local_generator
        scores = torch.rand(
            (batch, heads, sequence),
            dtype=torch.float32,
            device=keys.device,
            generator=local_generator,
        )
        largest = False

    keep_idx = torch.topk(
        scores,
        k=tokens_to_keep,
        dim=-1,
        largest=largest,
        sorted=False,
    ).indices
    # Restore chronological order after score-based top-k selection.
    return keep_idx.sort(dim=-1).values


def _gather_tokens(
    tensor: torch.Tensor,
    keep_idx: torch.Tensor,
) -> torch.Tensor:
    gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, tensor.shape[-1])
    return torch.gather(tensor, dim=2, index=gather_idx).contiguous()


@torch.no_grad()
def compress_cache(
    cache: Any,
    keep_ratio: float = 0.5,
    prune_after: int = 1024,
    strategy: CompressionStrategy = "low_l2",
    skip_layers: Sequence[int] = (),
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    """Compress a Hugging Face ``DynamicCache`` in place.

    Every initialized layer must contain compatible K and V tensors with shape
    ``[batch, num_key_value_heads, sequence, head_dim]``. Selection is performed
    independently for every batch item and KV head. ``low_l2`` keeps the keys
    with the smallest float32 L2 scores, ``high_l2`` keeps the largest, and
    ``random`` keeps a random subset. Exactly the same temporal indices are
    applied to K and V, then sorted so retained tokens stay in chronological
    order. Layers in ``skip_layers`` are never modified.

    Pass ``seed`` for repeatable random compression without changing PyTorch's
    global RNG state, or pass a device-compatible ``torch.Generator`` when the
    caller wants to manage the random stream.
    """

    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must satisfy 0 < keep_ratio <= 1")
    if prune_after < 0:
        raise ValueError("prune_after must be non-negative")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if seed is not None and generator is not None:
        raise ValueError("Pass either seed or generator, not both")

    skipped = set(skip_layers)
    layers = _cache_layers(cache)
    invalid_skips = sorted(skipped - set(range(len(layers))))
    if invalid_skips:
        raise ValueError(f"skip layer indices do not exist: {invalid_skips}")
    generators_by_device: dict[str, torch.Generator] = {}

    for layer_idx, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        _, _, sequence, _ = _validate_layer_tensors(keys, values, layer_idx)

        if layer_idx in skipped or sequence < prune_after or keep_ratio == 1:
            continue

        tokens_to_keep = min(math.ceil(keep_ratio * sequence), sequence)
        keep_idx = _select_indices(
            keys,
            tokens_to_keep=tokens_to_keep,
            strategy=strategy,
            seed=seed,
            generator=generator,
            generators_by_device=generators_by_device,
        )
        layer.keys = _gather_tokens(keys, keep_idx)
        layer.values = _gather_tokens(values, keep_idx)

    return cache


@torch.no_grad()
def compress_cache_to_budget(
    cache: Any,
    max_cache_tokens: int,
    strategy: CompressionStrategy = "low_l2",
    skip_layers: Sequence[int] = (),
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    """Compress each non-skipped cache layer to a fixed physical token budget.

    K and V must both have shape
    ``[batch, num_key_value_heads, sequence, head_dim]``. The score, shared K/V
    indices, temporal sorting, skip-layer behavior, and local random-source
    rules are identical to :func:`compress_cache`.
    """

    if max_cache_tokens <= 0:
        raise ValueError("max_cache_tokens must be positive")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if seed is not None and generator is not None:
        raise ValueError("Pass either seed or generator, not both")

    skipped = set(skip_layers)
    layers = _cache_layers(cache)
    invalid_skips = sorted(skipped - set(range(len(layers))))
    if invalid_skips:
        raise ValueError(f"skip layer indices do not exist: {invalid_skips}")
    generators_by_device: dict[str, torch.Generator] = {}

    for layer_idx, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        _, _, sequence, _ = _validate_layer_tensors(keys, values, layer_idx)

        if layer_idx in skipped or sequence <= max_cache_tokens:
            continue

        keep_idx = _select_indices(
            keys,
            tokens_to_keep=max_cache_tokens,
            strategy=strategy,
            seed=seed,
            generator=generator,
            generators_by_device=generators_by_device,
        )
        layer.keys = _gather_tokens(keys, keep_idx)
        layer.values = _gather_tokens(values, keep_idx)

    return cache
