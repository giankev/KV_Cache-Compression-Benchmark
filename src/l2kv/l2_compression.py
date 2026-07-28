"""L2-norm based KV-cache compression and random/high-L2 baselines."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import torch


CompressionStrategy = Literal["low_l2", "high_l2", "random"]


def _select_indices(
    keys: torch.Tensor,
    tokens_to_keep: int,
    strategy: CompressionStrategy,
    seed: int | None,
    generator: torch.Generator | None,
    generators_by_device: dict[str, torch.Generator],
) -> torch.Tensor:
    batch_size, num_heads, sequence_length, _ = keys.shape

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
            (batch_size, num_heads, sequence_length),
            dtype=torch.float32,
            device=keys.device,
            generator=local_generator,
        )
        largest = False

    selected_indices = torch.topk(
        scores,
        k=tokens_to_keep,
        dim=-1,
        largest=largest,
        sorted=False,
    ).indices
    return selected_indices.sort(dim=-1).values


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
    """Compress each non-skipped cache layer to a fixed token budget in place."""

    if max_cache_tokens <= 0:
        raise ValueError("max_cache_tokens must be positive")
    if strategy not in {"low_l2", "high_l2", "random"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    if seed is not None and generator is not None:
        raise ValueError("Pass either seed or generator, not both")

    layers = getattr(cache, "layers", None)
    if layers is None:
        raise ValueError("cache must expose a DynamicCache-compatible layers attribute")
    skipped = set(skip_layers)
    invalid_skips = sorted(skipped - set(range(len(layers))))
    if invalid_skips:
        raise ValueError(f"skip layer indices do not exist: {invalid_skips}")

    generators_by_device: dict[str, torch.Generator] = {}
    for layer_idx, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise ValueError(f"Layer {layer_idx} keys and values must be tensors")
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                f"Layer {layer_idx} K and V must be 4-D "
                "[batch, heads, sequence, head_dim]"
            )
        if keys.shape != values.shape:
            raise ValueError(
                f"Layer {layer_idx} K and V shapes must match; "
                f"got K={tuple(keys.shape)}, V={tuple(values.shape)}"
            )
        if keys.device != values.device:
            raise ValueError(f"Layer {layer_idx} K and V must be on the same device")

        sequence_length = int(keys.shape[2])
        if layer_idx in skipped or sequence_length <= max_cache_tokens:
            continue

        selected_indices = _select_indices(
            keys,
            tokens_to_keep=max_cache_tokens,
            strategy=strategy,
            seed=seed,
            generator=generator,
            generators_by_device=generators_by_device,
        )
        gather_indices = selected_indices.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            keys.shape[-1],
        )
        layer.keys = torch.gather(keys, dim=2, index=gather_indices).contiguous()
        layer.values = torch.gather(values, dim=2, index=gather_indices).contiguous()

    return cache
