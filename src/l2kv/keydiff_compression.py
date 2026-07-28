"""KeyDiff key-similarity based KV-cache compression."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F


def _select_keydiff_indices(
    keys: torch.Tensor,
    tokens_to_keep: int,
) -> torch.Tensor:
    """Select distinctive keys independently for every batch item and KV head."""

    scoring_keys = keys.float()
    normalized_keys = F.normalize(scoring_keys, p=2, dim=-1)
    anchor = normalized_keys.mean(dim=2, keepdim=True)
    scores = -F.cosine_similarity(scoring_keys, anchor, dim=-1)

    selected_indices = torch.topk(
        scores,
        k=tokens_to_keep,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    return selected_indices.sort(dim=-1).values


@torch.no_grad()
def compress_keydiff_cache_to_budget(
    cache: Any,
    max_cache_tokens: int,
    skip_layers: Sequence[int] = (),
) -> Any:
    """Compress non-skipped cache layers to a fixed token budget in place.

    KeyDiff retains keys with the lowest cosine similarity to an anchor formed
    by averaging the normalized cached keys. Selection is independent for each
    batch element and KV head. The selected temporal positions are then sorted
    before the same index set is gathered from keys and values.
    """

    if isinstance(max_cache_tokens, bool) or not isinstance(max_cache_tokens, int):
        raise ValueError("max_cache_tokens must be a positive integer")
    if max_cache_tokens <= 0:
        raise ValueError("max_cache_tokens must be a positive integer")

    layers = getattr(cache, "layers", None)
    if layers is None:
        raise ValueError("cache must expose a DynamicCache-compatible layers attribute")
    try:
        num_layers = len(layers)
    except TypeError as error:
        raise ValueError(
            "cache.layers must be a sized sequence of cache layers"
        ) from error

    if isinstance(skip_layers, (str, bytes)):
        raise ValueError("skip_layers must contain integer layer indices")
    try:
        skip_layer_indices = tuple(skip_layers)
    except TypeError as error:
        raise ValueError("skip_layers must contain integer layer indices") from error
    if any(
        isinstance(layer_idx, bool) or not isinstance(layer_idx, int)
        for layer_idx in skip_layer_indices
    ):
        raise ValueError("skip_layers must contain integer layer indices")

    skipped = set(skip_layer_indices)
    invalid_skips = sorted(skipped - set(range(num_layers)))
    if invalid_skips:
        raise ValueError(f"skip layer indices do not exist: {invalid_skips}")

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

        selected_indices = _select_keydiff_indices(
            keys,
            tokens_to_keep=max_cache_tokens,
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
