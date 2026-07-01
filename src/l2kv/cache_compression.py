from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import torch

CompressionStrategy = Literal["low_l2", "high_l2", "random"]


@torch.no_grad()
def compress_cache(
    cache: Any,
    keep_ratio: float = 0.5,
    prune_after: int = 1024,
    strategy: CompressionStrategy = "low_l2",
    skip_layers: Sequence[int] = (),
) -> Any:
    """Compress a HuggingFace DynamicCache in place.

    Expected layer tensor shapes:
      layer.keys:   [B, H_kv, T, D]
      layer.values: [B, H_kv, T, D]

    This preserves the notebook behavior:
      - low_l2 keeps the lowest key L2 scores.
      - high_l2 keeps the highest key L2 scores.
      - random keeps a random subset.
      - kept indices are restored to increasing temporal order.
    """

    skipped = set(skip_layers)

    for layer_idx, layer in enumerate(cache.layers):
        if layer_idx in skipped:
            continue

        K = layer.keys
        V = layer.values

        B, H_kv, T, D = K.shape

        if T < prune_after or keep_ratio >= 1.0:
            continue

        tokens_to_keep = max(1, min(math.ceil(keep_ratio * T), T))

        if strategy == "low_l2":
            scores = K.float().square().sum(dim=-1)
            largest = False
        elif strategy == "high_l2":
            scores = K.float().square().sum(dim=-1)
            largest = True
        elif strategy == "random":
            scores = torch.rand(B, H_kv, T, device=K.device)
            largest = False
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        keep_idx = torch.topk(
            scores,
            k=tokens_to_keep,
            dim=-1,
            largest=largest,
            sorted=False,
        ).indices

        keep_idx = keep_idx.sort(dim=-1).values
        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, D)

        layer.keys = torch.gather(K, dim=2, index=gather_idx).contiguous()
        layer.values = torch.gather(V, dim=2, index=gather_idx).contiguous()

    return cache
