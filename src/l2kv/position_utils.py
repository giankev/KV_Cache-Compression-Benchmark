from __future__ import annotations

import torch


def make_position_ids(
    start_position: int,
    length: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return logical position IDs with shape ``[1, length]``.

    Logical positions count all tokens processed by the model and therefore do
    not shrink when the physical KV cache is compressed.
    """

    if start_position < 0:
        raise ValueError("start_position must be >= 0")
    if length < 1:
        raise ValueError("length must be >= 1")
    return torch.arange(
        start_position,
        start_position + length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)


def make_cache_position(position_ids: torch.Tensor) -> torch.Tensor:
    """Return the 1-D cache position matching single-batch position IDs."""

    if position_ids.ndim != 2 or position_ids.shape[0] != 1:
        raise ValueError("position_ids must have shape [1, length]")
    return position_ids[0]
