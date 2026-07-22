from __future__ import annotations

from typing import Any


def get_default_skip_layers() -> tuple[int, ...]:
    """Layers left uncompressed by the basic benchmark."""

    return (0, 1)


BASIC_PASSKEY_CONFIGS: list[dict[str, Any]] = [
    {
        "config": "no_compression",
        "use_compression": False,
        "keep_ratio": 1.0,
        "strategy": "low_l2",
        "skip_layers": get_default_skip_layers(),
    },
    {
        "config": "low_l2_keep50",
        "use_compression": True,
        "keep_ratio": 0.5,
        "strategy": "low_l2",
        "skip_layers": get_default_skip_layers(),
    },
    {
        "config": "low_l2_keep10",
        "use_compression": True,
        "keep_ratio": 0.1,
        "strategy": "low_l2",
        "skip_layers": get_default_skip_layers(),
    },
    {
        "config": "random_keep50",
        "use_compression": True,
        "keep_ratio": 0.5,
        "strategy": "random",
        "skip_layers": get_default_skip_layers(),
    },
    {
        "config": "high_l2_keep50",
        "use_compression": True,
        "keep_ratio": 0.5,
        "strategy": "high_l2",
        "skip_layers": get_default_skip_layers(),
    },
]
