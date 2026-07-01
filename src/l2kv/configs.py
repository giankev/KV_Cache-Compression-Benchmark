from __future__ import annotations

from copy import deepcopy
from typing import Any

LIGHT_STANDARD_CONFIGS: list[dict[str, Any]] = [
    {
        "config": "no_compression",
        "use_compression": False,
        "keep_ratio": 1.0,
        "strategy": "low_l2",
        "skip_layers": (),
    },
    {
        "config": "low_l2_no_skip_keep10",
        "use_compression": True,
        "keep_ratio": 0.1,
        "strategy": "low_l2",
        "skip_layers": (),
    },
    {
        "config": "low_l2_skip01_keep10",
        "use_compression": True,
        "keep_ratio": 0.1,
        "strategy": "low_l2",
        "skip_layers": (0, 1),
    },
    {
        "config": "high_l2_no_skip_keep10",
        "use_compression": True,
        "keep_ratio": 0.1,
        "strategy": "high_l2",
        "skip_layers": (),
    },
    {
        "config": "random_no_skip_keep10",
        "use_compression": True,
        "keep_ratio": 0.1,
        "strategy": "random",
        "skip_layers": (),
    },
]

LIGHT_FINAL_CONFIGS: list[dict[str, Any]] = [
    {
        "config": "no_compression",
        "use_compression": False,
        "keep_ratio": 1.0,
        "strategy": "low_l2",
        "skip_layers": (),
    },
    {
        "config": "low_l2_no_skip_keep10",
        "use_compression": True,
        "keep_ratio": 0.1,
        "strategy": "low_l2",
        "skip_layers": (),
    },
    {
        "config": "low_l2_skip01_keep10",
        "use_compression": True,
        "keep_ratio": 0.1,
        "strategy": "low_l2",
        "skip_layers": (0, 1),
    },
]

LIGHT_CONFIG_GROUPS = {
    "light_standard": LIGHT_STANDARD_CONFIGS,
    "light_final": LIGHT_FINAL_CONFIGS,
}


def get_light_config_group(name: str) -> list[dict[str, Any]]:
    if name not in LIGHT_CONFIG_GROUPS:
        raise ValueError(f"Unknown config group: {name}")

    return deepcopy(LIGHT_CONFIG_GROUPS[name])


def get_light_config(name: str) -> dict[str, Any]:
    configs = {cfg["config"]: cfg for cfg in LIGHT_STANDARD_CONFIGS}
    if name not in configs:
        raise ValueError(f"Unknown config: {name}")

    return deepcopy(configs[name])
