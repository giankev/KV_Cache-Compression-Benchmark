from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import transformers


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def make_run_metadata(
    *,
    script: str,
    model_name: str,
    model: Any,
    requested_dtype: str | torch.dtype,
    attention_implementation: str | None,
    seed: int,
    lengths: Sequence[int],
    depths: Sequence[float] | None,
    configurations: Sequence[Any],
    skip_layers: Sequence[int],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect the reproducibility fields shared by executable scripts."""

    parameter = next(model.parameters())
    config = getattr(model.config, "text_config", model.config)
    actual_attention = (
        getattr(config, "_attn_implementation", None)
        or attention_implementation
    )
    metadata: dict[str, Any] = {
        "script": script,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_name": model_name,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "requested_dtype": requested_dtype,
        "model_dtype": str(parameter.dtype),
        "device": str(parameter.device),
        "device_map": getattr(model, "hf_device_map", None),
        "attention_implementation": actual_attention,
        "seed": seed,
        "lengths": list(lengths),
        "depths": list(depths) if depths is not None else None,
        "configurations": list(configurations),
        "skip_layers": list(skip_layers),
    }
    if extra:
        metadata.update(extra)
    return _json_ready(metadata)


def print_run_metadata(metadata: Mapping[str, Any]) -> None:
    """Print metadata in a compact, human-readable startup block."""

    print("Run metadata:")
    for key, value in metadata.items():
        rendered = json.dumps(value, sort_keys=True) if isinstance(
            value, (dict, list)
        ) else str(value)
        print(f"  {key}: {rendered}")


def save_run_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    """Save run metadata as formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
