from __future__ import annotations

from typing import Any


def get_cache_layer(cache: Any, layer_idx: int) -> tuple[Any, Any]:
    """Return key/value tensors for a DynamicCache layer.

    DynamicCache stores layer objects under cache.layers. The tuple/list fallback
    mirrors the notebook helper and is useful for older Transformers outputs.
    """

    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values

    return cache[layer_idx][0], cache[layer_idx][1]


def num_cache_layers(cache: Any) -> int:
    """Return the number of layers in a cache."""

    if hasattr(cache, "layers"):
        return len(cache.layers)

    return len(cache)


def kv_cache_size_mb(cache: Any) -> float:
    """Compute the actual memory used by key and value tensors in MiB."""

    total_bytes = 0

    for layer_idx in range(num_cache_layers(cache)):
        K, V = get_cache_layer(cache, layer_idx)
        total_bytes += K.numel() * K.element_size()
        total_bytes += V.numel() * V.element_size()

    return total_bytes / 1024**2


def theoretical_kv_cache_size_mb(
    model: Any,
    seq_len: int,
    batch_size: int = 1,
) -> float:
    """Compute uncompressed KV cache size in MiB from model config."""

    cfg = getattr(model.config, "text_config", model.config)

    L = cfg.num_hidden_layers
    H_q = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads

    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // H_q

    bytes_per_value = next(model.parameters()).element_size()

    total_bytes = (
        batch_size
        * L
        * 2
        * H_kv
        * seq_len
        * head_dim
        * bytes_per_value
    )

    return total_bytes / 1024**2


def cache_layer_lengths(cache: Any) -> list[int]:
    """Return the physical cache length for each layer."""

    lengths: list[int] = []
    for layer_idx in range(num_cache_layers(cache)):
        K, _ = get_cache_layer(cache, layer_idx)
        lengths.append(int(K.shape[2]))
    return lengths


def cache_seq_len(cache: Any, layer_idx: int = 0) -> int:
    """Return the physical cache length for one layer."""

    K, _ = get_cache_layer(cache, layer_idx)
    return int(K.shape[2])


def debug_cache_shapes(cache: Any) -> list[dict[str, Any]]:
    """Return per-layer key/value shape and dtype metadata."""

    rows: list[dict[str, Any]] = []
    for layer_idx in range(num_cache_layers(cache)):
        K, V = get_cache_layer(cache, layer_idx)
        rows.append(
            {
                "layer": layer_idx,
                "keys_shape": tuple(K.shape),
                "values_shape": tuple(V.shape),
                "keys_dtype": str(K.dtype),
                "values_dtype": str(V.dtype),
                "device": str(K.device),
            }
        )
    return rows
