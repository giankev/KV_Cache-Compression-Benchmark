from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from .model_utils import get_model_config
from .position_utils import make_cache_position, make_position_ids

PoolingMode = Literal["max", "avg", "mean"]
ObservationReduction = Literal["sum", "mean"]


@dataclass(frozen=True)
class SnapKVPrefillResult:
    """Outputs needed to compress a prompt cache and start greedy decoding."""

    cache: Any
    last_logits: torch.Tensor
    logical_position: int
    scores_by_layer: tuple[torch.Tensor | None, ...]
    prefix_length: int
    observation_length: int

    @property
    def logits(self) -> torch.Tensor:
        """Alias for callers that use the shorter name."""

        return self.last_logits

    @property
    def layer_scores(self) -> tuple[torch.Tensor | None, ...]:
        """Alias describing that scores are stored independently per layer."""

        return self.scores_by_layer


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return int(value)


def _validate_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_keep_ratio(keep_ratio: float) -> float:
    if (
        isinstance(keep_ratio, bool)
        or not isinstance(keep_ratio, Real)
        or not math.isfinite(float(keep_ratio))
        or not 0 < float(keep_ratio) <= 1
    ):
        raise ValueError("keep_ratio must satisfy 0 < keep_ratio <= 1")
    return float(keep_ratio)


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def compute_target_capacity(prompt_length: int, keep_ratio: float) -> int:
    """Return the fair total cache budget ``floor(prompt_length * ratio)``."""

    prompt_length = _validate_positive_int(prompt_length, "prompt_length")
    ratio = _validate_keep_ratio(keep_ratio)
    return math.floor(prompt_length * ratio)


def validate_snapkv_capacity(
    prompt_length: int,
    target_capacity: int,
    observation_window_size: int,
) -> int:
    """Validate a total cache budget and return its prefix-token allowance."""

    prompt_length = _validate_positive_int(prompt_length, "prompt_length")
    target_capacity = _validate_positive_int(target_capacity, "target_capacity")
    observation_window_size = _validate_positive_int(
        observation_window_size,
        "observation_window_size",
    )
    if observation_window_size > prompt_length:
        raise ValueError(
            "observation_window_size cannot exceed the prompt length"
        )
    if target_capacity > prompt_length:
        raise ValueError("target_capacity cannot exceed the prompt length")
    if target_capacity < observation_window_size:
        raise ValueError(
            "target_capacity must be at least observation_window_size so the "
            "complete observation window can be preserved"
        )
    return target_capacity - observation_window_size


def aggregate_gqa_attention(
    attention_prefix: torch.Tensor,
    num_key_value_heads: int,
    reduction: ObservationReduction = "sum",
) -> torch.Tensor:
    """Aggregate eager attention votes to native KV heads.

    ``attention_prefix`` has shape ``[batch, H_query, L_obs, L_prefix]``.
    Contiguous query-head groups belonging to one KV head are averaged, then
    observation queries are summed (paper-style voting) or averaged. The
    result has shape ``[batch, H_kv, L_prefix]``.
    """

    if not isinstance(attention_prefix, torch.Tensor):
        raise ValueError("attention_prefix must be a torch.Tensor")
    if attention_prefix.ndim != 4:
        raise ValueError(
            "attention_prefix must have shape "
            "[batch, query_heads, observation_tokens, prefix_tokens]"
        )
    _require_finite(attention_prefix, "attention_prefix")
    num_key_value_heads = _validate_positive_int(
        num_key_value_heads,
        "num_key_value_heads",
    )
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")

    batch, query_heads, observation_tokens, prefix_tokens = attention_prefix.shape
    if observation_tokens < 1 or prefix_tokens < 1:
        raise ValueError(
            "attention_prefix must contain at least one observation and prefix token"
        )
    if query_heads % num_key_value_heads != 0:
        raise ValueError(
            "The number of query heads must be divisible by num_key_value_heads; "
            f"got H_query={query_heads}, H_kv={num_key_value_heads}"
        )

    return _aggregate_gqa_attention_unchecked(
        attention_prefix,
        num_key_value_heads,
        reduction,
    )


def _aggregate_gqa_attention_unchecked(
    attention_prefix: torch.Tensor,
    num_key_value_heads: int,
    reduction: ObservationReduction,
) -> torch.Tensor:
    batch, query_heads, observation_tokens, prefix_tokens = attention_prefix.shape
    query_heads_per_kv = query_heads // num_key_value_heads
    grouped = attention_prefix.float().reshape(
        batch,
        num_key_value_heads,
        query_heads_per_kv,
        observation_tokens,
        prefix_tokens,
    )
    grouped = grouped.mean(dim=2)
    if reduction == "sum":
        return grouped.sum(dim=2)
    return grouped.mean(dim=2)


def pool_attention_scores(
    scores: torch.Tensor,
    kernel_size: int = 5,
    mode: PoolingMode = "max",
) -> torch.Tensor:
    """Pool per-head temporal scores without changing their sequence length."""

    if not isinstance(scores, torch.Tensor) or scores.ndim != 3:
        raise ValueError("scores must have shape [batch, kv_heads, prefix_tokens]")
    _require_finite(scores, "scores")
    kernel_size = _validate_positive_int(kernel_size, "kernel_size")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd to preserve score length")
    if mode not in {"max", "avg", "mean"}:
        raise ValueError("mode must be 'max' or 'avg'")
    if scores.shape[-1] < 1:
        raise ValueError("scores must contain at least one prefix token")

    float_scores = scores.float()
    padding = kernel_size // 2
    if mode == "max":
        pooled = F.max_pool1d(
            float_scores,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
    else:
        pooled = F.avg_pool1d(
            float_scores,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
    if pooled.shape != scores.shape:
        raise AssertionError(
            f"Pooling changed score shape from {tuple(scores.shape)} "
            f"to {tuple(pooled.shape)}"
        )
    _require_finite(pooled, "pooled scores")
    return pooled


def select_topk_indices(scores: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
    """Select the highest scores independently and restore chronological order."""

    if not isinstance(scores, torch.Tensor) or scores.ndim != 3:
        raise ValueError("scores must have shape [batch, kv_heads, prefix_tokens]")
    _require_finite(scores, "scores")
    tokens_to_keep = _validate_non_negative_int(tokens_to_keep, "tokens_to_keep")
    prefix_tokens = int(scores.shape[-1])
    if tokens_to_keep > prefix_tokens:
        raise ValueError(
            f"tokens_to_keep={tokens_to_keep} exceeds prefix length {prefix_tokens}"
        )
    if tokens_to_keep == 0:
        return torch.empty(
            (*scores.shape[:2], 0),
            dtype=torch.long,
            device=scores.device,
        )

    # Pooling deliberately creates plateaus. Stable sorting makes their
    # tie-break deterministic (earlier positions win) before chronological
    # order is restored for the gathered cache.
    selected = torch.argsort(
        scores,
        dim=-1,
        descending=True,
        stable=True,
    )[..., :tokens_to_keep]
    return selected.sort(dim=-1).values


def rewrite_kv_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    prefix_indices: torch.Tensor,
    observation_window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather identical prefix positions from K/V and append the full suffix."""

    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise ValueError("keys and values must be torch.Tensor objects")
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError(
            "keys and values must have shape [batch, kv_heads, sequence, head_dim]"
        )
    if keys.shape != values.shape:
        raise ValueError(
            f"keys and values must have identical shapes; got {tuple(keys.shape)} "
            f"and {tuple(values.shape)}"
        )
    if keys.device != values.device:
        raise ValueError("keys and values must be on the same device")
    observation_window_size = _validate_positive_int(
        observation_window_size,
        "observation_window_size",
    )
    sequence = int(keys.shape[2])
    if observation_window_size > sequence:
        raise ValueError("observation_window_size cannot exceed cache length")
    if not isinstance(prefix_indices, torch.Tensor) or prefix_indices.ndim != 3:
        raise ValueError("prefix_indices must have shape [batch, kv_heads, tokens]")
    if prefix_indices.dtype != torch.long:
        raise ValueError("prefix_indices must have dtype torch.long")
    if prefix_indices.device != keys.device:
        raise ValueError("prefix_indices must be on the cache tensor device")
    if prefix_indices.shape[:2] != keys.shape[:2]:
        raise ValueError(
            "prefix_indices batch and head dimensions must match keys and values"
        )

    prefix_length = sequence - observation_window_size
    if prefix_indices.numel() > 0:
        minimum = int(prefix_indices.min().item())
        maximum = int(prefix_indices.max().item())
        if minimum < 0 or maximum >= prefix_length:
            raise ValueError(
                "prefix_indices must refer only to positions before the "
                "observation window"
            )

    gather_index = prefix_indices.unsqueeze(-1).expand(
        -1,
        -1,
        -1,
        keys.shape[-1],
    )
    selected_keys = torch.gather(keys, dim=2, index=gather_index)
    selected_values = torch.gather(values, dim=2, index=gather_index)
    observation_keys = keys[:, :, prefix_length:, :]
    observation_values = values[:, :, prefix_length:, :]
    return (
        torch.cat((selected_keys, observation_keys), dim=2).contiguous(),
        torch.cat((selected_values, observation_values), dim=2).contiguous(),
    )


def _cache_layers(cache: Any) -> Sequence[Any]:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise ValueError("cache must expose DynamicCache-compatible layers")
    return layers


def _validate_cache_tensors(
    layer: Any,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        bool(getattr(layer, "is_sliding", False))
        or bool(getattr(layer, "is_compileable", False))
        or hasattr(layer, "cumulative_length")
    ):
        raise ValueError(
            f"Layer {layer_idx} is not a plain full-attention DynamicCache "
            "layer; sliding, static, and quantized cache layers are unsupported"
        )
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise ValueError(f"Layer {layer_idx} keys and values must be tensors")
    if keys.ndim != 4 or values.ndim != 4 or keys.shape != values.shape:
        raise ValueError(
            f"Layer {layer_idx} K/V must have matching 4-D shapes; "
            f"got K={getattr(keys, 'shape', None)}, V={getattr(values, 'shape', None)}"
        )
    if keys.device != values.device:
        raise ValueError(f"Layer {layer_idx} K/V must be on the same device")
    return keys, values


@torch.inference_mode()
def compress_snapkv_cache(
    cache: Any,
    scores_by_layer: Sequence[torch.Tensor | None],
    target_capacity: int | None = None,
    observation_window_size: int = 32,
    pooling_kernel_size: int = 5,
    pooling_mode: PoolingMode = "max",
    skip_layers: Sequence[int] = (),
    *,
    keep_ratio: float | None = None,
) -> Any:
    """Compress full-attention DynamicCache layers in place with SnapKV votes.

    ``target_capacity`` is the complete retained length: selected prefix tokens
    plus the untouched observation window. Alternatively, pass ``keep_ratio``
    to derive that capacity with :func:`compute_target_capacity`. Skipped layers
    keep their original tensor objects and may therefore remain longer than
    compressed layers.
    """

    observation_window_size = _validate_positive_int(
        observation_window_size,
        "observation_window_size",
    )
    pooling_kernel_size = _validate_positive_int(
        pooling_kernel_size,
        "pooling_kernel_size",
    )
    if pooling_kernel_size % 2 == 0:
        raise ValueError("pooling_kernel_size must be odd")
    if pooling_mode not in {"max", "avg", "mean"}:
        raise ValueError("pooling_mode must be 'max' or 'avg'")

    layers = _cache_layers(cache)
    if not layers:
        raise ValueError("cache must contain at least one initialized layer")
    if target_capacity is not None and keep_ratio is not None:
        raise ValueError("Pass either target_capacity or keep_ratio, not both")
    if target_capacity is None:
        if keep_ratio is None:
            raise ValueError("Either target_capacity or keep_ratio is required")
        prompt_lengths = {
            int(_validate_cache_tensors(layer, layer_idx)[0].shape[2])
            for layer_idx, layer in enumerate(layers)
        }
        if len(prompt_lengths) != 1:
            raise ValueError(
                "keep_ratio requires equal pre-compression lengths across layers"
            )
        target_capacity = compute_target_capacity(
            prompt_lengths.pop(),
            keep_ratio,
        )
        if target_capacity < observation_window_size:
            raise ValueError(
                "target_capacity must be at least observation_window_size so the "
                "complete observation window can be preserved"
            )
    target_capacity = _validate_positive_int(target_capacity, "target_capacity")

    if len(scores_by_layer) != len(layers):
        raise ValueError(
            "scores_by_layer must contain one entry for every cache layer; "
            f"got {len(scores_by_layer)} scores for {len(layers)} layers"
        )
    skipped: set[int] = set()
    for layer_idx in skip_layers:
        skipped.add(_validate_non_negative_int(layer_idx, "skip layer index"))
    invalid_skips = sorted(skipped - set(range(len(layers))))
    if invalid_skips:
        raise ValueError(f"skip layer indices do not exist: {invalid_skips}")

    for layer_idx, layer in enumerate(layers):
        keys, values = _validate_cache_tensors(layer, layer_idx)
        prompt_length = int(keys.shape[2])
        prefix_tokens_to_keep = validate_snapkv_capacity(
            prompt_length,
            target_capacity,
            observation_window_size,
        )

        if layer_idx in skipped or target_capacity == prompt_length:
            continue

        scores = scores_by_layer[layer_idx]
        if scores is None:
            raise ValueError(f"Layer {layer_idx} has no SnapKV attention scores")
        prefix_length = prompt_length - observation_window_size
        expected_shape = (keys.shape[0], keys.shape[1], prefix_length)
        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                f"Layer {layer_idx} scores must have shape {expected_shape}; "
                f"got {tuple(scores.shape)}"
            )
        if scores.device != keys.device:
            raise ValueError(
                f"Layer {layer_idx} scores and cache tensors must share a device"
            )
        _require_finite(scores, f"Layer {layer_idx} scores")

        pooled_scores = pool_attention_scores(
            scores,
            kernel_size=pooling_kernel_size,
            mode=pooling_mode,
        )
        prefix_indices = select_topk_indices(
            pooled_scores,
            tokens_to_keep=prefix_tokens_to_keep,
        )
        compressed_keys, compressed_values = rewrite_kv_cache(
            keys,
            values,
            prefix_indices,
            observation_window_size=observation_window_size,
        )
        layer.keys = compressed_keys
        layer.values = compressed_values

    return cache


def _as_single_batch_ids(
    token_ids: Sequence[int] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError("prompt_ids must contain one non-empty token sequence")
    return ids


def _forward_at_logical_position(
    model: Any,
    input_ids: torch.Tensor,
    cache: Any | None,
    logical_position: int,
    *,
    output_attentions: bool,
) -> Any:
    position_ids = make_position_ids(
        start_position=logical_position,
        length=int(input_ids.shape[1]),
        device=input_ids.device,
    )
    model_inputs: dict[str, Any] = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "cache_position": make_cache_position(position_ids),
        "use_cache": True,
        "return_dict": True,
        "output_attentions": output_attentions,
        "logits_to_keep": 1,
    }
    if cache is not None:
        model_inputs["past_key_values"] = cache
    return model(**model_inputs)


@torch.inference_mode()
def prefill_and_score_snapkv(
    model: Any,
    prompt_ids: Sequence[int] | torch.Tensor,
    observation_window_size: int = 32,
    chunk_size: int = 512,
    skip_layers: Sequence[int] = (),
    observation_reduction: ObservationReduction = "sum",
) -> SnapKVPrefillResult:
    """Prefill a prompt and collect SnapKV votes without retaining a 2-D map.

    Prefix chunks run without attention outputs. Observation tokens run one at
    a time using eager attention; their prefix votes are immediately reduced
    from query heads to native KV heads and accumulated in float32.
    """

    observation_window_size = _validate_positive_int(
        observation_window_size,
        "observation_window_size",
    )
    chunk_size = _validate_positive_int(chunk_size, "chunk_size")
    if observation_reduction not in {"sum", "mean"}:
        raise ValueError("observation_reduction must be 'sum' or 'mean'")

    config = get_model_config(model)
    attention_implementation = getattr(config, "_attn_implementation", None)
    if attention_implementation not in {None, "eager"}:
        raise ValueError(
            "SnapKV attention scoring requires attn_implementation='eager'; "
            f"got {attention_implementation!r}"
        )
    num_key_value_heads = getattr(config, "num_key_value_heads", None)
    if num_key_value_heads is None:
        raise ValueError("Model config must define num_key_value_heads")
    num_key_value_heads = _validate_positive_int(
        num_key_value_heads,
        "num_key_value_heads",
    )
    num_attention_heads = getattr(config, "num_attention_heads", None)
    if num_attention_heads is None:
        raise ValueError("Model config must define num_attention_heads")
    num_attention_heads = _validate_positive_int(
        num_attention_heads,
        "num_attention_heads",
    )
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads; "
            f"got H_query={num_attention_heads}, H_kv={num_key_value_heads}"
        )

    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must expose at least one parameter") from error
    prompt = _as_single_batch_ids(prompt_ids, device)
    prompt_length = int(prompt.shape[1])
    if observation_window_size >= prompt_length:
        raise ValueError(
            "observation_window_size must be smaller than the prompt length"
        )
    prefix_length = prompt_length - observation_window_size
    skipped = {
        _validate_non_negative_int(layer_idx, "skip layer index")
        for layer_idx in skip_layers
    }

    cache = None
    logical_position = 0
    for start in range(0, prefix_length, chunk_size):
        chunk = prompt[:, start : min(start + chunk_size, prefix_length)]
        output = _forward_at_logical_position(
            model,
            chunk,
            cache,
            logical_position,
            output_attentions=False,
        )
        cache = output.past_key_values
        logical_position += int(chunk.shape[1])
        del output

    accumulated: list[torch.Tensor | None] | None = None
    last_logits: torch.Tensor | None = None
    for observation_index in range(observation_window_size):
        token = prompt[:, prefix_length + observation_index : prefix_length + observation_index + 1]
        output = _forward_at_logical_position(
            model,
            token,
            cache,
            logical_position,
            output_attentions=True,
        )
        cache = output.past_key_values
        logical_position += 1
        last_logits = output.logits[:, -1, :].detach()
        attentions = getattr(output, "attentions", None)
        if attentions is None:
            raise ValueError(
                "The model returned no attention weights. Load it with "
                "attn_implementation='eager'."
            )
        if accumulated is None:
            accumulated = [None] * len(attentions)
        elif len(attentions) != len(accumulated):
            raise ValueError("The number of returned attention layers changed")

        expected_key_length = prefix_length + observation_index + 1
        for layer_idx, attention in enumerate(attentions):
            if layer_idx in skipped:
                continue
            if not isinstance(attention, torch.Tensor) or attention.ndim != 4:
                raise ValueError(
                    f"Layer {layer_idx} did not return a 4-D attention tensor"
                )
            if attention.shape[0] != 1 or attention.shape[2] != 1:
                raise ValueError(
                    f"Layer {layer_idx} attention must have shape "
                    "[1, query_heads, 1, key_tokens]"
                )
            if attention.shape[1] != num_attention_heads:
                raise ValueError(
                    f"Layer {layer_idx} returned {attention.shape[1]} query "
                    f"heads, but model config declares {num_attention_heads}"
                )
            if attention.shape[-1] != expected_key_length:
                raise ValueError(
                    f"Layer {layer_idx} returned key length {attention.shape[-1]}, "
                    f"expected {expected_key_length}. Sliding/static caches are not "
                    "supported by this simple SnapKV implementation."
                )
            # Shapes and GQA divisibility were validated above. Deferring the
            # finite-value check to cache compression avoids a host/device
            # synchronization for every layer and every observation token.
            vote = _aggregate_gqa_attention_unchecked(
                attention[..., :prefix_length],
                num_key_value_heads,
                "sum",
            )
            if accumulated[layer_idx] is None:
                accumulated[layer_idx] = vote
            else:
                accumulated[layer_idx].add_(vote)
            del vote

        if attentions:
            del attention
        del attentions
        del output

    if accumulated is None or last_logits is None or cache is None:
        raise AssertionError("SnapKV observation prefill did not produce outputs")
    if observation_reduction == "mean":
        for layer_idx, scores in enumerate(accumulated):
            if scores is not None:
                accumulated[layer_idx] = scores / observation_window_size

    return SnapKVPrefillResult(
        cache=cache,
        last_logits=last_logits,
        logical_position=logical_position,
        scores_by_layer=tuple(accumulated),
        prefix_length=prefix_length,
        observation_length=observation_window_size,
    )
