"""SnapKV attention-based KV-cache scoring and compression."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from .model_utils import get_model_config
from .position_utils import make_cache_position, make_position_ids


PoolingMode = Literal["max", "avg", "mean"]
ObservationReduction = Literal["sum", "mean"]


def _aggregate_gqa_attention(
    attention_prefix: torch.Tensor,
    num_key_value_heads: int,
    reduction: ObservationReduction,
) -> torch.Tensor:
    batch_size, num_query_heads, observation_length, prefix_length = (
        attention_prefix.shape
    )
    query_heads_per_kv_head = num_query_heads // num_key_value_heads

    # Average query heads belonging to the same KV head, then combine votes
    # across the observation window.
    scores = attention_prefix.float().reshape(
        batch_size,
        num_key_value_heads,
        query_heads_per_kv_head,
        observation_length,
        prefix_length,
    )
    scores = scores.mean(dim=2)
    if reduction == "sum":
        return scores.sum(dim=2)
    return scores.mean(dim=2)


def scores_from_block_attentions(
    attentions: Sequence[torch.Tensor],
    num_key_value_heads: int,
    observation_window_size: int,
    skip_layers: Sequence[int] = (),
    reduction: ObservationReduction = "sum",
) -> tuple[torch.Tensor | None, ...]:
    """Score the cache prefix from one online observation block."""

    if (
        isinstance(num_key_value_heads, bool)
        or not isinstance(num_key_value_heads, Integral)
        or num_key_value_heads < 1
    ):
        raise ValueError("num_key_value_heads must be an integer >= 1")
    if (
        isinstance(observation_window_size, bool)
        or not isinstance(observation_window_size, Integral)
        or observation_window_size < 1
    ):
        raise ValueError("observation_window_size must be an integer >= 1")
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")

    skipped = set(skip_layers)
    scores_by_layer: list[torch.Tensor | None] = []
    for layer_idx, attention in enumerate(attentions):
        if layer_idx in skipped:
            scores_by_layer.append(None)
            continue
        if not isinstance(attention, torch.Tensor) or attention.ndim != 4:
            raise ValueError(
                f"Layer {layer_idx} attention must have shape "
                "[batch, query_heads, observation_tokens, cache_tokens]"
            )

        _, num_query_heads, observation_length, cache_length = attention.shape
        if observation_length != observation_window_size:
            raise ValueError(
                f"Layer {layer_idx} returned {observation_length} observation "
                f"tokens, expected {observation_window_size}"
            )
        prefix_length = cache_length - observation_window_size
        if prefix_length < 1:
            raise ValueError("SnapKV scoring requires a non-empty cache prefix")
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by "
                f"num_key_value_heads; got H_query={num_query_heads}, "
                f"H_kv={num_key_value_heads}"
            )

        attention_prefix = attention[..., :prefix_length]
        if not bool(torch.isfinite(attention_prefix).all().item()):
            raise ValueError(f"Layer {layer_idx} attention must be finite")
        scores_by_layer.append(
            _aggregate_gqa_attention(
                attention_prefix,
                int(num_key_value_heads),
                reduction,
            )
        )

    return tuple(scores_by_layer)


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
    """Compress full-attention DynamicCache layers in place with SnapKV."""

    if (
        isinstance(observation_window_size, bool)
        or not isinstance(observation_window_size, Integral)
        or observation_window_size < 1
    ):
        raise ValueError("observation_window_size must be an integer >= 1")
    if (
        isinstance(pooling_kernel_size, bool)
        or not isinstance(pooling_kernel_size, Integral)
        or pooling_kernel_size < 1
    ):
        raise ValueError("pooling_kernel_size must be an integer >= 1")
    if pooling_kernel_size % 2 == 0:
        raise ValueError("pooling_kernel_size must be odd")
    if pooling_mode not in {"max", "avg", "mean"}:
        raise ValueError("pooling_mode must be 'max' or 'avg'")

    layers = getattr(cache, "layers", None)
    if layers is None or not layers:
        raise ValueError("cache must contain DynamicCache-compatible layers")

    if target_capacity is not None and keep_ratio is not None:
        raise ValueError("Pass either target_capacity or keep_ratio, not both")
    if target_capacity is None:
        if (
            isinstance(keep_ratio, bool)
            or not isinstance(keep_ratio, Real)
            or not math.isfinite(float(keep_ratio))
            or not 0 < float(keep_ratio) <= 1
        ):
            raise ValueError(
                "keep_ratio must satisfy 0 < keep_ratio <= 1 when no "
                "target_capacity is provided"
            )
        prompt_lengths: set[int] = set()
        for layer_idx, layer in enumerate(layers):
            if (
                bool(getattr(layer, "is_sliding", False))
                or bool(getattr(layer, "is_compileable", False))
                or hasattr(layer, "cumulative_length")
            ):
                raise ValueError(
                    f"Layer {layer_idx} is not a plain full-attention "
                    "DynamicCache layer"
                )
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if (
                not isinstance(keys, torch.Tensor)
                or not isinstance(values, torch.Tensor)
                or keys.ndim != 4
                or values.ndim != 4
                or keys.shape != values.shape
                or keys.device != values.device
            ):
                raise ValueError(
                    f"Layer {layer_idx} must contain matching 4-D K/V tensors "
                    "on one device"
                )
            prompt_lengths.add(int(keys.shape[2]))
        if len(prompt_lengths) != 1:
            raise ValueError(
                "keep_ratio requires equal pre-compression lengths across layers"
            )
        target_capacity = math.floor(prompt_lengths.pop() * float(keep_ratio))
    if (
        isinstance(target_capacity, bool)
        or not isinstance(target_capacity, Integral)
        or target_capacity < 1
    ):
        raise ValueError("target_capacity must be an integer >= 1")
    target_capacity = int(target_capacity)

    if len(scores_by_layer) != len(layers):
        raise ValueError(
            "scores_by_layer must contain one entry for every cache layer; "
            f"got {len(scores_by_layer)} scores for {len(layers)} layers"
        )
    skipped: set[int] = set()
    for layer_idx in skip_layers:
        if (
            isinstance(layer_idx, bool)
            or not isinstance(layer_idx, Integral)
            or layer_idx < 0
        ):
            raise ValueError("skip layer indices must be non-negative integers")
        skipped.add(int(layer_idx))
    invalid_skips = sorted(skipped - set(range(len(layers))))
    if invalid_skips:
        raise ValueError(f"skip layer indices do not exist: {invalid_skips}")

    for layer_idx, layer in enumerate(layers):
        if (
            bool(getattr(layer, "is_sliding", False))
            or bool(getattr(layer, "is_compileable", False))
            or hasattr(layer, "cumulative_length")
        ):
            raise ValueError(
                f"Layer {layer_idx} is not a plain full-attention DynamicCache "
                "layer; sliding, static, and quantized caches are unsupported"
            )
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise ValueError(f"Layer {layer_idx} keys and values must be tensors")
        if keys.ndim != 4 or values.ndim != 4 or keys.shape != values.shape:
            raise ValueError(
                f"Layer {layer_idx} K/V must have matching 4-D shapes; "
                f"got K={getattr(keys, 'shape', None)}, "
                f"V={getattr(values, 'shape', None)}"
            )
        if keys.device != values.device:
            raise ValueError(f"Layer {layer_idx} K/V must be on the same device")

        prompt_length = int(keys.shape[2])
        if observation_window_size > prompt_length:
            raise ValueError(
                "observation_window_size cannot exceed the prompt length"
            )
        if target_capacity > prompt_length:
            raise ValueError("target_capacity cannot exceed the prompt length")
        if target_capacity < observation_window_size:
            raise ValueError(
                "target_capacity must be at least observation_window_size so "
                "the complete observation window can be preserved"
            )
        if layer_idx in skipped or target_capacity == prompt_length:
            continue

        prefix_length = prompt_length - observation_window_size
        tokens_to_keep = target_capacity - observation_window_size
        scores = scores_by_layer[layer_idx]
        if not isinstance(scores, torch.Tensor):
            raise ValueError(f"Layer {layer_idx} has no SnapKV attention scores")
        expected_shape = (keys.shape[0], keys.shape[1], prefix_length)
        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                f"Layer {layer_idx} scores must have shape {expected_shape}; "
                f"got {tuple(scores.shape)}"
            )

        # With a sharded model, move only the scores to the cache layer's GPU.
        if scores.device != keys.device:
            scores = scores.to(device=keys.device, non_blocking=True)
        scores = scores.float()
        if not bool(torch.isfinite(scores).all().item()):
            raise ValueError(f"Layer {layer_idx} scores must be finite")

        padding = pooling_kernel_size // 2
        if pooling_mode == "max":
            scores = F.max_pool1d(
                scores,
                kernel_size=pooling_kernel_size,
                stride=1,
                padding=padding,
            )
        else:
            scores = F.avg_pool1d(
                scores,
                kernel_size=pooling_kernel_size,
                stride=1,
                padding=padding,
                count_include_pad=False,
            )
        if tuple(scores.shape) != expected_shape:
            raise AssertionError(
                f"Pooling changed score shape from {expected_shape} "
                f"to {tuple(scores.shape)}"
            )
        if not bool(torch.isfinite(scores).all().item()):
            raise ValueError(f"Layer {layer_idx} pooled scores must be finite")

        # Stable ranking gives deterministic early-position tie breaks; sorting
        # the chosen positions restores chronological cache order.
        if tokens_to_keep == 0:
            selected_indices = torch.empty(
                (*scores.shape[:2], 0),
                dtype=torch.long,
                device=scores.device,
            )
        else:
            selected_indices = torch.argsort(
                scores,
                dim=-1,
                descending=True,
                stable=True,
            )[..., :tokens_to_keep]
            selected_indices = selected_indices.sort(dim=-1).values
        gather_indices = selected_indices.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            keys.shape[-1],
        )
        selected_keys = torch.gather(keys, dim=2, index=gather_indices)
        selected_values = torch.gather(values, dim=2, index=gather_indices)

        observation_keys = keys[:, :, prefix_length:, :]
        observation_values = values[:, :, prefix_length:, :]
        layer.keys = torch.cat(
            (selected_keys, observation_keys),
            dim=2,
        ).contiguous()
        layer.values = torch.cat(
            (selected_values, observation_values),
            dim=2,
        ).contiguous()
        del (
            gather_indices,
            keys,
            observation_keys,
            observation_values,
            scores,
            selected_indices,
            selected_keys,
            selected_values,
            values,
        )

    return cache


@torch.inference_mode()
def prefill_and_score_snapkv(
    model: Any,
    prompt_ids: Sequence[int] | torch.Tensor,
    observation_window_size: int = 32,
    chunk_size: int = 512,
    skip_layers: Sequence[int] = (),
    observation_reduction: ObservationReduction = "sum",
) -> tuple[
    Any,
    torch.Tensor,
    int,
    tuple[torch.Tensor | None, ...],
]:
    """Prefill a prompt and collect per-layer SnapKV prefix scores."""

    if (
        isinstance(observation_window_size, bool)
        or not isinstance(observation_window_size, Integral)
        or observation_window_size < 1
    ):
        raise ValueError("observation_window_size must be an integer >= 1")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, Integral)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be an integer >= 1")
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
    num_attention_heads = getattr(config, "num_attention_heads", None)
    if (
        isinstance(num_key_value_heads, bool)
        or not isinstance(num_key_value_heads, Integral)
        or num_key_value_heads < 1
    ):
        raise ValueError("Model config must define a positive num_key_value_heads")
    if (
        isinstance(num_attention_heads, bool)
        or not isinstance(num_attention_heads, Integral)
        or num_attention_heads < 1
    ):
        raise ValueError("Model config must define a positive num_attention_heads")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads; "
            f"got H_query={num_attention_heads}, H_kv={num_key_value_heads}"
        )

    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must expose at least one parameter") from error
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("prompt_ids must contain one non-empty token sequence")

    prompt_length = int(prompt.shape[1])
    if observation_window_size >= prompt_length:
        raise ValueError(
            "observation_window_size must be smaller than the prompt length"
        )
    prefix_length = prompt_length - observation_window_size
    skipped: set[int] = set()
    for layer_idx in skip_layers:
        if (
            isinstance(layer_idx, bool)
            or not isinstance(layer_idx, Integral)
            or layer_idx < 0
        ):
            raise ValueError("skip layer indices must be non-negative integers")
        skipped.add(int(layer_idx))

    cache = None
    logical_position = 0
    for start in range(0, prefix_length, chunk_size):
        chunk = prompt[:, start : min(start + chunk_size, prefix_length)]
        position_ids = make_position_ids(
            logical_position,
            int(chunk.shape[1]),
            device,
        )
        model_inputs: dict[str, Any] = {
            "input_ids": chunk,
            "position_ids": position_ids,
            "cache_position": make_cache_position(position_ids),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "logits_to_keep": 1,
        }
        if cache is not None:
            model_inputs["past_key_values"] = cache
        output = model(**model_inputs)
        del model_inputs
        cache = output.past_key_values
        logical_position += int(chunk.shape[1])
        del output

    accumulated_scores: list[torch.Tensor | None] | None = None
    last_logits: torch.Tensor | None = None
    for observation_index in range(observation_window_size):
        token_start = prefix_length + observation_index
        token = prompt[:, token_start : token_start + 1]
        position_ids = make_position_ids(logical_position, 1, device)
        output = model(
            input_ids=token,
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=make_cache_position(position_ids),
            use_cache=True,
            return_dict=True,
            output_attentions=True,
            logits_to_keep=1,
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
        if accumulated_scores is None:
            accumulated_scores = [None] * len(attentions)
        elif len(attentions) != len(accumulated_scores):
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
                    f"expected {expected_key_length}. Sliding/static caches are "
                    "not supported."
                )

            vote = _aggregate_gqa_attention(
                attention[..., :prefix_length],
                int(num_key_value_heads),
                "sum",
            )
            if accumulated_scores[layer_idx] is None:
                accumulated_scores[layer_idx] = vote
            else:
                accumulated_scores[layer_idx].add_(vote)
            del vote

        if attentions:
            del attention
        del attentions
        del output

    if accumulated_scores is None or last_logits is None or cache is None:
        raise AssertionError("SnapKV observation prefill did not produce outputs")
    if observation_reduction == "mean":
        for layer_idx, scores in enumerate(accumulated_scores):
            if scores is not None:
                accumulated_scores[layer_idx] = scores / observation_window_size

    return (
        cache,
        last_logits,
        logical_position,
        tuple(accumulated_scores),
    )
