from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from .cache_metrics import get_cache_layer
from .model_utils import get_model_config


def debug_attention_l2_shapes(
    attn_full: torch.Tensor,
    l2: torch.Tensor,
    cache_len: int,
    layer_idx: int,
    q_head: int,
    kv_head: int,
) -> None:
    print(
        f"layer={layer_idx} | q_head={q_head} | kv_head={kv_head} | "
        f"attn_full.shape={tuple(attn_full.shape)} | "
        f"l2.shape={tuple(l2.shape)} | "
        f"cache_len={cache_len}"
    )


def extract_attention_over_cache(
    attn_full: torch.Tensor,
    cache_len: int,
) -> torch.Tensor:
    """Extract decode-step attention over cached tokens only."""

    attn_full = attn_full.detach().float().cpu().flatten()
    key_len = attn_full.shape[0]

    if key_len == cache_len + 1:
        return attn_full[:cache_len]

    if key_len == cache_len:
        return attn_full

    if key_len > cache_len + 1:
        print(
            f"Warning: key_len={key_len}, cache_len={cache_len}. "
            "Using the first cache_len positions."
        )
        return attn_full[:cache_len]

    raise ValueError(
        f"Attention too short: key_len={key_len}, cache_len={cache_len}. "
        "output_attentions with cache may not be returning full attention. "
        "Try loading the model with attn_implementation='eager'."
    )


def compute_alr_from_attention_and_l2(
    attn_scores: torch.Tensor,
    l2_scores: torch.Tensor,
    normalize_attention: bool = True,
) -> dict[str, float]:
    """Compute ALR metrics from attention scores and key L2 scores."""

    attn = attn_scores.detach().float().cpu().flatten()
    l2 = l2_scores.detach().float().cpu().flatten()

    if attn.shape[0] != l2.shape[0]:
        raise ValueError(
            f"Mismatch ALR: attn length={attn.shape[0]}, "
            f"l2 length={l2.shape[0]}"
        )

    if normalize_attention:
        attn = attn / (attn.sum() + 1e-8)

    l2_drop_order = torch.argsort(l2, descending=True)
    ideal_drop_order = torch.argsort(attn, descending=False)

    l2_cumulative_loss = torch.cumsum(attn[l2_drop_order], dim=0)
    ideal_cumulative_loss = torch.cumsum(attn[ideal_drop_order], dim=0)

    y = l2_cumulative_loss - ideal_cumulative_loss
    y = torch.clamp(y, min=0.0)

    return {
        "alr_sum": y.sum().item(),
        "alr_mean": y.mean().item(),
    }


@torch.no_grad()
def scan_alr_qwen_decode_step(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_tokens: int = 512,
    normalize_attention: bool = True,
    debug_shapes: bool = False,
) -> pd.DataFrame:
    """Compute ALR per prompt, layer, and query head for one decode step."""

    device = next(model.parameters()).device
    cfg = get_model_config(model)

    num_layers = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    group_size = num_q_heads // num_kv_heads

    rows: list[dict[str, Any]] = []

    for text_id, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        batch_size = inputs["input_ids"].shape[0]

        prefill_out = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )

        cache = prefill_out.past_key_values

        K0, _ = get_cache_layer(cache, 0)
        cache_len = K0.shape[2]

        next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        attention_mask_step = torch.ones(
            (batch_size, cache_len + 1),
            device=device,
            dtype=torch.long,
        )

        cache_position = torch.tensor(
            [cache_len],
            device=device,
            dtype=torch.long,
        )

        decode_out = model(
            input_ids=next_token,
            attention_mask=attention_mask_step,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

        if decode_out.attentions is None or decode_out.attentions[0] is None:
            raise RuntimeError(
                "Attention tensors are not available. "
                "Reload the model with attn_implementation='eager'."
            )

        for layer_idx in range(num_layers):
            K, _ = get_cache_layer(cache, layer_idx)
            layer_cache_len = K.shape[2]

            for q_head in range(num_q_heads):
                kv_head = q_head // group_size

                attn_full = (
                    decode_out.attentions[layer_idx][0, q_head, 0]
                    .detach()
                    .float()
                    .cpu()
                )

                keys = K[0, kv_head].detach().float().cpu()
                l2 = keys.pow(2).sum(dim=-1).sqrt()

                if debug_shapes:
                    debug_attention_l2_shapes(
                        attn_full=attn_full,
                        l2=l2,
                        cache_len=layer_cache_len,
                        layer_idx=layer_idx,
                        q_head=q_head,
                        kv_head=kv_head,
                    )

                attn_cache = extract_attention_over_cache(
                    attn_full=attn_full,
                    cache_len=layer_cache_len,
                )

                alr = compute_alr_from_attention_and_l2(
                    attn_scores=attn_cache,
                    l2_scores=l2,
                    normalize_attention=normalize_attention,
                )

                rows.append(
                    {
                        "text_id": text_id,
                        "layer": layer_idx,
                        "q_head": q_head,
                        "kv_head": kv_head,
                        "seq_len": layer_cache_len,
                        "alr_sum": alr["alr_sum"],
                        "alr_mean": alr["alr_mean"],
                    }
                )

    return pd.DataFrame(rows)


def summarize_alr_by_layer(alr_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Summarize ALR by layer using the notebook's relative threshold rule."""

    high_alr_threshold = alr_df["alr_mean"].quantile(0.75)

    summary = (
        alr_df.groupby("layer")
        .agg(
            mean_alr=("alr_mean", "mean"),
            median_alr=("alr_mean", "median"),
            max_alr=("alr_mean", "max"),
            min_alr=("alr_mean", "min"),
            std_alr=("alr_mean", "std"),
            high_alr_fraction=(
                "alr_mean",
                lambda x: (x > high_alr_threshold).mean(),
            ),
        )
        .reset_index()
        .sort_values("median_alr", ascending=False)
    )

    return summary, float(high_alr_threshold)


def suggest_skip_layers_from_alr(
    layer_summary: pd.DataFrame,
    top_k: int = 2,
    always_include_first_two: bool = True,
) -> tuple[int, ...]:
    """Suggest layers to skip by taking the highest median ALR layers."""

    skip_layers = (
        layer_summary.sort_values("median_alr", ascending=False)
        .head(top_k)["layer"]
        .tolist()
    )

    if always_include_first_two:
        skip_layers = sorted(set(skip_layers + [0, 1]))

    return tuple(int(layer) for layer in skip_layers)
