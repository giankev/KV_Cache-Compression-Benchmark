from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .cache_metrics import get_cache_layer
from .model_utils import get_model_config


def get_token_labels(
    tokenizer: Any,
    input_ids: torch.Tensor,
    clean: bool = True,
) -> list[str]:
    """Return tokenizer token labels, optionally cleaning common space markers."""

    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    if not clean:
        return raw_tokens

    def clean_token(tok: str) -> str:
        return (
            tok.replace("\u0120", "\u2420")
            .replace("\u2581", "\u2420")
            .replace("\n", "\\n")
        )

    return [clean_token(tok) for tok in raw_tokens]


def normalize_01(x: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to the [0, 1] range for visualization."""

    x = x.float()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Robust Pearson correlation, returning NaN for constant inputs."""

    x = x.float().cpu()
    y = y.float().cpu()

    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return float("nan")

    if x.std() < 1e-8 or y.std() < 1e-8:
        return float("nan")

    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


@torch.no_grad()
def plot_attention_l2_heatmap(
    model: Any,
    tokenizer: Any,
    text: str,
    layer_idx: int = 9,
    q_heads: Sequence[int] = (0, 4, 8, 12),
    max_tokens: int = 64,
    clean_tokens: bool = True,
    exclude_first_for_corr: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Any:
    """Plot causal attention maps above matching key L2 bars."""

    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("INPUT TEXT:")
    print(text)

    print("\nDECODED TEXT:")
    print(tokenizer.decode(inputs["input_ids"][0]))

    out = model(
        **inputs,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )

    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError(
            "Attention tensors are not available. "
            "Reload the model with attn_implementation='eager'."
        )

    input_ids = inputs["input_ids"][0].detach().cpu()
    token_labels = get_token_labels(tokenizer, input_ids, clean=clean_tokens)
    T = len(token_labels)

    print("\nTOKEN LABELS:")
    print(token_labels)

    cfg = get_model_config(model)

    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    group_size = num_q_heads // num_kv_heads

    K, _ = get_cache_layer(out.past_key_values, layer_idx)

    ncols = len(q_heads)

    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(4.8 * ncols, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    if ncols == 1:
        axes = axes.reshape(2, 1)

    for col, q_head in enumerate(q_heads):
        kv_head = q_head // group_size

        attn = out.attentions[layer_idx][0, q_head].detach().float().cpu()

        attn_np = attn.numpy()
        upper_mask = np.triu(np.ones_like(attn_np, dtype=bool), k=1)
        attn_masked = np.ma.array(attn_np, mask=upper_mask)

        keys = K[0, kv_head].detach().float().cpu()
        l2 = keys.pow(2).sum(dim=-1).sqrt()

        last_attn = attn[-1]

        if exclude_first_for_corr and T > 2:
            corr_start = 1
        else:
            corr_start = 0

        corr = safe_corr(
            last_attn[corr_start:],
            -l2[corr_start:],
        )

        ax_attn = axes[0, col]
        ax_l2 = axes[1, col]

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")

        ax_attn.imshow(attn_masked, aspect="auto", cmap=cmap)
        ax_attn.set_title(
            f"Layer {layer_idx} | q_head {q_head} | kv_head {kv_head}\n"
            f"corr(last_attn, -L2) = {corr:.3f}"
        )
        ax_attn.set_xlabel("Key token position")
        ax_attn.set_ylabel("Query token position")

        ax_l2.bar(range(T), l2)
        ax_l2.set_title("Key L2 norm")
        ax_l2.set_ylabel("L2")
        ax_l2.set_xticks(range(T))
        ax_l2.set_xticklabels(token_labels, rotation=90, fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_alr_heatmap(
    alr_df: Any,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Any:
    """Plot mean ALR by layer and query head."""

    pivot = alr_df.groupby(["layer", "q_head"])["alr_mean"].mean().unstack()

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean ALR")
    ax.set_xlabel("Query head")
    ax.set_ylabel("Layer")
    ax.set_title("ALR per layer/head - lower is better")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig
