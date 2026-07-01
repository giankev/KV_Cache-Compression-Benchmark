from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_config(model: Any) -> Any:
    """Return model.config.text_config when present, otherwise model.config."""

    return getattr(model.config, "text_config", model.config)


def load_model_and_tokenizer(
    model_name: str,
    dtype: str | torch.dtype = "auto",
    device_map: str | dict[str, Any] = "auto",
    attn_implementation: str | None = "eager",
    **model_kwargs: Any,
) -> tuple[Any, Any]:
    """Load a causal LM and tokenizer with notebook-compatible defaults."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device_map,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    load_kwargs.update(model_kwargs)

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    return model, tokenizer
