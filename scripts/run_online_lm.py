from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.cache_compression import compress_cache
from l2kv.cache_metrics import kv_cache_size_mb, theoretical_kv_cache_size_mb
from l2kv.configs import BASIC_PASSKEY_CONFIGS, get_default_skip_layers
from l2kv.model_utils import load_model_and_tokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NUM_TOKENS = 8192
PRUNE_AFTER = 1024

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "train"
STRATEGY = "low_l2"
PROGRESS_EVERY = 512

COLUMNS = [
    "config",
    "num_tokens",
    "next_token_accuracy",
    "perplexity",
    "compressed_cache_mb",
    "uncompressed_cache_mb",
    "memory_saved_percent",
]


def load_token_sequence(tokenizer: Any, num_tokens: int) -> torch.Tensor:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    text_parts: list[str] = []
    token_ids: list[int] = []

    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue

        text_parts.append(text)
        joined_text = "\n\n".join(text_parts)
        token_ids = tokenizer(joined_text, add_special_tokens=False).input_ids

        if len(token_ids) >= num_tokens + 1:
            break

    if len(token_ids) < num_tokens + 1:
        raise RuntimeError(
            f"Only found {len(token_ids)} tokens, need at least {num_tokens + 1}."
        )

    return torch.tensor(token_ids[: num_tokens + 1], dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def evaluate_config(
    model: Any,
    token_ids: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    config_name = config["config"]
    skip_layers = get_default_skip_layers()
    device = next(model.parameters()).device

    token_ids = token_ids.to(device)
    past_key_values = None
    total_loss = 0.0
    correct = 0

    for t in range(NUM_TOKENS):
        if t > 0 and t % PROGRESS_EVERY == 0:
            print(f"{config_name}: processed {t}/{NUM_TOKENS}")

        input_ids = token_ids[:, t : t + 1]
        label = token_ids[:, t + 1]

        model_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "cache_position": torch.tensor([t], device=device, dtype=torch.long),
        }

        if past_key_values is None:
            past_len = 0
        else:
            past_len = past_key_values.layers[0].keys.shape[2]

        model_kwargs["attention_mask"] = torch.ones(
            (1, past_len + 1),
            device=device,
            dtype=torch.long,
        )

        outputs = model(**model_kwargs)
        logits = outputs.logits[:, -1, :]

        loss = F.cross_entropy(logits.float(), label)
        prediction = logits.argmax(dim=-1)

        total_loss += float(loss.item())
        correct += int((prediction == label).item())

        past_key_values = outputs.past_key_values
        if config["use_compression"]:
            past_key_values = compress_cache(
                past_key_values,
                keep_ratio=config["keep_ratio"],
                prune_after=PRUNE_AFTER,
                strategy=STRATEGY,
                skip_layers=skip_layers,
            )

    print(f"{config_name}: processed {NUM_TOKENS}/{NUM_TOKENS}")

    mean_loss = total_loss / NUM_TOKENS
    compressed_cache_mb = kv_cache_size_mb(past_key_values)
    uncompressed_cache_mb = theoretical_kv_cache_size_mb(
        model=model,
        seq_len=NUM_TOKENS,
        batch_size=1,
    )
    memory_saved_percent = 100 * (
        1 - compressed_cache_mb / (uncompressed_cache_mb + 1e-8)
    )

    return {
        "config": config_name,
        "num_tokens": NUM_TOKENS,
        "next_token_accuracy": correct / NUM_TOKENS,
        "perplexity": math.exp(mean_loss),
        "compressed_cache_mb": compressed_cache_mb,
        "uncompressed_cache_mb": uncompressed_cache_mb,
        "memory_saved_percent": memory_saved_percent,
    }


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"Loading {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    print(f"Loading {DATASET_NAME}/{DATASET_CONFIG} ({DATASET_SPLIT})")
    token_ids = load_token_sequence(tokenizer, NUM_TOKENS)
    print(f"Loaded {token_ids.shape[1]} tokens for online evaluation")
    print(f"Using skip layers for compressed configs: {get_default_skip_layers()}")

    rows = [
        evaluate_config(
            model=model,
            token_ids=token_ids,
            config=config,
        )
        for config in BASIC_PASSKEY_CONFIGS
    ]

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["next_token_accuracy"] = df["next_token_accuracy"].round(4)
    df["perplexity"] = df["perplexity"].round(4)
    df["compressed_cache_mb"] = df["compressed_cache_mb"].round(2)
    df["uncompressed_cache_mb"] = df["uncompressed_cache_mb"].round(2)
    df["memory_saved_percent"] = df["memory_saved_percent"].round(2)

    output_path = results_dir / "online_lm_summary.csv"
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
