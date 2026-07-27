from __future__ import annotations

import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from l2kv.cache_compression import compress_cache_to_budget
from l2kv.cache_metrics import (
    cache_layer_lengths,
    get_cache_layer,
    kv_cache_size_mb,
    theoretical_kv_cache_size_mb,
)
from l2kv.model_utils import get_model_config, load_model_and_tokenizer
from l2kv.position_utils import make_cache_position, make_position_ids
from l2kv.runtime_metadata import (
    make_run_metadata,
    print_run_metadata,
    save_run_metadata,
)
from l2kv.snapkv import (
    compress_snapkv_cache,
    scores_from_block_attentions,
)


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NUM_TOKENS = 8192
MAX_CACHE_TOKENS = 2000
BLOCK_SIZE = 32
CHECKPOINT_EVERY = 512

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"

SEED = 0
DTYPE = "auto"
ATTN_IMPLEMENTATION = "eager"

L2_SKIP_LAYERS = (0, 1)
SNAPKV_SKIP_LAYERS: tuple[int, ...] = ()
OBSERVATION_WINDOW_SIZE = 32
POOLING_KERNEL_SIZE = 5
POOLING_MODE = "max"

ONLINE_LM_CONFIGS = (
    {"config": "no_compression", "strategy": "none", "skip_layers": ()},
    {"config": "low_l2", "strategy": "low_l2", "skip_layers": L2_SKIP_LAYERS},
    {"config": "random", "strategy": "random", "skip_layers": L2_SKIP_LAYERS},
    {"config": "high_l2", "strategy": "high_l2", "skip_layers": L2_SKIP_LAYERS},
    {
        "config": "snapkv",
        "strategy": "snapkv",
        "skip_layers": SNAPKV_SKIP_LAYERS,
    },
)

CURVE_COLUMNS = [
    "model_name",
    "config",
    "processed_tokens",
    "max_cache_tokens",
    "log_ppl",
    "perplexity",
    "next_token_accuracy",
    "cache_mb",
    "memory_saved_mb",
    "memory_saved_percent",
    "elapsed_seconds",
]
SUMMARY_COLUMNS = [
    "model_name",
    "config",
    "num_tokens",
    "max_cache_tokens",
    "final_log_ppl",
    "final_perplexity",
    "final_next_token_accuracy",
    "final_cache_mb",
    "final_memory_saved_mb",
    "final_memory_saved_percent",
    "elapsed_seconds",
]


def load_token_sequence(tokenizer: Any, num_tokens: int = NUM_TOKENS) -> torch.Tensor:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    separator_ids = tokenizer("\n\n", add_special_tokens=False).input_ids
    token_ids: list[int] = []

    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        if token_ids:
            token_ids.extend(separator_ids)
        token_ids.extend(
            tokenizer(text, add_special_tokens=False).input_ids
        )
        if len(token_ids) >= num_tokens + 1:
            break

    if len(token_ids) < num_tokens + 1:
        raise RuntimeError(
            f"Only found {len(token_ids)} tokens, need at least {num_tokens + 1}."
        )
    return torch.tensor(
        token_ids[: num_tokens + 1],
        dtype=torch.long,
    ).unsqueeze(0)


def make_block(
    token_ids: torch.Tensor,
    *,
    start: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    end = min(start + block_size, int(token_ids.shape[1]) - 1)
    return token_ids[:, start:end], token_ids[:, start + 1 : end + 1]


def should_compress(
    previous_cache_length: int,
    block_length: int,
    max_cache_tokens: int = MAX_CACHE_TOKENS,
) -> bool:
    return previous_cache_length + block_length > max_cache_tokens


def make_block_causal_mask(
    previous_cache_length: int,
    block_length: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Allow the physical cache prefix and causally mask the current block."""

    mask = torch.zeros(
        (1, 1, block_length, previous_cache_length + block_length),
        dtype=dtype,
        device=device,
    )
    mask[..., previous_cache_length:] = torch.triu(
        torch.full(
            (block_length, block_length),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        ),
        diagonal=1,
    )
    return mask


def compute_metrics(
    *,
    total_nll: float,
    correct_next_tokens: int,
    num_predictions: int,
) -> tuple[float, float, float]:
    log_ppl = total_nll / num_predictions
    return (
        log_ppl,
        math.exp(log_ppl),
        correct_next_tokens / num_predictions,
    )


@torch.inference_mode()
def evaluate_config(
    model: Any,
    token_ids: torch.Tensor,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_name = str(config["config"])
    strategy = str(config["strategy"])
    skip_layers = tuple(config["skip_layers"])
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    num_key_value_heads = int(get_model_config(model).num_key_value_heads)
    input_tokens = token_ids.to(device)
    cache = None
    total_nll = 0.0
    correct_next_tokens = 0
    num_predictions = 0
    logical_position = 0
    curve_rows: list[dict[str, Any]] = []
    if strategy == "random":
        torch.manual_seed(SEED)
    cuda_devices = {
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    }
    for cuda_device in cuda_devices:
        torch.cuda.synchronize(cuda_device)
    started = perf_counter()

    for start in range(0, NUM_TOKENS, BLOCK_SIZE):
        block_ids, labels = make_block(
            input_tokens,
            start=start,
            block_size=BLOCK_SIZE,
        )
        block_length = int(block_ids.shape[1])
        previous_lengths = cache_layer_lengths(cache) if cache is not None else []
        previous_cache_length = previous_lengths[0] if previous_lengths else 0
        collect_snapkv_scores = strategy == "snapkv" and should_compress(
            previous_cache_length,
            block_length,
        )
        position_ids = make_position_ids(
            logical_position,
            block_length,
            device,
        )

        attention_mask = None
        changed_attention_types: list[tuple[Any, str]] = []
        if previous_lengths and any(
            length != logical_position for length in previous_lengths
        ):
            if strategy == "snapkv":
                attention_mask = make_block_causal_mask(
                    previous_cache_length,
                    block_length,
                    dtype=model_dtype,
                    device=device,
                )
            elif strategy in {"low_l2", "random", "high_l2"}:
                decoder_layers = model.model.layers
                if len(decoder_layers) != len(previous_lengths):
                    raise AssertionError(
                        "Qwen decoder layers do not match the KV cache layers"
                    )
                attention_mask = {}
                # Qwen accepts a mask mapping by attention type. Temporarily
                # distinguish full skip layers from fixed-budget layers because
                # their current-block suffix starts at a different KV index.
                for layer_idx, (layer, previous_length) in enumerate(
                    zip(decoder_layers, previous_lengths, strict=True)
                ):
                    original_type = layer.attention_type
                    attention_type = original_type
                    if layer_idx in skip_layers:
                        attention_type = f"{original_type}_online_lm_full"
                        changed_attention_types.append((layer, original_type))
                        layer.attention_type = attention_type
                    if attention_type not in attention_mask:
                        attention_mask[attention_type] = make_block_causal_mask(
                            previous_length,
                            block_length,
                            dtype=model_dtype,
                            device=device,
                        )

        try:
            outputs = model(
                input_ids=block_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                position_ids=position_ids,
                cache_position=make_cache_position(position_ids),
                use_cache=True,
                output_attentions=collect_snapkv_scores,
                return_dict=True,
            )
        finally:
            for layer, original_type in changed_attention_types:
                layer.attention_type = original_type
        logits = outputs.logits
        total_nll += float(
            F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            ).item()
        )
        predictions = logits.argmax(dim=-1)
        correct_next_tokens += int((predictions == labels).sum().item())
        num_predictions += int(labels.numel())
        logical_position += block_length
        cache = outputs.past_key_values

        if strategy in {"low_l2", "random", "high_l2"}:
            active_lengths = [
                length
                for layer_idx, length in enumerate(cache_layer_lengths(cache))
                if layer_idx not in skip_layers
            ]
            if any(length > MAX_CACHE_TOKENS for length in active_lengths):
                cache = compress_cache_to_budget(
                    cache,
                    max_cache_tokens=MAX_CACHE_TOKENS,
                    strategy=strategy,
                    skip_layers=skip_layers,
                )
            lengths = cache_layer_lengths(cache)
            for layer_idx, length in enumerate(lengths):
                expected = (
                    logical_position
                    if layer_idx in skip_layers
                    else min(logical_position, MAX_CACHE_TOKENS)
                )
                if length != expected:
                    raise AssertionError(
                        f"{config_name} layer {layer_idx} has {length} tokens, "
                        f"expected {expected}"
                    )
        elif strategy == "snapkv":
            if collect_snapkv_scores:
                if outputs.attentions is None:
                    raise AssertionError("SnapKV compression requires attentions")
                observation_snapshots = [
                    (
                        get_cache_layer(cache, layer_idx)[0][
                            ..., -OBSERVATION_WINDOW_SIZE:, :
                        ].clone(),
                        get_cache_layer(cache, layer_idx)[1][
                            ..., -OBSERVATION_WINDOW_SIZE:, :
                        ].clone(),
                    )
                    for layer_idx in range(len(outputs.attentions))
                ]
                scores_by_layer = scores_from_block_attentions(
                    outputs.attentions,
                    num_key_value_heads=num_key_value_heads,
                    observation_window_size=OBSERVATION_WINDOW_SIZE,
                    skip_layers=skip_layers,
                    reduction="sum",
                )
                del outputs
                cache = compress_snapkv_cache(
                    cache,
                    scores_by_layer=scores_by_layer,
                    target_capacity=MAX_CACHE_TOKENS,
                    observation_window_size=OBSERVATION_WINDOW_SIZE,
                    pooling_kernel_size=POOLING_KERNEL_SIZE,
                    pooling_mode=POOLING_MODE,
                    skip_layers=skip_layers,
                )
                for layer_idx, (expected_keys, expected_values) in enumerate(
                    observation_snapshots
                ):
                    keys, values = get_cache_layer(cache, layer_idx)
                    if keys.shape != values.shape:
                        raise AssertionError(
                            f"SnapKV layer {layer_idx} K/V shapes differ"
                        )
                    if int(keys.shape[2]) != MAX_CACHE_TOKENS:
                        raise AssertionError(
                            f"SnapKV layer {layer_idx} has {keys.shape[2]} "
                            f"tokens, expected {MAX_CACHE_TOKENS}"
                        )
                    if not torch.equal(
                        keys[..., -OBSERVATION_WINDOW_SIZE:, :],
                        expected_keys,
                    ) or not torch.equal(
                        values[..., -OBSERVATION_WINDOW_SIZE:, :],
                        expected_values,
                    ):
                        raise AssertionError(
                            f"SnapKV layer {layer_idx} changed the observation "
                            "window"
                        )
                del observation_snapshots
                del scores_by_layer
            else:
                lengths = cache_layer_lengths(cache)
                if any(length != logical_position for length in lengths):
                    raise AssertionError(
                        "SnapKV compressed before the KV budget was exceeded"
                    )
        else:
            lengths = cache_layer_lengths(cache)
            if any(length != logical_position for length in lengths):
                raise AssertionError(
                    "The no-compression cache does not match logical position"
                )

        if strategy != "snapkv" or not collect_snapkv_scores:
            del outputs
        del logits
        del predictions

        if (
            logical_position % CHECKPOINT_EVERY == 0
            or logical_position == NUM_TOKENS
        ):
            for cuda_device in cuda_devices:
                torch.cuda.synchronize(cuda_device)
            log_ppl, perplexity, accuracy = compute_metrics(
                total_nll=total_nll,
                correct_next_tokens=correct_next_tokens,
                num_predictions=num_predictions,
            )
            cache_mb = kv_cache_size_mb(cache)
            baseline_cache_mb = theoretical_kv_cache_size_mb(
                model,
                seq_len=logical_position,
                batch_size=1,
            )
            memory_saved_mb = baseline_cache_mb - cache_mb
            memory_saved_percent = (
                100.0 * (1.0 - cache_mb / baseline_cache_mb)
                if baseline_cache_mb
                else 0.0
            )
            curve_rows.append(
                {
                    "model_name": MODEL_NAME,
                    "config": config_name,
                    "processed_tokens": logical_position,
                    "max_cache_tokens": MAX_CACHE_TOKENS,
                    "log_ppl": log_ppl,
                    "perplexity": perplexity,
                    "next_token_accuracy": accuracy,
                    "cache_mb": cache_mb,
                    "memory_saved_mb": memory_saved_mb,
                    "memory_saved_percent": memory_saved_percent,
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            print(
                f"{config_name} | processed={logical_position}/{NUM_TOKENS} | "
                f"log_ppl={log_ppl:.4f} | accuracy={accuracy:.4f}"
            )

    if logical_position != NUM_TOKENS or num_predictions != NUM_TOKENS:
        raise AssertionError(
            f"{config_name} processed {logical_position} positions and "
            f"{num_predictions} predictions, expected {NUM_TOKENS}"
        )
    final = curve_rows[-1]
    summary = {
        "model_name": MODEL_NAME,
        "config": config_name,
        "num_tokens": NUM_TOKENS,
        "max_cache_tokens": MAX_CACHE_TOKENS,
        "final_log_ppl": final["log_ppl"],
        "final_perplexity": final["perplexity"],
        "final_next_token_accuracy": final["next_token_accuracy"],
        "final_cache_mb": final["cache_mb"],
        "final_memory_saved_mb": final["memory_saved_mb"],
        "final_memory_saved_percent": final["memory_saved_percent"],
        "elapsed_seconds": final["elapsed_seconds"],
    }
    del cache
    return curve_rows, summary


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    curve_path = results_dir / "online_lm_curve.csv"
    summary_path = results_dir / "online_lm_summary.csv"
    metadata_path = results_dir / "online_lm_metadata.json"

    print(f"Loading {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME,
        dtype=DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    metadata = make_run_metadata(
        script=Path(__file__).name,
        model_name=MODEL_NAME,
        model=model,
        requested_dtype=DTYPE,
        attention_implementation=ATTN_IMPLEMENTATION,
        seed=SEED,
        lengths=[NUM_TOKENS],
        depths=None,
        configurations=ONLINE_LM_CONFIGS,
        skip_layers=(),
        extra={
            "dataset_name": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "dataset_split": DATASET_SPLIT,
            "num_tokens": NUM_TOKENS,
            "max_cache_tokens": MAX_CACHE_TOKENS,
            "block_size": BLOCK_SIZE,
            "checkpoint_every": CHECKPOINT_EVERY,
            "l2_skip_layers": L2_SKIP_LAYERS,
            "snapkv": {
                "target_cache_tokens": MAX_CACHE_TOKENS,
                "observation_window_size": OBSERVATION_WINDOW_SIZE,
                "pooling_kernel_size": POOLING_KERNEL_SIZE,
                "pooling_mode": POOLING_MODE,
                "skip_layers": SNAPKV_SKIP_LAYERS,
            },
        },
    )
    print_run_metadata(metadata)
    save_run_metadata(metadata_path, metadata)
    pd.DataFrame(columns=CURVE_COLUMNS).to_csv(curve_path, index=False)
    pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)

    print(f"Loading {DATASET_NAME}/{DATASET_CONFIG} ({DATASET_SPLIT})")
    token_ids = load_token_sequence(tokenizer)
    print(f"Loaded one deterministic stream of {token_ids.shape[1]} tokens")

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for config in ONLINE_LM_CONFIGS:
        config_curve, config_summary = evaluate_config(model, token_ids, config)
        curve_rows.extend(config_curve)
        summary_rows.append(config_summary)
        pd.DataFrame(curve_rows, columns=CURVE_COLUMNS).to_csv(
            curve_path,
            index=False,
        )
        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(
            summary_path,
            index=False,
        )

    baseline_before_budget = {
        row["processed_tokens"]: row
        for row in curve_rows
        if row["config"] == "no_compression"
        and row["processed_tokens"] <= MAX_CACHE_TOKENS
    }
    for row in curve_rows:
        baseline = baseline_before_budget.get(row["processed_tokens"])
        if baseline is None or row["config"] == "no_compression":
            continue
        if not math.isclose(
            row["log_ppl"],
            baseline["log_ppl"],
            rel_tol=1e-5,
            abs_tol=1e-6,
        ) or not math.isclose(
            row["next_token_accuracy"],
            baseline["next_token_accuracy"],
            abs_tol=1e-12,
        ):
            raise AssertionError(
                f"{row['config']} metrics diverged before reaching the KV budget"
            )

    curve_df = pd.DataFrame(curve_rows, columns=CURVE_COLUMNS)
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    curve_df.to_csv(curve_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved {curve_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
