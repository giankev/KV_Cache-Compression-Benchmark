from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd
import torch

from .cache_compression import CompressionStrategy, compress_cache
from .cache_metrics import kv_cache_size_mb, theoretical_kv_cache_size_mb

PromptKind = Literal["standard", "distractor"]


def make_passkey_prompt(
    tokenizer: Any,
    target_tokens: int = 2048,
    seed: int = 0,
    depth: float = 0.5,
) -> tuple[str, str, str]:
    random.seed(seed)

    pass_key = str(random.randint(10000, 99999))

    intro = (
        "There is an important information hidden inside a lot of irrelevant text. "
        "Find it and remember it. I will ask you about it at the end.\n\n"
    )

    needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key.\n\n"
    question = "What is the pass key? The pass key is"

    garbage_sentence = (
        "The grass is green. The sky is blue. The sun is yellow. "
        "Here we go. There and back again. "
    )

    fixed_text = intro + needle + question
    fixed_tokens = len(tokenizer(fixed_text, add_special_tokens=False).input_ids)

    garbage_budget = max(0, target_tokens - fixed_tokens)

    garbage_text = garbage_sentence * ((garbage_budget // 20) + 100)
    garbage_ids = tokenizer(garbage_text, add_special_tokens=False).input_ids[
        :garbage_budget
    ]

    split = int(len(garbage_ids) * depth)

    prefix = tokenizer.decode(garbage_ids[:split], skip_special_tokens=True)
    suffix = tokenizer.decode(garbage_ids[split:], skip_special_tokens=True)

    context = intro + prefix + "\n\n" + needle + suffix + "\n\n"

    return context, question, pass_key


def make_distractor_passkey_prompt(
    tokenizer: Any,
    target_tokens: int = 8192,
    seed: int = 0,
    depth: float = 0.5,
    passkey_digits: int = 5,
) -> tuple[str, str, str]:
    random.seed(seed)

    pass_key = "".join(str(random.randint(0, 9)) for _ in range(passkey_digits))

    intro = (
        "There is one important pass key hidden inside a long document. "
        "Many other numbers are irrelevant distractors. "
        "Find the true pass key and remember it.\n\n"
    )

    needle = (
        f"IMPORTANT INFORMATION: the true pass key is {pass_key}. "
        f"Only {pass_key} is the pass key.\n\n"
    )

    question = "What is the true pass key? Answer only the pass key:"

    distractor_templates = [
        "The ticket number is {num}. ",
        "The reference code is {num}. ",
        "The room number is {num}. ",
        "The invoice ID is {num}. ",
        "The archive label is {num}. ",
        "The temporary access code is {num}. ",
        "The user identifier is {num}. ",
        "The checkpoint number is {num}. ",
    ]

    def random_number() -> str:
        while True:
            num = "".join(str(random.randint(0, 9)) for _ in range(passkey_digits))
            if num != pass_key:
                return num

    garbage_parts: list[str] = []

    while True:
        template = random.choice(distractor_templates)
        num = random_number()
        garbage_parts.append(template.format(num=num))

        garbage_text = "".join(garbage_parts)
        fixed_text = intro + needle + question

        total_tokens = len(
            tokenizer(
                fixed_text + garbage_text,
                add_special_tokens=False,
            ).input_ids
        )

        if total_tokens >= target_tokens:
            break

    fixed_text = intro + needle + question
    fixed_tokens = len(tokenizer(fixed_text, add_special_tokens=False).input_ids)

    garbage_budget = max(0, target_tokens - fixed_tokens)

    garbage_ids = tokenizer(
        "".join(garbage_parts),
        add_special_tokens=False,
    ).input_ids[:garbage_budget]

    split = int(len(garbage_ids) * depth)

    prefix = tokenizer.decode(garbage_ids[:split], skip_special_tokens=True)
    suffix = tokenizer.decode(garbage_ids[split:], skip_special_tokens=True)

    context = intro + prefix + "\n\n" + needle + suffix + "\n\n"

    return context, question, pass_key


def extract_first_number(text: str) -> str:
    match = re.search(r"\d+", text)
    if match is None:
        return ""
    return match.group(0)


@torch.no_grad()
def generate_passkey_answer(
    model: Any,
    tokenizer: Any,
    context: str,
    question: str,
    max_new_tokens: int = 8,
    use_compression: bool = False,
    keep_ratio: float = 0.6,
    prune_after: int = 1024,
    chunk_size: int = 512,
    strategy: CompressionStrategy = "low_l2",
    skip_layers: Sequence[int] = (),
) -> dict[str, Any]:
    device = next(model.parameters()).device

    context_inputs = tokenizer(context, return_tensors="pt").to(device)
    question_inputs = tokenizer(
        question,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    context_ids = context_inputs["input_ids"]
    question_ids = question_inputs["input_ids"]

    batch_size = context_ids.shape[0]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    cache = None
    logical_pos = 0
    out = None

    for start in range(0, context_ids.shape[1], chunk_size):
        chunk = context_ids[:, start : start + chunk_size]
        chunk_len = chunk.shape[1]

        if cache is None:
            attention_mask = torch.ones(
                (batch_size, chunk_len),
                device=device,
                dtype=torch.long,
            )

            out = model(
                input_ids=chunk,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

        else:
            past_len = cache.layers[0].keys.shape[2]

            attention_mask = torch.ones(
                (batch_size, past_len + chunk_len),
                device=device,
                dtype=torch.long,
            )

            cache_position = torch.arange(
                logical_pos,
                logical_pos + chunk_len,
                device=device,
                dtype=torch.long,
            )

            out = model(
                input_ids=chunk,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )

        cache = out.past_key_values
        logical_pos += chunk_len

    if use_compression:
        cache = compress_cache(
            cache,
            keep_ratio=keep_ratio,
            prune_after=prune_after,
            strategy=strategy,
            skip_layers=skip_layers,
        )

    q_len = question_ids.shape[1]
    past_len = cache.layers[0].keys.shape[2]

    attention_mask = torch.ones(
        (batch_size, past_len + q_len),
        device=device,
        dtype=torch.long,
    )

    cache_position = torch.arange(
        logical_pos,
        logical_pos + q_len,
        device=device,
        dtype=torch.long,
    )

    out = model(
        input_ids=question_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )

    cache = out.past_key_values
    logical_pos += q_len

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    prefill_time = time.perf_counter() - t0

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for step in range(max_new_tokens):
        generated.append(next_token)

        if step == max_new_tokens - 1:
            break

        past_len = cache.layers[0].keys.shape[2]

        attention_mask_step = torch.ones(
            (batch_size, past_len + 1),
            device=device,
            dtype=torch.long,
        )

        cache_position = torch.tensor(
            [logical_pos],
            device=device,
            dtype=torch.long,
        )

        out = model(
            input_ids=next_token,
            attention_mask=attention_mask_step,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

        cache = out.past_key_values
        logical_pos += 1

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    decode_time = time.perf_counter() - t0

    generated_ids = torch.cat(generated, dim=-1)
    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    final_cache_mb = kv_cache_size_mb(cache)

    uncompressed_cache_mb = theoretical_kv_cache_size_mb(
        model=model,
        seq_len=logical_pos,
        batch_size=batch_size,
    )

    memory_saved_percent = 100 * (
        1 - final_cache_mb / (uncompressed_cache_mb + 1e-8)
    )

    return {
        "generated_text": generated_text,
        "context_tokens": context_ids.shape[1],
        "question_tokens": question_ids.shape[1],
        "total_prompt_tokens": context_ids.shape[1] + question_ids.shape[1],
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "decode_tok_s": max_new_tokens / max(decode_time, 1e-8),
        "final_cache_len": cache.layers[0].keys.shape[2],
        "final_cache_mb": final_cache_mb,
        "uncompressed_cache_mb": uncompressed_cache_mb,
        "memory_saved_percent": memory_saved_percent,
    }


def run_passkey_benchmark(
    model: Any,
    tokenizer: Any,
    eval_configs: Sequence[Mapping[str, Any]],
    context_lengths: Sequence[int],
    depths: Sequence[float],
    seeds: Sequence[int],
    prompt_kind: PromptKind = "standard",
    max_new_tokens: int = 8,
    prune_after: int = 1024,
    chunk_size: int = 512,
    strategy: CompressionStrategy = "low_l2",
    passkey_digits: int = 8,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Run the passkey benchmark loop from the notebook."""

    rows: list[dict[str, Any]] = []

    for context_len in context_lengths:
        for depth in depths:
            for seed in seeds:
                if prompt_kind == "standard":
                    context, question, answer = make_passkey_prompt(
                        tokenizer,
                        target_tokens=context_len,
                        seed=seed,
                        depth=depth,
                    )
                elif prompt_kind == "distractor":
                    context, question, answer = make_distractor_passkey_prompt(
                        tokenizer,
                        target_tokens=context_len,
                        seed=seed,
                        depth=depth,
                        passkey_digits=passkey_digits,
                    )
                else:
                    raise ValueError(f"Unknown prompt_kind: {prompt_kind}")

                for cfg in eval_configs:
                    print(
                        f"{cfg['config']} | "
                        f"context={context_len} | depth={depth} | seed={seed}"
                    )

                    cfg_strategy = cfg.get("strategy", strategy)
                    cfg_prune_after = cfg.get("prune_after", prune_after)

                    res = generate_passkey_answer(
                        model,
                        tokenizer,
                        context,
                        question,
                        max_new_tokens=cfg.get("max_new_tokens", max_new_tokens),
                        use_compression=cfg["use_compression"],
                        keep_ratio=cfg["keep_ratio"],
                        prune_after=cfg_prune_after,
                        chunk_size=cfg.get("chunk_size", chunk_size),
                        strategy=cfg_strategy,
                        skip_layers=cfg["skip_layers"],
                    )

                    generated = res["generated_text"].strip()
                    prediction = extract_first_number(generated)
                    if prompt_kind == "distractor":
                        correct = prediction == answer
                    else:
                        correct = generated.startswith(answer)

                    row = {
                        "config": cfg["config"],
                        "context_len": context_len,
                        "depth": depth,
                        "seed": seed,
                        "answer": answer,
                        "generated": generated,
                        "correct": correct,
                        "keep_ratio": cfg["keep_ratio"],
                        "compression_ratio": 1 - cfg["keep_ratio"],
                        "strategy": cfg_strategy,
                        "skip_layers": tuple(cfg["skip_layers"]),
                        "prune_after": cfg_prune_after,
                        "final_cache_mb": res["final_cache_mb"],
                        "uncompressed_cache_mb": res["uncompressed_cache_mb"],
                        "memory_saved_percent": res["memory_saved_percent"],
                        "context_tokens": res["context_tokens"],
                        "question_tokens": res["question_tokens"],
                        "total_prompt_tokens": res["total_prompt_tokens"],
                        "prefill_time": res["prefill_time"],
                        "decode_time": res["decode_time"],
                        "decode_tok_s": res["decode_tok_s"],
                        "final_cache_len": res["final_cache_len"],
                    }

                    if prompt_kind == "distractor":
                        row["prediction"] = prediction

                    rows.append(row)

    df = pd.DataFrame(rows)

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
