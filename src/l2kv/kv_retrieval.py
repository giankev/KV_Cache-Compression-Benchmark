from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch


_KEY_ADJECTIVES = (
    "amber",
    "azure",
    "bronze",
    "calm",
    "coral",
    "crimson",
    "dusky",
    "emerald",
    "gentle",
    "golden",
    "indigo",
    "ivory",
    "quiet",
    "silver",
    "swift",
    "violet",
)

_RECORD_TEMPLATE = "Record {key}: REGISTER_CONTENT is {value}.\n"
_QUESTION_TEMPLATE = "\n{key} REGISTER_CONTENT? 5 digits:\n"
_PADDING_TEXT = "Background context remains irrelevant.\n"


@dataclass(frozen=True)
class KVRecord:
    """One structured key-value record and its exact token placement."""

    key: str
    value: str
    text: str
    token_ids: tuple[int, ...]
    token_position: int


@dataclass(frozen=True)
class KVRetrievalPrompt:
    """Exact-token synthetic key-value retrieval example.

    ``prompt_ids`` already includes the final question. The target key appears
    once among ``records`` and once more in that question, as required for a
    retrieval query; the target value appears in exactly one record.
    """

    prompt_ids: tuple[int, ...]
    question_ids: tuple[int, ...]
    records: tuple[KVRecord, ...]
    target_key: str
    target_value: str
    target_record_index: int
    target_record_token_position: int
    question_token_position: int
    depth_target: float
    observation_window_size: int

    @property
    def context_ids(self) -> tuple[int, ...]:
        """Alias matching the terminology used by the passkey benchmark."""

        return self.prompt_ids

    @property
    def context_len_actual(self) -> int:
        return len(self.prompt_ids)

    @property
    def record_keys(self) -> tuple[str, ...]:
        return tuple(record.key for record in self.records)

    @property
    def record_values(self) -> tuple[str, ...]:
        return tuple(record.value for record in self.records)

    @property
    def depth_actual(self) -> float:
        if len(self.records) == 1:
            return 0.0
        return self.target_record_index / (len(self.records) - 1)


def _validate_prompt_arguments(
    target_tokens: int,
    seed: int,
    depth: float,
    observation_window_size: int,
) -> None:
    if (
        isinstance(target_tokens, bool)
        or not isinstance(target_tokens, Integral)
        or target_tokens < 1
    ):
        raise ValueError("target_tokens must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if (
        isinstance(depth, bool)
        or not isinstance(depth, Real)
        or not math.isfinite(float(depth))
        or not 0 <= float(depth) <= 1
    ):
        raise ValueError("depth must satisfy 0 <= depth <= 1")
    if (
        isinstance(observation_window_size, bool)
        or not isinstance(observation_window_size, Integral)
        or observation_window_size < 1
    ):
        raise ValueError("observation_window_size must be an integer >= 1")
    if observation_window_size > target_tokens:
        raise ValueError(
            "observation_window_size must not exceed target_tokens"
        )
    if target_tokens > 90_000:
        raise ValueError(
            "target_tokens is too large for globally unique five-digit values"
        )


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = (
        encoded["input_ids"]
        if isinstance(encoded, dict)
        else encoded.input_ids
    )

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().cpu().tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Expected a single tokenized sequence")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _base26_code(number: int, width: int = 4) -> str:
    if number < 0:
        raise ValueError("number must be non-negative")

    letters = ["a"] * width
    remaining = number
    for position in range(width - 1, -1, -1):
        remaining, digit = divmod(remaining, 26)
        letters[position] = chr(ord("a") + digit)
    if remaining:
        raise ValueError("Too many records for the available key space")
    return "".join(letters)


def _make_key(index: int) -> str:
    adjective = _KEY_ADJECTIVES[index % len(_KEY_ADJECTIVES)]
    return f"{adjective}-{_base26_code(index)}"


def _make_record(
    tokenizer: Any,
    key: str,
    value: int,
) -> tuple[str, str, tuple[int, ...]]:
    value_text = f"{value:05d}"
    text = _RECORD_TEMPLATE.format(key=key, value=value_text)
    token_ids = tuple(_encode_without_special_tokens(tokenizer, text))
    if not token_ids:
        raise ValueError("A key-value record produced no token IDs")
    return value_text, text, token_ids


def _repeat_token_ids(unit_ids: list[int], length: int) -> list[int]:
    if length == 0:
        return []
    if not unit_ids:
        raise ValueError("The padding text produced no token IDs")
    repeats = math.ceil(length / len(unit_ids))
    return (unit_ids * repeats)[:length]


def _target_index(num_records: int, depth: float) -> int:
    if num_records < 2:
        raise ValueError("At least two records are required to define depth")
    scaled = float(depth) * (num_records - 1)
    return min(int(math.floor(scaled + 0.5)), num_records - 1)


def make_kv_retrieval_prompt(
    tokenizer: Any,
    target_tokens: int = 2048,
    seed: int = 0,
    depth: float = 0.5,
    observation_window_size: int = 64,
) -> KVRetrievalPrompt:
    """Build an exact-length LongEval-Lines-style retrieval prompt.

    Every complete distractor has the same format as the target record. Text
    components are tokenized once and then concatenated as token IDs; the
    function never decodes and re-encodes the prompt. Any small residual budget
    is filled with nonnumeric background-token IDs immediately before the final
    question. The whole question is therefore inside the observation window.
    All randomness is local to ``seed``.
    """

    _validate_prompt_arguments(
        target_tokens=target_tokens,
        seed=seed,
        depth=depth,
        observation_window_size=observation_window_size,
    )

    rng = random.Random(int(seed))
    candidate_count = min(int(target_tokens), 90_000)
    key_indices = list(range(candidate_count))
    rng.shuffle(key_indices)
    values = rng.sample(range(10_000, 100_000), k=candidate_count)

    target_key = _make_key(key_indices[0])
    question_text = _QUESTION_TEMPLATE.format(key=target_key)
    question_ids = tuple(_encode_without_special_tokens(tokenizer, question_text))
    if not question_ids:
        raise ValueError("The final retrieval question produced no token IDs")
    if len(question_ids) > observation_window_size:
        raise ValueError(
            "observation_window_size is too small to contain the final question: "
            f"question has {len(question_ids)} tokens, window has "
            f"{observation_window_size}"
        )

    records_budget = int(target_tokens) - len(question_ids)
    pending_records: list[tuple[str, str, str, tuple[int, ...]]] = []
    used_record_tokens = 0

    for key_index, value in zip(key_indices, values, strict=True):
        key = _make_key(key_index)
        value_text, record_text, record_ids = _make_record(
            tokenizer=tokenizer,
            key=key,
            value=value,
        )
        if used_record_tokens + len(record_ids) > records_budget:
            break
        pending_records.append((key, value_text, record_text, record_ids))
        used_record_tokens += len(record_ids)

    if len(pending_records) < 2:
        raise ValueError(
            "target_tokens is too small for two complete records and the final "
            "question"
        )

    target_record_index = _target_index(len(pending_records), float(depth))
    target_record = pending_records.pop(0)
    pending_records.insert(target_record_index, target_record)

    record_ids_flat: list[int] = []
    records: list[KVRecord] = []
    for key, value_text, record_text, record_ids in pending_records:
        token_position = len(record_ids_flat)
        records.append(
            KVRecord(
                key=key,
                value=value_text,
                text=record_text,
                token_ids=record_ids,
                token_position=token_position,
            )
        )
        record_ids_flat.extend(record_ids)

    padding_length = records_budget - len(record_ids_flat)
    padding_ids = _repeat_token_ids(
        _encode_without_special_tokens(tokenizer, _PADDING_TEXT),
        padding_length,
    )
    question_token_position = len(record_ids_flat) + len(padding_ids)
    prompt_ids = tuple(record_ids_flat + padding_ids + list(question_ids))

    if len(prompt_ids) != target_tokens:
        raise AssertionError(
            f"Retrieval prompt length mismatch: {len(prompt_ids)} != "
            f"{target_tokens}"
        )
    if question_token_position < target_tokens - observation_window_size:
        raise AssertionError("The final question is outside the observation window")

    target_value = target_record[1]
    target_record_token_position = records[
        target_record_index
    ].token_position
    record_keys = [record.key for record in records]
    record_values = [record.value for record in records]
    if len(record_keys) != len(set(record_keys)):
        raise AssertionError("Generated record keys are not unique")
    if len(record_values) != len(set(record_values)):
        raise AssertionError("Generated record values are not unique")
    if record_keys.count(target_key) != 1:
        raise AssertionError("The target key must occur in exactly one record")
    if record_values.count(target_value) != 1:
        raise AssertionError("The target value must occur in exactly one record")

    return KVRetrievalPrompt(
        prompt_ids=prompt_ids,
        question_ids=question_ids,
        records=tuple(records),
        target_key=target_key,
        target_value=target_value,
        target_record_index=target_record_index,
        target_record_token_position=target_record_token_position,
        question_token_position=question_token_position,
        depth_target=float(depth),
        observation_window_size=int(observation_window_size),
    )


def extract_first_five_digit_number(text: str) -> str:
    """Return the first standalone five-digit number, or an empty string."""

    match = re.search(r"(?<!\d)\d{5}(?!\d)", text)
    return "" if match is None else match.group(0)


def is_correct_prediction(prediction: str, target_value: str) -> bool:
    return prediction == target_value
