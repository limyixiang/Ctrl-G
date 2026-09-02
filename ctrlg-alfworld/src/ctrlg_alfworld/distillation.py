"""Prepare ALFWorld action-span samples for Ctrl-G's LVD/EM scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from .prompts import ACTION_CLOSE, ACTION_OPEN, DECISION_CLOSE, DECISION_OPEN


def validate_tokenizer_contract(tokenizer, records: list[dict]) -> None:
    """Reject tokenizer/tag/vocabulary mismatches before expensive LVD work."""

    vocab_size = tokenizer.vocab_size
    eos_token_id = tokenizer.eos_token_id
    if vocab_size is None or eos_token_id is None:
        raise ValueError("tokenizer must define vocab_size and eos_token_id")
    if not 0 <= eos_token_id < vocab_size:
        raise ValueError(
            f"EOS token id {eos_token_id} is outside HMM vocabulary {vocab_size}"
        )

    added_vocab = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
    tags = {ACTION_OPEN, ACTION_CLOSE, DECISION_OPEN, DECISION_CLOSE}
    collisions = sorted(tag for tag in tags if tag in added_vocab)
    if collisions:
        raise ValueError(
            "experiment delimiters must not be tokenizer added tokens: "
            + ", ".join(collisions)
        )

    for record_index, record in enumerate(records):
        if record["hmm_sequence_token_ids"][-1] != eos_token_id:
            raise ValueError(
                f"record {record_index} HMM sequence does not end in tokenizer EOS"
            )
        for field in (
            "hmm_prefix_token_ids",
            "tail_token_ids",
            "hmm_sequence_token_ids",
        ):
            invalid = [
                token
                for token in record[field]
                if token < 0 or token >= vocab_size
            ]
            if invalid:
                raise ValueError(
                    f"record {record_index} field {field} contains IDs outside "
                    f"HMM vocabulary {vocab_size}: {invalid[:5]}"
                )


def load_eligible_records(path: str | Path) -> list[dict]:
    records = []
    with open(path) as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("distill_eligible"):
                validate_record(record, source=f"{path}:{line_number}")
                records.append(record)
    if not records:
        raise ValueError(f"no distillation-eligible records in {path}")
    return records


def validate_record(record: dict, *, source: str = "record") -> None:
    required = (
        "use_decision",
        "prompt_token_ids",
        "head_token_ids",
        "hmm_prefix_token_ids",
        "tail_token_ids",
        "hmm_sequence_token_ids",
        "raw_tail",
        "tail_span_exact",
        "head_truncated",
        "tail_truncated",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"{source} is missing fields: {', '.join(missing)}")

    prefix = list(record["hmm_prefix_token_ids"])
    head = list(record["head_token_ids"])
    tail = list(record["tail_token_ids"])
    sequence = list(record["hmm_sequence_token_ids"])
    if not prefix:
        raise ValueError(f"{source} has an empty HMM prefix")
    if record["head_truncated"] or record["tail_truncated"]:
        raise ValueError(f"{source} contains a truncated generated span")
    if not record["tail_span_exact"]:
        raise ValueError(f"{source} action tail is not exactly token-aligned")
    if not record["raw_tail"].endswith("</action>"):
        raise ValueError(f"{source} action tail does not end exactly in </action>")
    if head[-len(prefix) :] != prefix:
        raise ValueError(f"{source} HMM prefix is not a suffix of generated head")
    if sequence[: len(prefix)] != prefix:
        raise ValueError(f"{source} distilled sequence does not start with HMM prefix")
    if sequence[len(prefix) : len(prefix) + len(tail)] != tail:
        raise ValueError(f"{source} distilled sequence does not contain generated tail")
    if len(sequence) != len(prefix) + len(tail) + 1:
        raise ValueError(
            f"{source} distilled sequence must equal prefix + tail + one EOS"
        )


def split_records(
    records: list[dict], *, dev_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split by environment state, keeping sibling samples together."""

    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between zero and one")
    if len(records) < 2:
        raise ValueError("at least two records are required")

    groups: dict[tuple, list[dict]] = {}
    for index, record in enumerate(records):
        if "episode" in record and "step" in record:
            key = ("state", record["episode"], record["step"])
        else:
            # Backward compatibility for manually constructed records: each is
            # its own group, which matches the old record-level behavior.
            key = ("record", index)
        groups.setdefault(key, []).append(record)

    if len(groups) < 2:
        raise ValueError("at least two environment-state groups are required")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    dev_group_count = max(1, round(len(keys) * dev_fraction))
    dev_group_count = min(dev_group_count, len(keys) - 1)
    dev_keys = set(keys[:dev_group_count])
    train = [
        record
        for key in keys
        if key not in dev_keys
        for record in groups[key]
    ]
    dev = [
        record
        for key in keys
        if key in dev_keys
        for record in groups[key]
    ]
    return train, dev


def pad_sequences(records: list[dict], *, eos_token_id: int, length: int | None = None) -> torch.Tensor:
    sequences = [list(record["hmm_sequence_token_ids"]) for record in records]
    target_length = length or max(len(sequence) for sequence in sequences)
    if any(len(sequence) > target_length for sequence in sequences):
        raise ValueError("requested sequence length truncates a distilled span")
    return torch.tensor(
        [
            sequence + [eos_token_id] * (target_length - len(sequence))
            for sequence in sequences
        ],
        dtype=torch.long,
    )


def extract_lvd_embeddings(model, records: list[dict], *, eos_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract prompt-conditioned embeddings aligned to HMM sequence tokens.

    For target token at absolute position ``k``, causal-LM hidden state ``k-1``
    is the representation used to predict it. The model therefore sees the full
    original prompt and native thinking even though only the post-think prefix
    and action sequence are returned to the HMM trainer.
    """

    device = model.device
    sequence_tensors = []
    embedding_tensors = []
    for record in records:
        validate_record(record)
        prompt = list(record["prompt_token_ids"])
        head = list(record["head_token_ids"])
        prefix = list(record["hmm_prefix_token_ids"])
        tail = list(record["tail_token_ids"])
        target_sequence = list(record["hmm_sequence_token_ids"])

        prefix_start_in_head = len(head) - len(prefix)
        target_start = len(prompt) + prefix_start_in_head
        full_ids = prompt + head + tail + [eos_token_id]
        expected = prefix + tail + [eos_token_id]
        if target_sequence != expected:
            raise ValueError(
                "hmm_sequence_token_ids must equal prefix + tail + EOS"
            )
        if target_start <= 0:
            raise ValueError("distilled span lacks a causal predecessor token")

        input_ids = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = output.hidden_states[-1][0]
        target_end = target_start + len(target_sequence)
        aligned = hidden[target_start - 1 : target_end - 1].detach().cpu()
        if aligned.shape[0] != len(target_sequence):
            raise RuntimeError("hidden-state/token alignment length mismatch")
        sequence_tensors.append(torch.tensor(target_sequence, dtype=torch.long))
        embedding_tensors.append(aligned)

    max_length = max(sequence.shape[0] for sequence in sequence_tensors)
    hidden_size = embedding_tensors[0].shape[-1]
    sequences = torch.full(
        (len(records), max_length), eos_token_id, dtype=torch.long
    )
    embeddings = torch.zeros(
        (len(records), max_length, hidden_size),
        dtype=embedding_tensors[0].dtype,
    )
    for index, (sequence, embedding) in enumerate(
        zip(sequence_tensors, embedding_tensors)
    ):
        sequences[index, : sequence.shape[0]] = sequence
        embeddings[index, : embedding.shape[0]] = embedding
    return sequences, embeddings
