"""Prepare ALFWorld action-span samples for Ctrl-G's LVD/EM scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from .hmm_context import HMM_SEQUENCE_FORMAT, HMM_CONTEXT_FORMAT, PADDING_TOKEN_ID, build_hmm_context
from .collection import ROLLOUT_SAMPLE_FORMAT, attach_hmm_context

from .prompts import ACTION_CLOSE, ACTION_OPEN, PROMPT_FORMAT, STATE_CLOSE, STATE_OPEN, Step


SampleKey = tuple[int, int, int]


def load_advance_selections(
    path: str | Path,
) -> tuple[dict[SampleKey, str], dict[str, int]]:
    """Load executed sampled turns from rollout episode traces.

    Deterministic fallback steps have no corresponding generated sample and are
    counted but omitted from the returned selection map.
    """

    selected_actions: dict[SampleKey, str] = {}
    selected_states: set[tuple[int, int]] = set()
    trace_steps = 0
    fallback_steps = 0
    terminal_failure_steps = 0
    with open(path) as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            episode_record = json.loads(line)
            if "episode" not in episode_record:
                raise ValueError(f"{path}:{line_number} is missing episode")
            episode = episode_record["episode"]
            advance_trace = episode_record.get("advance_trace")
            if not isinstance(advance_trace, list):
                raise ValueError(
                    f"{path}:{line_number} is missing an advance_trace list"
                )
            for trace_index, trace in enumerate(advance_trace):
                trace_steps += 1
                if not isinstance(trace, dict) or "step" not in trace:
                    raise ValueError(
                        f"{path}:{line_number} advance_trace[{trace_index}] "
                        "is missing step"
                    )
                step = trace["step"]
                state_key = (episode, step)
                if state_key in selected_states:
                    raise ValueError(
                        f"{path}:{line_number} selects more than one turn for "
                        f"episode {episode}, step {step}"
                    )
                selected_states.add(state_key)

                sample = trace.get("sample")
                if sample is None:
                    if trace.get("termination_reason"):
                        terminal_failure_steps += 1
                    else:
                        fallback_steps += 1
                    continue
                action = trace.get("action")
                if not isinstance(action, str) or not action:
                    raise ValueError(
                        f"{path}:{line_number} advance_trace[{trace_index}] "
                        "is missing a sampled action"
                    )
                selected_actions[(episode, step, sample)] = action

    if trace_steps == 0:
        raise ValueError(f"no advance-trace steps in {path}")
    return selected_actions, {
        "advance_trace_steps": trace_steps,
        "sampled_advance_steps": len(selected_actions),
        "fallback_advance_steps": fallback_steps,
        "terminal_failure_steps": terminal_failure_steps,
    }


def validate_tokenizer_contract(tokenizer, records: list[dict]) -> None:
    """Reject tokenizer/tag/vocabulary mismatches before expensive LVD work."""

    # ``tokenizer.vocab_size`` is only the base vocabulary for Hugging Face
    # tokenizers. Special tokens may be added above that range (Qwen3.5's EOS
    # is one example), so the HMM must use the complete tokenizer size.
    vocab_size = len(tokenizer)
    eos_token_id = tokenizer.eos_token_id
    if vocab_size is None or eos_token_id is None:
        raise ValueError("tokenizer must define vocab_size and eos_token_id")
    if not 0 <= eos_token_id < vocab_size:
        raise ValueError(
            f"EOS token id {eos_token_id} is outside HMM vocabulary {vocab_size}"
        )

    added_vocab = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
    tags = {ACTION_OPEN, ACTION_CLOSE, STATE_OPEN, STATE_CLOSE}
    collisions = sorted(tag for tag in tags if tag in added_vocab)
    if collisions:
        raise ValueError(
            "experiment delimiters must not be tokenizer added tokens: "
            + ", ".join(collisions)
        )

    for record_index, record in enumerate(records):
        if record.get("hmm_completed", True) and record["hmm_sequence_token_ids"][-1] != eos_token_id:
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


def load_eligible_records(
    path: str | Path,
    *,
    selected_actions: dict[SampleKey, str] | None = None,
) -> list[dict]:
    records = []
    seen_selected: set[SampleKey] = set()
    with open(path) as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if selected_actions is not None:
                missing_key_fields = [
                    field
                    for field in ("episode", "step", "sample")
                    if field not in record
                ]
                if missing_key_fields:
                    raise ValueError(
                        f"{path}:{line_number} is missing selection fields: "
                        + ", ".join(missing_key_fields)
                    )
                key = (record["episode"], record["step"], record["sample"])
                if key not in selected_actions:
                    continue
                if key in seen_selected:
                    raise ValueError(
                        f"{path}:{line_number} duplicates selected sample {key}"
                    )
                seen_selected.add(key)
                if record.get("action") != selected_actions[key]:
                    raise ValueError(
                        f"{path}:{line_number} action does not match the "
                        f"advance trace for selected sample {key}"
                    )
            if record.get("distill_eligible"):
                if (record.get("rollout_sample_format") != ROLLOUT_SAMPLE_FORMAT
                        and record.get("hmm_sequence_format") != HMM_SEQUENCE_FORMAT
                        and record.get("parse_ok") is not True):
                    raise ValueError(
                        f"{path}:{line_number} distillation-eligible sample "
                        "is not well formed"
                    )
                validate_record(record, source=f"{path}:{line_number}")
                records.append(record)
    if selected_actions is not None:
        missing_selected = set(selected_actions) - seen_selected
        if missing_selected:
            preview = ", ".join(str(key) for key in sorted(missing_selected)[:5])
            raise ValueError(
                f"{path} is missing selected samples referenced by the advance "
                f"trace: {preview}"
            )
    if not records:
        raise ValueError(f"no distillation-eligible records in {path}")
    return records


def validate_prompt_regime(records: list[dict]) -> dict:
    """Validate raw collection provenance independently of dataset context settings."""

    if not records:
        raise ValueError("at least one record is required to validate prompt regime")
    if any(record.get("prompt_format") != PROMPT_FORMAT for record in records):
        raise ValueError(f"HMM samples must use prompt_format {PROMPT_FORMAT!r}")
    fields = (
        "schema_version", "policy_version", "constrained_state_collection",
        "no_oracle_filtering", "skills_sha256",
        "model", "tokenizer",
        "prompt_format",
        "collection_max_attempts",
    )
    result = {}
    for field in fields:
        values = {record.get(field) for record in records}
        if len(values) != 1 or None in values:
            raise ValueError(f"records do not share one {field}")
        result[field] = values.pop()
    if result["prompt_format"] != PROMPT_FORMAT:
        raise ValueError(f"HMM samples must use prompt_format {PROMPT_FORMAT!r}")
    if result["constrained_state_collection"] is not False:
        raise ValueError("HMM samples must use unmasked state/action collection; recollect masked data")
    if all(record.get("rollout_sample_format") == ROLLOUT_SAMPLE_FORMAT for record in records):
        result["rollout_sample_format"] = ROLLOUT_SAMPLE_FORMAT
    elif all(record.get("hmm_sequence_format") == HMM_SEQUENCE_FORMAT
             and record.get("hmm_context_format") == HMM_CONTEXT_FORMAT for record in records):
        result["rollout_sample_format"] = "contextualized_unmasked_attempt_v1"
    else:
        raise ValueError("HMM samples use an obsolete rollout_sample_format or hmm_sequence_format")
    if result["no_oracle_filtering"] is not True:
        raise ValueError("HMM samples must not be filtered by the environment oracle")
    # Older unmasked files lack this marker and remain reusable. New files carry
    # it into dataset/checkpoint provenance; mixing selection regimes is explicit.
    execution_policies = {record.get("collection_execution_policy") for record in records}
    if execution_policies != {None}:
        if len(execution_policies) != 1 or None in execution_policies:
            raise ValueError("records do not share one collection_execution_policy")
        result["collection_execution_policy"] = execution_policies.pop()
    return result


def validate_record(record: dict, *, source: str = "record") -> None:
    contextual = record.get("hmm_sequence_format") == HMM_SEQUENCE_FORMAT
    if contextual or record.get("rollout_sample_format") == ROLLOUT_SAMPLE_FORMAT:
        required = ("hmm_target_token_ids", "hmm_completed", "hmm_appended_eos",
                    "hmm_target_start_in_head", "prompt_token_ids", "head_token_ids", "tail_token_ids")
        required += (("hmm_context_token_ids", "hmm_sequence_token_ids", "hmm_sequence_length")
                     if contextual else ("hmm_action_start_in_target",))
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"{source} is missing fields: {', '.join(missing)}")
        target = record["hmm_target_token_ids"]
        if not target:
            raise ValueError(f"{source} has an empty observed target")
        if contextual:
            context, sequence = record["hmm_context_token_ids"], record["hmm_sequence_token_ids"]
            if not context or sequence != context + target:
                raise ValueError(f"{source} must equal nonempty context + observed target")
            if len(sequence) != record["hmm_sequence_length"]:
                raise ValueError(f"{source} sequence length mismatch")
        start = record["hmm_target_start_in_head"]
        teacher = record["head_token_ids"] + record["tail_token_ids"]
        observed = target[:-1] if record["hmm_appended_eos"] else target
        if start < 0 or teacher[start:start+len(observed)] != observed:
            raise ValueError(f"{source} target is not an exact observed teacher span")
        if record["hmm_appended_eos"] and not record.get("hmm_protocol_complete"):
            raise ValueError(f"{source} appends EOS to an incomplete attempt")
        if not contextual:
            boundary = record["hmm_action_start_in_target"]
            if boundary is not None and (not isinstance(boundary, int) or not 0 <= boundary <= len(target)):
                raise ValueError(f"{source} has an invalid action boundary")
        return
    required = (
        "use_state",
        "prompt_token_ids",
        "head_token_ids",
        "hmm_prefix_token_ids",
        "tail_token_ids",
        "hmm_sequence_token_ids",
        "raw_tail",
        "tail_span_exact",
        "head_truncated",
        "state_truncated",
        "used_state_repair",
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
    if record["state_truncated"] or record["used_state_repair"]:
        raise ValueError(
            f"{source} contains a truncated or repaired state span"
        )
    if record["tail_truncated"]:
        raise ValueError(f"{source} contains a truncated action span")
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


def build_contextual_records(tokenizer, records, history_path, *, k=1, max_tokens=1024):
    """Join attempts to their earlier executed turns, rebuilding C_t at build time."""
    if k < 0 or max_tokens < 1:
        raise ValueError("HMM context k must be nonnegative and token budget positive")
    episodes = {}
    with open(history_path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            episode = json.loads(line)
            required = ("episode", "gamefile", "task_description", "initial_observation", "steps")
            if not isinstance(episode, dict) or any(key not in episode for key in required):
                raise ValueError(f"{history_path}:{line_number} is missing observable episode history")
            if episode["episode"] in episodes:
                raise ValueError(f"duplicate episode {episode['episode']} in history")
            if not isinstance(episode["steps"], list) or any(
                    not isinstance(step, dict) or step.get("step") != index
                    for index, step in enumerate(episode["steps"])):
                raise ValueError(f"episode {episode['episode']} history steps are not contiguous")
            if not all(isinstance(episode[key], str) for key in
                       ("gamefile", "task_description", "initial_observation")):
                raise ValueError(f"episode {episode['episode']} has invalid observable text")
            episodes[episode["episode"]] = episode
    contexts, seen, output = {}, set(), []
    for record in records:
        key = (record["episode"], record["step"], record["sample"])
        if key in seen:
            raise ValueError(f"duplicate attempt {key}")
        seen.add(key)
        episode = episodes.get(record["episode"])
        if episode is None or episode["gamefile"] != record["gamefile"]:
            raise ValueError(f"attempt {key} does not match its episode/game history")
        step = record["step"]
        if not isinstance(step, int) or not 0 <= step < len(episode["steps"]):
            raise ValueError(f"attempt {key} has no matching history step")
        current = episode["steps"][step]
        if record.get("selected_for_execution") and (
                current.get("selected_sample") != record["sample"]
                or current.get("action") != record.get("action")):
            raise ValueError(f"attempt {key} does not match the executed sample")
        context_key = key[:2]
        if context_key not in contexts:
            previous = episode["steps"][:step]
            if any(not isinstance(turn.get("action"), str) or not turn["action"]
                   or not isinstance(turn.get("observation"), str) for turn in previous):
                raise ValueError(f"attempt {key} follows an unexecuted or incomplete history turn")
            observation = previous[-1]["observation"] if previous else episode["initial_observation"]
            recent = previous[-k:] if k else []
            history = [Step(thought="", state="", action=turn["action"],
                            observation=turn["observation"]) for turn in recent]
            contexts[context_key] = build_hmm_context(tokenizer, episode["task_description"],
                episode["initial_observation"], observation, history, k=k, max_tokens=max_tokens)
        contextual = attach_hmm_context(record, contexts[context_key])
        contextual.update(hmm_context_steps=k, max_hmm_context_tokens=max_tokens)
        validate_record(contextual, source=f"attempt {key}")
        output.append(contextual)
    return output


def summarize_context_usage(records, max_tokens):
    """Count each environment step once, rather than overweighting retries."""
    steps = {(record["episode"], record["step"]): record for record in records}
    lengths = torch.tensor([len(record["hmm_context_token_ids"]) for record in steps.values()], dtype=torch.float64)
    quantiles = lengths.quantile(torch.tensor([.5, .95, .99], dtype=torch.float64)).tolist()
    truncated = sum(record["hmm_context_truncated"] for record in steps.values())
    return {"steps": len(steps), "budget_tokens": max_tokens,
            "used_tokens": dict(mean=float(lengths.mean()), p50=quantiles[0], p95=quantiles[1],
                                p99=quantiles[2], max=int(lengths.max())),
            "mean_budget_utilization": float(lengths.mean()) / max_tokens,
            "truncated_steps": truncated, "truncation_rate": truncated / len(steps)}


def split_records(
    records: list[dict], *, dev_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split by game instance, keeping all steps and sibling attempts together."""

    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between zero and one")
    if len(records) < 2:
        raise ValueError("at least two records are required")

    groups: dict[tuple, list[dict]] = {}
    for index, record in enumerate(records):
        if record.get("hmm_sequence_format") == HMM_SEQUENCE_FORMAT:
            key = ("game", record["gamefile"])
        elif "episode" in record and "step" in record:
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
            sequence + [PADDING_TOKEN_ID if records[0].get("hmm_sequence_format") == HMM_SEQUENCE_FORMAT else eos_token_id] * (target_length - len(sequence))
            for sequence in sequences
        ],
        dtype=torch.long,
    )


def extract_lvd_embeddings(model, records: list[dict], *, eos_token_id: int, sequence_length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract prompt-conditioned embeddings aligned to HMM sequence tokens.

    For target token at absolute position ``k``, causal-LM hidden state ``k-1``
    is the representation used to predict it. The model therefore sees the full
    original prompt and native thinking even though only the post-think prefix
    and action sequence are returned to the HMM trainer.
    """

    device = model.device
    base_model = getattr(model, "base_model", None)
    if base_model is model:
        base_model = None
    max_length = sequence_length or max(len(r["hmm_sequence_token_ids"]) for r in records)
    sequences = pad_sequences(records, eos_token_id=eos_token_id, length=max_length)
    embeddings = None
    for row_index, record in enumerate(records):
        validate_record(record)
        prompt = list(record["prompt_token_ids"])
        head = list(record["head_token_ids"])
        prefix = list(record["hmm_prefix_token_ids"])
        tail = list(record["tail_token_ids"])
        target_sequence = list(record["hmm_sequence_token_ids"])

        contextual = record.get("hmm_sequence_format") == HMM_SEQUENCE_FORMAT
        context = list(record.get("hmm_context_token_ids", []))
        target = list(record.get("hmm_target_token_ids", target_sequence))
        prefix_start_in_head = (record["hmm_target_start_in_head"] if contextual
                                else len(head) - len(prefix))
        target_start = len(prompt) + prefix_start_in_head
        full_ids = prompt + head + tail + ([eos_token_id] if
            not contextual or record["hmm_appended_eos"] else [])
        if contextual:
            full_ids = full_ids[:target_start + len(target)]
        expected = context + target if contextual else prefix + tail + [eos_token_id]
        if target_sequence != expected:
            raise ValueError(
                "hmm_sequence_token_ids must equal prefix + tail + EOS"
            )
        if target_start <= 0:
            raise ValueError("distilled span lacks a causal predecessor token")

        input_ids = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            if base_model is None:
                # Lightweight test doubles and older model wrappers may not
                # expose the Hugging Face ``base_model`` interface.
                output = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = output.hidden_states[-1][0]
            else:
                # Calling the causal-LM wrapper would materialize logits for
                # every input position over the full vocabulary. Requesting
                # all hidden states also retains every decoder layer. LVD only
                # needs the final decoder representation, so call the base
                # model directly to keep long prompts within GPU memory.
                output = base_model(
                    input_ids=input_ids,
                    output_hidden_states=False,
                    use_cache=False,
                )
                hidden = output.last_hidden_state[0]
        target_end = target_start + len(target)
        aligned = hidden[target_start - 1 : target_end - 1].detach().cpu()
        if contextual:
            context_input = torch.tensor([[prompt[-1]] + context], device=device)
            with torch.no_grad():
                if base_model is None:
                    context_output = model(input_ids=context_input, output_hidden_states=True, use_cache=False)
                    context_hidden = context_output.hidden_states[-1][0]
                else:
                    context_output = base_model(input_ids=context_input, output_hidden_states=False, use_cache=False)
                    context_hidden = context_output.last_hidden_state[0]
            aligned = torch.cat((context_hidden[:-1].detach().cpu(), aligned))
            del context_input, context_output, context_hidden
        if aligned.shape[0] != len(target_sequence):
            raise RuntimeError("hidden-state/token alignment length mismatch")
        if embeddings is None:
            embeddings = torch.zeros((len(records), max_length, aligned.shape[-1]), dtype=aligned.dtype)
        embeddings[row_index, :len(target_sequence)] = aligned
        del aligned
        del output, hidden, input_ids

    return sequences, embeddings
