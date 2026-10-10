"""Explicit HMM evidence, independent of the unchanged LLM prompt."""

from dataclasses import dataclass
import json

HMM_SEQUENCE_FORMAT = "context_state_action_censored_v1"
HMM_CONTEXT_FORMAT = "observable_trajectory_json_v1"
PADDING_TOKEN_ID = -2  # Different from the HMM's -1 missing-token evidence.


@dataclass(frozen=True)
class HMMContext:
    text: str
    token_ids: tuple[int, ...]
    truncated: bool
    history_turns: int


def build_hmm_context(tokenizer, task_description, initial_observation,
                      current_observation, history, *, k=1, max_tokens=1024):
    if k < 0 or max_tokens < 1:
        raise ValueError("HMM context k must be nonnegative and token budget positive")
    recent = list(history[-k:]) if k else []
    value = {
        "task": task_description,
        "initial_observation": initial_observation,
        "history": [{"action": h.action, "observation": h.observation}
                    for h in recent],
        "current_observation": current_observation,
    }
    # The final history feedback is already represented by current_observation.
    if value["history"] and value["history"][-1]["observation"] == current_observation:
        value["history"][-1].pop("observation")

    def encode():
        text = "<context>" + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "</context>\n"
        return text, tokenizer.encode(text, add_special_tokens=False)

    truncated = False
    text, ids = encode()
    while len(ids) > max_tokens and value["history"]:
        value["history"].pop(0)
        truncated = True
        text, ids = encode()
    # Retain task and current feedback. Shorten only the initial description
    # after dropping complete history entries, and record any shortening.
    initial_ids = tokenizer.encode(initial_observation, add_special_tokens=False)
    while len(ids) > max_tokens and initial_ids:
        initial_ids = initial_ids[:max(0, len(initial_ids) - max(1, len(ids) - max_tokens))]
        value["initial_observation"] = tokenizer.decode(initial_ids, skip_special_tokens=False)
        truncated = True
        text, ids = encode()
    if len(ids) > max_tokens:
        raise ValueError("HMM context budget cannot hold the task and current observation")
    return HMMContext(text, tuple(ids), truncated, len(value["history"]))
