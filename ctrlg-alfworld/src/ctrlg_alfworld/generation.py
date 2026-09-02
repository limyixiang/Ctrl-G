"""Parsing and token-span bookkeeping for one ALFWorld model turn.

The HMM boundary is intentionally different from the base-model prompt
boundary.  Native thinking stays outside the distilled sequence; generated
post-thinking text through ``<action>`` is the HMM prefix; and the DFA reads
only the action body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prompts import (
    ACTION_CLOSE,
    ACTION_OPEN,
    DECISION_CLOSE,
    DECISION_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
)


THINK_RE = re.compile(
    rf"{re.escape(THINK_OPEN)}(.*?){re.escape(THINK_CLOSE)}", re.DOTALL
)
DECISION_RE = re.compile(
    rf"{re.escape(DECISION_OPEN)}(.*?){re.escape(DECISION_CLOSE)}", re.DOTALL
)
ACTION_RE = re.compile(
    rf"{re.escape(ACTION_OPEN)}(.*?){re.escape(ACTION_CLOSE)}", re.DOTALL
)


@dataclass(frozen=True)
class ParsedTurn:
    raw_head: str
    raw_tail: str
    thought: str
    decision: str
    hmm_prefix_text: str
    action: str
    native_think_closed: bool
    decision_format_ok: bool
    action_open_found: bool
    action_close_found: bool
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def parse_ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class GeneratedChunk:
    text: str
    token_ids: tuple[int, ...]
    stop_found: bool
    truncated: bool
    latency_seconds: float

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class TurnGeneration:
    parsed: ParsedTurn
    head_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    tail_token_ids: tuple[int, ...]
    hmm_prefix_token_ids: tuple[int, ...]
    head_latency_seconds: float
    action_latency_seconds: float
    used_head_repair: bool = False
    hmm_applied: bool = False
    hmm_skip_reason: str | None = None
    head_stop_found: bool = False
    head_truncated: bool = False
    tail_stop_found: bool = False
    tail_truncated: bool = False
    tail_span_exact: bool = False

    @property
    def total_generated_tokens(self) -> int:
        return len(self.head_token_ids) + len(self.tail_token_ids)

    @property
    def total_latency_seconds(self) -> float:
        return self.head_latency_seconds + self.action_latency_seconds


def parse_turn(raw_head: str, raw_tail: str, *, use_decision: bool) -> ParsedTurn:
    """Parse the two-phase generation without silently repairing it.

    ``raw_head`` ends at and includes ``<action>`` when generation followed the
    contract. ``raw_tail`` starts with the action body and ends at and includes
    ``</action>``.  Malformed output is represented by flags/errors so it can be
    measured instead of rewritten into apparently valid data.
    """

    errors: list[str] = []
    native_think_closed = THINK_CLOSE in raw_head
    action_open_found = ACTION_OPEN in raw_head
    action_close_found = ACTION_CLOSE in raw_tail
    if action_close_found:
        post_action = raw_tail.split(ACTION_CLOSE, 1)[1]
        if post_action:
            errors.append("unexpected_post_action_text")

    think_match = THINK_RE.search(raw_head)
    if think_match:
        thought = think_match.group(1).strip()
    elif native_think_closed:
        # Qwen chat templates may place the opening think tag in the rendered
        # prompt, leaving only its body and closing tag in generated text.
        thought = raw_head.split(THINK_CLOSE, 1)[0].strip()
    else:
        thought = ""
        errors.append("missing_think_close")

    post_think_start = (
        raw_head.rfind(THINK_CLOSE) + len(THINK_CLOSE)
        if native_think_closed
        else 0
    )
    action_open_start = raw_head.find(ACTION_OPEN, post_think_start)
    if action_open_start < 0:
        hmm_prefix_text = raw_head[post_think_start:]
        pre_action_text = hmm_prefix_text
        errors.append("missing_action_open")
    else:
        action_open_end = action_open_start + len(ACTION_OPEN)
        hmm_prefix_text = raw_head[post_think_start:action_open_end]
        pre_action_text = raw_head[post_think_start:action_open_start]

    decision_match = DECISION_RE.search(pre_action_text)
    decision = decision_match.group(1).strip() if decision_match else ""
    if use_decision:
        decision_format_ok = decision_match is not None and bool(decision)
        if not decision_format_ok:
            errors.append("missing_or_empty_decision")
    else:
        decision_format_ok = pre_action_text.strip() == ""
        if not decision_format_ok:
            errors.append("unexpected_pre_action_text")

    combined = raw_head + raw_tail
    action_match = ACTION_RE.search(combined)
    if action_match:
        action = action_match.group(1).strip()
    else:
        action = raw_tail.split(ACTION_CLOSE, 1)[0].strip()
        if not action_close_found:
            errors.append("missing_action_close")
    if not action:
        errors.append("empty_action")

    return ParsedTurn(
        raw_head=raw_head,
        raw_tail=raw_tail,
        thought=thought,
        decision=decision,
        hmm_prefix_text=hmm_prefix_text,
        action=action,
        native_think_closed=native_think_closed,
        decision_format_ok=decision_format_ok,
        action_open_found=action_open_found,
        action_close_found=action_close_found,
        errors=tuple(errors),
    )


def token_boundary_for_char_offset(tokenizer, token_ids: list[int], offset: int) -> int:
    """Return the token boundary whose decoded character offset is ``offset``.

    Exact alignment matters for distillation: a substring that begins inside a
    BPE token cannot be cropped faithfully. Such samples are rejected rather
    than decoded and retokenized.
    """

    if offset < 0:
        raise ValueError("offset must be non-negative")
    for index in range(len(token_ids) + 1):
        text = tokenizer.decode(token_ids[:index], skip_special_tokens=False)
        if len(text) == offset:
            return index
        if len(text) > offset:
            break
    raise ValueError(f"character offset {offset} is not an exact token boundary")


def hmm_prefix_token_ids(tokenizer, head_token_ids: list[int], raw_head: str) -> list[int]:
    """Extract the exact post-think-through-``<action>`` HMM prefix tokens."""

    decoded = tokenizer.decode(head_token_ids, skip_special_tokens=False)
    if decoded != raw_head:
        raise ValueError("raw_head does not exactly match head_token_ids")
    if THINK_CLOSE not in raw_head or ACTION_OPEN not in raw_head:
        raise ValueError("head must contain </think> and <action>")

    start_char = raw_head.rfind(THINK_CLOSE) + len(THINK_CLOSE)
    action_start = raw_head.find(ACTION_OPEN, start_char)
    if action_start < 0:
        raise ValueError("<action> must occur after </think>")
    end_char = action_start + len(ACTION_OPEN)

    start_token = token_boundary_for_char_offset(tokenizer, head_token_ids, start_char)
    end_token = token_boundary_for_char_offset(tokenizer, head_token_ids, end_char)
    return head_token_ids[start_token:end_token]


def exact_action_tail_token_ids(
    tokenizer, tail_token_ids: list[int], raw_tail: str
) -> tuple[list[int], list[int]]:
    """Split exact generated action-body and ``</action>`` token spans.

    The closer must terminate the decoded tail and both its start and end must
    be token boundaries. A token decoding as ``</action>,`` is therefore
    rejected instead of silently adding the comma to the HMM suffix.
    """

    decoded = tokenizer.decode(tail_token_ids, skip_special_tokens=False)
    if decoded != raw_tail:
        raise ValueError("raw_tail does not exactly match tail_token_ids")
    if not raw_tail.endswith(ACTION_CLOSE):
        raise ValueError("generated tail must end exactly with </action>")
    close_start = len(raw_tail) - len(ACTION_CLOSE)
    body_end_token = token_boundary_for_char_offset(
        tokenizer, tail_token_ids, close_start
    )
    close_end_token = token_boundary_for_char_offset(
        tokenizer, tail_token_ids, len(raw_tail)
    )
    return (
        tail_token_ids[:body_end_token],
        tail_token_ids[body_end_token:close_end_token],
    )
