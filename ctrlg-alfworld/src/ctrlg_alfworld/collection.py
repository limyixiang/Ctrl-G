"""Lossless attempt records for a context-conditioned, joint state/action HMM."""

from .hmm_context import HMM_CONTEXT_FORMAT, HMM_SEQUENCE_FORMAT

ROLLOUT_SAMPLE_FORMAT = "unmasked_state_action_attempt_v1"


def executable_attempt(turn):
    """Use public structure only; environment admissibility is never consulted."""
    return (turn.state_stop_found and not turn.used_state_repair
            and turn.state_schema_valid and turn.tail_stop_found
            and turn.tail_span_exact and turn.action_grammar_valid
            and bool(turn.parsed.action))


def observed_attempt_span(turn, eos_token_id):
    """Save the exact target independently of the future HMM context window."""
    target = list(turn.hmm_target_token_ids)
    observed_eos = bool(target and target[-1] == eos_token_id)
    protocol_complete = bool(turn.state_stop_found and turn.tail_stop_found
                             and turn.tail_span_exact and not turn.used_state_repair)
    appended_eos = protocol_complete and not observed_eos
    if appended_eos:
        target.append(eos_token_id)
    # Fixed openers alone provide no sampled state/action evidence.
    eligible = bool(turn.state_token_ids or turn.tail_token_ids)
    return {
        "rollout_sample_format": ROLLOUT_SAMPLE_FORMAT,
        "hmm_target_token_ids": target,
        "hmm_target_start_in_head": turn.hmm_target_start_in_head,
        "hmm_action_start_in_target": (len(turn.head_token_ids) - turn.hmm_target_start_in_head
                                       if not turn.used_state_repair else None),
        "hmm_completed": observed_eos or protocol_complete,
        "hmm_protocol_complete": protocol_complete,
        "hmm_appended_eos": appended_eos,
        "distill_eligible": eligible,
        "distill_exclusion_reasons": [] if eligible else ["no_observed_target_tokens"],
        "selected_for_execution": executable_attempt(turn),
    }


def attach_hmm_context(record, context):
    """Construct a training sequence without changing the observed target."""
    target = record["hmm_target_token_ids"]
    if "hmm_action_start_in_target" in record:
        action_start = record["hmm_action_start_in_target"]
    else:
        # Reuse older unmasked collections that saved a particular context.
        action_start = record["hmm_action_start"]
        if action_start is not None:
            action_start -= len(record["hmm_context_token_ids"])
    return {
        **record,
        "hmm_sequence_format": HMM_SEQUENCE_FORMAT,
        "hmm_context_format": HMM_CONTEXT_FORMAT,
        "hmm_context_text": context.text,
        "hmm_context_token_ids": list(context.token_ids),
        "hmm_context_truncated": context.truncated,
        "hmm_context_history_turns": context.history_turns,
        "hmm_action_start": None if action_start is None else len(context.token_ids) + action_start,
        "hmm_sequence_token_ids": list(context.token_ids) + target,
        "hmm_sequence_length": len(context.token_ids) + len(target),
    }


def training_span(turn, context, eos_token_id):
    return attach_hmm_context(observed_attempt_span(turn, eos_token_id), context)
