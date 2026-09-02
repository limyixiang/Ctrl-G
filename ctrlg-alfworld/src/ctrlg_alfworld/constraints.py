"""Finite admissible-action constraints for ALFWorld.

The authoritative language at each step is TextWorld's current
``admissible_commands`` list.  No symbolic state tracker is involved.  A trie
over tokenized commands provides both the pure DFA baseline and the DFA passed
to Ctrl-G's HMM-aware logits processor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import LogitsProcessor

from .prompts import ACTION_CLOSE


def tokenize_continuation(tokenizer, prompt_text: str, continuation: str) -> list[int]:
    """Tokenize a continuation at a fixed prompt boundary.

    Prefer the suffix of joint tokenization when the prompt tokens remain an
    exact prefix.  Some BPE tokenizers merge across the seam; in that case use
    standalone continuation tokens and verify that appending them reconstructs
    the requested decoded suffix.  Original generation token IDs should still
    be used for distillation records; this helper is for finite DFA candidates.
    """

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(
        prompt_text + continuation, add_special_tokens=False
    )
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return list(full_ids[len(prompt_ids) :])

    continuation_ids = list(
        tokenizer.encode(continuation, add_special_tokens=False)
    )
    decoded = tokenizer.decode(
        prompt_ids + continuation_ids, skip_special_tokens=False
    )
    if not decoded.endswith(continuation):
        raise ValueError(
            "Cannot construct a boundary-stable tokenization for continuation "
            f"{continuation!r}"
        )
    return continuation_ids


def build_trie_dfa(token_sequences: Sequence[Sequence[int]], vocab_size: int) -> dict:
    """Build a total Ctrl-G DFA accepting exactly the supplied token paths."""

    if not token_sequences:
        raise ValueError("admissible action set is empty")

    root, dead = 0, 1
    next_node = 2
    children: dict[int, dict[int, int]] = {root: {}}
    accept_states: set[int] = set()

    for sequence in token_sequences:
        if not sequence:
            raise ValueError("empty action token sequence is not allowed")
        node = root
        for token in sequence:
            if token < 0 or token >= vocab_size:
                raise ValueError(f"token id {token} is outside vocabulary")
            target = children.setdefault(node, {}).get(token)
            if target is None:
                target = next_node
                next_node += 1
                children[node][token] = target
                children[target] = {}
            node = target
        accept_states.add(node)

    edges: list[tuple[int, int, np.ndarray]] = []
    for node, outgoing in children.items():
        used = np.zeros(vocab_size, dtype=bool)
        for token, target in outgoing.items():
            token_set = np.zeros(vocab_size, dtype=bool)
            token_set[token] = True
            used[token] = True
            edges.append((node, target, token_set))
        edges.append((node, dead, ~used))
    edges.append((dead, dead, np.ones(vocab_size, dtype=bool)))

    return {
        "edges": edges,
        "initial_state": root,
        "accept_states": accept_states,
    }


def build_action_dfa(
    allowed_actions: Sequence[str], tokenizer, vocab_size: int, prompt_text: str
) -> tuple[dict, dict[str, list[int]]]:
    """Compile the action body language; ``</action>`` is not in this DFA."""

    if not allowed_actions:
        raise ValueError("TextWorld returned no admissible commands")
    token_sequences = {
        action: tokenize_continuation(tokenizer, prompt_text, action)
        for action in allowed_actions
    }
    dfa = build_trie_dfa(list(token_sequences.values()), vocab_size)
    return dfa, token_sequences


def tokenize_fixed_suffix(
    tokenizer, prefix_ids: Sequence[int], suffix: str
) -> list[int]:
    """Tokenize a fixed suffix after already-committed prefix token IDs.

    Joint retokenization is not allowed to rewrite the final action token. The
    returned IDs are appended to the exact action-body path used by the DFA and
    verified to decode as a literal suffix.
    """

    suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
    before = tokenizer.decode(list(prefix_ids), skip_special_tokens=False)
    after = tokenizer.decode(
        list(prefix_ids) + suffix_ids, skip_special_tokens=False
    )
    if after != before + suffix:
        raise ValueError(
            f"fixed suffix {suffix!r} is not compositional at this token boundary"
        )
    return suffix_ids


def dfa_accepts(dfa_graph: dict, token_ids: Sequence[int]) -> bool:
    """Small CPU reference implementation used by tests and diagnostics."""

    transitions: dict[int, list[tuple[int, np.ndarray]]] = {}
    for source, target, token_set in dfa_graph["edges"]:
        transitions.setdefault(source, []).append((target, token_set))

    state = dfa_graph["initial_state"]
    for token in token_ids:
        next_state = None
        for target, token_set in transitions.get(state, []):
            if token_set[token]:
                next_state = target
                break
        if next_state is None:
            return False
        state = next_state
    return state in dfa_graph["accept_states"]


class FiniteActionLogitsProcessor(LogitsProcessor):
    """Hard-mask generation to a finite set of action-plus-closer token paths.

    This is the no-HMM baseline.  It uses the same admissible command strings
    as the Ctrl-G condition but contributes no learned future-mass estimate.
    """

    def __init__(
        self,
        tokenizer,
        prompt_text: str,
        allowed_actions: Sequence[str],
        prompt_length: int,
        eos_token_id: int,
    ):
        if not allowed_actions:
            raise ValueError("TextWorld returned no admissible commands")
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        action_sequences = {
            action: tokenize_continuation(tokenizer, prompt_text, action)
            for action in allowed_actions
        }
        self.paths = []
        for action in allowed_actions:
            body_ids = action_sequences[action]
            suffix_ids = tokenize_fixed_suffix(
                tokenizer, prompt_ids + body_ids, ACTION_CLOSE
            )
            self.paths.append(tuple(body_ids + suffix_ids))
        if any(not path for path in self.paths):
            raise ValueError("empty constrained continuation")

    def __call__(self, input_ids, scores):
        masked = torch.full_like(scores, -torch.inf)
        for row_index, row in enumerate(input_ids):
            prefix = tuple(row[self.prompt_length :].tolist())
            allowed_next: set[int] = set()
            complete = False
            for path in self.paths:
                if len(prefix) <= len(path) and path[: len(prefix)] == prefix:
                    if len(prefix) == len(path):
                        complete = True
                    else:
                        allowed_next.add(path[len(prefix)])
            if complete:
                allowed_next.add(self.eos_token_id)
            if not allowed_next:
                # This should be unreachable after hard masking. EOS is safer
                # than returning an all-negative-infinity row to generate().
                allowed_next.add(self.eos_token_id)
            indices = torch.tensor(
                sorted(allowed_next), dtype=torch.long, device=scores.device
            )
            masked[row_index, indices] = scores[row_index, indices]
        return masked
