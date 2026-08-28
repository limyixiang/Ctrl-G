"""Compile a finite set of allowed action strings into a Ctrl-G DFA.

At each step the admissible actions form a finite set of strings, so the
tightest constraint is a token-level trie DFA that accepts exactly those token
sequences. This avoids general regex->token-DFA machinery entirely: SKILLS.md
regex/templates are grounded against the tracked state *first* (skills.py),
and only the resulting finite string set is compiled here.

DFA graph format (see Ctrl-G ctrlg/dfa.py):
    {'edges': [(u, v, np.bool_[vocab_size]), ...],
     'initial_state': s0, 'accept_states': {s, ...}}
"""

from __future__ import annotations

import numpy as np


def tokenize_continuation(tokenizer, prompt_text: str, continuation: str) -> list[int]:
    """Token ids of `continuation` exactly as it would be tokenized when
    generated after `prompt_text` (robust to BPE boundary merges)."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + continuation, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Boundary merge across the prompt/continuation seam. Find the longest
        # common prefix and treat the remainder as the continuation.
        i = 0
        while i < min(len(prompt_ids), len(full_ids)) and prompt_ids[i] == full_ids[i]:
            i += 1
        return full_ids[i:]
    return full_ids[len(prompt_ids):]


def build_trie_dfa(token_seqs: list[list[int]], vocab_size: int) -> dict:
    """Exact-match DFA over a finite set of token sequences.

    Total DFA: a sink 'dead' state absorbs every token not on a trie edge
    (mirrors the style of Ctrl-G's built-in builders; the logits processor
    never actually takes those edges since they lead to non-accepting sink).
    """
    if not token_seqs:
        raise ValueError("No allowed token sequences - constraint would be UNSAT.")

    # trie construction: node 0 = root, node 1 = dead sink
    ROOT, DEAD = 0, 1
    next_id = 2
    children: dict[int, dict[int, int]] = {ROOT: {}}
    accept: set[int] = set()

    for seq in token_seqs:
        if not seq:
            accept.add(ROOT)
            continue
        node = ROOT
        for tok in seq:
            nxt = children.setdefault(node, {}).get(tok)
            if nxt is None:
                nxt = next_id
                next_id += 1
                children[node][tok] = nxt
                children.setdefault(nxt, {})
            node = nxt
        accept.add(node)

    edges = []
    for node, kids in children.items():
        # group child transitions by target state (one bitset per (u, v))
        by_target: dict[int, list[int]] = {}
        for tok, tgt in kids.items():
            by_target.setdefault(tgt, []).append(tok)
        used = np.zeros((vocab_size,), dtype=bool)
        for tgt, toks in by_target.items():
            tset = np.zeros((vocab_size,), dtype=bool)
            tset[toks] = True
            used |= tset
            edges.append((node, tgt, tset))
        # everything else falls into the dead sink
        rest = ~used
        if rest.any():
            edges.append((node, DEAD, rest))
    edges.append((DEAD, DEAD, np.ones((vocab_size,), dtype=bool)))

    return {"edges": edges, "initial_state": ROOT, "accept_states": accept}


def dfa_accepts(dfa_graph: dict, seq: list[int]) -> bool:
    """CPU reference: does the DFA accept this token sequence? (for tests)"""
    trans = {}
    for u, v, tset in dfa_graph["edges"]:
        trans.setdefault(u, []).append((v, tset))
    state = dfa_graph["initial_state"]
    for tok in seq:
        nxt = None
        for v, tset in trans.get(state, []):
            if tset[tok]:
                nxt = v
                break
        if nxt is None:
            return False
        state = nxt
    return state in dfa_graph["accept_states"]


def build_step_dfa(
    allowed_actions: list[str],
    tokenizer,
    vocab_size: int,
    prompt_tail: str = "<tool>",
):
    """Compile this step's allowed actions into (dfa_graph, action->token_seq).

    `prompt_tail` is the text immediately preceding the constrained span
    (used for boundary-correct tokenization of each action).
    """
    seqs = {}
    for action in allowed_actions:
        seqs[action] = tokenize_continuation(tokenizer, prompt_tail, action)
    dfa_graph = build_trie_dfa(list(seqs.values()), vocab_size)
    return dfa_graph, seqs
