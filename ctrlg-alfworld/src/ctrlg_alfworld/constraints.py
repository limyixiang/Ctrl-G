"""Regular-language compilation and hard token-DFA decoding.

The languages in this module depend only on ``SKILLS.md`` and on values the
model emitted in its decision block. Environment admissible commands are never
an input to these compilers.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import interegular
import numpy as np
import yaml
from interegular.fsm import anything_else

from .prompts import ACTION_CLOSE, DECISION_CLOSE
from .skills import DecisionSchema, PLACEHOLDER_RE, PolicyRule, SkillSet

ATOM_PATTERN = r"[a-z]+"
ENTITY_PATTERN = r"[a-z]+ [1-9][0-9]*"
_TOKEN_TEXT_CACHE: dict[tuple, dict[int, str]] = {}


def _literal(value: Any) -> str:
    # Python escapes literal spaces as ``\ ``, but vLLM's Rust-style regex
    # converter warns about that nonstandard escape on every request. A plain
    # space is literal in both regex dialects.
    return re.escape(str(value)).replace(r"\ ", " ")


def _choice(patterns: Sequence[str]) -> str:
    if not patterns:
        return r"(?!)"
    if len(patterns) == 1:
        return patterns[0]
    return "(?:" + "|".join(patterns) + ")"


def action_body_pattern(skillset: SkillSet, *, take_type: str | None = None) -> str:
    """Builds one regex describing every action allowed by the action templates in skills.md"""
    patterns = []
    for action in skillset.actions:
        template = action.template
        cursor = 0
        pieces: list[str] = []
        for match in PLACEHOLDER_RE.finditer(template):
            pieces.append(_literal(template[cursor : match.start()]))
            if action.name == "take" and match.group(1) == "obj" and take_type:
                pieces.append(_literal(take_type) + r" [1-9][0-9]*")
            else:
                pieces.append(ENTITY_PATTERN)
            cursor = match.end()
        pieces.append(_literal(template[cursor:]))
        patterns.append("".join(pieces))
    return _choice(patterns)


def _schema_node_pattern(node: dict[str, Any], skillset: SkillSet) -> str:
    schema_type = node["type"]
    if schema_type == "literal":
        return _literal(node["value"])
    if schema_type == "enum":
        return _choice([_literal(value) for value in node["values"]])
    if schema_type == "lowercase_atom":
        return ATOM_PATTERN
    if schema_type == "numbered_entity":
        return _choice(
            [ENTITY_PATTERN]
            + [_literal(value) for value in node.get("alternatives", [])]
        )
    if schema_type == "action":
        return _choice(
            [action_body_pattern(skillset)]
            + [_literal(value) for value in node.get("alternatives", [])]
        )
    if schema_type == "record":
        return _fields_pattern(node["fields"], skillset)
    if schema_type == "bounded_list":
        item = _schema_node_pattern(node["item"], skillset)
        minimum = node.get("min_items", 0)
        maximum = node["max_items"]
        if maximum == 0:
            return r"\[\]"
        if minimum == 0:
            content = rf"(?:{item}(?:, {item}){{0,{maximum - 1}}})?"
        else:
            required_tail = minimum - 1
            optional_tail = maximum - minimum
            content = rf"{item}(?:, {item}){{{required_tail},{required_tail + optional_tail}}}"
        return rf"\[{content}\]"
    raise ValueError(f"unsupported schema type {schema_type!r}")


def _fields_pattern(fields: Sequence[dict[str, Any]], skillset: SkillSet) -> str:
    pairs = [
        _literal(field["name"] + ": ")
        + _schema_node_pattern(field["schema"], skillset)
        for field in fields
    ]
    return r"\{" + ", ".join(pairs) + r"\}"


def decision_body_pattern(schema: DecisionSchema, skillset: SkillSet) -> str:
    return _fields_pattern(schema.fields, skillset)


def decision_span_pattern(schema: DecisionSchema, skillset: SkillSet) -> str:
    return decision_body_pattern(schema, skillset) + _literal(DECISION_CLOSE)


def action_span_pattern(skillset: SkillSet, *, take_type: str | None = None) -> str:
    return action_body_pattern(skillset, take_type=take_type) + _literal(ACTION_CLOSE)


def regex_fsm(pattern: str):
    try:
        return interegular.parse_pattern(pattern).to_fsm()
    except Exception as exc:
        raise ValueError(f"cannot compile regular language {pattern!r}: {exc}") from exc


def _fsm_step(fsm, state, text: str):
    for character in text:
        transition = fsm.alphabet[character]
        if transition is None or transition not in fsm.map.get(state, {}):
            return None
        state = fsm.map[state][transition]
    return state


def _token_text(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def lift_character_fsm(fsm, tokenizer, vocab_size: int | None = None) -> dict:
    """Lift an interegular character FSM to a total token DFA.

    Empty-decoding and special tokens are deliberately routed to the dead
    state. EOS is supplied separately by the generation machinery.
    """

    size = len(tokenizer) if vocab_size is None else vocab_size
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    cache_key = (
        getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
        size,
        tuple(sorted(token_id for token_id in special_ids if token_id < size)),
    )
    token_texts = _TOKEN_TEXT_CACHE.get(cache_key)
    if token_texts is None:
        ordinary_ids = [token_id for token_id in range(size) if token_id not in special_ids]
        if hasattr(tokenizer, "batch_decode"):
            decoded_tokens = []
            for start in range(0, len(ordinary_ids), 4096):
                batch = [[token_id] for token_id in ordinary_ids[start : start + 4096]]
                try:
                    decoded_tokens.extend(tokenizer.batch_decode(
                        batch, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ))
                except TypeError:
                    decoded_tokens.extend(tokenizer.batch_decode(
                        batch, skip_special_tokens=False,
                    ))
            token_texts = dict(zip(ordinary_ids, decoded_tokens))
        else:
            token_texts = {
                token_id: _token_text(tokenizer, token_id) for token_id in ordinary_ids
            }
        token_texts = {key: value for key, value in token_texts.items() if value}
        _TOKEN_TEXT_CACHE[cache_key] = token_texts
    tokens_by_first: dict[str, list[int]] = {}
    for token_id, text in token_texts.items():
        tokens_by_first.setdefault(text[0], []).append(token_id)

    char_to_token_state: dict[Any, int] = {fsm.initial: 0}
    queue = deque([fsm.initial])
    transition_rows: dict[int, dict[int, int]] = {}
    while queue:
        char_state = queue.popleft()
        source = char_to_token_state[char_state]
        row: dict[int, int] = {}
        candidate_ids: set[int] = set()
        outgoing_keys = fsm.map.get(char_state, {})
        for transition_key in outgoing_keys:
            symbols = fsm.alphabet.by_transition.get(transition_key, ())
            if anything_else in symbols:
                candidate_ids.update(token_texts)
                break
            for symbol in symbols:
                candidate_ids.update(tokens_by_first.get(symbol, ()))
        for token_id in candidate_ids:
            text = token_texts[token_id]
            target_char = _fsm_step(fsm, char_state, text)
            if target_char is None:
                continue
            if target_char not in char_to_token_state:
                char_to_token_state[target_char] = len(char_to_token_state)
                queue.append(target_char)
            row[token_id] = char_to_token_state[target_char]
        transition_rows[source] = row

    dead = len(char_to_token_state)
    edges: list[tuple[int, int, np.ndarray]] = []
    for source in range(dead):
        grouped: dict[int, list[int]] = {}
        for token_id, target in transition_rows.get(source, {}).items():
            grouped.setdefault(target, []).append(token_id)
        used = np.zeros(size, dtype=bool)
        for target, token_ids in grouped.items():
            token_set = np.zeros(size, dtype=bool)
            token_set[token_ids] = True
            used[token_ids] = True
            edges.append((source, target, token_set))
        edges.append((source, dead, ~used))
    edges.append((dead, dead, np.ones(size, dtype=bool)))
    accepting = {
        token_state
        for char_state, token_state in char_to_token_state.items()
        if char_state in fsm.finals
    }
    graph = {
        "edges": edges,
        "initial_state": 0,
        "accept_states": accepting,
        "dead_state": dead,
        "vocab_size": size,
    }
    graph["min_tokens_to_accept"] = _min_tokens_to_accept(graph)
    if graph["min_tokens_to_accept"].get(0) is None:
        raise ValueError("tokenizer cannot realize any string in the language")
    return graph


def _transition_table(graph: dict) -> list[np.ndarray]:
    cached = graph.get("transition_table")
    if cached is not None:
        return cached
    count = max(max(source, target) for source, target, _ in graph["edges"]) + 1
    dtype = np.uint16 if count <= np.iinfo(np.uint16).max else np.uint32
    dead = graph.get("dead_state", count - 1)
    table = [np.full(graph["vocab_size"], dead, dtype=dtype) for _ in range(count)]
    for source, target, token_set in graph["edges"]:
        table[source][token_set] = target
    graph["transition_table"] = table
    return table


def _min_tokens_to_accept(graph: dict) -> dict[int, int]:
    reverse: dict[int, set[int]] = {}
    dead = graph.get("dead_state")
    for source, target, token_set in graph["edges"]:
        if target != dead and token_set.any():
            reverse.setdefault(target, set()).add(source)
    distance = {state: 0 for state in graph["accept_states"]}
    queue = deque(distance)
    while queue:
        target = queue.popleft()
        for source in reverse.get(target, ()):
            if source not in distance:
                distance[source] = distance[target] + 1
                queue.append(source)
    return distance


def build_trie_dfa(token_sequences: Sequence[Sequence[int]], vocab_size: int) -> dict:
    if not token_sequences:
        raise ValueError("token language is empty")
    root, dead, next_node = 0, 1, 2
    children: dict[int, dict[int, int]] = {root: {}}
    accepting: set[int] = set()
    for sequence in token_sequences:
        if not sequence:
            raise ValueError("empty token sequence is not allowed")
        node = root
        for token in sequence:
            if not 0 <= token < vocab_size:
                raise ValueError(f"token id {token} is outside vocabulary")
            target = children.setdefault(node, {}).get(token)
            if target is None:
                target, next_node = next_node, next_node + 1
                children[node][token] = target
                children[target] = {}
            node = target
        accepting.add(node)
    edges = []
    for source, outgoing in children.items():
        used = np.zeros(vocab_size, dtype=bool)
        for token, target in outgoing.items():
            token_set = np.zeros(vocab_size, dtype=bool)
            token_set[token] = True
            used[token] = True
            edges.append((source, target, token_set))
        edges.append((source, dead, ~used))
    edges.append((dead, dead, np.ones(vocab_size, dtype=bool)))
    graph = {
        "edges": edges, "initial_state": root, "accept_states": accepting,
        "dead_state": dead, "vocab_size": vocab_size,
    }
    graph["min_tokens_to_accept"] = _min_tokens_to_accept(graph)
    return graph


def tokenize_continuation(tokenizer, prompt_text: str, continuation: str) -> list[int]:
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + continuation, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return list(full_ids[len(prompt_ids):])
    continuation_ids = list(tokenizer.encode(continuation, add_special_tokens=False))
    if tokenizer.decode(prompt_ids + continuation_ids, skip_special_tokens=False).endswith(continuation):
        return continuation_ids
    raise ValueError(f"cannot tokenize continuation at prompt boundary: {continuation!r}")


def tokenize_fixed_suffix(tokenizer, prefix_ids: Sequence[int], suffix: str) -> list[int]:
    suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
    before = tokenizer.decode(list(prefix_ids), skip_special_tokens=False)
    after = tokenizer.decode(list(prefix_ids) + suffix_ids, skip_special_tokens=False)
    if after != before + suffix:
        raise ValueError(f"fixed suffix {suffix!r} is not compositional")
    return suffix_ids


def dfa_accepts(graph: dict, token_ids: Sequence[int]) -> bool:
    table = _transition_table(graph)
    state = graph["initial_state"]
    for token in token_ids:
        if not 0 <= token < graph["vocab_size"]:
            return False
        state = int(table[state][token])
        if state < 0 or state == graph.get("dead_state"):
            return False
    return state in graph["accept_states"]


class HardDFALogitsProcessor:
    """Hard mask with token-budget reachability to an accepting state."""

    def __init__(self, graph: dict, prompt_length: int, max_new_tokens: int, eos_token_id: int):
        self.graph = graph
        self.table = _transition_table(graph)
        self.prompt_length = prompt_length
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        minimum = graph["min_tokens_to_accept"].get(graph["initial_state"])
        if minimum is None or minimum > max_new_tokens:
            raise ValueError("DFA cannot reach acceptance within the token budget")

    def __call__(self, input_ids, scores):
        import torch

        masked = torch.full_like(scores, -torch.inf)
        dead = self.graph.get("dead_state")
        distances = self.graph["min_tokens_to_accept"]
        distance_vector = np.full(
            len(self.table), self.max_new_tokens + 1, dtype=np.int32
        )
        for state, distance in distances.items():
            distance_vector[state] = distance
        for row_index, row in enumerate(input_ids):
            generated = row[self.prompt_length:].tolist()
            state = self.graph["initial_state"]
            for token in generated:
                state = int(self.table[state][token])
            if state in self.graph["accept_states"]:
                masked[row_index, self.eos_token_id] = scores[row_index, self.eos_token_id]
                continue
            remaining_after = self.max_new_tokens - len(generated) - 1
            targets = self.table[state]
            allowed = np.flatnonzero(
                (targets != dead) & (distance_vector[targets] <= remaining_after)
            )
            if allowed.size == 0:
                raise RuntimeError("DFA has no accepting continuation within budget")
            indices = torch.as_tensor(allowed, device=scores.device, dtype=torch.long)
            masked[row_index, indices] = scores[row_index, indices]
        return masked


def parse_decision(decision_text: str, schema: DecisionSchema, skillset: SkillSet) -> dict[str, Any]:
    if re.fullmatch(decision_body_pattern(schema, skillset), decision_text) is None:
        raise ValueError(f"decision does not match the {schema.task!r} schema")
    value = yaml.safe_load(decision_text)
    if not isinstance(value, dict):
        raise ValueError("decision is not a mapping")
    return value


def _condition_matches(condition: dict[str, Any], decision: dict[str, Any]) -> bool:
    value = decision.get(condition["field"])
    if "equals" in condition:
        return value == condition["equals"]
    if "not_in" in condition:
        return value not in condition["not_in"]
    if "is_type" in condition:
        return isinstance(value, str) and re.fullmatch(ENTITY_PATTERN, value) is not None
    raise ValueError("unsupported policy condition")


@dataclass(frozen=True)
class PolicyCompilation:
    fsm: Any
    pattern: str
    activated: tuple[str, ...]
    shadowed: tuple[str, ...]
    evaluable: bool
    fallback_reason: str | None
    required_action: str | None = None
    forbidden_action: str | None = None
    take_type: str | None = None


def _bind_required_action(rule: PolicyRule, decision: dict[str, Any], skillset: SkillSet) -> str:
    spec = rule.effect["require_action"]
    action_template = next(
        action for action in skillset.actions if action.name == spec["action"]
    )
    values = {}
    for placeholder, field_name in spec["bindings"].items():
        if placeholder not in action_template.placeholders:
            raise ValueError(f"policy {rule.name} binds unknown placeholder {placeholder}")
        value = decision.get(field_name)
        if not isinstance(value, str) or re.fullmatch(ENTITY_PATTERN, value) is None:
            raise ValueError(f"policy {rule.name} cannot bind {field_name!r}")
        values[placeholder] = value
    if set(values) != set(action_template.placeholders):
        raise ValueError(f"policy {rule.name} does not bind every action placeholder")
    return action_template.template.format(**values)


def compile_policy_language(decision: dict[str, Any], skillset: SkillSet) -> PolicyCompilation:
    generic_pattern = action_span_pattern(skillset)
    generic_fsm = regex_fsm(generic_pattern)
    matching = [
        rule for rule in skillset.policies
        if all(_condition_matches(condition, decision) for condition in rule.when)
    ]
    required = [rule for rule in matching if "require_action" in rule.effect]
    if required:
        winner = required[0]
        shadowed = tuple(rule.name for rule in matching if rule is not winner and rule.priority < winner.priority)
        try:
            action = _bind_required_action(winner, decision, skillset)
            pattern = _literal(action + ACTION_CLOSE)
            fsm = regex_fsm(pattern)
            if (fsm & generic_fsm).empty():
                raise ValueError("required action is outside the generic action grammar")
            return PolicyCompilation(
                fsm=fsm, pattern=pattern, activated=(winner.name,), shadowed=shadowed,
                evaluable=True, fallback_reason=None, required_action=action,
            )
        except ValueError as exc:
            return PolicyCompilation(
                fsm=generic_fsm, pattern=generic_pattern,
                activated=tuple(rule.name for rule in matching), shadowed=shadowed,
                evaluable=False, fallback_reason=f"unbound_required_action: {exc}",
            )

    active_names = tuple(rule.name for rule in matching)
    take_type = None
    forbidden = None
    try:
        for rule in matching:
            if "restrict_take_type" in rule.effect:
                field_name = rule.effect["restrict_take_type"]["field"]
                value = decision.get(field_name)
                if not isinstance(value, str) or re.fullmatch(ATOM_PATTERN, value) is None:
                    raise ValueError(f"policy {rule.name} cannot bind {field_name!r}")
                take_type = value
            elif "forbid_exact_action" in rule.effect:
                field_name = rule.effect["forbid_exact_action"]["field"]
                value = decision.get(field_name)
                if not isinstance(value, str):
                    raise ValueError(f"policy {rule.name} cannot bind {field_name!r}")
                forbidden = value
        pattern = action_span_pattern(skillset, take_type=take_type)
        fsm = regex_fsm(pattern)
        if forbidden is not None:
            fsm = fsm - regex_fsm(_literal(forbidden + ACTION_CLOSE))
        if fsm.empty():
            raise ValueError("policy intersection is empty")
        return PolicyCompilation(
            fsm=fsm, pattern=pattern, activated=active_names, shadowed=(),
            evaluable=True, fallback_reason=None, forbidden_action=forbidden,
            take_type=take_type,
        )
    except ValueError as exc:
        return PolicyCompilation(
            fsm=generic_fsm, pattern=generic_pattern, activated=active_names,
            shadowed=(), evaluable=False,
            fallback_reason=f"unsatisfiable_policy_language: {exc}",
        )


def action_grammar_valid(action: str, skillset: SkillSet) -> bool:
    return re.fullmatch(action_body_pattern(skillset), action) is not None


def policy_satisfied(action: str, compilation: PolicyCompilation) -> bool | None:
    if not compilation.evaluable:
        return None
    return compilation.fsm.accepts(action + ACTION_CLOSE)
