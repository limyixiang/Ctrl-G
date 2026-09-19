from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ACTION_BLOCK_RE = re.compile(r"```action\s*\n(.*?)```", re.DOTALL)
SCHEMA_BLOCK_RE = re.compile(r"```decision-schema\s*\n(.*?)```", re.DOTALL)
POLICY_BLOCK_RE = re.compile(r"```policy\s*\n(.*?)```", re.DOTALL)
PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}") # {obj} or {recep}
PRECOND_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\((.*)\))?$")

SCHEMA_VERSION = 1
POLICY_VERSION = 1
TASK_KEYS = {"put", "clean", "heat", "cool", "examine", "puttwo"}
SUPPORTED_SCHEMA_TYPES = {
    "literal", "enum", "lowercase_atom", "numbered_entity", "action",
    "record", "bounded_list",
}
SUPPORTED_CONDITION_OPERATORS = {"equals", "is_type", "not_in"}
SUPPORTED_EFFECTS = {"require_action", "restrict_take_type", "forbid_exact_action"}


def _load_mapping(block: str, kind: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed {kind} YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{kind} block must be a YAML mapping")
    return value


@dataclass(frozen=True)
class Precondition:
    pred: str
    args: tuple[str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> "Precondition":
        match = PRECOND_RE.match(text.strip())
        if match is None:
            raise ValueError(f"Cannot parse precondition: {text!r}")
        predicate, argument_text = match.group(1), match.group(2)
        arguments = () if not argument_text else tuple(
            item.strip() for item in argument_text.split(",")
        )
        return cls(pred=predicate, args=arguments)


@dataclass(frozen=True)
class Action:
    name: str
    template: str
    preconditions: tuple[Precondition, ...] = ()

    @property
    def placeholders(self) -> list[str]:
        return PLACEHOLDER_RE.findall(self.template)


@dataclass(frozen=True)
class DecisionSchema:
    task: str
    version: int
    fields: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PolicyRule:
    name: str
    version: int
    priority: int
    when: tuple[dict[str, Any], ...]
    effect: dict[str, Any]


def _validate_schema_node(node: Any, path: str) -> None:
    if not isinstance(node, dict):
        raise ValueError(f"{path} must be a schema mapping")
    schema_type = node.get("type")
    if schema_type not in SUPPORTED_SCHEMA_TYPES:
        raise ValueError(f"{path} has unsupported schema type {schema_type!r}")
    allowed_keys = {
        "literal": {"type", "value"},
        "enum": {"type", "values"},
        "lowercase_atom": {"type"},
        "numbered_entity": {"type", "alternatives"},
        "action": {"type", "alternatives"},
        "record": {"type", "fields"},
        "bounded_list": {"type", "item", "min_items", "max_items"},
    }[schema_type]
    extra = set(node) - allowed_keys
    if extra:
        raise ValueError(f"{path} has unsupported keys: {sorted(extra)}")
    if schema_type == "literal":
        if not isinstance(node.get("value"), (str, int)):
            raise ValueError(f"{path}.value must be a string or integer")
    elif schema_type == "enum":
        values = node.get("values")
        if not isinstance(values, list) or not values or not all(
            isinstance(value, (str, int)) for value in values
        ):
            raise ValueError(f"{path}.values must be a nonempty scalar list")
    elif schema_type in {"numbered_entity", "action"}:
        alternatives = node.get("alternatives", [])
        if not isinstance(alternatives, list) or not all(
            isinstance(value, str) for value in alternatives
        ):
            raise ValueError(f"{path}.alternatives must be a string list")
    elif schema_type == "record":
        _validate_fields(node.get("fields"), f"{path}.fields")
    elif schema_type == "bounded_list":
        minimum = node.get("min_items", 0)
        maximum = node.get("max_items")
        if not isinstance(minimum, int) or not isinstance(maximum, int):
            raise ValueError(f"{path} list bounds must be integers")
        if minimum < 0 or maximum < minimum or maximum > 64:
            raise ValueError(f"{path} has invalid list bounds {minimum}..{maximum}")
        _validate_schema_node(node.get("item"), f"{path}.item")


def _validate_fields(fields: Any, path: str) -> None:
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"{path} must be a nonempty ordered list")
    names: set[str] = set()
    for index, field_spec in enumerate(fields):
        item_path = f"{path}[{index}]"
        if not isinstance(field_spec, dict) or set(field_spec) != {"name", "schema"}:
            raise ValueError(f"{item_path} must contain exactly name and schema")
        name = field_spec["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
            raise ValueError(f"{item_path}.name is invalid")
        if name in names:
            raise ValueError(f"duplicate field {name!r} in {path}")
        names.add(name)
        _validate_schema_node(field_spec["schema"], f"{item_path}.schema")


def _validate_task_schema_contract(task: str, fields: list[dict[str, Any]]) -> None:
    by_name = {field["name"]: field["schema"] for field in fields}
    if fields[0]["name"] != "task" or fields[0]["schema"] != {
        "type": "literal", "value": task
    }:
        raise ValueError(f"decision schema {task!r} must start with its task literal")
    required = {
        "previous_action": "action",
        "arrived_receptacle_state": "enum",
        "held_relation": "enum",
    }
    for name, schema_type in required.items():
        if by_name.get(name, {}).get("type") != schema_type:
            raise ValueError(f"decision schema {task!r} requires {name}: {schema_type}")
    search = by_name.get("search", {})
    if search.get("type") != "record":
        raise ValueError(f"decision schema {task!r} requires a search record")
    search_fields = {item["name"]: item["schema"] for item in search["fields"]}
    for name in ("locations_searched", "locations_to_search"):
        spec = search_fields.get(name, {})
        if spec.get("type") != "bounded_list" or spec.get("max_items") != 64:
            raise ValueError(f"decision schema {task!r} requires {name} max_items: 64")
    if task == "puttwo":
        targets = by_name.get("targets", {})
        if (
            targets.get("type") != "bounded_list"
            or targets.get("min_items") != 2
            or targets.get("max_items") != 2
            or targets.get("item", {}).get("type") != "record"
        ):
            raise ValueError("puttwo targets must be exactly two independent records")


def _validate_condition(condition: Any, path: str) -> None:
    if not isinstance(condition, dict) or not isinstance(condition.get("field"), str):
        raise ValueError(f"{path} must be a mapping with a field")
    operators = set(condition) - {"field"}
    if len(operators) != 1:
        raise ValueError(f"{path} must contain exactly one condition operator")
    operator = next(iter(operators))
    if operator not in SUPPORTED_CONDITION_OPERATORS:
        raise ValueError(f"{path} has unsupported operator {operator!r}")
    if operator == "not_in" and not isinstance(condition[operator], list):
        raise ValueError(f"{path}.not_in must be a list")
    if operator == "is_type" and condition[operator] != "numbered_entity":
        raise ValueError(f"{path} supports only is_type: numbered_entity")


def _validate_effect(effect: Any, action_names: set[str], path: str) -> None:
    if not isinstance(effect, dict) or len(effect) != 1:
        raise ValueError(f"{path} must contain exactly one effect")
    effect_name, effect_spec = next(iter(effect.items()))
    if effect_name not in SUPPORTED_EFFECTS:
        raise ValueError(f"{path} has unsupported effect {effect_name!r}")
    if not isinstance(effect_spec, dict):
        raise ValueError(f"{path}.{effect_name} must be a mapping")
    if effect_name == "require_action":
        if set(effect_spec) != {"action", "bindings"}:
            raise ValueError(f"{path}.require_action requires action and bindings")
        if effect_spec["action"] not in action_names:
            raise ValueError(
                f"{path}.require_action references unknown action template "
                f"{effect_spec['action']!r}"
            )
        if not isinstance(effect_spec["bindings"], dict):
            raise ValueError(f"{path}.require_action.bindings must be a mapping")
    elif set(effect_spec) != {"field"} or not isinstance(effect_spec["field"], str):
        raise ValueError(f"{path}.{effect_name} requires one field")


@dataclass(frozen=True)
class SkillSet:
    actions: tuple[Action, ...]
    decision_schemas: dict[str, DecisionSchema]
    policies: tuple[PolicyRule, ...]
    raw_markdown: str
    schema_version: int = SCHEMA_VERSION
    policy_version: int = POLICY_VERSION

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.raw_markdown.encode("utf-8")).hexdigest()

    @classmethod
    def from_file(cls, path: str | Path) -> "SkillSet":
        raw = Path(path).read_text(encoding="utf-8")
        actions: list[Action] = []
        action_names: set[str] = set()
        for index, block in enumerate(ACTION_BLOCK_RE.findall(raw)):
            spec = _load_mapping(block, f"action[{index}]")
            if set(spec) - {"name", "template", "preconditions"}:
                raise ValueError(f"action[{index}] contains unsupported keys")
            name, template = spec.get("name"), spec.get("template")
            if not isinstance(name, str) or not isinstance(template, str):
                raise ValueError(f"action[{index}] requires string name and template")
            if name in action_names:
                raise ValueError(f"duplicate action template {name!r}")
            placeholders = PLACEHOLDER_RE.findall(template)
            if len(placeholders) != len(set(placeholders)):
                raise ValueError(f"action template {name!r} repeats a placeholder")
            action_names.add(name)
            actions.append(Action(
                name=name,
                template=template,
                preconditions=tuple(
                    Precondition.parse(value)
                    for value in (spec.get("preconditions") or [])
                ),
            ))
        if not actions:
            raise ValueError(f"No ```action blocks found in {path}")

        schemas: dict[str, DecisionSchema] = {}
        for index, block in enumerate(SCHEMA_BLOCK_RE.findall(raw)):
            spec = _load_mapping(block, f"decision-schema[{index}]")
            if set(spec) != {"version", "task", "fields"}:
                raise ValueError(
                    f"decision-schema[{index}] must contain version, task, and fields"
                )
            if spec["version"] != SCHEMA_VERSION:
                raise ValueError(f"unsupported decision schema version {spec['version']!r}")
            task = spec["task"]
            if not isinstance(task, str):
                raise ValueError(f"decision-schema[{index}].task must be a string")
            if task in schemas:
                raise ValueError(f"duplicate decision schema {task!r}")
            _validate_fields(spec["fields"], f"decision-schema[{task}].fields")
            _validate_task_schema_contract(task, spec["fields"])
            schemas[task] = DecisionSchema(task, SCHEMA_VERSION, tuple(spec["fields"]))
        if not schemas:
            raise ValueError(f"No ```decision-schema blocks found in {path}")
        if set(schemas) != TASK_KEYS:
            raise ValueError(
                "decision schemas must define exactly: " + ", ".join(sorted(TASK_KEYS))
            )

        policies: list[PolicyRule] = []
        policy_names: set[str] = set()
        for index, block in enumerate(POLICY_BLOCK_RE.findall(raw)):
            spec = _load_mapping(block, f"policy[{index}]")
            if set(spec) != {"version", "name", "priority", "when", "effect"}:
                raise ValueError(
                    f"policy[{index}] must contain version, name, priority, when, effect"
                )
            if spec["version"] != POLICY_VERSION:
                raise ValueError(f"unsupported policy version {spec['version']!r}")
            name = spec["name"]
            if not isinstance(name, str) or name in policy_names:
                raise ValueError(f"duplicate or invalid policy name {name!r}")
            if not isinstance(spec["priority"], int) or not isinstance(spec["when"], list):
                raise ValueError(f"policy {name!r} needs integer priority and list when")
            for condition_index, condition in enumerate(spec["when"]):
                _validate_condition(condition, f"policy[{name}].when[{condition_index}]")
            _validate_effect(spec["effect"], action_names, f"policy[{name}].effect")
            policy_names.add(name)
            policies.append(PolicyRule(
                name=name, version=POLICY_VERSION, priority=spec["priority"],
                when=tuple(spec["when"]), effect=spec["effect"],
            ))
        if not policies:
            raise ValueError(f"No ```policy blocks found in {path}")
        policies.sort(key=lambda rule: (-rule.priority, rule.name))
        return cls(
            actions=tuple(actions), decision_schemas=schemas,
            policies=tuple(policies), raw_markdown=raw,
        )
