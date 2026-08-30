from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ACTION_BLOCK_RE = re.compile(r"```action\s*\n(.*?)```", re.DOTALL)
PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
# precondition syntax: pred_name(arg1, arg2, ...) or bare pred_name
PRECOND_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\((.*)\))?$")

@dataclass
class Precondition:
    pred: str
    args: tuple[str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> "Precondition":
        m = PRECOND_RE.match(text.strip())
        if m is None:
            raise ValueError(f"Cannot parse precondition: {text!r}")
        pred, argstr = m.group(1), m.group(2)
        args = ()
        if argstr:
            args = tuple(a.strip() for a in argstr.split(","))
        return cls(pred=pred, args=args)

@dataclass
class Skill:
    name: str
    template: str
    preconditions: list[Precondition] = field(default_factory=list)

    @property
    def placeholders(self) -> list[str]:
        return PLACEHOLDER_RE.findall(self.template)

    def ground(self, domains: dict[str, list[str]], check) -> list[str]:
        raise NotImplementedError

@dataclass
class SkillSet:
    skills: list[Skill]
    raw_markdown: str

    @classmethod
    def from_file(cls, path: str | Path) -> SkillSet:
        raw = Path(path).read_text()
        skills = []
        for block in ACTION_BLOCK_RE.findall(raw):
            spec = yaml.safe_load(block)
            skills.append(
                Skill(
                    name=spec["name"],
                    template=spec["template"],
                    preconditions=[
                        Precondition.parse(p) for p in spec.get("preconditions") or []
                    ]
                )
            )
        if not skills:
            raise ValueError(f"No ```action blocks found in {path}")
        return cls(skills=skills, raw_markdown=raw)

    def ground_all(self, domains: dict[str, list[str]], check) -> list[str]:
        raise NotImplementedError
    