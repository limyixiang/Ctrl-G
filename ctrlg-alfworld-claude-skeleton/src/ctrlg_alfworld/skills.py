"""Parse SKILLS.md into machine-usable skill specs.

SKILLS.md is both prompt content (injected verbatim into the system prompt)
and the source of per-step constraints: each ```tool fenced block is YAML with
`name`, `template`, `preconditions`.

Templates contain placeholders in braces, e.g. "take {obj} from {recep}".
Placeholder names are free-form; grounding domains are supplied by the state
tracker (see state.AlfWorldState.domains()).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TOOL_BLOCK_RE = re.compile(r"```tool\s*\n(.*?)```", re.DOTALL)
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
        """Enumerate concrete action strings whose preconditions all hold.

        domains: placeholder name -> candidate values (e.g. {"recep": [...]})
        check: callable(pred: str, args: tuple[str, ...]) -> bool.
            Called with placeholder args substituted by candidate values.
            Should return True when the predicate holds OR is unknown
            (sound-but-not-complete pruning).
        """
        phs = self.placeholders
        if not phs:
            binding_iter = [()]
        else:
            pools = []
            for ph in phs:
                pool = domains.get(ph)
                if pool is None:
                    # unknown placeholder: fall back to union of all domains
                    pool = sorted(set(itertools.chain.from_iterable(domains.values())))
                pools.append(pool)
            binding_iter = itertools.product(*pools)

        actions = []
        for values in binding_iter:
            binding = dict(zip(phs, values))
            ok = True
            for pc in self.preconditions:
                args = tuple(
                    binding.get(PLACEHOLDER_RE.match(a).group(1), a)
                    if PLACEHOLDER_RE.match(a)
                    else a
                    for a in pc.args
                )
                if not check(pc.pred, args):
                    ok = False
                    break
            if ok:
                action = self.template
                for ph, v in binding.items():
                    action = action.replace("{" + ph + "}", v)
                actions.append(action)
        return actions


@dataclass
class SkillSet:
    skills: list[Skill]
    raw_markdown: str

    @classmethod
    def from_file(cls, path: str | Path) -> "SkillSet":
        raw = Path(path).read_text()
        skills = []
        for block in TOOL_BLOCK_RE.findall(raw):
            spec = yaml.safe_load(block)
            skills.append(
                Skill(
                    name=spec["name"],
                    template=spec["template"],
                    preconditions=[
                        Precondition.parse(p) for p in spec.get("preconditions") or []
                    ],
                )
            )
        if not skills:
            raise ValueError(f"No ```tool blocks found in {path}")
        return cls(skills=skills, raw_markdown=raw)

    def ground_all(self, domains: dict[str, list[str]], check) -> list[str]:
        """All admissible grounded actions across skills (deduped, ordered)."""
        seen, out = set(), []
        for skill in self.skills:
            for action in skill.ground(domains, check):
                if action not in seen:
                    seen.add(action)
                    out.append(action)
        return out
