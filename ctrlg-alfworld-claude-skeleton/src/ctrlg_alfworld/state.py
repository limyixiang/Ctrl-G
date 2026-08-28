"""Symbolic state tracking for AlfWorld text games.

The tracker consumes (action, observation) pairs and maintains a partial world
state: known receptacles, current location, held object, per-receptacle
contents, open/closed status. It exposes:

- `domains()`: candidate values for template placeholders ({obj}, {recep}, ...)
- `check(pred, args)`: predicate evaluation for skill preconditions.

Soundness convention: `check` returns True whenever the predicate holds OR the
tracker cannot prove it false. Constraints built from this state therefore only
remove provably-illegal actions and can never exclude the correct one due to
missing information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Receptacle types that can be opened/closed in ALFRED.
OPENABLE_TYPES = ("cabinet", "drawer", "fridge", "microwave", "safe", "box", "laundryhamper")

SEE_RE = re.compile(r"you see (.*?)(?:\.|$)", re.IGNORECASE)
ARTICLE_SPLIT_RE = re.compile(r",? and a |,? and an |, a |, an ")
ENTITY_RE = re.compile(r"^[a-z][a-z]* \d+$")


def _parse_entity_list(text: str) -> list[str]:
    """Parse 'a apple 1, a bread 2, and a mug 3' into entity names."""
    text = text.strip()
    if text.startswith(("a ", "an ")):
        text = text.split(" ", 1)[1]
    if "nothing" in text:
        return []
    parts = ARTICLE_SPLIT_RE.split(text)
    return [p.strip().rstrip(".") for p in parts if ENTITY_RE.match(p.strip().rstrip("."))]


@dataclass
class AlfWorldState:
    receptacles: list[str] = field(default_factory=list)
    location: str | None = None
    holding: str | None = None
    contents: dict[str, list[str]] = field(default_factory=dict)  # only known receps
    is_open: dict[str, bool] = field(default_factory=dict)  # only known receps

    # ------------------------------------------------------------------ init
    def reset(self, initial_observation: str) -> None:
        """Parse the room description in the first observation."""
        self.receptacles, self.location, self.holding = [], None, None
        self.contents, self.is_open = {}, {}
        m = SEE_RE.search(initial_observation.replace("\n", " "))
        if m:
            self.receptacles = _parse_entity_list(m.group(1))

    # ---------------------------------------------------------------- update
    def update(self, action: str, observation: str) -> None:
        obs = observation.strip()
        if obs == "Nothing happens." or not action:
            return
        act = action.strip().lower()

        if act.startswith("go to "):
            self.location = act[len("go to "):].strip()
            self._parse_location_obs(obs)
        elif act.startswith("open "):
            recep = act[len("open "):].strip()
            self.is_open[recep] = True
            self._parse_location_obs(obs, recep=recep)
        elif act.startswith("close "):
            self.is_open[act[len("close "):].strip()] = False
        elif act.startswith("take "):
            m = re.match(r"take (.+) from (.+)", act)
            if m and "you pick up" in obs.lower():
                obj, recep = m.group(1).strip(), m.group(2).strip()
                self.holding = obj
                if recep in self.contents and obj in self.contents[recep]:
                    self.contents[recep].remove(obj)
        elif act.startswith("put "):
            m = re.match(r"put (.+) in/on (.+)", act)
            if m and "you put" in obs.lower():
                obj, recep = m.group(1).strip(), m.group(2).strip()
                self.holding = None
                self.contents.setdefault(recep, [])
                if obj not in self.contents[recep]:
                    self.contents[recep].append(obj)
        elif act in ("look", "inventory") or act.startswith(("examine ",)):
            self._parse_location_obs(obs)
        # clean/heat/cool/use: object identity is preserved in AlfWorld obs,
        # no tracked state change needed.

    def _parse_location_obs(self, obs: str, recep: str | None = None) -> None:
        """Parse 'On the countertop 1, you see ...' / 'In it, you see ...'."""
        target = recep or self.location
        flat = obs.replace("\n", " ")
        m = re.search(r"(?:On|In) the ([a-z]+ \d+),? you see (.*?)(?:\.|$)", flat)
        if m:
            target = m.group(1)
            self.contents[target] = _parse_entity_list(m.group(2))
            return
        m = re.search(r"(?:In it|On it),? you see (.*?)(?:\.|$)", flat)
        if m and target:
            self.contents[target] = _parse_entity_list(m.group(1))
            return
        m = re.search(r"The ([a-z]+ \d+) is (open|closed)", flat)
        if m:
            self.is_open[m.group(1)] = m.group(2) == "open"

    # --------------------------------------------------------------- queries
    def visible_objects(self) -> list[str]:
        objs = []
        if self.location and self.location in self.contents:
            objs.extend(self.contents[self.location])
        if self.holding and self.holding not in objs:
            objs.append(self.holding)
        return objs

    def domains(self) -> dict[str, list[str]]:
        objs = self.visible_objects()
        return {
            "recep": list(self.receptacles),
            "obj": objs,
            "thing": objs + list(self.receptacles),
        }

    def check(self, pred: str, args: tuple[str, ...]) -> bool:
        """True if the predicate holds or cannot be proven false."""
        if pred == "at":
            return self.location is None or self.location == args[0]
        if pred == "not_at":
            return self.location is None or self.location != args[0]
        if pred == "seen":
            return not self.receptacles or args[0] in self.receptacles
        if pred == "holding":
            return self.holding == args[0]
        if pred == "hand_empty":
            return self.holding is None
        if pred == "visible":
            return args[0] in self.visible_objects() or (
                self.location is not None and self.location not in self.contents
            )
        if pred == "obj_at_current":
            if self.location is None or self.location not in self.contents:
                return True  # contents unknown -> cannot prune
            return args[0] in self.contents[self.location]
        if pred == "openable":
            return any(args[0].startswith(t) for t in OPENABLE_TYPES)
        if pred == "closed":
            return not self.is_open.get(args[0], False) if args[0] in self.is_open else True
        if pred == "open":
            return self.is_open.get(args[0], True)
        if pred == "recep_type":
            return args[0].startswith(args[1])
        # unknown predicate: never prune
        return True
