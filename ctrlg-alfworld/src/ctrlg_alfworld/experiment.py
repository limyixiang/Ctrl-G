"""The matched two-condition Ctrl-G/ALFWorld experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConditionName(str, Enum):
    DECISION_PROMPT = "decision_prompt"
    DECISION_CTRLG = "decision_ctrlg"


@dataclass(frozen=True)
class ExperimentCondition:
    name: ConditionName
    use_hmm: bool
    use_decision: bool = True
    use_decision_dfa: bool = False
    use_action_dfa: bool = False


CONDITIONS: dict[ConditionName, ExperimentCondition] = {
    ConditionName.DECISION_PROMPT: ExperimentCondition(
        ConditionName.DECISION_PROMPT, use_hmm=False
    ),
    ConditionName.DECISION_CTRLG: ExperimentCondition(
        ConditionName.DECISION_CTRLG,
        use_hmm=True,
        use_decision_dfa=True,
        use_action_dfa=True,
    ),
}


def get_condition(value: str | ConditionName) -> ExperimentCondition:
    try:
        name = value if isinstance(value, ConditionName) else ConditionName(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in ConditionName)
        raise ValueError(f"Unknown condition {value!r}; choose one of: {choices}") from exc
    return CONDITIONS[name]


def condition_choices() -> list[str]:
    return [item.value for item in ConditionName]
