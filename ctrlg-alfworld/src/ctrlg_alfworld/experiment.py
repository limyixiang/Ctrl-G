"""The matched two-condition Ctrl-G/ALFWorld experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConditionName(str, Enum):
    DECISION_DFA = "decision_dfa"
    DECISION_DFA_HMM = "decision_dfa_hmm"


@dataclass(frozen=True)
class ExperimentCondition:
    name: ConditionName
    use_hmm: bool
    use_decision: bool = True
    use_dfa: bool = True


CONDITIONS: dict[ConditionName, ExperimentCondition] = {
    ConditionName.DECISION_DFA: ExperimentCondition(
        ConditionName.DECISION_DFA, use_hmm=False
    ),
    ConditionName.DECISION_DFA_HMM: ExperimentCondition(
        ConditionName.DECISION_DFA_HMM, use_hmm=True
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
