from .skills import Action, SkillSet, Precondition
from .prompts import (
    Step, ACTION_OPEN, ACTION_CLOSE, DECISION_OPEN, DECISION_CLOSE,
    SYSTEM_INSTRUCTION, build_user_prompt, render_prompt,
    task_key_from_gamefile
)
from .experiment import (
    ConditionName, ExperimentCondition, condition_choices,
    get_condition
)
from .generation import ParsedTurn, parse_turn

def __getattr__(name):
    # Keep lightweight schema/policy tooling importable without initializing
    # PyTorch/Transformers. Public rollout symbols remain lazily available.
    if name in {"run_episode", "EpisodeRecord", "StepRecord", "process_ob"}:
        from . import agent_loop

        return getattr(agent_loop, name)
    raise AttributeError(name)
