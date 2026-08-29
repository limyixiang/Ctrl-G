import json
from dataclasses import dataclass
from pathlib import Path

ACTION_OPEN, ACTION_CLOSE = "<action>", "</action>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

SYSTEM_INSTRUCTION = "You are an expert agent operating in the ALFRED Embodied Environment."

SKILL_TEMPLATE = """## Skill Knowledge
Below is a skill document with learned strategies. Use these guidelines to inform your decisions:

{skill_content}
"""

ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = str(ROOT / "templates/SKILLS.md")
TEMPLATED_OBS_NO_HIST_PATH = str(ROOT / "templates/rollout_no_hostory.md")
TEMPLATED_OBS_WITH_HIST_PATH = str(ROOT / "rollout_with_history.md")

@dataclass
class Step:
    thought: str
    action: str
    observation: str

def _load_md(path: str):
    with open(path) as f:
        return f.read()

def build_user_prompt(task_description: str, current_observation: str, admissible_actions: str, obs_history: list[Step]):
    parts = []
    skill_content = _load_md(SKILL_PATH)
    skill = SKILL_TEMPLATE.format(skill_content)
    parts.append(skill)

    if len(obs_history) > 0:
        obs_template = _load_md(TEMPLATED_OBS_WITH_HIST_PATH)
    else:
        obs_template = _load_md(TEMPLATED_OBS_NO_HIST_PATH)
    obs_template.format(
        task_description=task_description,
        step_count=len(obs_history),
        history_length=min(2, len(obs_history)),
        action_history=obs_history[-2:],
        current_step=len(obs_history)+1,
        current_observation=current_observation,
        admissible_actions=admissible_actions,
        think_open=THINK_OPEN,
        think_close=THINK_CLOSE,
        action_open=ACTION_OPEN,
        action_close=ACTION_CLOSE
    )
    parts.append(obs_template)

    return "\n".join(parts)
