from dataclasses import dataclass
from pathlib import Path

ACTION_OPEN, ACTION_CLOSE = "<action>", "</action>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

SYSTEM_INSTRUCTION = "You are an expert agent operating in the ALFRED Embodied Environment."

SKILL_TEMPLATE = """## Skill Knowledge
Below is a skill document with learned strategies. Use these guidelines to inform your decisions:

{skill_content}
"""

ROOT = Path(__file__).resolve().parents[2]
TEMPLATED_OBS_NO_HIST_PATH = str(ROOT / "templates/rollout_no_history.md")
TEMPLATED_OBS_WITH_HIST_PATH = str(ROOT / "templates/rollout_with_history.md")

def _load_md(path: str):
    with open(path) as f:
        return f.read()

@dataclass
class Step:
    thought: str
    action: str
    observation: str

def build_user_prompt(skill_content: str, task_description: str, current_observation: str, admissible_actions: str, obs_history: list[Step]):
    parts = []
    skill = SKILL_TEMPLATE.format(skill_content=skill_content)
    parts.append(skill)

    recent_obs_history = obs_history[-3:]
    action_history = []
    for idx, h in enumerate(recent_obs_history):
        if idx == 0:
            action_history.append(f"OBS: {h.observation.strip()}")
        else:
            if h.thought:
                action_history.append(f"{THINK_OPEN}{h.thought}{THINK_CLOSE}")
            action_history.append(f"{ACTION_OPEN}{h.action}{ACTION_CLOSE}")
            if idx != len(recent_obs_history) - 1:
                action_history.append(f"OBS: {h.observation.strip()}")
    action_history = "\n".join(action_history)

    if len(obs_history) > 0:
        obs_template = _load_md(TEMPLATED_OBS_WITH_HIST_PATH)
    else:
        obs_template = _load_md(TEMPLATED_OBS_NO_HIST_PATH)
    obs_template = obs_template.format(
        task_description=task_description,
        step_count=len(obs_history),
        history_length=min(2, len(obs_history)),
        action_history=action_history,
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


def render_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ReAct gamefile-prefix -> few-shot key mapping (ysymyth/ReAct alfworld.ipynb)
TASK_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

def task_key_from_gamefile(gamefile: str) -> str:
    name = "/".join(gamefile.split("/")[-3:-1])
    for prefix, key in TASK_PREFIXES.items():
        if name.startswith(prefix):
            return key
    raise ValueError(f"Unknown task type for gamefile: {gamefile}")
