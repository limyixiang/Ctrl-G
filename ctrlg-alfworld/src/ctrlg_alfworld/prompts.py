from dataclasses import dataclass
from pathlib import Path

ACTION_OPEN, ACTION_CLOSE = "<action>", "</action>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
DECISION_OPEN, DECISION_CLOSE = "<decision>", "</decision>"

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
    decision: str = ""

def build_user_prompt(
    skill_content: str,
    task_description: str,
    current_observation: str,
    obs_history: list[Step],
    *,
    use_decision: bool,
    admissible_actions: list[str] | None = None,
    show_admissible_actions: bool = False,
):
    """Build one prompt for the matched decision-agent experiment.

    TextWorld's admissible commands stay out of the model-visible prompt in the
    default experiment. They are passed separately to the decoder to build the
    DFA. ``show_admissible_actions`` enables a matched prompt-visible regime
    that must be used consistently for sample collection and evaluation.

    Native model thinking is enabled by :func:`render_prompt`, so this prompt
    must not request a second, manually generated ``<think>`` block.
    """
    parts = []
    skill = SKILL_TEMPLATE.format(skill_content=skill_content)
    parts.append(skill)

    recent_obs_history = obs_history[-3:]
    action_history = []
    for h in recent_obs_history:
        # Native/hidden reasoning is never replayed. Decision conditions retain
        # their explicit decisions as persistent memory; no-decision prompts
        # retain the original action/observation-only history.
        if use_decision and h.decision.strip():
            action_history.append(
                f"{DECISION_OPEN}{h.decision.strip()}{DECISION_CLOSE}"
            )
        action_history.append(f"{ACTION_OPEN}{h.action}{ACTION_CLOSE}")
        action_history.append(f"OBS: {h.observation.strip()}")
    action_history = "\n".join(action_history)

    if show_admissible_actions:
        # The environment's command order is not semantically meaningful. Keep
        # it from becoming an accidental prompt variable while leaving the
        # original list untouched for decoding and rollout fallback behavior.
        actions = sorted(admissible_actions or [])
        admissible_actions_section = (
            "Your admissible actions in the current situation are: "
            f"[{', '.join(actions)}]."
        )
    else:
        admissible_actions_section = ""

    if use_decision:
        decision_instruction = (
            f"After thinking, emit one short decision in {DECISION_OPEN} "
            f"{DECISION_CLOSE}, then emit exactly one action in "
            f"{ACTION_OPEN} {ACTION_CLOSE}."
        )
    else:
        decision_instruction = (
            f"After thinking, emit exactly one action in {ACTION_OPEN} "
            f"{ACTION_CLOSE}, with no text between the thinking span and "
            f"{ACTION_OPEN}."
        )

    if len(obs_history) > 0:
        obs_template = _load_md(TEMPLATED_OBS_WITH_HIST_PATH)
    else:
        obs_template = _load_md(TEMPLATED_OBS_NO_HIST_PATH)
    obs_template = obs_template.format(
        task_description=task_description,
        step_count=len(obs_history),
        history_length=min(3, len(obs_history)),
        action_history=action_history,
        current_step=len(obs_history)+1,
        current_observation=current_observation,
        admissible_actions_section=admissible_actions_section,
        decision_instruction=decision_instruction,
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
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)


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
