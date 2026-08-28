"""The episode loop shared by rollouts (data collection) and evaluation.

Modes:
  unconstrained          - free generation (baseline; also produces the
                           distillation prompt dump)
  constrained            - Ctrl-G on the <tool> span, allowed set from
                           SKILLS.md preconditions x tracked state
  constrained-oracle     - Ctrl-G with the env's admissible_commands as the
                           allowed set (upper bound / tracker debugging)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .prompts import (
    Step,
    build_system_prompt,
    build_transcript,
    load_few_shot,
    render_prompt,
    task_key_from_gamefile,
)
from .skills import SkillSet
from .state import AlfWorldState


def process_ob(ob: str) -> str:
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob


@dataclass
class StepRecord:
    step: int
    thought: str
    action: str
    observation: str
    allowed_actions: list[str]
    admissible_gt: list[str]      # env ground truth (for constraint soundness eval)
    action_was_admissible: bool
    prefix_to_tool: str | None    # distillation prompt for this step


@dataclass
class EpisodeRecord:
    gamefile: str
    task_key: str
    mode: str
    success: bool
    num_steps: int
    steps: list[StepRecord] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def run_episode(env, backend, skillset: SkillSet, few_shot_path: str,
                mode: str = "unconstrained", max_steps: int = 50,
                verbose: bool = False) -> EpisodeRecord:
    ob, info = env.reset()
    initial_obs = "\n".join(ob[0].split("\n\n")[1:])
    gamefile = info["extra.gamefile"][0]
    task_key = task_key_from_gamefile(gamefile)

    system_prompt = build_system_prompt(
        skillset.raw_markdown, load_few_shot(few_shot_path, task_key)
    )

    state = AlfWorldState()
    state.reset(initial_obs)

    steps: list[Step] = []
    records: list[StepRecord] = []
    success = False

    for t in range(max_steps):
        transcript = build_transcript(initial_obs, steps)
        prompt_text = render_prompt(backend.tokenizer, system_prompt, transcript)

        admissible_gt = list(info.get("admissible_commands", [[]])[0])
        if mode == "constrained":
            allowed = skillset.ground_all(state.domains(), state.check)
        elif mode == "constrained-oracle":
            allowed = admissible_gt
        else:
            allowed = skillset.ground_all(state.domains(), state.check)  # logged only

        if mode.startswith("constrained"):
            thought, action, prefix = backend.generate_turn_constrained(
                prompt_text, allowed
            )
        else:
            thought, action, prefix = backend.generate_turn_unconstrained(prompt_text)

        if action is None:
            action = "look"  # malformed output fallback

        ob, reward, done, info = env.step([action])
        obs = process_ob(ob[0])
        state.update(action, obs)

        records.append(StepRecord(
            step=t, thought=thought, action=action, observation=obs,
            allowed_actions=allowed, admissible_gt=admissible_gt,
            action_was_admissible=(not admissible_gt) or action in admissible_gt,
            prefix_to_tool=prefix,
        ))
        steps.append(Step(thought=thought, action=action, observation=obs))

        if verbose:
            print(f"[{t}] {action}\n    {obs}")

        if done[0]:
            success = bool(info["won"][0])
            break

    return EpisodeRecord(
        gamefile=gamefile, task_key=task_key, mode=mode,
        success=success, num_steps=len(records), steps=records,
    )
