from dataclasses import asdict, dataclass, field

from .prompts import (
    Step,
    SYSTEM_INSTRUCTION,
    build_user_prompt,
    render_prompt,
    task_key_from_gamefile
)
from .skills import SkillSet


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
    # allowed_actions: list[str]
    admissible_gt: list[str]
    action_was_admissible: bool | None
    prefix_to_action: str | None # distillation prompt


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

def run_episode(env, backend, skillset: SkillSet, mode: str = "unconstrained", max_steps: int = 50, greedy: bool = True, verbose: bool = False) -> EpisodeRecord:
    ob, info = env.reset()
    initial_obs = "\n".join(ob[0].split("\n\n")[1:])
    task_description = ob[0].split("\n\n")[2]

    gamefile = info["extra.gamefile"][0]
    task_key = task_key_from_gamefile(gamefile)

    system_prompt = SYSTEM_INSTRUCTION

    steps: list[Step] = []
    records: list[StepRecord] = []
    success = False

    obs = initial_obs

    # if verbose:
        # print(initial_obs)

    for t in range(max_steps):
        admissible_actions = list(info.get("admissible_commands", [[]])[0])
        admissible_actions_str = ", ".join(admissible_actions)

        user_prompt = build_user_prompt(
            skill_content=skillset.raw_markdown,
            task_description=task_description,
            current_observation=obs,
            admissible_actions=admissible_actions_str,
            obs_history=steps
        )

        prompt_text = render_prompt(backend.tokenizer, system_prompt, user_prompt)

        if verbose and t == 0:
            print(user_prompt)

        if mode == "unconstrained":
            thought, action, prefix = backend.generate_turn_unconstrained(prompt_text, greedy=greedy)
        else:
            raise NotImplementedError

        if action is None or action == "":
            action = "look" # malformed output fallback

        ob, reward, done, info = env.step([action])
        obs = process_ob(ob[0])

        records.append(StepRecord(
            step=t, thought=thought, action=action, observation=obs, admissible_gt=admissible_actions, action_was_admissible= action in admissible_actions if admissible_actions else None, prefix_to_action=prefix
        ))
        steps.append(Step(thought=thought, action=action, observation=obs))

        if verbose:
            print(f"[{t}] {thought}\n   {action}\n    {obs}")

        if done[0]:
            success = bool(info["won"][0])
            break

    return EpisodeRecord(
        gamefile=gamefile, task_key=task_key, mode=mode, success=success, num_steps=len(records), steps=records
    )