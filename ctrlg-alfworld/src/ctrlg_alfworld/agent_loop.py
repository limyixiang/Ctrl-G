from dataclasses import asdict, dataclass, field

from .prompts import (
    Step,
    SYSTEM_INSTRUCTION,
    build_user_prompt,
    render_prompt,
    task_key_from_gamefile
)
from .experiment import ExperimentCondition, get_condition
from .skills import SkillSet


TASK_DESCRIPTION_PREFIX = "Your task is to:"


def parse_initial_observation(raw_observation: str) -> tuple[str, str]:
    """Split ALFWorld's welcome, room observation, and task description."""

    parts = raw_observation.split("\n\n")
    if len(parts) < 3:
        raise ValueError(
            "ALFWorld reset observation must contain welcome, room, and task sections"
        )
    initial_observation = "\n\n".join(parts[1:-1]).strip()
    task_description = parts[-1].strip()
    if task_description.startswith(TASK_DESCRIPTION_PREFIX):
        task_description = task_description[len(TASK_DESCRIPTION_PREFIX) :].strip()
    if not initial_observation or not task_description:
        raise ValueError("ALFWorld reset observation has an empty room or task section")
    return initial_observation, task_description


def process_ob(ob: str) -> str:
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob

@dataclass
class StepRecord:
    step: int
    condition: str
    thought: str
    decision: str
    action: str
    observation: str
    admissible_gt: list[str]
    action_was_admissible: bool
    parse_ok: bool
    parse_errors: list[str]
    used_head_repair: bool
    head_truncated: bool
    tail_truncated: bool
    tail_span_exact: bool
    hmm_applied: bool
    hmm_skip_reason: str | None
    hmm_prefix_text: str
    hmm_prefix_token_ids: list[int]
    action_token_ids: list[int]
    tail_token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    head_latency_seconds: float
    action_latency_seconds: float


@dataclass
class EpisodeRecord:
    gamefile: str
    task_key: str
    condition: str
    success: bool
    num_steps: int
    steps: list[StepRecord] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

def run_episode(
    env,
    backend,
    skillset: SkillSet,
    condition: str | ExperimentCondition,
    max_steps: int = 50,
    greedy_head: bool = True,
    show_admissible_actions: bool = False,
    verbose: bool = False,
) -> EpisodeRecord:
    condition = (
        condition
        if isinstance(condition, ExperimentCondition)
        else get_condition(condition)
    )
    ob, info = env.reset()
    initial_obs, task_description = parse_initial_observation(ob[0])

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
        user_prompt = build_user_prompt(
            skill_content=skillset.raw_markdown,
            task_description=task_description,
            current_observation=obs,
            obs_history=steps,
            use_decision=condition.use_decision,
            admissible_actions=admissible_actions,
            show_admissible_actions=show_admissible_actions,
        )

        prompt_text = render_prompt(backend.tokenizer, system_prompt, user_prompt)
        prompt_tokens = len(
            backend.tokenizer.encode(prompt_text, add_special_tokens=False)
        )

        if verbose and t == 0:
            print(user_prompt)

        turn = backend.generate_turn(
            prompt_text,
            admissible_actions,
            use_decision=condition.use_decision,
            use_hmm=condition.use_hmm,
            greedy_head=greedy_head,
        )
        action = turn.parsed.action

        ob, reward, done, info = env.step([action])
        obs = process_ob(ob[0])

        records.append(StepRecord(
            step=t,
            condition=condition.name.value,
            thought=turn.parsed.thought,
            decision=turn.parsed.decision,
            action=action,
            observation=obs,
            admissible_gt=admissible_actions,
            action_was_admissible=action in admissible_actions,
            parse_ok=turn.parsed.parse_ok,
            parse_errors=list(turn.parsed.errors),
            used_head_repair=turn.used_head_repair,
            head_truncated=turn.head_truncated,
            tail_truncated=turn.tail_truncated,
            tail_span_exact=turn.tail_span_exact,
            hmm_applied=turn.hmm_applied,
            hmm_skip_reason=turn.hmm_skip_reason,
            hmm_prefix_text=turn.parsed.hmm_prefix_text,
            hmm_prefix_token_ids=list(turn.hmm_prefix_token_ids),
            action_token_ids=list(turn.action_token_ids),
            tail_token_ids=list(turn.tail_token_ids),
            prompt_tokens=prompt_tokens,
            generated_tokens=turn.total_generated_tokens,
            head_latency_seconds=turn.head_latency_seconds,
            action_latency_seconds=turn.action_latency_seconds,
        ))
        steps.append(Step(
            thought=turn.parsed.thought,
            decision=turn.parsed.decision,
            action=action,
            observation=obs,
        ))

        if verbose:
            print(
                f"[{t}] decision={turn.parsed.decision!r} "
                f"parse_ok={turn.parsed.parse_ok}\n"
                f"   {action}\n    {obs}"
            )

        if done[0]:
            success = bool(info["won"][0])
            break

    return EpisodeRecord(
        gamefile=gamefile,
        task_key=task_key,
        condition=condition.name.value,
        success=success,
        num_steps=len(records),
        steps=records,
    )
