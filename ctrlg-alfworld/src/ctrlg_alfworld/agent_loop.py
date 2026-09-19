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
from .constraints import (
    action_grammar_valid,
    compile_policy_language,
    parse_decision,
    policy_satisfied,
)


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
    used_thought_repair: bool
    used_decision_repair: bool
    head_truncated: bool
    thought_truncated: bool
    decision_truncated: bool
    tail_truncated: bool
    tail_span_exact: bool
    hmm_applied: bool
    hmm_skip_reason: str | None
    hmm_prefix_text: str
    hmm_prefix_token_ids: list[int]
    action_token_ids: list[int]
    tail_token_ids: list[int]
    prompt_tokens: int
    thought_tokens: int
    decision_tokens: int
    generated_tokens: int
    head_latency_seconds: float
    thought_latency_seconds: float
    decision_latency_seconds: float
    action_latency_seconds: float
    decision_schema_valid: bool
    decision_fields: dict
    action_grammar_valid: bool
    activated_policies: list[str]
    shadowed_policies: list[str]
    policy_evaluable: bool
    policy_satisfied: bool | None
    policy_fallback_reason: str | None


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
            initial_observation=initial_obs,
            current_observation=obs,
            obs_history=steps,
            use_decision=condition.use_decision,
        )

        prompt_text = render_prompt(backend.tokenizer, system_prompt, user_prompt)
        prompt_tokens = len(
            backend.tokenizer.encode(prompt_text, add_special_tokens=False)
        )

        if verbose and t == 0:
            print(user_prompt)

        turn = backend.generate_turn(
            prompt_text,
            skillset,
            task_key,
            constrained=condition.use_decision_dfa,
            greedy_head=greedy_head,
        )
        action = turn.parsed.action

        decision_schema_valid = turn.decision_schema_valid
        decision_fields = turn.decision_fields
        activated_policies = turn.activated_policies
        shadowed_policies = turn.shadowed_policies
        policy_evaluable = turn.policy_evaluable
        policy_adherence = turn.policy_satisfied
        policy_fallback_reason = turn.policy_fallback_reason
        grammar_valid = turn.action_grammar_valid
        if not condition.use_decision_dfa:
            grammar_valid = action_grammar_valid(action, skillset)
            try:
                decision_fields = parse_decision(
                    turn.parsed.decision, skillset.decision_schemas[task_key], skillset
                )
                decision_schema_valid = True
                policy = compile_policy_language(decision_fields, skillset)
                activated_policies = policy.activated
                shadowed_policies = policy.shadowed
                policy_evaluable = policy.evaluable
                policy_adherence = policy_satisfied(action, policy)
                policy_fallback_reason = policy.fallback_reason
            except ValueError as exc:
                decision_fields = {}
                decision_schema_valid = False
                policy_evaluable = False
                policy_adherence = None
                policy_fallback_reason = f"invalid_decision_schema: {exc}"

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
            used_thought_repair=turn.used_thought_repair,
            used_decision_repair=turn.used_decision_repair,
            head_truncated=turn.head_truncated,
            thought_truncated=turn.thought_truncated,
            decision_truncated=turn.decision_truncated,
            tail_truncated=turn.tail_truncated,
            tail_span_exact=turn.tail_span_exact,
            hmm_applied=turn.hmm_applied,
            hmm_skip_reason=turn.hmm_skip_reason,
            hmm_prefix_text=turn.parsed.hmm_prefix_text,
            hmm_prefix_token_ids=list(turn.hmm_prefix_token_ids),
            action_token_ids=list(turn.action_token_ids),
            tail_token_ids=list(turn.tail_token_ids),
            prompt_tokens=prompt_tokens,
            thought_tokens=len(turn.thought_token_ids),
            decision_tokens=len(turn.decision_token_ids),
            generated_tokens=turn.total_generated_tokens,
            head_latency_seconds=turn.head_latency_seconds,
            thought_latency_seconds=turn.thought_latency_seconds,
            decision_latency_seconds=turn.decision_latency_seconds,
            action_latency_seconds=turn.action_latency_seconds,
            decision_schema_valid=decision_schema_valid,
            decision_fields=decision_fields,
            action_grammar_valid=grammar_valid,
            activated_policies=list(activated_policies),
            shadowed_policies=list(shadowed_policies),
            policy_evaluable=policy_evaluable,
            policy_satisfied=policy_adherence,
            policy_fallback_reason=policy_fallback_reason,
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
