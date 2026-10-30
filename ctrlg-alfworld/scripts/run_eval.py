import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld import SkillSet, run_episode
from ctrlg_alfworld.backends import GenConfig, HFBackend
from ctrlg_alfworld.experiment import condition_choices, get_condition
from ctrlg_alfworld.prompts import PROMPT_FORMAT
from ctrlg_alfworld.hmm_context import HMM_CONTEXT_FORMAT, HMM_SEQUENCE_FORMAT
from ctrlg_alfworld.provenance import (
    artifact_sha256,
    file_sha256,
    git_revision,
    json_sha256,
    runtime_versions,
    source_tree_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
HMM_TRAINING_METADATA = "training_data_metadata.json"


def resolve_hmm_path(
    condition,
    *,
    hmm: str | None,
) -> str | None:
    """Require the matched state-format HMM only for its treatment cell."""

    if not condition.use_hmm:
        return None
    if hmm is None:
        raise ValueError(f"{condition.name.value} requires --hmm")
    return hmm


def validate_hmm_provenance(
    hmm_path: str | None, *, skillset: SkillSet, model: str,
    hmm_context_steps=1, max_hmm_context_tokens=1024,
    allow_skills_mismatch: bool = False,
) -> tuple[str | None, dict | None]:
    """Validate joint-HMM provenance, optionally allowing a different skill hash."""

    if hmm_path is None:
        return None, None
    checkpoint = Path(hmm_path)
    candidates = (
        checkpoint / HMM_TRAINING_METADATA,
        checkpoint.parent / HMM_TRAINING_METADATA,
    )
    metadata_path = next((path for path in candidates if path.is_file()), None)
    if metadata_path is None:
        raise ValueError(f"{checkpoint} has no {HMM_TRAINING_METADATA}")

    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    training_skills_hash = metadata.get("skills_sha256")
    if not isinstance(training_skills_hash, str) or not training_skills_hash:
        raise ValueError("HMM provenance requires a recorded skills_sha256")
    expected = {
        "skills_sha256": skillset.content_sha256,
        "schema_version": skillset.schema_version,
        "policy_version": skillset.policy_version,
        "no_oracle_filtering": True,
        "constrained_state_collection": False,
        "hmm_sequence_format": HMM_SEQUENCE_FORMAT,
        "hmm_context_format": HMM_CONTEXT_FORMAT,
        "hmm_context_steps": hmm_context_steps,
        "max_hmm_context_tokens": max_hmm_context_tokens,
        "selection_policy": "all_observed_attempts",
        "model": model,
        "tokenizer": model,
        "prompt_format": PROMPT_FORMAT,
    }
    mismatches = [
        f"{key}: checkpoint={metadata.get(key)!r}, evaluation={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
        and not (key == "skills_sha256" and allow_skills_mismatch)
    ]
    if mismatches:
        raise ValueError("HMM provenance mismatch: " + "; ".join(mismatches))
    return str(metadata_path.resolve()), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hmm", default=None)
    parser.add_argument("--condition", required=True, choices=condition_choices())
    parser.add_argument("--config", default=str(ROOT / "configs/config_tw.yaml"))
    parser.add_argument("--skills", default=str(ROOT / "templates/SKILLS.md"))
    parser.add_argument(
        "--allow-skills-mismatch", action="store_true",
        help="Allow an HMM trained with a different skills_sha256; retain all other provenance checks",
    )
    parser.add_argument(
        "--split",
        default="eval_out_of_distribution",
        choices=["eval_in_distribution", "eval_out_of_distribution"],
    )
    parser.add_argument("--num_episodes", type=int, default=134)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--beam_size", type=int, default=1,
        help="Matched action beam width in both arms (default: 1)",
    )
    parser.add_argument("--max_thought_tokens", type=int, default=1024)
    parser.add_argument("--max_state_tokens", type=int, default=512)
    parser.add_argument("--max_action_tokens", type=int, default=32)
    parser.add_argument("--min_action_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--max_hmm_prefix_tokens", type=int, default=None)
    parser.add_argument("--hmm_context_steps", type=int, default=1)
    parser.add_argument("--max_hmm_context_tokens", type=int, default=1024)
    parser.add_argument("--sample_actions", action="store_true")
    parser.add_argument("--sample_head", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--out", default="out/eval")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    condition = get_condition(args.condition)
    skillset = SkillSet.from_file(args.skills)
    try:
        hmm_path = resolve_hmm_path(
            condition,
            hmm=args.hmm,
        )
        hmm_training_metadata, hmm_provenance = (
            validate_hmm_provenance(
                hmm_path, skillset=skillset, model=args.model,
                hmm_context_steps=args.hmm_context_steps, max_hmm_context_tokens=args.max_hmm_context_tokens,
                allow_skills_mismatch=args.allow_skills_mismatch,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import alfworld.agents.environment as environment

    with open(args.config) as file:
        config = yaml.safe_load(file)
    config["general"]["random_seed"] = args.seed
    env_factory = environment.get_environment(config["env"]["type"])(
        config, train_eval=args.split
    )
    # ALFWorld discovers games via os.walk without sorting. Sort explicitly so
    # separate Slurm array jobs receive the same ordered episode manifest.
    env_factory.game_files = sorted(env_factory.game_files)
    env_factory.num_games = len(env_factory.game_files)
    env = env_factory.init_env(batch_size=1)

    generation_config = GenConfig(
        max_thought_tokens=args.max_thought_tokens,
        max_state_tokens=args.max_state_tokens,
        max_action_tokens=args.max_action_tokens,
        min_action_tokens=args.min_action_tokens,
        beam_size=args.beam_size,
        do_sample=args.sample_actions,
        temperature=args.temperature,
        rollout_temperature=args.rollout_temperature,
        seed=args.seed,
        max_hmm_prefix_tokens=args.max_hmm_prefix_tokens,
        hmm_context_steps=args.hmm_context_steps, max_hmm_context_tokens=args.max_hmm_context_tokens,
    )
    backend = HFBackend(
        args.model,
        hmm_path=hmm_path,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        gen_config=generation_config,
    )

    output_directory = Path(args.out)
    output_directory.mkdir(parents=True, exist_ok=True)
    episodes_path = output_directory / f"eval_{condition.name.value}.jsonl"
    summary_path = output_directory / f"summary_{condition.name.value}.json"

    per_type = defaultdict(lambda: [0, 0])
    rule_activations = Counter()
    rule_violations = Counter()
    episode_gamefiles = []
    state_hmm_steps = 0
    context_truncated_steps = 0
    state_hmm_skips = Counter()
    action_hmm_skips = Counter()
    totals = {
        "episodes": 0,
        "successes": 0,
        "actions": 0,
        "admissible_actions": 0,
        "parsed_turns": 0,
        "state_schema_valid_turns": 0,
        "action_grammar_valid_turns": 0,
        "policy_evaluable_turns": 0,
        "policy_satisfied_turns": 0,
        "policy_fallback_turns": 0,
        "hmm_applied_turns": 0,
        "head_truncated_turns": 0,
        "thought_truncated_turns": 0,
        "state_truncated_turns": 0,
        "thought_repair_turns": 0,
        "state_repair_turns": 0,
        "tail_truncated_turns": 0,
        "exact_tail_span_turns": 0,
        "prompt_tokens": 0,
        "thought_tokens": 0,
        "state_tokens": 0,
        "generated_tokens": 0,
        "head_latency_seconds": 0.0,
        "thought_latency_seconds": 0.0,
        "state_latency_seconds": 0.0,
        "action_latency_seconds": 0.0,
    }

    track_cuda_memory = (
        str(args.device).startswith("cuda") and torch.cuda.is_available()
    )
    with open(episodes_path, "w") as output_file:
        for episode_index in range(args.num_episodes):
            if track_cuda_memory:
                torch.cuda.reset_peak_memory_stats(args.device)
            record = run_episode(
                env,
                backend,
                skillset,
                condition,
                max_steps=args.max_steps,
                greedy_head=not args.sample_head,
                verbose=args.verbose,
            )
            output_file.write(json.dumps(record.to_dict()) + "\n")
            output_file.flush()
            episode_gamefiles.append(record.gamefile)

            totals["episodes"] += 1
            totals["successes"] += int(record.success)
            per_type[record.task_key][0] += int(record.success)
            per_type[record.task_key][1] += 1
            for step in record.steps:
                state_hmm_steps += int(step.state_hmm_applied)
                context_truncated_steps += int(step.hmm_context_truncated)
                if step.state_hmm_skip_reason:
                    state_hmm_skips[step.state_hmm_skip_reason] += 1
                if step.hmm_skip_reason:
                    action_hmm_skips[step.hmm_skip_reason] += 1
                totals["actions"] += 1
                totals["admissible_actions"] += int(step.action_was_admissible)
                totals["parsed_turns"] += int(step.parse_ok)
                totals["state_schema_valid_turns"] += int(step.state_schema_valid)
                totals["action_grammar_valid_turns"] += int(step.action_grammar_valid)
                totals["policy_evaluable_turns"] += int(step.policy_evaluable)
                totals["policy_satisfied_turns"] += int(step.policy_satisfied is True)
                totals["policy_fallback_turns"] += int(step.policy_fallback_reason is not None)
                rule_activations.update(step.activated_policies)
                if step.policy_satisfied is False:
                    rule_violations.update(step.activated_policies)
                totals["hmm_applied_turns"] += int(step.hmm_applied)
                totals["head_truncated_turns"] += int(step.head_truncated)
                totals["thought_truncated_turns"] += int(step.thought_truncated)
                totals["state_truncated_turns"] += int(step.state_truncated)
                totals["thought_repair_turns"] += int(step.used_thought_repair)
                totals["state_repair_turns"] += int(step.used_state_repair)
                totals["tail_truncated_turns"] += int(step.tail_truncated)
                totals["exact_tail_span_turns"] += int(step.tail_span_exact)
                totals["prompt_tokens"] += step.prompt_tokens
                totals["thought_tokens"] += step.thought_tokens
                totals["state_tokens"] += step.state_tokens
                totals["generated_tokens"] += step.generated_tokens
                totals["head_latency_seconds"] += step.head_latency_seconds
                totals["thought_latency_seconds"] += step.thought_latency_seconds
                totals["state_latency_seconds"] += step.state_latency_seconds
                totals["action_latency_seconds"] += step.action_latency_seconds

            action_count = max(totals["actions"], 1)
            memory_status = ""
            if track_cuda_memory:
                gib = 1024 ** 3
                memory_status = (
                    f" cuda_allocated={torch.cuda.memory_allocated(args.device) / gib:.1f}GiB"
                    f" cuda_reserved={torch.cuda.memory_reserved(args.device) / gib:.1f}GiB"
                    f" cuda_episode_peak={torch.cuda.max_memory_allocated(args.device) / gib:.1f}GiB"
                )
            print(
                f"[{episode_index + 1}/{args.num_episodes}] "
                f"SR={totals['successes'] / totals['episodes']:.3f} "
                f"admissible={totals['admissible_actions'] / action_count:.3f} "
                f"parse={totals['parsed_turns'] / action_count:.3f}"
                f"{memory_status}"
            )

    action_count = max(totals["actions"], 1)
    summary = {
        "condition": condition.name.value,
        "factors": {
            "use_state": condition.use_state,
            "use_state_dfa": condition.use_state_dfa,
            "use_action_dfa": condition.use_action_dfa,
            "use_hmm": condition.use_hmm,
        },
        "model": args.model,
        "tokenizer": args.model,
        "prompt_format": PROMPT_FORMAT,
        "hmm": hmm_path,
        "hmm_sha256": artifact_sha256(hmm_path) if hmm_path else None,
        "state_hmm_applied_steps": state_hmm_steps,
        "state_hmm_skip_reasons": dict(state_hmm_skips),
        "action_hmm_skip_reasons": dict(action_hmm_skips),
        "hmm_context_truncated_steps": context_truncated_steps,
        "hmm_scope": "state_and_action" if condition.use_hmm else "none",
        "hmm_training_metadata": hmm_training_metadata,
        "hmm_training_metadata_sha256": (
            file_sha256(hmm_training_metadata) if hmm_training_metadata else None
        ),
        "hmm_provenance": hmm_provenance,
        "allow_skills_mismatch": args.allow_skills_mismatch,
        "split": args.split,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "beam_size": args.beam_size,
        "max_thought_tokens": args.max_thought_tokens,
        "max_state_tokens": args.max_state_tokens,
        "max_action_tokens": args.max_action_tokens,
        "min_action_tokens": args.min_action_tokens,
        "temperature": args.temperature,
        "rollout_temperature": args.rollout_temperature,
        "max_hmm_prefix_tokens": args.max_hmm_prefix_tokens,
        "hmm_sequence_format": HMM_SEQUENCE_FORMAT,
        "hmm_context_format": HMM_CONTEXT_FORMAT,
        "hmm_context_steps": args.hmm_context_steps,
        "max_hmm_context_tokens": args.max_hmm_context_tokens,
        "num_episodes": args.num_episodes,
        "sample_actions": args.sample_actions,
        "sample_head": args.sample_head,
        "device": args.device,
        "dtype": args.dtype,
        "episode_gamefiles": episode_gamefiles,
        "episode_manifest_sha256": json_sha256(episode_gamefiles),
        "config": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "skills": str(Path(args.skills).resolve()),
        "skills_sha256": skillset.content_sha256,
        "schema_version": skillset.schema_version,
        "policy_version": skillset.policy_version,
        "git_revision": git_revision(ROOT.parent),
        "source_tree_sha256": source_tree_sha256(ROOT),
        "runtime": runtime_versions(),
        "metrics": {
            **totals,
            "success_rate": totals["successes"] / max(totals["episodes"], 1),
            "admissibility_rate": totals["admissible_actions"] / action_count,
            "parse_rate": totals["parsed_turns"] / action_count,
            "state_schema_validity_rate": totals["state_schema_valid_turns"] / action_count,
            "action_grammar_validity_rate": totals["action_grammar_valid_turns"] / action_count,
            "policy_evaluability_rate": totals["policy_evaluable_turns"] / action_count,
            "policy_adherence_rate": totals["policy_satisfied_turns"] / max(totals["policy_evaluable_turns"], 1),
            "policy_fallback_rate": totals["policy_fallback_turns"] / action_count,
            "state_hmm_applied_rate": state_hmm_steps / action_count,
            "hmm_context_truncation_rate": context_truncated_steps / action_count,
            "hmm_applied_rate": totals["hmm_applied_turns"] / action_count,
            "head_truncation_rate": totals["head_truncated_turns"] / action_count,
            "thought_truncation_rate": (
                totals["thought_truncated_turns"] / action_count
            ),
            "state_truncation_rate": (
                totals["state_truncated_turns"] / action_count
            ),
            "thought_repair_rate": totals["thought_repair_turns"] / action_count,
            "state_repair_rate": totals["state_repair_turns"] / action_count,
            "tail_truncation_rate": totals["tail_truncated_turns"] / action_count,
            "exact_tail_span_rate": totals["exact_tail_span_turns"] / action_count,
            "mean_prompt_tokens_per_action": totals["prompt_tokens"] / action_count,
            "mean_thought_tokens_per_action": (
                totals["thought_tokens"] / action_count
            ),
            "mean_state_tokens_per_action": (
                totals["state_tokens"] / action_count
            ),
            "mean_generated_tokens_per_action": totals["generated_tokens"] / action_count,
            "mean_head_latency_seconds": totals["head_latency_seconds"] / action_count,
            "mean_thought_latency_seconds": (
                totals["thought_latency_seconds"] / action_count
            ),
            "mean_state_latency_seconds": (
                totals["state_latency_seconds"] / action_count
            ),
            "mean_action_latency_seconds": totals["action_latency_seconds"] / action_count,
            "per_rule_activations": dict(sorted(rule_activations.items())),
            "per_rule_violations": dict(sorted(rule_violations.items())),
        },
        "per_task_type": {
            key: {
                "successes": successes,
                "episodes": count,
                "success_rate": successes / count,
            }
            for key, (successes, count) in sorted(per_type.items())
        },
    }
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
