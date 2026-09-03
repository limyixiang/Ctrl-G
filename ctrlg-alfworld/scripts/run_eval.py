import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld import SkillSet, run_episode
from ctrlg_alfworld.backends import GenConfig, HFBackend
from ctrlg_alfworld.experiment import condition_choices, get_condition
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
    """Require the matched decision-format HMM only for its treatment cell."""

    if not condition.use_hmm:
        return None
    if hmm is None:
        raise ValueError(f"{condition.name.value} requires --hmm")
    return hmm


def validate_hmm_prompt_regime(
    hmm_path: str | None, *, show_admissible_actions: bool
) -> tuple[str | None, bool | None]:
    """Validate new checkpoints while allowing legacy checkpoints without metadata."""

    if hmm_path is None:
        return None, None
    checkpoint = Path(hmm_path)
    candidates = (
        checkpoint / HMM_TRAINING_METADATA,
        checkpoint.parent / HMM_TRAINING_METADATA,
    )
    metadata_path = next((path for path in candidates if path.is_file()), None)
    if metadata_path is None:
        return None, None

    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    trained_with_actions = metadata.get("show_admissible_actions")
    if not isinstance(trained_with_actions, bool):
        raise ValueError(
            f"{metadata_path} does not define boolean show_admissible_actions"
        )
    if trained_with_actions != show_admissible_actions:
        raise ValueError(
            "HMM prompt regime does not match evaluation: "
            f"checkpoint metadata has show_admissible_actions={trained_with_actions}, "
            f"evaluation has show_admissible_actions={show_admissible_actions}"
        )
    return str(metadata_path.resolve()), trained_with_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hmm", default=None)
    parser.add_argument("--condition", required=True, choices=condition_choices())
    parser.add_argument("--config", default=str(ROOT / "configs/config_tw.yaml"))
    parser.add_argument("--skills", default=str(ROOT / "templates/SKILLS.md"))
    parser.add_argument(
        "--split",
        default="eval_out_of_distribution",
        choices=["eval_in_distribution", "eval_out_of_distribution"],
    )
    parser.add_argument("--num_episodes", type=int, default=134)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam_size", type=int, default=8)
    parser.add_argument("--max_head_tokens", type=int, default=512)
    parser.add_argument("--max_action_tokens", type=int, default=24)
    parser.add_argument("--min_action_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--max_hmm_prefix_tokens", type=int, default=None)
    parser.add_argument("--sample_actions", action="store_true")
    parser.add_argument("--sample_head", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--show_admissible_actions",
        action="store_true",
        help=(
            "Include current admissible commands in the model-visible prompt. "
            "Disabled by default and must match HMM sample collection."
        ),
    )
    parser.add_argument("--out", default="out/eval")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    condition = get_condition(args.condition)
    try:
        hmm_path = resolve_hmm_path(
            condition,
            hmm=args.hmm,
        )
        hmm_training_metadata, hmm_training_show_admissible_actions = (
            validate_hmm_prompt_regime(
                hmm_path,
                show_admissible_actions=args.show_admissible_actions,
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

    skillset = SkillSet.from_file(args.skills)
    generation_config = GenConfig(
        max_head_tokens=args.max_head_tokens,
        max_action_tokens=args.max_action_tokens,
        min_action_tokens=args.min_action_tokens,
        beam_size=args.beam_size,
        do_sample=args.sample_actions,
        temperature=args.temperature,
        rollout_temperature=args.rollout_temperature,
        seed=args.seed,
        max_hmm_prefix_tokens=args.max_hmm_prefix_tokens,
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
    episode_gamefiles = []
    totals = {
        "episodes": 0,
        "successes": 0,
        "actions": 0,
        "admissible_actions": 0,
        "parsed_turns": 0,
        "hmm_applied_turns": 0,
        "head_truncated_turns": 0,
        "tail_truncated_turns": 0,
        "exact_tail_span_turns": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "head_latency_seconds": 0.0,
        "action_latency_seconds": 0.0,
    }

    with open(episodes_path, "w") as output_file:
        for episode_index in range(args.num_episodes):
            record = run_episode(
                env,
                backend,
                skillset,
                condition,
                max_steps=args.max_steps,
                greedy_head=not args.sample_head,
                show_admissible_actions=args.show_admissible_actions,
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
                totals["actions"] += 1
                totals["admissible_actions"] += int(step.action_was_admissible)
                totals["parsed_turns"] += int(step.parse_ok)
                totals["hmm_applied_turns"] += int(step.hmm_applied)
                totals["head_truncated_turns"] += int(step.head_truncated)
                totals["tail_truncated_turns"] += int(step.tail_truncated)
                totals["exact_tail_span_turns"] += int(step.tail_span_exact)
                totals["prompt_tokens"] += step.prompt_tokens
                totals["generated_tokens"] += step.generated_tokens
                totals["head_latency_seconds"] += step.head_latency_seconds
                totals["action_latency_seconds"] += step.action_latency_seconds

            action_count = max(totals["actions"], 1)
            print(
                f"[{episode_index + 1}/{args.num_episodes}] "
                f"SR={totals['successes'] / totals['episodes']:.3f} "
                f"admissible={totals['admissible_actions'] / action_count:.3f} "
                f"parse={totals['parsed_turns'] / action_count:.3f}"
            )

    action_count = max(totals["actions"], 1)
    summary = {
        "condition": condition.name.value,
        "factors": {
            "use_decision": condition.use_decision,
            "use_dfa": condition.use_dfa,
            "use_hmm": condition.use_hmm,
            "show_admissible_actions": args.show_admissible_actions,
        },
        "model": args.model,
        "hmm": hmm_path,
        "hmm_sha256": artifact_sha256(hmm_path) if hmm_path else None,
        "hmm_training_metadata": hmm_training_metadata,
        "hmm_training_metadata_sha256": (
            file_sha256(hmm_training_metadata) if hmm_training_metadata else None
        ),
        "hmm_training_show_admissible_actions": (
            hmm_training_show_admissible_actions
        ),
        "split": args.split,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "beam_size": args.beam_size,
        "max_head_tokens": args.max_head_tokens,
        "max_action_tokens": args.max_action_tokens,
        "min_action_tokens": args.min_action_tokens,
        "temperature": args.temperature,
        "rollout_temperature": args.rollout_temperature,
        "max_hmm_prefix_tokens": args.max_hmm_prefix_tokens,
        "num_episodes": args.num_episodes,
        "sample_actions": args.sample_actions,
        "sample_head": args.sample_head,
        "show_admissible_actions": args.show_admissible_actions,
        "device": args.device,
        "dtype": args.dtype,
        "episode_gamefiles": episode_gamefiles,
        "episode_manifest_sha256": json_sha256(episode_gamefiles),
        "config": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "skills": str(Path(args.skills).resolve()),
        "skills_sha256": file_sha256(args.skills),
        "git_revision": git_revision(ROOT.parent),
        "source_tree_sha256": source_tree_sha256(ROOT),
        "runtime": runtime_versions(),
        "metrics": {
            **totals,
            "success_rate": totals["successes"] / max(totals["episodes"], 1),
            "admissibility_rate": totals["admissible_actions"] / action_count,
            "parse_rate": totals["parsed_turns"] / action_count,
            "hmm_applied_rate": totals["hmm_applied_turns"] / action_count,
            "head_truncation_rate": totals["head_truncated_turns"] / action_count,
            "tail_truncation_rate": totals["tail_truncated_turns"] / action_count,
            "exact_tail_span_rate": totals["exact_tail_span_turns"] / action_count,
            "mean_prompt_tokens_per_action": totals["prompt_tokens"] / action_count,
            "mean_generated_tokens_per_action": totals["generated_tokens"] / action_count,
            "mean_head_latency_seconds": totals["head_latency_seconds"] / action_count,
            "mean_action_latency_seconds": totals["action_latency_seconds"] / action_count,
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
