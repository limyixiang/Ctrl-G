"""Validate and summarize the shared-baseline grid from slurm/eval_grid.sh.

python ctrlg-alfworld/scripts/summarize_results.py --results results/alfworld/grid_4b --out results/alfworld/grid_4b/grid_summary.json

Legacy flat, two-condition result directories are also supported.
Sibling eval_<condition>.jsonl files supply episode-level and paired statistics.
Without those files, only recorded aggregate metrics are reported; unavailable
episode statistics and paired comparisons are null. Gamefile lists are used for
validation but never copied into the report.
Use --allow-source-mismatch to permit differing recorded source hashes;
all other validation remains enabled and the hashes are included in the output.
Use --allow-skills-mismatch to permit HMM training skills to differ from the
shared evaluation skills; both hashes are reported and other checks remain active.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld.experiment import condition_choices, get_condition
from ctrlg_alfworld.prompts import PROMPT_FORMAT
from ctrlg_alfworld.hmm_context import HMM_CONTEXT_FORMAT, HMM_SEQUENCE_FORMAT


# Directory name: (evaluation condition, expected HMM context length).
GRID_ARMS = {
    "k8": ("state_ctrlg", 8),
    "k4": ("state_ctrlg", 4),
    "prompt_only": ("state_prompt", None),
}


COMPARABILITY_FIELDS = (
    "model", "tokenizer", "prompt_format", "split", "seed", "max_steps", "beam_size", "max_thought_tokens",
    "hmm_sequence_format", "hmm_context_format", "hmm_context_steps", "max_hmm_context_tokens",
    "max_state_tokens",
    "max_action_tokens", "min_action_tokens", "temperature",
    "rollout_temperature", "max_hmm_prefix_tokens", "num_episodes",
    "sample_actions", "sample_head", "device",
    "dtype", "config_sha256", "skills_sha256", "schema_version",
    "policy_version", "source_tree_sha256",
    "episode_manifest_sha256",
)


METRIC_NAMES = (
    "episodes", "successes", "failures", "success_rate", "actions",
    "admissible_actions", "inadmissible_actions", "admissibility_rate",
    "mean_actions_per_episode", "mean_inadmissible_actions_per_episode",
    "parse_rate", "state_schema_validity_rate", "action_grammar_validity_rate",
    "policy_evaluable_turns", "policy_evaluability_rate", "policy_adherence_rate",
    "policy_fallback_rate", "hmm_applied_rate", "state_hmm_applied_rate",
    "hmm_context_truncation_rate", "head_truncation_rate",
    "thought_truncation_rate", "state_truncation_rate",
    "thought_repair_rate", "state_repair_rate", "tail_truncation_rate",
    "exact_tail_span_rate", "prompt_tokens", "thought_tokens", "state_tokens",
    "generated_tokens", "mean_prompt_tokens_per_action",
    "mean_thought_tokens_per_action", "mean_state_tokens_per_action",
    "mean_generated_tokens_per_action", "mean_prompt_tokens_per_episode",
    "mean_generated_tokens_per_episode", "mean_head_latency_seconds",
    "mean_thought_latency_seconds", "mean_state_latency_seconds",
    "mean_action_latency_seconds", "generation_latency_seconds",
    "mean_generation_latency_seconds_per_action",
    "mean_generation_latency_seconds_per_episode",
)

RESOURCE_FIELDS = (
    "prompt_tokens", "thought_tokens", "state_tokens", "generated_tokens",
    "head_latency_seconds", "action_latency_seconds",
)


def divide(numerator, denominator):
    """Use null for undefined rates instead of implying a measured zero."""
    return numerator / denominator if denominator else None


def distribution(values: list) -> dict:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "p90": ordered[math.ceil(0.9 * len(values)) - 1] if values else None,
        "min": ordered[0] if values else None,
        "max": ordered[-1] if values else None,
    }


def load_episodes(path: Path, summary: dict) -> list[dict] | None:
    """Reject stale/truncated trajectories before using them in a matched report."""
    if not path.exists():
        return None
    episodes = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid episode JSON: {exc.msg}") from exc
            if not isinstance(episode, dict) or not isinstance(episode.get("success"), bool):
                raise ValueError(f"{location}: expected an episode with boolean success")
            if not isinstance(episode.get("gamefile"), str) or not isinstance(episode.get("task_key"), str):
                raise ValueError(f"{location}: expected gamefile and task_key strings")
            steps = episode.get("steps")
            if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
                raise ValueError(f"{location}: expected a list of step objects")
            if type(episode.get("num_steps")) is not int or episode["num_steps"] != len(steps):
                raise ValueError(f"{location}: num_steps does not match steps")
            if len(steps) > summary["max_steps"]:
                raise ValueError(f"{location}: episode exceeds max_steps")
            if episode.get("condition", summary["condition"]) != summary["condition"]:
                raise ValueError(f"{location}: episode condition differs from summary")
            for step in steps:
                if "action_was_admissible" in step and not isinstance(step["action_was_admissible"], bool):
                    raise ValueError(f"{location}: action_was_admissible must be boolean")
                reason = step.get("policy_fallback_reason")
                if reason is not None and not isinstance(reason, str):
                    raise ValueError(f"{location}: policy_fallback_reason must be a string or null")
                for field in RESOURCE_FIELDS:
                    if field in step and (
                        type(step[field]) not in (int, float)
                        or not math.isfinite(step[field]) or step[field] < 0
                    ):
                        raise ValueError(f"{location}: {field} must be finite and nonnegative")
            # Large thoughts, states, token arrays and HMM contexts are not
            # needed for aggregation. Retain only the compact measurement data.
            episodes.append({
                **{field: episode[field] for field in ("gamefile", "task_key", "success", "num_steps")},
                "steps": [
                    {field: step[field] for field in RESOURCE_FIELDS + (
                        "action_was_admissible", "policy_fallback_reason",
                    ) if field in step}
                    for step in steps
                ],
            })
    if [episode["gamefile"] for episode in episodes] != summary["episode_gamefiles"]:
        raise ValueError(f"{path}: ordered episode gamefiles differ from summary")
    totals = episode_totals(episodes)
    if len(episodes) != summary["num_episodes"]:
        raise ValueError(f"{path}: episode count differs from num_episodes")
    for field in ("episodes", "successes", "actions", "admissible_actions"):
        recorded = summary["metrics"].get(field)
        if recorded is not None and field in totals and recorded != totals[field]:
            raise ValueError(f"{path}: {field} differs from summary metrics")
    if episodes and not math.isclose(
        summary["metrics"]["success_rate"], totals["successes"] / len(episodes),
        rel_tol=1e-9, abs_tol=1e-12,
    ):
        raise ValueError(f"{path}: success_rate differs from summary metrics")
    return episodes


def episode_totals(episodes: list[dict]) -> dict:
    steps = [step for episode in episodes for step in episode["steps"]]
    totals = {
        "episodes": len(episodes),
        "successes": sum(episode["success"] for episode in episodes),
        "actions": len(steps),
    }
    if all("action_was_admissible" in step for step in steps):
        totals["admissible_actions"] = sum(step["action_was_admissible"] for step in steps)
    for field in RESOURCE_FIELDS:
        if all(field in step for step in steps):
            totals[field] = sum(step[field] for step in steps)
    return totals


def add_averages(metrics: dict) -> None:
    episodes, actions = metrics["episodes"], metrics.get("actions")
    if "successes" in metrics:
        metrics["failures"] = episodes - metrics["successes"]
        metrics["success_rate"] = divide(metrics["successes"], episodes)
    if actions is not None:
        metrics["mean_actions_per_episode"] = divide(actions, episodes)
        if "admissible_actions" in metrics:
            metrics["inadmissible_actions"] = actions - metrics["admissible_actions"]
            metrics["admissibility_rate"] = divide(metrics["admissible_actions"], actions)
            metrics["mean_inadmissible_actions_per_episode"] = divide(metrics["inadmissible_actions"], episodes)
    for field in ("prompt_tokens", "generated_tokens"):
        if field in metrics:
            metrics[f"mean_{field}_per_episode"] = divide(metrics[field], episodes)
    if actions is not None:
        for field in ("prompt_tokens", "thought_tokens", "state_tokens", "generated_tokens"):
            if field in metrics:
                metrics[f"mean_{field}_per_action"] = divide(metrics[field], actions)
    # Head latency already includes thought and state latency. Adding those
    # component timers again would double count part of generation.
    if all(field in metrics for field in ("head_latency_seconds", "action_latency_seconds")):
        total = metrics["head_latency_seconds"] + metrics["action_latency_seconds"]
        metrics["generation_latency_seconds"] = total
        metrics["mean_generation_latency_seconds_per_action"] = divide(total, actions)
        metrics["mean_generation_latency_seconds_per_episode"] = divide(total, episodes)
    if actions == 0:
        for field in metrics:
            if (
                (field.endswith("_rate") and field != "success_rate")
                or field.endswith("_per_action")
                or field in (
                    "mean_head_latency_seconds", "mean_thought_latency_seconds",
                    "mean_state_latency_seconds", "mean_action_latency_seconds",
                )
            ):
                metrics[field] = None
    if metrics.get("policy_evaluable_turns") == 0:
        metrics["policy_adherence_rate"] = None


def summarize_episodes(episodes: list[dict], max_steps: int) -> dict:
    successes = [episode for episode in episodes if episode["success"]]
    failures = [episode for episode in episodes if not episode["success"]]
    budgets = sorted({budget for budget in (10, 20, 30, max_steps) if 0 < budget <= max_steps})
    at_limit = sum(episode["num_steps"] == max_steps for episode in failures)
    admissibility_known = all(
        "action_was_admissible" in step for episode in episodes for step in episode["steps"]
    )
    bad_episodes = sum(
        any(step.get("action_was_admissible") is False for step in episode["steps"])
        for episode in episodes
    )
    fallback_reasons = Counter()
    fallback_known = True
    for episode in episodes:
        for step in episode["steps"]:
            fallback_known &= "policy_fallback_reason" in step
            if step.get("policy_fallback_reason"):
                fallback_reasons[step["policy_fallback_reason"].split(":", 1)[0]] += 1
    return {
        "actions_per_episode": {
            group: distribution([episode["num_steps"] for episode in records])
            for group, records in (("all", episodes), ("successful", successes), ("failed", failures))
        },
        "success_within_actions": {
            str(budget): {
                "successes": sum(episode["num_steps"] <= budget for episode in successes),
                "success_rate": divide(sum(episode["num_steps"] <= budget for episode in successes), len(episodes)),
            }
            for budget in budgets
        },
        "failed_at_step_limit": at_limit,
        "fraction_of_failures_at_step_limit": divide(at_limit, len(failures)),
        "episodes_with_inadmissible_actions": bad_episodes if admissibility_known else None,
        "episodes_with_inadmissible_actions_rate": divide(bad_episodes, len(episodes)) if admissibility_known else None,
        "policy_fallback_reasons": dict(sorted(fallback_reasons.items())) if fallback_known else None,
    }


def summarize_condition(summary: dict, episodes: list[dict] | None) -> dict:
    metrics = dict(summary["metrics"])
    metrics.setdefault("episodes", len(summary["episode_gamefiles"]))
    if episodes is not None:
        metrics.update(episode_totals(episodes))
    add_averages(metrics)
    result = {field: metrics[field] for field in METRIC_NAMES if field in metrics}
    result["episode_details_available"] = episodes is not None
    result["episode_statistics"] = (
        summarize_episodes(episodes, summary["max_steps"]) if episodes is not None else None
    )
    if episodes is None:
        result["per_task_type"] = summary.get("per_task_type", {})
    else:
        groups = defaultdict(list)
        for episode in episodes:
            groups[episode["task_key"]].append(episode)
        per_task = {}
        for task_key, records in sorted(groups.items()):
            task_metrics = episode_totals(records)
            add_averages(task_metrics)
            per_task[task_key] = {
                field: task_metrics[field] for field in (
                    "episodes", "successes", "failures", "success_rate", "actions",
                    "mean_actions_per_episode", "inadmissible_actions",
                    "mean_prompt_tokens_per_episode", "mean_generated_tokens_per_episode",
                    "mean_generation_latency_seconds_per_episode",
                ) if field in task_metrics
            }
            for label, success in (("successful", True), ("failed", False)):
                values = [record["num_steps"] for record in records if record["success"] is success]
                per_task[task_key][f"mean_actions_per_{label}_episode"] = mean(values) if values else None
        result["per_task_type"] = per_task
    result["hmm_skip_reasons"] = {
        "state": summary.get("state_hmm_skip_reasons", {}),
        "action": summary.get("action_hmm_skip_reasons", {}),
    }
    return result


def compare_episodes(treatment: list[dict] | None, baseline: list[dict] | None) -> dict | None:
    if treatment is None or baseline is None:
        return None
    if [(ep["gamefile"], ep["task_key"]) for ep in treatment] != [
        (ep["gamefile"], ep["task_key"]) for ep in baseline
    ]:
        raise ValueError("paired trajectories have different ordered episodes or task types")
    pairs = list(zip(treatment, baseline))
    both_successful = [(a, b) for a, b in pairs if a["success"] and b["success"]]
    action_deltas = [a["num_steps"] - b["num_steps"] for a, b in both_successful]
    return {
        "episodes": len(pairs),
        "ctrlg_only_successes": sum(a["success"] and not b["success"] for a, b in pairs),
        "baseline_only_successes": sum(b["success"] and not a["success"] for a, b in pairs),
        "both_successes": len(both_successful),
        "both_failures": sum(not a["success"] and not b["success"] for a, b in pairs),
        "actions_on_both_successful": {
            "episodes": len(both_successful),
            "ctrlg_mean": mean([a["num_steps"] for a, _ in both_successful]) if both_successful else None,
            "baseline_mean": mean([b["num_steps"] for _, b in both_successful]) if both_successful else None,
            "ctrlg_minus_baseline": distribution(action_deltas),
            "ctrlg_fewer_actions": sum(delta < 0 for delta in action_deltas),
            "same_actions": sum(delta == 0 for delta in action_deltas),
            "ctrlg_more_actions": sum(delta > 0 for delta in action_deltas),
        },
    }


def validate_comparable(
    summaries: dict[str, dict], *, hmm_context_steps: int | None = None,
    allow_source_mismatch: bool = False,
    allow_skills_mismatch: bool = False,
) -> None:
    """Validate a pair, allowing the grid's specified HMM context length."""

    conditions = condition_choices()
    baseline = summaries["state_prompt"]
    problems = []
    for condition in conditions:
        candidate = summaries[condition]
        missing = [
            field for field in COMPARABILITY_FIELDS + ("episode_gamefiles", "factors")
            if field not in candidate
        ]
        if missing:
            problems.append(f"{condition}: missing required fields {', '.join(missing)}")
            continue
        evaluation_skills_hash = candidate["skills_sha256"]
        if not isinstance(evaluation_skills_hash, str) or not evaluation_skills_hash:
            problems.append(f"{condition}: requires a recorded skills_sha256")
        if candidate.get("condition") != condition:
            problems.append(f"{condition}: expected condition {condition!r}")
        expected = get_condition(condition)
        if candidate["prompt_format"] != PROMPT_FORMAT:
            problems.append(f"{condition}: expected prompt_format {PROMPT_FORMAT!r}")
        expected_factors = {
            "use_state": True,
            "use_state_dfa": expected.use_state_dfa,
            "use_action_dfa": expected.use_action_dfa,
            "use_hmm": expected.use_hmm,
        }
        for factor, value in expected_factors.items():
            if candidate["factors"].get(factor) != value:
                problems.append(
                    f"{condition}: factor {factor}={candidate['factors'].get(factor)!r}; "
                    f"expected {value!r}"
                )

    candidate = summaries["state_ctrlg"]
    for field in COMPARABILITY_FIELDS:
        # The shared prompt-only baseline does not use HMM context. Each grid
        # treatment must instead match its arm and its own checkpoint metadata.
        if field == "hmm_context_steps" and hmm_context_steps is not None:
            continue
        if field == "source_tree_sha256" and allow_source_mismatch:
            continue
        if candidate.get(field) != baseline.get(field):
            problems.append(
                f"state_ctrlg: {field}={candidate.get(field)!r} differs "
                f"from state_prompt={baseline.get(field)!r}"
            )
    if hmm_context_steps is not None and candidate.get("hmm_context_steps") != hmm_context_steps:
        problems.append(f"state_ctrlg: expected hmm_context_steps={hmm_context_steps}")
    if candidate.get("episode_gamefiles") != baseline.get("episode_gamefiles"):
        problems.append("state_ctrlg: ordered episode gamefiles differ")
    if baseline.get("hmm") is not None or baseline.get("hmm_sha256") is not None:
        problems.append("state_prompt: baseline must not select an HMM checkpoint")
    if candidate.get("hmm") is None or candidate.get("hmm_sha256") is None:
        problems.append("state_ctrlg: matched HMM checkpoint and hash are required")
    provenance = candidate.get("hmm_provenance") or {}
    training_skills_hash = provenance.get("skills_sha256")
    if not isinstance(training_skills_hash, str) or not training_skills_hash:
        problems.append("state_ctrlg: HMM provenance requires a recorded skills_sha256")
    for field in ("model", "tokenizer", "prompt_format", "skills_sha256", "schema_version", "policy_version"):
        if field == "skills_sha256" and allow_skills_mismatch:
            continue
        if provenance.get(field) != candidate.get(field):
            problems.append(f"state_ctrlg: HMM provenance {field} mismatch")
    if provenance.get("no_oracle_filtering") is not True:
        problems.append("state_ctrlg: HMM provenance permits oracle filtering")
    if provenance.get("constrained_state_collection") is not False:
        problems.append("state_ctrlg: HMM was not collected from unmasked attempts")
    if provenance.get("selection_policy") != "all_observed_attempts":
        problems.append("state_ctrlg: HMM must retain all observed attempts")
    for field in ("hmm_sequence_format", "hmm_context_format", "hmm_context_steps", "max_hmm_context_tokens"):
        if provenance.get(field) != candidate.get(field):
            problems.append(f"state_ctrlg: HMM provenance {field} mismatch")
    if provenance.get("hmm_sequence_format") != HMM_SEQUENCE_FORMAT or provenance.get("hmm_context_format") != HMM_CONTEXT_FORMAT:
        problems.append("state_ctrlg: obsolete HMM sequence/context format")
    if candidate.get("hmm_scope") != "state_and_action":
        problems.append("state_ctrlg: expected state and action HMM guidance")
    if problems:
        raise ValueError(
            "condition summaries are not a comparable matched pair:\n- "
            + "\n- ".join(problems)
        )


def validate_grid(
    summaries: dict[str, dict], *, allow_source_mismatch: bool = False,
    allow_skills_mismatch: bool = False,
) -> None:
    missing = [arm for arm in GRID_ARMS if arm not in summaries]
    if missing:
        raise ValueError(f"missing grid arms: {', '.join(missing)}")
    for arm, (condition, context_steps) in GRID_ARMS.items():
        if condition == "state_prompt":
            continue
        try:
            validate_comparable(
                {"state_prompt": summaries["prompt_only"], condition: summaries[arm]},
                hmm_context_steps=context_steps,
                allow_source_mismatch=allow_source_mismatch,
                allow_skills_mismatch=allow_skills_mismatch,
            )
        except ValueError as exc:
            raise ValueError(f"{arm}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", required=True,
        help="Grid root containing k8/, k4/, and prompt_only/ (or a legacy pair directory)",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--allow-source-mismatch", action="store_true",
        help="Allow differing source_tree_sha256 values; retain all other checks and report the hashes",
    )
    parser.add_argument(
        "--allow-skills-mismatch", action="store_true",
        help="Allow HMM training skills_sha256 to differ from evaluation; evaluation arms must still match",
    )
    args = parser.parse_args()
    root = Path(args.results)
    is_grid = any((root / arm).is_dir() for arm in GRID_ARMS) or not any(
        (root / f"summary_{condition}.json").is_file()
        for condition in condition_choices()
    )
    paths = (
        {arm: root / arm / f"summary_{condition}.json"
         for arm, (condition, _) in GRID_ARMS.items()}
        if is_grid else
        {condition: root / f"summary_{condition}.json" for condition in condition_choices()}
    )
    summaries = {}
    for arm, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing condition summary: {path}")
        with path.open(encoding="utf-8") as input_file:
            summaries[arm] = json.load(input_file)
    if is_grid:
        validate_grid(
            summaries, allow_source_mismatch=args.allow_source_mismatch,
            allow_skills_mismatch=args.allow_skills_mismatch,
        )
    else:
        validate_comparable(
            summaries, allow_source_mismatch=args.allow_source_mismatch,
            allow_skills_mismatch=args.allow_skills_mismatch,
        )
    baseline = "prompt_only" if is_grid else "state_prompt"
    treatments = ("k8", "k4") if is_grid else ("state_ctrlg",)

    episodes = {
        arm: load_episodes(path.with_name(f"eval_{summaries[arm]['condition']}.jsonl"), summaries[arm])
        for arm, path in paths.items()
    }
    table = {
        arm: summarize_condition(summary, episodes[arm]) for arm, summary in summaries.items()
    }
    output = {
        "report_schema_version": 2,
        "design": "matched three-arm grid comparison" if is_grid else "matched two-condition comparison",
        "estimand": "each Ctrl-G arm minus prompt_only" if is_grid else "state_ctrlg minus state_prompt",
        "model": summaries[baseline]["model"],
        "split": summaries[baseline]["split"],
        "seed": summaries[baseline]["seed"],
        "max_steps": summaries[baseline]["max_steps"],
        "validated_episode_manifest_sha256": summaries[baseline][
            "episode_manifest_sha256"
        ],
        "conditions": table,
        "paired_comparisons": {
            arm: compare_episodes(episodes[arm], episodes[baseline]) for arm in treatments
        },
        "per_rule": {
            condition: {
                "activations": summaries[condition]["metrics"].get(
                    "per_rule_activations", {}
                ),
                "violations": summaries[condition]["metrics"].get(
                    "per_rule_violations", {}
                ),
            }
            for condition in summaries
        },
    }
    artifacts = {
        arm: {"path": summaries[arm]["hmm"], "sha256": summaries[arm]["hmm_sha256"]}
        for arm in treatments
    }
    effects = {
        arm: {
            metric: table[arm][metric] - table[baseline][metric]
            for metric in METRIC_NAMES
            if type(table[arm].get(metric)) in (int, float)
            and type(table[baseline].get(metric)) in (int, float)
            and metric != "episodes"
        }
        for arm in treatments
    }
    if is_grid:
        output["hmm_artifacts"] = {
            arm: {**artifacts[arm], "hmm_context_steps": summaries[arm]["hmm_context_steps"]}
            for arm in treatments
        }
        output["ctrlg_effect"] = effects
    else:
        output["hmm_artifact"] = artifacts["state_ctrlg"]
        output["ctrlg_effect"] = effects["state_ctrlg"]
    if args.allow_source_mismatch:
        source_hashes = {
            arm: summary["source_tree_sha256"] for arm, summary in summaries.items()
        }
        output["source_tree_validation"] = {
            "allow_source_mismatch": True,
            "matched": len(set(source_hashes.values())) == 1,
            "sha256_by_arm": source_hashes,
        }
    if args.allow_skills_mismatch:
        evaluation_hash = summaries[baseline]["skills_sha256"]
        training_hashes = {
            arm: summaries[arm]["hmm_provenance"]["skills_sha256"]
            for arm in treatments
        }
        output["skills_validation"] = {
            "allow_skills_mismatch": True,
            "evaluation_sha256": evaluation_hash,
            "hmm_training_sha256_by_arm": training_hashes,
            "matched_by_arm": {
                arm: training_hash == evaluation_hash
                for arm, training_hash in training_hashes.items()
            },
        }
    rendered = json.dumps(output, indent=2, allow_nan=False)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
