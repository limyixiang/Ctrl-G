"""Validate and summarize the shared-baseline grid from slurm/eval_grid.sh.

python ctrlg-alfworld/scripts/summarize_results.py --results results/alfworld/grid_4b --out results/alfworld/grid_4b/grid_summary.json

Legacy flat, two-condition result directories are also supported.
"""

import argparse
import json
import sys
from pathlib import Path

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


def validate_comparable(
    summaries: dict[str, dict], *, hmm_context_steps: int | None = None
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
    for field in ("model", "tokenizer", "prompt_format", "skills_sha256", "schema_version", "policy_version"):
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


def validate_grid(summaries: dict[str, dict]) -> None:
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
        with open(path) as input_file:
            summaries[arm] = json.load(input_file)
    if is_grid:
        validate_grid(summaries)
    else:
        validate_comparable(summaries)
    baseline = "prompt_only" if is_grid else "state_prompt"
    treatments = ("k8", "k4") if is_grid else ("state_ctrlg",)

    metric_names = (
        "success_rate", "admissibility_rate", "state_schema_validity_rate",
        "action_grammar_validity_rate", "policy_adherence_rate",
        "policy_fallback_rate", "hmm_applied_rate", "state_hmm_applied_rate",
        "hmm_context_truncation_rate",
        "head_truncation_rate", "tail_truncation_rate", "exact_tail_span_rate",
        "mean_prompt_tokens_per_action", "mean_thought_tokens_per_action",
        "mean_state_tokens_per_action", "mean_generated_tokens_per_action",
        "mean_head_latency_seconds", "mean_state_latency_seconds", "mean_action_latency_seconds",
    )
    table = {
        condition: {
            metric: summaries[condition]["metrics"][metric] for metric in metric_names
        }
        for condition in summaries
    }
    output = {
        "design": "matched three-arm grid comparison" if is_grid else "matched two-condition comparison",
        "estimand": "each Ctrl-G arm minus prompt_only" if is_grid else "state_ctrlg minus state_prompt",
        "validated_episode_manifest_sha256": summaries[baseline][
            "episode_manifest_sha256"
        ],
        "conditions": table,
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
        arm: {metric: table[arm][metric] - table[baseline][metric] for metric in metric_names}
        for arm in treatments
    }
    if is_grid:
        output["model"] = summaries[baseline]["model"]
        output["hmm_artifacts"] = {
            arm: {**artifacts[arm], "hmm_context_steps": summaries[arm]["hmm_context_steps"]}
            for arm in treatments
        }
        output["ctrlg_effect"] = effects
    else:
        output["hmm_artifact"] = artifacts["state_ctrlg"]
        output["ctrlg_effect"] = effects["state_ctrlg"]
    rendered = json.dumps(output, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
