"""Validate and summarize prompt-only versus policy-constrained Ctrl-G.

python ctrlg-alfworld/scripts/summarize_results.py --results results/alfworld/pair_4b --out results/alfworld/pair_4b/paired_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld.experiment import condition_choices, get_condition
from ctrlg_alfworld.prompts import PROMPT_FORMAT
from ctrlg_alfworld.hmm_context import HMM_CONTEXT_FORMAT, HMM_SEQUENCE_FORMAT


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


def validate_comparable(summaries: dict[str, dict]) -> None:
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
        if candidate.get(field) != baseline.get(field):
            problems.append(
                f"state_ctrlg: {field}={candidate.get(field)!r} differs "
                f"from state_prompt={baseline.get(field)!r}"
            )
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    summaries = {}
    for condition in condition_choices():
        path = Path(args.results) / f"summary_{condition}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing condition summary: {path}")
        with open(path) as input_file:
            summaries[condition] = json.load(input_file)
    validate_comparable(summaries)

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
        for condition in condition_choices()
    }
    output = {
        "design": "matched two-condition comparison",
        "estimand": "state_ctrlg minus state_prompt",
        "validated_episode_manifest_sha256": summaries["state_prompt"][
            "episode_manifest_sha256"
        ],
        "hmm_artifact": {
            "path": summaries["state_ctrlg"]["hmm"],
            "sha256": summaries["state_ctrlg"]["hmm_sha256"],
        },
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
            for condition in condition_choices()
        },
        "ctrlg_effect": {
            metric: table["state_ctrlg"][metric] - table["state_prompt"][metric]
            for metric in metric_names
        },
    }
    rendered = json.dumps(output, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
