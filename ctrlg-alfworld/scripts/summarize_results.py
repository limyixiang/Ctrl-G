"""Validate and summarize prompt-only versus policy-constrained Ctrl-G."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld.experiment import condition_choices, get_condition


COMPARABILITY_FIELDS = (
    "model", "tokenizer", "prompt_format", "split", "seed", "max_steps", "beam_size", "max_thought_tokens",
    "max_decision_tokens",
    "max_action_tokens", "min_action_tokens", "temperature",
    "rollout_temperature", "max_hmm_prefix_tokens", "num_episodes",
    "sample_actions", "sample_head", "device",
    "dtype", "config_sha256", "skills_sha256", "schema_version",
    "policy_version", "source_tree_sha256",
    "episode_manifest_sha256",
)


def validate_comparable(summaries: dict[str, dict]) -> None:
    conditions = condition_choices()
    baseline = summaries["decision_prompt"]
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
        expected_factors = {
            "use_decision": True,
            "use_decision_dfa": expected.use_decision_dfa,
            "use_action_dfa": expected.use_action_dfa,
            "use_hmm": expected.use_hmm,
        }
        for factor, value in expected_factors.items():
            if candidate["factors"].get(factor) != value:
                problems.append(
                    f"{condition}: factor {factor}={candidate['factors'].get(factor)!r}; "
                    f"expected {value!r}"
                )

    candidate = summaries["decision_ctrlg"]
    for field in COMPARABILITY_FIELDS:
        if candidate.get(field) != baseline.get(field):
            problems.append(
                f"decision_ctrlg: {field}={candidate.get(field)!r} differs "
                f"from decision_prompt={baseline.get(field)!r}"
            )
    if candidate.get("episode_gamefiles") != baseline.get("episode_gamefiles"):
        problems.append("decision_ctrlg: ordered episode gamefiles differ")
    if baseline.get("hmm") is not None or baseline.get("hmm_sha256") is not None:
        problems.append("decision_prompt: baseline must not select an HMM checkpoint")
    if candidate.get("hmm") is None or candidate.get("hmm_sha256") is None:
        problems.append("decision_ctrlg: matched HMM checkpoint and hash are required")
    provenance = candidate.get("hmm_provenance") or {}
    for field in ("model", "tokenizer", "prompt_format", "skills_sha256", "schema_version", "policy_version"):
        if provenance.get(field) != candidate.get(field):
            problems.append(f"decision_ctrlg: HMM provenance {field} mismatch")
    if provenance.get("no_oracle_filtering") is not True:
        problems.append("decision_ctrlg: HMM provenance permits oracle filtering")
    if provenance.get("constrained_decision_collection") is not True:
        problems.append("decision_ctrlg: HMM was not collected under hard decisions")
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
        "success_rate", "admissibility_rate", "decision_schema_validity_rate",
        "action_grammar_validity_rate", "policy_adherence_rate",
        "policy_fallback_rate", "hmm_applied_rate",
        "head_truncation_rate", "tail_truncation_rate", "exact_tail_span_rate",
        "mean_prompt_tokens_per_action", "mean_thought_tokens_per_action",
        "mean_decision_tokens_per_action", "mean_generated_tokens_per_action",
        "mean_head_latency_seconds", "mean_action_latency_seconds",
    )
    table = {
        condition: {
            metric: summaries[condition]["metrics"][metric] for metric in metric_names
        }
        for condition in condition_choices()
    }
    output = {
        "design": "matched two-condition comparison",
        "estimand": "decision_ctrlg minus decision_prompt",
        "validated_episode_manifest_sha256": summaries["decision_prompt"][
            "episode_manifest_sha256"
        ],
        "hmm_artifact": {
            "path": summaries["decision_ctrlg"]["hmm"],
            "sha256": summaries["decision_ctrlg"]["hmm_sha256"],
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
            metric: table["decision_ctrlg"][metric] - table["decision_prompt"][metric]
            for metric in metric_names
        },
    }
    rendered = json.dumps(output, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
