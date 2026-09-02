"""Validate and summarize the matched DFA versus DFA+HMM comparison."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld.experiment import condition_choices, get_condition


COMPARABILITY_FIELDS = (
    "model", "split", "seed", "max_steps", "beam_size", "max_head_tokens",
    "max_action_tokens", "min_action_tokens", "temperature",
    "rollout_temperature", "max_hmm_prefix_tokens", "num_episodes",
    "sample_actions", "sample_head", "show_admissible_actions", "device",
    "dtype", "config_sha256", "skills_sha256", "source_tree_sha256",
    "episode_manifest_sha256",
)


def validate_comparable(summaries: dict[str, dict]) -> None:
    conditions = condition_choices()
    baseline = summaries["decision_dfa"]
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
            "use_dfa": True,
            "use_hmm": expected.use_hmm,
        }
        for factor, value in expected_factors.items():
            if candidate["factors"].get(factor) != value:
                problems.append(
                    f"{condition}: factor {factor}={candidate['factors'].get(factor)!r}; "
                    f"expected {value!r}"
                )

    candidate = summaries["decision_dfa_hmm"]
    for field in COMPARABILITY_FIELDS:
        if candidate.get(field) != baseline.get(field):
            problems.append(
                f"decision_dfa_hmm: {field}={candidate.get(field)!r} differs "
                f"from decision_dfa={baseline.get(field)!r}"
            )
    if candidate.get("episode_gamefiles") != baseline.get("episode_gamefiles"):
        problems.append("decision_dfa_hmm: ordered episode gamefiles differ")
    if baseline.get("hmm") is not None or baseline.get("hmm_sha256") is not None:
        problems.append("decision_dfa: baseline must not select an HMM checkpoint")
    if candidate.get("hmm") is None or candidate.get("hmm_sha256") is None:
        problems.append("decision_dfa_hmm: matched HMM checkpoint and hash are required")
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
        "success_rate", "admissibility_rate", "parse_rate", "hmm_applied_rate",
        "head_truncation_rate", "tail_truncation_rate", "exact_tail_span_rate",
        "mean_prompt_tokens_per_action", "mean_generated_tokens_per_action",
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
        "estimand": "decision_dfa_hmm minus decision_dfa",
        "validated_episode_manifest_sha256": summaries["decision_dfa"][
            "episode_manifest_sha256"
        ],
        "hmm_artifact": {
            "path": summaries["decision_dfa_hmm"]["hmm"],
            "sha256": summaries["decision_dfa_hmm"]["hmm_sha256"],
        },
        "conditions": table,
        "hmm_effect": {
            metric: table["decision_dfa_hmm"][metric] - table["decision_dfa"][metric]
            for metric in metric_names
        },
    }
    rendered = json.dumps(output, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
