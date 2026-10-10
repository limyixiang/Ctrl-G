import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_results", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def summaries():
    shared = {
        "hmm_sequence_format": "context_state_action_censored_v1",
        "hmm_context_format": "observable_trajectory_json_v1",
        "hmm_context_steps": 1, "max_hmm_context_tokens": 1024,
        "model": "m", "tokenizer": "m",
        "prompt_format": "state_with_persistent_history_no_oracle_v1",
        "split": "eval_out_of_distribution", "seed": 42,
        "max_steps": 50, "beam_size": 8, "max_thought_tokens": 1024,
        "max_state_tokens": 512, "max_action_tokens": 32,
        "min_action_tokens": 1, "temperature": 1.0,
        "rollout_temperature": 0.7, "max_hmm_prefix_tokens": None,
        "num_episodes": 2, "sample_actions": False, "sample_head": False,
        "device": "cuda", "dtype": "bfloat16", "config_sha256": "c",
        "skills_sha256": "s", "schema_version": 1, "policy_version": 1,
        "source_tree_sha256": "src", "episode_manifest_sha256": "manifest",
        "episode_gamefiles": ["a", "b"],
    }
    baseline = {
        **shared, "condition": "state_prompt", "hmm": None,
        "hmm_sha256": None,
        "factors": {"use_state": True, "use_state_dfa": False,
                    "use_action_dfa": False, "use_hmm": False},
    }
    treatment = {
        **shared, "hmm_scope": "state_and_action", "condition": "state_ctrlg", "hmm": "h",
        "hmm_sha256": "hh",
        "factors": {"use_state": True, "use_state_dfa": True,
                    "use_action_dfa": True, "use_hmm": True},
        "hmm_provenance": {
            "skills_sha256": "s", "schema_version": 1, "policy_version": 1,
            "no_oracle_filtering": True,
            "constrained_state_collection": False,
                "hmm_sequence_format": "context_state_action_censored_v1",
                "hmm_context_format": "observable_trajectory_json_v1",
                "hmm_context_steps": 1, "max_hmm_context_tokens": 1024,
                "selection_policy": "all_observed_attempts",
            "model": "m", "tokenizer": "m",
            "prompt_format": "state_with_persistent_history_no_oracle_v1",
        },
    }
    return {"state_prompt": baseline, "state_ctrlg": treatment}


class SummaryTests(unittest.TestCase):
    def test_valid_pair(self):
        module.validate_comparable(summaries())

    def test_matching_legacy_prompt_formats_are_rejected(self):
        values = summaries()
        legacy = "decision_with_persistent_history_no_oracle_v1"
        for candidate in values.values():
            candidate["prompt_format"] = legacy
        values["state_ctrlg"]["hmm_provenance"]["prompt_format"] = legacy
        with self.assertRaisesRegex(ValueError, "prompt_format"):
            module.validate_comparable(values)

    def test_manifest_mismatch_rejected(self):
        values = summaries()
        values["state_ctrlg"]["episode_gamefiles"] = ["b", "a"]
        with self.assertRaisesRegex(ValueError, "gamefiles"):
            module.validate_comparable(values)

    def test_policy_or_hmm_provenance_mismatch_rejected(self):
        values = summaries()
        values["state_ctrlg"]["hmm_provenance"]["policy_version"] = 2
        with self.assertRaisesRegex(ValueError, "policy_version"):
            module.validate_comparable(values)

    def test_model_and_skills_mismatch_rejected(self):
        for field in ("model", "skills_sha256", "prompt_format"):
            values = summaries()
            values["state_ctrlg"][field] += "-different"
            with self.assertRaisesRegex(ValueError, field):
                module.validate_comparable(values)


if __name__ == "__main__":
    unittest.main()
