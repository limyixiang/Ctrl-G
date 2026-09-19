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
        "model": "m", "tokenizer": "m",
        "prompt_format": "decision_with_persistent_history_no_oracle_v1",
        "split": "eval_out_of_distribution", "seed": 42,
        "max_steps": 50, "beam_size": 8, "max_thought_tokens": 1024,
        "max_decision_tokens": 512, "max_action_tokens": 32,
        "min_action_tokens": 1, "temperature": 1.0,
        "rollout_temperature": 0.7, "max_hmm_prefix_tokens": None,
        "num_episodes": 2, "sample_actions": False, "sample_head": False,
        "device": "cuda", "dtype": "bfloat16", "config_sha256": "c",
        "skills_sha256": "s", "schema_version": 1, "policy_version": 1,
        "source_tree_sha256": "src", "episode_manifest_sha256": "manifest",
        "episode_gamefiles": ["a", "b"],
    }
    baseline = {
        **shared, "condition": "decision_prompt", "hmm": None,
        "hmm_sha256": None,
        "factors": {"use_decision": True, "use_decision_dfa": False,
                    "use_action_dfa": False, "use_hmm": False},
    }
    treatment = {
        **shared, "condition": "decision_ctrlg", "hmm": "h",
        "hmm_sha256": "hh",
        "factors": {"use_decision": True, "use_decision_dfa": True,
                    "use_action_dfa": True, "use_hmm": True},
        "hmm_provenance": {
            "skills_sha256": "s", "schema_version": 1, "policy_version": 1,
            "no_oracle_filtering": True,
            "constrained_decision_collection": True,
            "model": "m", "tokenizer": "m",
            "prompt_format": "decision_with_persistent_history_no_oracle_v1",
        },
    }
    return {"decision_prompt": baseline, "decision_ctrlg": treatment}


class SummaryTests(unittest.TestCase):
    def test_valid_pair(self):
        module.validate_comparable(summaries())

    def test_manifest_mismatch_rejected(self):
        values = summaries()
        values["decision_ctrlg"]["episode_gamefiles"] = ["b", "a"]
        with self.assertRaisesRegex(ValueError, "gamefiles"):
            module.validate_comparable(values)

    def test_policy_or_hmm_provenance_mismatch_rejected(self):
        values = summaries()
        values["decision_ctrlg"]["hmm_provenance"]["policy_version"] = 2
        with self.assertRaisesRegex(ValueError, "policy_version"):
            module.validate_comparable(values)

    def test_model_and_skills_mismatch_rejected(self):
        for field in ("model", "skills_sha256", "prompt_format"):
            values = summaries()
            values["decision_ctrlg"][field] += "-different"
            with self.assertRaisesRegex(ValueError, field):
                module.validate_comparable(values)


if __name__ == "__main__":
    unittest.main()
