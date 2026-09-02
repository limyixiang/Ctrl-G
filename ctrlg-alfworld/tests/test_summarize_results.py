import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_results", SCRIPT)
summarize_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize_results)


def make_summaries():
    shared = {
        "model": "model", "split": "eval_out_of_distribution", "seed": 42,
        "max_steps": 50, "beam_size": 8, "max_head_tokens": 512,
        "max_action_tokens": 24, "min_action_tokens": 1, "temperature": 1.0,
        "rollout_temperature": 0.7, "max_hmm_prefix_tokens": None,
        "num_episodes": 2, "sample_actions": False, "sample_head": False,
        "show_admissible_actions": False, "device": "cuda", "dtype": "bfloat16",
        "config_sha256": "config", "skills_sha256": "skills",
        "source_tree_sha256": "source", "episode_manifest_sha256": "manifest",
        "episode_gamefiles": ["game-a", "game-b"],
    }
    return {
        "decision_dfa": {
            **shared,
            "hmm": None,
            "hmm_sha256": None,
            "factors": {"use_decision": True, "use_dfa": True, "use_hmm": False},
        },
        "decision_dfa_hmm": {
            **shared,
            "hmm": "decision-checkpoint",
            "hmm_sha256": "sha-decision-checkpoint",
            "factors": {"use_decision": True, "use_dfa": True, "use_hmm": True},
        },
    }


class SummarizeResultsTests(unittest.TestCase):
    def test_accepts_matching_pair(self):
        summarize_results.validate_comparable(make_summaries())

    def test_rejects_different_episode_order(self):
        summaries = make_summaries()
        summaries["decision_dfa_hmm"]["episode_gamefiles"] = ["game-b", "game-a"]
        with self.assertRaisesRegex(ValueError, "episode gamefiles differ"):
            summarize_results.validate_comparable(summaries)

    def test_rejects_different_generation_setting(self):
        summaries = make_summaries()
        summaries["decision_dfa_hmm"]["beam_size"] = 4
        with self.assertRaisesRegex(ValueError, "beam_size"):
            summarize_results.validate_comparable(summaries)

    def test_requires_hmm_artifact_only_in_hmm_cell(self):
        summaries = make_summaries()
        summaries["decision_dfa_hmm"]["hmm"] = None
        with self.assertRaisesRegex(ValueError, "checkpoint and hash are required"):
            summarize_results.validate_comparable(summaries)

    def test_rejects_condition_without_decision_memory(self):
        summaries = make_summaries()
        summaries["decision_dfa"]["factors"]["use_decision"] = False
        with self.assertRaisesRegex(ValueError, "use_decision"):
            summarize_results.validate_comparable(summaries)


if __name__ == "__main__":
    unittest.main()
