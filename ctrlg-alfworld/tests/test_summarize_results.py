import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


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
        "metrics": dict.fromkeys((
            "success_rate", "admissibility_rate", "state_schema_validity_rate",
            "action_grammar_validity_rate", "policy_adherence_rate",
            "policy_fallback_rate", "hmm_applied_rate", "state_hmm_applied_rate",
            "hmm_context_truncation_rate", "head_truncation_rate",
            "tail_truncation_rate", "exact_tail_span_rate",
            "mean_prompt_tokens_per_action", "mean_thought_tokens_per_action",
            "mean_state_tokens_per_action", "mean_generated_tokens_per_action",
            "mean_head_latency_seconds", "mean_state_latency_seconds",
            "mean_action_latency_seconds",
        ), 0.0),
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


def grid_summaries():
    pair = summaries()
    values = {"prompt_only": pair["state_prompt"]}
    for arm, steps in (("k8", 8), ("k4", 4)):
        candidate = deepcopy(pair["state_ctrlg"])
        candidate["hmm_context_steps"] = steps
        candidate["hmm_provenance"]["hmm_context_steps"] = steps
        candidate["hmm"] = f"checkpoint_{arm}"
        candidate["hmm_sha256"] = f"hash_{arm}"
        values[arm] = candidate
    return values


class SummaryTests(unittest.TestCase):
    def write_summaries(self, root, values, *, grid=True):
        for arm, summary in values.items():
            directory = root / arm if grid else root
            directory.mkdir(exist_ok=True)
            path = directory / f"summary_{summary['condition']}.json"
            path.write_text(json.dumps(summary))

    def run_main(self, root, *extra_args):
        output_path = root / "summary.json"
        stdout = io.StringIO()
        with patch.object(sys, "argv", [
            str(SCRIPT), "--results", str(root), "--out", str(output_path), *extra_args,
        ]), redirect_stdout(stdout):
            module.main()
        output = json.loads(stdout.getvalue())
        self.assertEqual(output, json.loads(output_path.read_text()))
        return output

    def test_valid_pair(self):
        module.validate_comparable(summaries())

    def test_legacy_pair_requires_matching_context_lengths(self):
        values = summaries()
        values["state_ctrlg"]["hmm_context_steps"] = 8
        values["state_ctrlg"]["hmm_provenance"]["hmm_context_steps"] = 8
        with self.assertRaisesRegex(ValueError, "hmm_context_steps"):
            module.validate_comparable(values)

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

    def test_valid_grid_with_distinct_context_lengths(self):
        module.validate_grid(grid_summaries())

    def test_grid_rejects_wrong_arm_context_length(self):
        for arm, wrong_steps in (("k8", 4), ("k4", 8)):
            with self.subTest(arm=arm):
                values = grid_summaries()
                values[arm]["hmm_context_steps"] = wrong_steps
                values[arm]["hmm_provenance"]["hmm_context_steps"] = wrong_steps
                with self.assertRaisesRegex(ValueError, f"(?s){arm}:.*hmm_context_steps"):
                    module.validate_grid(values)

    def test_grid_validates_each_checkpoint_provenance(self):
        for arm in ("k8", "k4"):
            with self.subTest(arm=arm):
                values = grid_summaries()
                values[arm]["hmm_provenance"]["hmm_context_steps"] = 1
                with self.assertRaisesRegex(ValueError, "HMM provenance hmm_context_steps"):
                    module.validate_grid(values)

    def test_grid_requires_shared_model_settings_and_episodes(self):
        for arm in ("k8", "k4"):
            for field, wrong_value in (
                ("model", "other-model"), ("seed", 43),
                ("max_hmm_context_tokens", 512),
                ("episode_manifest_sha256", "other-manifest"),
                ("episode_gamefiles", ["b", "a"]),
            ):
                with self.subTest(arm=arm, field=field):
                    values = grid_summaries()
                    values[arm][field] = wrong_value
                    with self.assertRaises(ValueError):
                        module.validate_grid(values)

    def test_grid_rejects_wrong_condition(self):
        values = grid_summaries()
        values["k4"]["condition"] = "state_prompt"
        with self.assertRaisesRegex(ValueError, "expected condition 'state_ctrlg'"):
            module.validate_grid(values)

    def test_grid_baseline_must_not_select_hmm(self):
        values = grid_summaries()
        values["prompt_only"]["hmm"] = "checkpoint"
        with self.assertRaisesRegex(ValueError, "baseline must not select an HMM"):
            module.validate_grid(values)

    def test_grid_cli_reports_both_effects_and_artifacts(self):
        values = grid_summaries()
        for arm, rate, latency in (
            ("prompt_only", 0.25, 4.0), ("k8", 0.75, 6.0), ("k4", 0.5, 5.0),
        ):
            values[arm]["metrics"]["success_rate"] = rate
            values[arm]["metrics"]["mean_action_latency_seconds"] = latency
        values["k8"]["metrics"]["per_rule_activations"] = {"rule": 2}
        values["k4"]["metrics"]["per_rule_violations"] = {"rule": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summaries(root, values)
            output = self.run_main(root)
        self.assertEqual(output["model"], "m")
        self.assertEqual(output["validated_episode_manifest_sha256"], "manifest")
        self.assertEqual(set(output["conditions"]), {"k8", "k4", "prompt_only"})
        for arm, rate_delta, latency_delta, steps in (
            ("k8", 0.5, 2.0, 8), ("k4", 0.25, 1.0, 4),
        ):
            self.assertEqual(output["ctrlg_effect"][arm]["success_rate"], rate_delta)
            self.assertEqual(output["ctrlg_effect"][arm]["mean_action_latency_seconds"], latency_delta)
            self.assertEqual(output["hmm_artifacts"][arm], {
                "path": f"checkpoint_{arm}", "sha256": f"hash_{arm}",
                "hmm_context_steps": steps,
            })
        self.assertEqual(output["per_rule"]["k8"]["activations"], {"rule": 2})
        self.assertEqual(output["per_rule"]["k4"]["violations"], {"rule": 1})
        self.assertEqual(output["per_rule"]["prompt_only"]["violations"], {})

    def test_grid_cli_requires_all_arm_files(self):
        for arm in ("k8", "k4", "prompt_only"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                values = grid_summaries()
                del values[arm]
                self.write_summaries(root, values)
                with self.assertRaisesRegex(FileNotFoundError, arm):
                    self.run_main(root)

    def test_legacy_pair_cli_preserves_output_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summaries(root, summaries(), grid=False)
            output = self.run_main(root)
        self.assertEqual(output["design"], "matched two-condition comparison")
        self.assertEqual(output["estimand"], "state_ctrlg minus state_prompt")
        self.assertEqual(set(output["conditions"]), {"state_ctrlg", "state_prompt"})
        self.assertEqual(output["hmm_artifact"], {"path": "h", "sha256": "hh"})
        self.assertEqual(output["ctrlg_effect"]["success_rate"], 0.0)

    def test_source_mismatch_requires_override_and_records_hashes(self):
        for arm in ("k8", "k4", "state_ctrlg"):
            grid = arm != "state_ctrlg"
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                values = grid_summaries() if grid else summaries()
                values[arm]["source_tree_sha256"] = "different-source"
                root = Path(directory)
                self.write_summaries(root, values, grid=grid)
                with self.assertRaisesRegex(ValueError, "source_tree_sha256"):
                    self.run_main(root)
                output = self.run_main(root, "--allow-source-mismatch")
                self.assertEqual(output["source_tree_validation"], {
                    "allow_source_mismatch": True,
                    "matched": False,
                    "sha256_by_arm": {
                        name: summary["source_tree_sha256"]
                        for name, summary in values.items()
                    },
                })

    def test_source_override_with_matching_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summaries(root, grid_summaries())
            output = self.run_main(root, "--allow-source-mismatch")
        self.assertTrue(output["source_tree_validation"]["matched"])

    def test_source_override_preserves_other_checks(self):
        for grid in (True, False):
            for field, value, error in (
                ("model", "other-model", "model"),
                ("episode_gamefiles", ["b", "a"], "ordered episode gamefiles"),
                ("hmm_provenance", {}, "HMM provenance"),
                ("hmm_context_steps", 2, "hmm_context_steps"),
            ):
                with self.subTest(grid=grid, field=field):
                    values = grid_summaries() if grid else summaries()
                    arm = "k4" if grid else "state_ctrlg"
                    values[arm]["source_tree_sha256"] = "different-source"
                    values[arm][field] = value
                    validate = module.validate_grid if grid else module.validate_comparable
                    with self.assertRaisesRegex(ValueError, error):
                        validate(values, allow_source_mismatch=True)

    def test_source_override_still_requires_recorded_hash(self):
        values = grid_summaries()
        del values["k8"]["source_tree_sha256"]
        with self.assertRaisesRegex(ValueError, "missing required fields source_tree_sha256"):
            module.validate_grid(values, allow_source_mismatch=True)


if __name__ == "__main__":
    unittest.main()
