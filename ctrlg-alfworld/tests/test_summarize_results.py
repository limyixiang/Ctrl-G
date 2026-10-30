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


def episode_record(gamefile, success, num_steps, *, task_key="put", bad_actions=0):
    return {
        "gamefile": gamefile, "task_key": task_key, "success": success,
        "num_steps": num_steps,
        "steps": [
            {
                "action_was_admissible": index >= bad_actions,
                "prompt_tokens": 100, "thought_tokens": 5,
                "state_tokens": 3, "generated_tokens": 10,
                "head_latency_seconds": 2.0, "thought_latency_seconds": 1.0,
                "state_latency_seconds": 1.0, "action_latency_seconds": 0.5,
                "policy_fallback_reason": None,
            }
            for index in range(num_steps)
        ],
    }


def evaluation_summaries(episodes, *, grid=True):
    values = deepcopy(grid_summaries() if grid else summaries())
    for arm, records in episodes.items():
        summary = values[arm]
        # The legacy-pair fixtures share a metrics object; real JSON files do not.
        summary["metrics"] = dict(summary["metrics"])
        summary["num_episodes"] = len(records)
        summary["episode_gamefiles"] = [record["gamefile"] for record in records]
        summary["metrics"].update(module.episode_totals(records))
        summary["metrics"]["success_rate"] = (
            sum(record["success"] for record in records) / len(records) if records else 0.0
        )
    return values


class SummaryTests(unittest.TestCase):
    def write_summaries(self, root, values, *, grid=True, episodes=None):
        for arm, summary in values.items():
            directory = root / arm if grid else root
            directory.mkdir(exist_ok=True)
            path = directory / f"summary_{summary['condition']}.json"
            path.write_text(json.dumps(summary))
            if episodes is not None and arm in episodes:
                trace_path = directory / f"eval_{summary['condition']}.jsonl"
                trace_path.write_text("".join(json.dumps(record) + "\n" for record in episodes[arm]))

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

    def test_skills_mismatch_requires_override_and_reports_both_hashes(self):
        for grid in (True, False):
            with self.subTest(grid=grid), tempfile.TemporaryDirectory() as directory:
                values = grid_summaries() if grid else summaries()
                treatments = ("k8", "k4") if grid else ("state_ctrlg",)
                for arm in treatments:
                    values[arm]["hmm_provenance"]["skills_sha256"] = f"training-{arm}"
                original = deepcopy(values)
                root = Path(directory)
                self.write_summaries(root, values, grid=grid)
                with self.assertRaisesRegex(ValueError, "HMM provenance skills_sha256"):
                    self.run_main(root)
                # Source-code override does not also permit different training skills.
                with self.assertRaisesRegex(ValueError, "HMM provenance skills_sha256"):
                    self.run_main(root, "--allow-source-mismatch")
                output = self.run_main(root, "--allow-skills-mismatch")
                self.assertEqual(output["skills_validation"], {
                    "allow_skills_mismatch": True,
                    "evaluation_sha256": "s",
                    "hmm_training_sha256_by_arm": {arm: f"training-{arm}" for arm in treatments},
                    "matched_by_arm": {arm: False for arm in treatments},
                })
                self.assertEqual(values, original)
                for arm, summary in values.items():
                    summary_dir = root / arm if grid else root
                    path = summary_dir / f"summary_{summary['condition']}.json"
                    self.assertEqual(json.loads(path.read_text()), summary)

    def test_skills_override_reports_matching_and_differing_hmm_arms(self):
        values = grid_summaries()
        values["k4"]["hmm_provenance"]["skills_sha256"] = "old-skills"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summaries(root, values)
            output = self.run_main(root, "--allow-skills-mismatch")
        self.assertEqual(output["skills_validation"]["matched_by_arm"], {"k8": True, "k4": False})

    def test_skills_override_preserves_other_checks(self):
        for grid in (True, False):
            for provenance_field, wrong_value in (
                ("model", "other-model"), ("tokenizer", "other-tokenizer"),
                ("prompt_format", "other-prompt"), ("schema_version", 2), ("policy_version", 2),
                ("hmm_sequence_format", "old-format"), ("hmm_context_format", "old-context"),
                ("hmm_context_steps", 2), ("max_hmm_context_tokens", 512),
                ("no_oracle_filtering", False), ("constrained_state_collection", True),
                ("selection_policy", "executed_only"),
                ("skills_sha256", None), ("skills_sha256", ""), ("skills_sha256", 42),
            ):
                with self.subTest(grid=grid, field=provenance_field, value=wrong_value):
                    values = grid_summaries() if grid else summaries()
                    arm = "k4" if grid else "state_ctrlg"
                    values[arm]["hmm_provenance"].update({
                        "skills_sha256": "old-skills", provenance_field: wrong_value,
                    })
                    validate = module.validate_grid if grid else module.validate_comparable
                    with self.assertRaises(ValueError):
                        validate(values, allow_skills_mismatch=True)

    def test_skills_override_requires_recorded_training_hash(self):
        values = grid_summaries()
        del values["k8"]["hmm_provenance"]["skills_sha256"]
        with self.assertRaisesRegex(ValueError, "skills_sha256"):
            module.validate_grid(values, allow_skills_mismatch=True)

    def test_skills_override_requires_recorded_evaluation_hash(self):
        for missing_hash in (None, "", 42):
            with self.subTest(missing_hash=missing_hash):
                values = grid_summaries()
                for summary in values.values():
                    summary["skills_sha256"] = missing_hash
                with self.assertRaisesRegex(ValueError, "requires a recorded skills_sha256"):
                    module.validate_grid(values, allow_skills_mismatch=True)

    def test_skills_override_requires_same_evaluation_skills_and_source(self):
        for grid in (True, False):
            for field in ("skills_sha256", "source_tree_sha256", "episode_manifest_sha256", "seed"):
                with self.subTest(grid=grid, field=field):
                    values = grid_summaries() if grid else summaries()
                    arm = "k8" if grid else "state_ctrlg"
                    values[arm]["hmm_provenance"]["skills_sha256"] = "old-skills"
                    values[arm][field] = "different"
                    validate = module.validate_grid if grid else module.validate_comparable
                    with self.assertRaisesRegex(ValueError, field):
                        validate(values, allow_skills_mismatch=True)

    def test_skills_and_source_overrides_can_be_combined(self):
        values = grid_summaries()
        values["k4"]["hmm_provenance"]["skills_sha256"] = "old-skills"
        values["k4"]["source_tree_sha256"] = "different-source"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summaries(root, values)
            output = self.run_main(root, "--allow-skills-mismatch", "--allow-source-mismatch")
        self.assertFalse(output["skills_validation"]["matched_by_arm"]["k4"])
        self.assertFalse(output["source_tree_validation"]["matched"])


    def test_episode_efficiency_costs_and_task_breakdowns(self):
        episodes = {
            "prompt_only": [episode_record("a", True, 6), episode_record("b", False, 50, task_key="heat")],
            "k8": [episode_record("a", True, 4, bad_actions=1), episode_record("b", False, 50, task_key="heat")],
            "k4": [episode_record("a", True, 5), episode_record("b", False, 50, task_key="heat")],
        }
        episodes["k8"][0]["steps"][0]["policy_fallback_reason"] = "invalid_state_schema: details"
        values = evaluation_summaries(episodes)
        values["k8"]["state_hmm_skip_reasons"] = {"invalid_state": 1}
        values["k8"]["metrics"]["policy_evaluability_rate"] = 0.75
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summaries(root, values, episodes=episodes)
            output = self.run_main(root)
        arm = output["conditions"]["k8"]
        self.assertEqual((arm["episodes"], arm["successes"], arm["failures"], arm["actions"]), (2, 1, 1, 54))
        self.assertEqual(arm["mean_actions_per_episode"], 27)
        self.assertEqual(arm["inadmissible_actions"], 1)
        self.assertEqual(arm["mean_inadmissible_actions_per_episode"], 0.5)
        self.assertEqual(arm["prompt_tokens"], 5400)
        self.assertEqual(arm["mean_prompt_tokens_per_episode"], 2700)
        self.assertEqual(arm["generated_tokens"], 540)
        self.assertEqual(arm["mean_generated_tokens_per_action"], 10)
        self.assertEqual(arm["mean_generated_tokens_per_episode"], 270)
        self.assertEqual(arm["generation_latency_seconds"], 135)
        self.assertEqual(arm["mean_generation_latency_seconds_per_action"], 2.5)
        self.assertEqual(arm["mean_generation_latency_seconds_per_episode"], 67.5)
        self.assertEqual(arm["policy_evaluability_rate"], 0.75)
        self.assertTrue(arm["episode_details_available"])
        stats = arm["episode_statistics"]
        self.assertEqual(stats["actions_per_episode"]["all"], {
            "count": 2, "mean": 27, "median": 27, "p90": 50, "min": 4, "max": 50,
        })
        self.assertEqual(stats["actions_per_episode"]["successful"]["mean"], 4)
        self.assertEqual(stats["actions_per_episode"]["failed"]["mean"], 50)
        self.assertEqual(stats["success_within_actions"]["10"], {"successes": 1, "success_rate": 0.5})
        self.assertEqual(stats["failed_at_step_limit"], 1)
        self.assertEqual(stats["fraction_of_failures_at_step_limit"], 1.0)
        self.assertEqual(stats["episodes_with_inadmissible_actions"], 1)
        self.assertEqual(stats["policy_fallback_reasons"], {"invalid_state_schema": 1})
        self.assertEqual(arm["hmm_skip_reasons"]["state"], {"invalid_state": 1})
        self.assertEqual(arm["per_task_type"]["put"]["mean_actions_per_successful_episode"], 4)
        self.assertIsNone(arm["per_task_type"]["put"]["mean_actions_per_failed_episode"])
        self.assertEqual(arm["per_task_type"]["heat"]["failures"], 1)
        self.assertEqual(output["ctrlg_effect"]["k8"]["mean_actions_per_episode"], -1)
        self.assertEqual(output["paired_comparisons"]["k8"]["actions_on_both_successful"]["ctrlg_minus_baseline"]["mean"], -2)
        self.assertNotIn("episode_gamefiles", json.dumps(output))

    def test_paired_outcomes_and_actions_use_only_common_successes(self):
        baseline = [
            episode_record("a", True, 8), episode_record("b", False, 50),
            episode_record("c", True, 12), episode_record("d", False, 40),
        ]
        treatment = [
            episode_record("a", True, 6), episode_record("b", True, 20),
            episode_record("c", False, 50), episode_record("d", False, 50),
        ]
        episodes = {"state_prompt": baseline, "state_ctrlg": treatment}
        values = evaluation_summaries(episodes, grid=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summaries(root, values, grid=False, episodes=episodes)
            output = self.run_main(root)
        comparison = output["paired_comparisons"]["state_ctrlg"]
        self.assertEqual(comparison["episodes"], 4)
        for metric in ("ctrlg_only_successes", "baseline_only_successes", "both_successes", "both_failures"):
            self.assertEqual(comparison[metric], 1)
        actions = comparison["actions_on_both_successful"]
        self.assertEqual((actions["ctrlg_mean"], actions["baseline_mean"]), (6, 8))
        self.assertEqual(actions["ctrlg_minus_baseline"]["mean"], -2)
        self.assertEqual(actions["ctrlg_fewer_actions"], 1)

    def test_summary_only_keeps_aggregates_and_marks_episode_details_unavailable(self):
        values = deepcopy(grid_summaries())
        for summary in values.values():
            summary["metrics"].update({
                "episodes": 2, "successes": 1, "actions": 60, "admissible_actions": 57,
                "prompt_tokens": 6000, "generated_tokens": 600,
                "head_latency_seconds": 120, "action_latency_seconds": 30,
            })
            summary["per_task_type"] = {"put": {"episodes": 2, "successes": 1, "success_rate": 0.5}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summaries(root, values)
            output = self.run_main(root)
        arm = output["conditions"]["k8"]
        self.assertEqual(arm["mean_actions_per_episode"], 30)
        self.assertEqual(arm["inadmissible_actions"], 3)
        self.assertEqual(arm["mean_generation_latency_seconds_per_episode"], 75)
        self.assertEqual(arm["per_task_type"]["put"]["successes"], 1)
        self.assertFalse(arm["episode_details_available"])
        self.assertIsNone(arm["episode_statistics"])
        self.assertIsNone(output["paired_comparisons"]["k8"])

    def test_empty_evaluations_have_null_rates_and_no_nan(self):
        episodes = {arm: [] for arm in module.GRID_ARMS}
        values = evaluation_summaries(episodes)
        for summary in values.values():
            summary["metrics"]["parse_rate"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summaries(root, values, episodes=episodes)
            output = self.run_main(root)
        arm = output["conditions"]["k8"]
        self.assertEqual((arm["episodes"], arm["actions"]), (0, 0))
        self.assertIsNone(arm["success_rate"])
        self.assertIsNone(arm["parse_rate"])
        self.assertIsNone(arm["mean_action_latency_seconds"])
        self.assertIsNone(arm["mean_actions_per_episode"])
        self.assertIsNone(arm["episode_statistics"]["actions_per_episode"]["all"]["mean"])
        self.assertIsNone(output["paired_comparisons"]["k8"]["actions_on_both_successful"]["ctrlg_mean"])
        self.assertNotIn("success_rate", output["ctrlg_effect"]["k8"])
        json.dumps(output, allow_nan=False)

    def test_action_budgets_include_boundary_success_and_exclude_late_success(self):
        episodes = [episode_record("a", True, 10), episode_record("b", True, 50)]
        stats = module.summarize_episodes(episodes, 50)
        self.assertEqual(stats["success_within_actions"]["10"]["success_rate"], 0.5)
        self.assertEqual(stats["success_within_actions"]["50"]["success_rate"], 1.0)
        self.assertEqual(stats["failed_at_step_limit"], 0)
        self.assertIsNone(stats["fraction_of_failures_at_step_limit"])
        self.assertIsNone(stats["actions_per_episode"]["failed"]["mean"])
        self.assertEqual(list(module.summarize_episodes(episodes[:1], 10)["success_within_actions"]), ["10"])

    def test_missing_optional_measurements_are_not_reported_as_zero(self):
        records = [episode_record("a", False, 1)]
        records[0]["steps"] = [{}]
        totals = module.episode_totals(records)
        self.assertNotIn("generated_tokens", totals)
        self.assertNotIn("admissible_actions", totals)
        stats = module.summarize_episodes(records, 50)
        self.assertIsNone(stats["episodes_with_inadmissible_actions"])
        self.assertIsNone(stats["policy_fallback_reasons"])

    def test_zero_action_episodes_still_have_defined_episode_rates(self):
        metrics = {
            "episodes": 2, "successes": 0, "actions": 0, "prompt_tokens": 0,
            "admissible_actions": 0, "policy_evaluable_turns": 0,
        }
        module.add_averages(metrics)
        self.assertEqual(metrics["success_rate"], 0)
        self.assertEqual(metrics["mean_actions_per_episode"], 0)
        self.assertEqual(metrics["mean_prompt_tokens_per_episode"], 0)
        self.assertIsNone(metrics["admissibility_rate"])
        self.assertIsNone(metrics["policy_adherence_rate"])
        self.assertIsNone(metrics["mean_prompt_tokens_per_action"])

    def test_trace_loader_discards_large_unused_fields(self):
        records = [episode_record("a", True, 1)]
        records[0]["task_description"] = "An objective"
        records[0]["steps"][0].update({"thought": "long reasoning", "hmm_prefix_token_ids": [1, 2, 3]})
        summary = evaluation_summaries({"state_ctrlg": records}, grid=False)["state_ctrlg"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval_state_ctrlg.jsonl"
            path.write_text(json.dumps(records[0]) + "\n")
            loaded = module.load_episodes(path, summary)
        self.assertEqual(loaded[0]["num_steps"], 1)
        self.assertNotIn("task_description", loaded[0])
        self.assertNotIn("thought", loaded[0]["steps"][0])
        self.assertNotIn("hmm_prefix_token_ids", loaded[0]["steps"][0])

    def test_missing_one_trace_only_disables_its_paired_comparison(self):
        records = [episode_record("a", True, 1)]
        all_episodes = {arm: records for arm in module.GRID_ARMS}
        values = evaluation_summaries(all_episodes)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summaries(root, values, episodes={"k8": records, "prompt_only": records})
            output = self.run_main(root)
        self.assertEqual(output["paired_comparisons"]["k8"]["both_successes"], 1)
        self.assertIsNone(output["paired_comparisons"]["k4"])
        self.assertIsNone(output["conditions"]["k4"]["episode_statistics"])
        self.assertEqual(output["conditions"]["k4"]["actions"], 1)

    def test_trace_validation_rejects_stale_and_malformed_records(self):
        records = [episode_record("a", True, 4), episode_record("b", False, 50)]
        summary = evaluation_summaries({"state_ctrlg": records}, grid=False)["state_ctrlg"]
        invalid = [
            ([records[1], records[0]], "ordered episode gamefiles"),
            (records[:1], "ordered episode gamefiles"),
            ([{**records[0], "success": 1}, records[1]], "boolean success"),
            ([{**records[0], "num_steps": 5}, records[1]], "num_steps"),
            ([{**records[0], "condition": "state_prompt"}, records[1]], "condition"),
            ([episode_record("a", True, 51), records[1]], "exceeds max_steps"),
        ]
        nonfinite = deepcopy(records)
        nonfinite[0]["steps"][0]["head_latency_seconds"] = float("nan")
        invalid.append((nonfinite, "finite and nonnegative"))
        for bad_records, message in invalid:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "eval_state_ctrlg.jsonl"
                path.write_text("".join(json.dumps(record) + "\n" for record in bad_records))
                with self.assertRaisesRegex(ValueError, message):
                    module.load_episodes(path, summary)
        for field in ("episodes", "successes", "actions", "admissible_actions", "success_rate"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                stale = deepcopy(summary)
                stale["metrics"][field] += 1
                path = Path(directory) / "eval_state_ctrlg.jsonl"
                path.write_text("".join(json.dumps(record) + "\n" for record in records))
                with self.assertRaisesRegex(ValueError, field):
                    module.load_episodes(path, stale)

    def test_invalid_json_has_file_and_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval_state_ctrlg.jsonl"
            path.write_text("\n{\n")
            with self.assertRaisesRegex(ValueError, "eval_state_ctrlg.jsonl:2: invalid episode JSON"):
                module.load_episodes(path, summaries()["state_ctrlg"])

    def test_pairing_rejects_task_type_mismatch(self):
        with self.assertRaisesRegex(ValueError, "task types"):
            module.compare_episodes([episode_record("a", True, 1)], [episode_record("a", True, 1, task_key="heat")])


if __name__ == "__main__":
    unittest.main()
