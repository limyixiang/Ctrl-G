import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from ctrlg_alfworld.experiment import get_condition
from ctrlg_alfworld.skills import SkillSet


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_eval", ROOT / "scripts" / "run_eval.py")
run_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_eval)


class EvalRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skills = SkillSet.from_file(ROOT / "templates" / "SKILLS.md")

    def hmm_metadata(self):
        return {
            "skills_sha256": self.skills.content_sha256,
            "schema_version": self.skills.schema_version,
            "policy_version": self.skills.policy_version,
            "no_oracle_filtering": True,
            "constrained_state_collection": False,
            "hmm_sequence_format": "context_state_action_censored_v1",
            "hmm_context_format": "observable_trajectory_json_v1",
            "hmm_context_steps": 1, "max_hmm_context_tokens": 1024,
            "selection_policy": "all_observed_attempts",
            "model": "model-a",
            "tokenizer": "model-a",
            "prompt_format": "state_with_persistent_history_no_oracle_v1",
        }

    def test_prompt_condition_uses_no_hmm(self):
        self.assertIsNone(run_eval.resolve_hmm_path(
            get_condition("state_prompt"), hmm="checkpoint"
        ))

    def test_ctrlg_requires_hmm(self):
        self.assertEqual(run_eval.resolve_hmm_path(
            get_condition("state_ctrlg"), hmm="checkpoint"
        ), "checkpoint")
        with self.assertRaisesRegex(ValueError, "requires --hmm"):
            run_eval.resolve_hmm_path(get_condition("state_ctrlg"), hmm=None)

    def test_hmm_provenance_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-1"
            checkpoint.mkdir()
            metadata = self.hmm_metadata()
            path = root / run_eval.HMM_TRAINING_METADATA
            path.write_text(json.dumps(metadata))
            resolved, loaded = run_eval.validate_hmm_provenance(
                str(checkpoint), skillset=self.skills, model="model-a"
            )
            self.assertEqual(resolved, str(path.resolve()))
            self.assertEqual(loaded, metadata)
            for key, incompatible in (
                ("constrained_state_collection", True),
                ("hmm_sequence_format", "old_format"),
                ("hmm_context_format", "old_context"),
                ("hmm_context_steps", 2),
                ("max_hmm_context_tokens", 64),
                ("selection_policy", "executed_sample_only"),
            ):
                with self.subTest(field=key):
                    old = metadata[key]
                    metadata[key] = incompatible
                    path.write_text(json.dumps(metadata))
                    with self.assertRaisesRegex(ValueError, key):
                        run_eval.validate_hmm_provenance(str(checkpoint), skillset=self.skills, model="model-a")
                    metadata[key] = old
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "model"):
                run_eval.validate_hmm_provenance(
                    str(checkpoint), skillset=self.skills, model="model-b"
                )
            metadata["prompt_format"] = "decision_with_persistent_history_no_oracle_v1"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "prompt_format"):
                run_eval.validate_hmm_provenance(
                    str(checkpoint), skillset=self.skills, model="model-a"
                )

    def test_skills_override_preserves_original_metadata(self):
        for training_hash in (self.skills.content_sha256, "old-skills"):
            with self.subTest(training_hash=training_hash), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / run_eval.HMM_TRAINING_METADATA
                metadata = {**self.hmm_metadata(), "skills_sha256": training_hash}
                path.write_text(json.dumps(metadata))
                original_bytes = path.read_bytes()
                if training_hash != self.skills.content_sha256:
                    with self.assertRaisesRegex(ValueError, "skills_sha256"):
                        run_eval.validate_hmm_provenance(directory, skillset=self.skills, model="model-a")
                _, loaded = run_eval.validate_hmm_provenance(
                    directory, skillset=self.skills, model="model-a", allow_skills_mismatch=True,
                )
                self.assertEqual(loaded, metadata)
                self.assertEqual(path.read_bytes(), original_bytes)

    def test_skills_override_preserves_other_provenance_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / run_eval.HMM_TRAINING_METADATA
            metadata = {**self.hmm_metadata(), "skills_sha256": "old-skills"}
            for key, incompatible in (
                ("model", "other-model"), ("tokenizer", "other-tokenizer"),
                ("schema_version", 2), ("policy_version", 2),
                ("prompt_format", "old-prompt"), ("no_oracle_filtering", False),
                ("constrained_state_collection", True),
                ("hmm_sequence_format", "old-format"), ("hmm_context_format", "old-context"),
                ("hmm_context_steps", 2), ("max_hmm_context_tokens", 512),
                ("selection_policy", "executed_sample_only"),
                ("skills_sha256", None), ("skills_sha256", ""), ("skills_sha256", 42),
            ):
                with self.subTest(field=key, value=incompatible):
                    path.write_text(json.dumps({**metadata, key: incompatible}))
                    with self.assertRaisesRegex(ValueError, key):
                        run_eval.validate_hmm_provenance(
                            directory, skillset=self.skills, model="model-a", allow_skills_mismatch=True,
                        )
            del metadata["skills_sha256"]
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "skills_sha256"):
                run_eval.validate_hmm_provenance(
                    directory, skillset=self.skills, model="model-a", allow_skills_mismatch=True,
                )

    def test_cli_records_override_and_both_skill_hashes(self):
        alfworld = ModuleType("alfworld")
        agents = ModuleType("alfworld.agents")
        environment = ModuleType("alfworld.agents.environment")
        alfworld.agents = agents
        agents.environment = environment
        factory = SimpleNamespace(game_files=[], init_env=Mock())
        environment.get_environment = Mock(return_value=Mock(return_value=factory))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {**self.hmm_metadata(), "skills_sha256": "old-skills"}
            metadata_path = root / run_eval.HMM_TRAINING_METADATA
            metadata_path.write_text(json.dumps(metadata))
            config = root / "config.yaml"
            config.write_text("general: {}\nenv: {type: TestEnv}\n")
            for condition in ("state_prompt", "state_ctrlg"):
                with self.subTest(condition=condition), patch.dict(sys.modules, {
                    "alfworld": alfworld, "alfworld.agents": agents,
                    "alfworld.agents.environment": environment,
                }), patch.object(sys, "argv", [
                    str(ROOT / "scripts/run_eval.py"), "--model", "model-a",
                    "--hmm", directory, "--condition", condition,
                    "--config", str(config), "--out", directory,
                    "--num_episodes", "0", "--device", "cpu", "--allow-skills-mismatch",
                ]), patch.object(run_eval, "HFBackend"), patch.object(run_eval, "runtime_versions", return_value={}):
                    run_eval.main()
                    summary = json.loads((root / f"summary_{condition}.json").read_text())
                    self.assertTrue(summary["allow_skills_mismatch"])
                    self.assertEqual(summary["skills_sha256"], self.skills.content_sha256)
                    self.assertEqual(summary["hmm_provenance"], metadata if condition == "state_ctrlg" else None)
            self.assertEqual(json.loads(metadata_path.read_text()), metadata)

    def test_missing_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for allow_skills_mismatch in (False, True):
                with self.subTest(allow_skills_mismatch=allow_skills_mismatch):
                    with self.assertRaisesRegex(ValueError, "training_data_metadata"):
                        run_eval.validate_hmm_provenance(
                            directory, skillset=self.skills, model="model-a",
                            allow_skills_mismatch=allow_skills_mismatch,
                        )


if __name__ == "__main__":
    unittest.main()
