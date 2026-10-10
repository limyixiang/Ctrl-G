import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

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
            metadata = {
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

    def test_missing_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "training_data_metadata"):
                run_eval.validate_hmm_provenance(
                    directory, skillset=self.skills, model="model-a"
                )


if __name__ == "__main__":
    unittest.main()
