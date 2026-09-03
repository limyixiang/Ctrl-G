import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from ctrlg_alfworld.experiment import get_condition


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
SPEC = importlib.util.spec_from_file_location("run_eval", SCRIPT)
run_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_eval)


class EvalRoutingTests(unittest.TestCase):
    def test_baseline_uses_no_hmm(self):
        self.assertIsNone(
            run_eval.resolve_hmm_path(get_condition("decision_dfa"), hmm="checkpoint")
        )

    def test_hmm_condition_uses_matched_artifact(self):
        self.assertEqual(
            run_eval.resolve_hmm_path(
                get_condition("decision_dfa_hmm"), hmm="decision-checkpoint"
            ),
            "decision-checkpoint",
        )

    def test_missing_matched_artifact_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires --hmm"):
            run_eval.resolve_hmm_path(get_condition("decision_dfa_hmm"), hmm=None)

    def test_hmm_prompt_regime_metadata_must_match_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            checkpoint = model_dir / "checkpoint-1"
            checkpoint.mkdir()
            metadata = model_dir / run_eval.HMM_TRAINING_METADATA
            metadata.write_text(json.dumps({"show_admissible_actions": True}))

            path, shown = run_eval.validate_hmm_prompt_regime(
                str(checkpoint), show_admissible_actions=True
            )
            self.assertEqual(path, str(metadata.resolve()))
            self.assertTrue(shown)
            with self.assertRaisesRegex(ValueError, "does not match evaluation"):
                run_eval.validate_hmm_prompt_regime(
                    str(checkpoint), show_admissible_actions=False
                )

    def test_legacy_hmm_without_prompt_metadata_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                run_eval.validate_hmm_prompt_regime(
                    directory, show_admissible_actions=False
                ),
                (None, None),
            )


if __name__ == "__main__":
    unittest.main()
