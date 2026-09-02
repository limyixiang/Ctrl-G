import importlib.util
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


if __name__ == "__main__":
    unittest.main()
