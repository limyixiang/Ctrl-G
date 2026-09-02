import unittest

from ctrlg_alfworld.experiment import ConditionName, condition_choices, get_condition


class ExperimentConditionTests(unittest.TestCase):
    def test_active_pair_differs_only_by_hmm(self):
        baseline = get_condition("decision_dfa")
        constrained = get_condition("decision_dfa_hmm")
        self.assertTrue(baseline.use_decision and constrained.use_decision)
        self.assertTrue(baseline.use_dfa and constrained.use_dfa)
        self.assertFalse(baseline.use_hmm)
        self.assertTrue(constrained.use_hmm)

    def test_only_two_stable_condition_names_are_active(self):
        self.assertEqual(
            condition_choices(),
            [ConditionName.DECISION_DFA.value, ConditionName.DECISION_DFA_HMM.value],
        )

    def test_unknown_condition_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown condition"):
            get_condition("unknown")


if __name__ == "__main__":
    unittest.main()
