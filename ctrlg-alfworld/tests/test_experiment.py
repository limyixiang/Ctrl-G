import unittest

from ctrlg_alfworld.experiment import ConditionName, condition_choices, get_condition


class ExperimentConditionTests(unittest.TestCase):
    def test_active_pair_is_prompt_only_versus_ctrlg(self):
        baseline = get_condition("decision_prompt")
        constrained = get_condition("decision_ctrlg")
        self.assertTrue(baseline.use_decision and constrained.use_decision)
        self.assertFalse(baseline.use_decision_dfa or baseline.use_action_dfa)
        self.assertTrue(constrained.use_decision_dfa and constrained.use_action_dfa)
        self.assertFalse(baseline.use_hmm)
        self.assertTrue(constrained.use_hmm)

    def test_only_two_stable_condition_names_are_active(self):
        self.assertEqual(
            condition_choices(),
            [ConditionName.DECISION_PROMPT.value, ConditionName.DECISION_CTRLG.value],
        )

    def test_unknown_condition_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown condition"):
            get_condition("unknown")


if __name__ == "__main__":
    unittest.main()
