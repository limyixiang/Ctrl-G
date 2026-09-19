import unittest

from ctrlg_alfworld.prompts import Step, build_user_prompt


class PromptTests(unittest.TestCase):
    def make_prompt(self, history=None):
        return build_user_prompt(
            skill_content="Keep searching systematically.",
            task_description="put a mug on shelf 1",
            initial_observation="You are in a kitchen.",
            current_observation="You see a countertop 1.",
            obs_history=history or [],
            use_decision=True,
        )

    def test_initial_prompt_contains_task_and_no_oracle_list(self):
        prompt = self.make_prompt()
        self.assertIn("put a mug on shelf 1", prompt)
        self.assertNotIn("Your admissible actions", prompt)

    def test_history_starts_with_initial_observation(self):
        history = [Step(thought="private", action="look", observation="You see a shelf 1.")]
        prompt = self.make_prompt(history)
        self.assertLess(prompt.index("Your initial observation was"), prompt.index("Step 1: look"))
        self.assertNotIn("private", prompt)

    def test_condition_flag_does_not_change_prompt_bytes(self):
        arguments = dict(
            skill_content="raw skill bytes", task_description="find cup",
            initial_observation="room", current_observation="room", obs_history=[]
        )
        baseline = build_user_prompt(**arguments, use_decision=True)
        treatment = build_user_prompt(**arguments, use_decision=True)
        self.assertEqual(baseline.encode(), treatment.encode())

    def test_admissible_parameters_are_not_accepted(self):
        with self.assertRaises(TypeError):
            build_user_prompt(
                skill_content="x", task_description="x", initial_observation="x",
                current_observation="x", obs_history=[], use_decision=True,
                admissible_actions=["look"],
            )


if __name__ == "__main__":
    unittest.main()
