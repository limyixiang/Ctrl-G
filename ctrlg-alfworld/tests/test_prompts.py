import unittest

from ctrlg_alfworld.prompts import Step, build_user_prompt


class PromptTests(unittest.TestCase):
    def make_prompt(self, *, use_decision, show_actions=False, history=None):
        return build_user_prompt(
            skill_content="Keep searching systematically.",
            task_description="put a mug on shelf 1",
            current_observation="You see a countertop 1.",
            obs_history=history or [],
            use_decision=use_decision,
            admissible_actions=["look", "go to countertop 1"],
            show_admissible_actions=show_actions,
        )

    def test_initial_prompt_contains_task(self):
        prompt = self.make_prompt(use_decision=False)
        self.assertIn("put a mug on shelf 1", prompt)

    def test_core_prompt_does_not_leak_admissible_list(self):
        prompt = self.make_prompt(use_decision=False)
        self.assertNotIn("go to countertop 1", prompt)
        self.assertNotIn("Your admissible actions", prompt)

    def test_prompt_list_requires_explicit_control_flag(self):
        prompt = self.make_prompt(use_decision=False, show_actions=True)
        self.assertIn("go to countertop 1", prompt)

    def test_native_thinking_is_not_requested_as_manual_tag(self):
        prompt = self.make_prompt(use_decision=False)
        self.assertNotIn("<think>", prompt)
        self.assertNotIn("</think>", prompt)

    def test_decision_factor_changes_output_contract(self):
        without = self.make_prompt(use_decision=False)
        with_decision = self.make_prompt(use_decision=True)
        self.assertNotIn("<decision>", without)
        self.assertIn("<decision>", with_decision)

    def test_decision_history_replays_decision_before_matching_action_and_observation(self):
        history = [
            Step(
                thought="private native reasoning",
                decision="search elsewhere",
                action="look",
                observation="You see a shelf 1.",
            )
        ]
        prompt = self.make_prompt(use_decision=True, history=history)
        decision = "<decision>search elsewhere</decision>"
        action = "<action>look</action>"
        observation = "OBS: You see a shelf 1."
        self.assertLess(prompt.index(decision), prompt.index(action))
        self.assertLess(prompt.index(action), prompt.index(observation))
        self.assertNotIn("private native reasoning", prompt)

    def test_generic_action_only_history_remains_action_observation_only(self):
        history = [
            Step(
                thought="private native reasoning",
                decision="search elsewhere",
                action="look",
                observation="You see a shelf 1.",
            )
        ]
        prompt = self.make_prompt(use_decision=False, history=history)
        self.assertNotIn("search elsewhere", prompt)
        self.assertNotIn("<decision>", prompt)
        self.assertIn("<action>look</action>\nOBS: You see a shelf 1.", prompt)
        self.assertNotIn("private native reasoning", prompt)

    def test_empty_prior_decision_is_not_replayed(self):
        history = [Step(thought="hidden", decision="  ", action="look", observation="Room.")]
        prompt = self.make_prompt(use_decision=True, history=history)
        # The current-turn output instruction contains one opening tag; an
        # empty history entry must not add another one.
        self.assertEqual(prompt.count("<decision>"), 1)
