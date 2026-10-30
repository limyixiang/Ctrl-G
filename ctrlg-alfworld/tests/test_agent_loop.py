import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ctrlg_alfworld.agent_loop import parse_initial_observation, run_episode
from ctrlg_alfworld.experiment import condition_choices, get_condition
from ctrlg_alfworld.generation import TurnGeneration, parse_turn
from ctrlg_alfworld.skills import PolicyRule, SkillSet


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "\n".join(message["content"] for message in messages) + "\n<think>"


STATE = (
    "{task: put, target_type: cup, target_id: unknown, target_location: unknown, "
    "held: none, at: countertop 1, dest: unknown, previous_action: none, "
    "arrived_receptacle_state: open, held_relation: none, search: "
    "{locations_searched: [], locations_to_search: [countertop 1]}, phase: search}"
)


class FakeBackend:
    def __init__(self):
        self.tokenizer = CharacterTokenizer()
        self.calls = []
        self.prompt_texts = []
        self.contexts = []

    def generate_turn(self, prompt_text, skillset, task_key, *, constrained, greedy_head, hmm_context):
        self.calls.append((task_key, constrained, skillset.content_sha256))
        self.prompt_texts.append(prompt_text)
        self.contexts.append(hmm_context)
        head = f"choose</think><state>{STATE}</state><action>"
        tail = "look</action>"
        parsed = parse_turn(head, tail, use_state=True)
        return TurnGeneration(
            parsed=parsed,
            head_token_ids=tuple(self.tokenizer.encode(head)),
            action_token_ids=tuple(self.tokenizer.encode("look")),
            tail_token_ids=tuple(self.tokenizer.encode(tail)),
            hmm_prefix_token_ids=tuple(self.tokenizer.encode(parsed.hmm_prefix_text)),
            head_latency_seconds=0.01, action_latency_seconds=0.02,
            state_schema_valid=constrained,
            state_fields={},
            action_grammar_valid=constrained,
            policy_evaluable=constrained,
            policy_satisfied=constrained,
        )


class OneStepEnvironment:
    def reset(self):
        observation = (
            "-= Welcome to TextWorld, ALFRED! =-\n\nYou see a countertop 1.\n\n"
            "Your task is to: look around"
        )
        return [observation], {
            "extra.gamefile": ["/games/pick_and_place_simple/trial/game.tw-pddl"],
            "admissible_commands": [["look", "go to countertop 1"]],
        }

    def step(self, actions):
        self.actions = actions
        return ["Task complete."], [1], [True], {"won": [True]}


class AgentLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skills = SkillSet.from_file(
            Path(__file__).resolve().parents[1] / "templates" / "SKILLS.md"
        )

    def test_initial_observation_separates_room_and_task(self):
        room, task = parse_initial_observation(
            "welcome\n\nYou are in a kitchen.\n\nYou see a countertop 1.\n\n"
            "Your task is to: put a mug on shelf 1."
        )
        self.assertEqual(room, "You are in a kitchen.\n\nYou see a countertop 1.")
        self.assertEqual(task, "put a mug on shelf 1.")

    def test_decoder_never_receives_admissible_commands(self):
        prompts = []
        for name in condition_choices():
            backend = FakeBackend()
            record = run_episode(OneStepEnvironment(), backend, self.skills, name)
            condition = get_condition(name)
            self.assertTrue(record.success)
            self.assertEqual(record.task_description, "look around")
            self.assertEqual(backend.calls[0][0:2], ("put", condition.use_state_dfa))
            self.assertNotIn("look", repr(backend.calls[0]))
            self.assertNotIn("Your admissible actions", backend.prompt_texts[0])
            self.assertEqual(record.steps[0].admissible_gt, ["look", "go to countertop 1"])
            self.assertTrue(record.steps[0].action_was_admissible)
            saved = record.to_dict()["steps"][0]
            self.assertEqual(saved["state"], STATE)
            self.assertTrue(saved["state_schema_valid"])
            self.assertNotIn("decision", saved)
            self.assertNotIn("decision_fields", saved)
            prompts.append(backend.prompt_texts[0])
        self.assertEqual(prompts[0].encode(), prompts[1].encode())

    def test_prompt_arm_records_rejected_requirements(self):
        state = STATE.replace("previous_action: none", "previous_action: open countertop 1").replace(
            "arrived_receptacle_state: open", "arrived_receptacle_state: closed",
        )
        with patch(f"{__name__}.STATE", state):
            record = run_episode(OneStepEnvironment(), FakeBackend(), self.skills, "state_prompt")
        saved = record.to_dict()["steps"][0]
        self.assertEqual(saved["rejected_required_policies"],
                         {"open_declared_closed_receptacle": "forbidden_by_restrictions"})
        self.assertTrue(saved["state_schema_valid"])
        self.assertTrue(saved["policy_evaluable"])

    def test_prompt_arm_keeps_valid_state_when_policy_language_is_empty(self):
        policies = tuple(PolicyRule(
            f"ban_{action.name}", 1, 90, (),
            {"forbid_action": {"action": action.name, "bindings": {}}},
        ) for action in self.skills.actions)
        skills = replace(self.skills, policies=policies)
        record = run_episode(OneStepEnvironment(), FakeBackend(), skills, "state_prompt")
        saved = record.to_dict()["steps"][0]
        self.assertTrue(saved["state_schema_valid"])
        self.assertEqual(saved["state_fields"]["held"], "none")
        self.assertFalse(saved["policy_evaluable"])
        self.assertIsNone(saved["policy_satisfied"])
        self.assertIn("empty_restricted_language", saved["policy_fallback_reason"])


if __name__ == "__main__":
    unittest.main()
