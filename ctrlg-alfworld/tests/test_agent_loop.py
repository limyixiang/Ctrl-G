import unittest
from pathlib import Path

from ctrlg_alfworld.agent_loop import parse_initial_observation, run_episode
from ctrlg_alfworld.experiment import condition_choices, get_condition
from ctrlg_alfworld.generation import TurnGeneration, parse_turn
from ctrlg_alfworld.skills import SkillSet


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "\n".join(message["content"] for message in messages) + "\n<think>"


DECISION = (
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

    def generate_turn(self, prompt_text, skillset, task_key, *, constrained, greedy_head):
        self.calls.append((task_key, constrained, skillset.content_sha256))
        self.prompt_texts.append(prompt_text)
        head = f"choose</think><decision>{DECISION}</decision><action>"
        tail = "look</action>"
        parsed = parse_turn(head, tail, use_decision=True)
        return TurnGeneration(
            parsed=parsed,
            head_token_ids=tuple(self.tokenizer.encode(head)),
            action_token_ids=tuple(self.tokenizer.encode("look")),
            tail_token_ids=tuple(self.tokenizer.encode(tail)),
            hmm_prefix_token_ids=tuple(self.tokenizer.encode(parsed.hmm_prefix_text)),
            head_latency_seconds=0.01, action_latency_seconds=0.02,
            decision_schema_valid=constrained,
            decision_fields={},
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
            self.assertEqual(backend.calls[0][0:2], ("put", condition.use_decision_dfa))
            self.assertNotIn("look", repr(backend.calls[0]))
            self.assertNotIn("Your admissible actions", backend.prompt_texts[0])
            self.assertEqual(record.steps[0].admissible_gt, ["look", "go to countertop 1"])
            self.assertTrue(record.steps[0].action_was_admissible)
            prompts.append(backend.prompt_texts[0])
        self.assertEqual(prompts[0].encode(), prompts[1].encode())


if __name__ == "__main__":
    unittest.main()
