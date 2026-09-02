import unittest
from types import SimpleNamespace

from ctrlg_alfworld.agent_loop import run_episode
from ctrlg_alfworld.experiment import condition_choices, get_condition
from ctrlg_alfworld.generation import TurnGeneration, parse_turn


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking
    ):
        self.enable_thinking = enable_thinking
        return "\n".join(message["content"] for message in messages) + "\n<think>"


class FakeBackend:
    def __init__(self):
        self.tokenizer = CharacterTokenizer()
        self.calls = []

    def generate_turn(
        self,
        prompt_text,
        allowed_actions,
        *,
        use_decision,
        use_hmm,
        greedy_head,
    ):
        self.calls.append((use_decision, use_hmm, tuple(allowed_actions)))
        if use_decision:
            head = (
                "choose a legal action</think>"
                "<decision>inspect the room</decision><action>"
            )
        else:
            head = "choose a legal action</think><action>"
        tail = "look</action>"
        parsed = parse_turn(head, tail, use_decision=use_decision)
        return TurnGeneration(
            parsed=parsed,
            head_token_ids=tuple(self.tokenizer.encode(head)),
            action_token_ids=tuple(self.tokenizer.encode("look")),
            tail_token_ids=tuple(self.tokenizer.encode(tail)),
            hmm_prefix_token_ids=tuple(
                self.tokenizer.encode(parsed.hmm_prefix_text)
            ),
            head_latency_seconds=0.01,
            action_latency_seconds=0.02,
        )


class OneStepEnvironment:
    def reset(self):
        observation = "header\n\nYou see a countertop 1.\n\nlook around"
        info = {
            "extra.gamefile": [
                "/games/pick_and_place_simple/trial/game.tw-pddl"
            ],
            "admissible_commands": [["look", "go to countertop 1"]],
        }
        return [observation], info

    def step(self, actions):
        if actions != ["look"]:
            raise AssertionError(f"unexpected action: {actions}")
        return ["Task complete."], [1], [True], {"won": [True]}


class AgentLoopTests(unittest.TestCase):
    def test_every_condition_uses_the_same_dfa_language_and_logs_metrics(self):
        for condition_name in condition_choices():
            with self.subTest(condition=condition_name):
                backend = FakeBackend()
                record = run_episode(
                    OneStepEnvironment(),
                    backend,
                    SimpleNamespace(raw_markdown="minimal skills"),
                    condition_name,
                )
                condition = get_condition(condition_name)
                self.assertTrue(record.success)
                self.assertEqual(record.condition, condition_name)
                self.assertEqual(
                    backend.calls,
                    [
                        (
                            condition.use_decision,
                            condition.use_hmm,
                            ("look", "go to countertop 1"),
                        )
                    ],
                )
                step = record.steps[0]
                self.assertTrue(step.action_was_admissible)
                self.assertTrue(step.parse_ok)
                self.assertEqual(step.generated_tokens, len(step.tail_token_ids) + len(backend.tokenizer.encode(step.thought + "</think>" + step.hmm_prefix_text)))
                self.assertAlmostEqual(step.head_latency_seconds, 0.01)
                self.assertAlmostEqual(step.action_latency_seconds, 0.02)
