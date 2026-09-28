import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ctrlg_alfworld.backends import (
    BaseBackend,
    DecisionGenerationExhausted,
    GenConfig,
    HFBackend,
    StructuredOutputMismatchError,
    VLLMBackend,
)
from ctrlg_alfworld.constraints import build_trie_dfa
from ctrlg_alfworld.generation import GeneratedChunk, TurnGeneration, parse_turn
from ctrlg_alfworld.prompts import ACTION_CLOSE
from ctrlg_alfworld.skills import SkillSet


class CharacterTokenizer:
    name_or_path = "character"
    all_special_ids = [0]
    eos_token_id = 0

    def __len__(self):
        return 128

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        values = [ord(char) for char in text]
        if return_tensors:
            import torch
            return torch.tensor([values])
        return values

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "".join(chr(item) for item in ids if item)


def chunk(text, tokenizer, stop=True):
    return GeneratedChunk(
        text=text, token_ids=tuple(tokenizer.encode(text)), stop_found=stop,
        truncated=not stop, latency_seconds=0.1,
    )


DECISION = (
    "{task: put, target_type: cup, target_id: unknown, target_location: unknown, "
    "held: none, at: countertop 1, dest: unknown, previous_action: look, "
    "arrived_receptacle_state: open, held_relation: none, search: "
    "{locations_searched: [], locations_to_search: [countertop 1]}, phase: search}"
)


class BackendIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skills = SkillSet.from_file(
            Path(__file__).resolve().parents[1] / "templates" / "SKILLS.md"
        )

    def test_prompt_condition_routes_directly_to_unconstrained_generation(self):
        backend = object.__new__(HFBackend)
        sentinel = object()
        backend.generate_turn_unconstrained = lambda *args, **kwargs: sentinel
        self.assertIs(HFBackend.generate_turn(
            backend, "prompt", self.skills, "put", constrained=False
        ), sentinel)

    def test_strict_head_builds_and_uses_hard_decision_dfa(self):
        tokenizer = CharacterTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.vocab_size = len(tokenizer)
        backend.cfg = GenConfig(max_decision_tokens=512)
        backend._generate_until = lambda *args, **kwargs: chunk("reason</think>", tokenizer)
        captured = {}

        def generate(prompt, graph, **kwargs):
            captured.update(kwargs)
            ids = tokenizer.encode(DECISION + "</decision>")
            from ctrlg_alfworld.constraints import dfa_accepts
            self.assertTrue(dfa_accepts(graph, ids))
            return chunk(DECISION + "</decision>", tokenizer)

        backend._generate_dfa_span = generate
        head = backend._strict_head("P", self.skills, "put", greedy=True)
        self.assertIn("</decision>", head.chunk.text)
        self.assertEqual(captured["stop_string"], "</decision>")
        self.assertEqual(captured["max_new_tokens"], 512)

    def test_strict_condition_uses_policy_dfa_and_hmm_without_oracle_actions(self):
        tokenizer = CharacterTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.vocab_size = len(tokenizer)
        backend.hmm_model = object()
        backend.device = "cpu"
        backend.cfg = GenConfig(max_action_tokens=32)
        thought = chunk("reason</think>", tokenizer)
        decision = chunk(DECISION + "</decision>", tokenizer)
        head = BaseBackend._assemble_head(backend, thought, decision, use_decision=True)
        backend._strict_head = lambda *args, **kwargs: head
        seen = {}

        def hmm(prompt, graph, prefix):
            seen["graph"] = graph
            seen["prefix"] = prefix
            return chunk("inventory</action>", tokenizer)

        backend._generate_hmm_span = hmm
        turn = HFBackend.generate_turn(
            backend, "P", self.skills, "put", constrained=True
        )
        self.assertTrue(turn.decision_schema_valid)
        self.assertTrue(turn.action_grammar_valid)
        self.assertTrue(turn.hmm_applied)
        self.assertIn("preserve_target_lexical_type", turn.activated_policies)
        self.assertEqual(turn.parsed.action, "inventory")
        self.assertNotIn("admissible", repr(seen))

    def test_hmm_reranks_only_completed_policy_actions(self):
        import torch

        tokenizer = CharacterTokenizer()
        valid = tokenizer.encode("look</action>")
        invalid = tokenizer.encode("lookx</action>")
        graph = build_trie_dfa([valid], len(tokenizer))
        rows = [
            tokenizer.encode("P") + valid + [0, ord("$"), ord("$")],
            tokenizer.encode("P") + invalid + [0],
        ]
        width = max(map(len, rows))

        class Model:
            def generate(self, **kwargs):
                return torch.tensor([row + [0] * (width - len(row)) for row in rows])

        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.model = Model()
        backend.hmm_model = SimpleNamespace(eos_token_id=0)
        backend.vocab_size = len(tokenizer)
        backend.device = "cpu"
        backend.cfg = GenConfig(beam_size=2)

        ranked_inputs = []

        def rank(_model, candidates, _prompt, _suffix):
            ranked_inputs.extend(candidates)
            return candidates

        with (
            patch("ctrlg.DFAModel") as dfa_class,
            patch("ctrlg.ConstraintLogitsProcessor"),
            patch("ctrlg.rank_generated_ids", side_effect=rank),
        ):
            dfa_class.return_value.to.return_value = object()
            chunk = backend._generate_hmm_span("P", graph, [])

        self.assertEqual(ranked_inputs, [tuple(valid)])
        self.assertEqual(chunk.text, "look</action>")

    def test_vllm_sends_structured_regex_and_preserves_token_ids(self):
        tokenizer = CharacterTokenizer()
        backend = object.__new__(VLLMBackend)
        backend.tokenizer = tokenizer
        backend.model = "m"
        backend.cfg = GenConfig(seed=7)
        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                text = "look</action>"
                return SimpleNamespace(choices=[SimpleNamespace(
                    text=text, token_ids=tokenizer.encode(text), model_extra=None
                )])

        backend.client = SimpleNamespace(completions=Completions())
        result = backend._generate_until(
            "prompt", [ACTION_CLOSE], 32, 0.7,
            seed=11, guided_regex=r"look</action>",
        )
        self.assertEqual(
            captured["extra_body"]["structured_outputs"],
            {"regex": r"look</action>"},
        )
        self.assertNotIn("guided_regex", captured["extra_body"])
        self.assertEqual(captured["extra_body"]["seed"], 11)
        self.assertEqual(result.token_ids, tuple(tokenizer.encode("look</action>")))
        self.assertTrue(result.stop_found)

    def test_vllm_structured_regex_fails_closed(self):
        tokenizer = CharacterTokenizer()
        backend = object.__new__(VLLMBackend)
        backend.tokenizer = tokenizer
        backend.model = "m"
        backend.cfg = GenConfig(seed=7)

        class Completions:
            def create(self, **kwargs):
                text = "\nlook</action>"
                return SimpleNamespace(choices=[SimpleNamespace(
                    text=text, token_ids=tokenizer.encode(text), model_extra=None
                )])

        backend.client = SimpleNamespace(completions=Completions())
        with self.assertRaisesRegex(RuntimeError, "outside the requested"):
            backend._generate_until(
                "prompt", [ACTION_CLOSE], 32, 0.7,
                seed=11, guided_regex=r"look</action>",
            )

    def test_vllm_batch_preserves_input_order_and_per_candidate_seeds(self):
        backend = object.__new__(VLLMBackend)
        seen = []

        def generate(prompt, stops, cap, temperature, *, seed, guided_regex=None):
            seen.append((prompt, seed, guided_regex))
            return GeneratedChunk(str(seed), (seed,), True, False, 0.0)

        backend._generate_until = generate
        results = backend._generate_batch_until(
            ["a", "b", "c"], ["stop"], 10, 0.7, [31, 32, 33],
            guided_regex="regex",
        )
        self.assertEqual([result.text for result in results], ["31", "32", "33"])
        self.assertEqual(sorted(seen), [
            ("a", 31, "regex"), ("b", 32, "regex"), ("c", 33, "regex")
        ])

    def test_vllm_training_retries_failed_decision_with_distinct_seeds(self):
        tokenizer = CharacterTokenizer()
        backend = object.__new__(VLLMBackend)
        backend.tokenizer = tokenizer
        backend.cfg = GenConfig(
            max_decision_tokens=512,
            max_action_tokens=32,
            seed=17,
            decision_max_attempts=5,
            decision_retry_temperature=0.2,
        )
        decision_calls = []

        def generate(prompt, stops, cap, temperature, *, seed, guided_regex=None):
            if guided_regex is not None:
                decision_calls.append((temperature, seed))
                if len(decision_calls) < 3:
                    raise StructuredOutputMismatchError(
                        chunk("repeated", tokenizer, stop=False)
                    )
                return chunk(DECISION + "</decision>", tokenizer)
            if stops == ["</think>"]:
                return chunk("reason</think>", tokenizer)
            return chunk("look</action>", tokenizer)

        backend._generate_until = generate
        turns = backend.generate_turns_training(
            "prompt", self.skills, "put", count=1, seed_context=(2, 3, 0)
        )

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].decision_attempts, 3)
        self.assertTrue(turns[0].decision_retry_used)
        self.assertEqual(
            turns[0].decision_failure_reasons,
            ("max_tokens_before_accept", "max_tokens_before_accept"),
        )
        self.assertEqual([item[0] for item in decision_calls], [None, 0.2, 0.2])
        self.assertEqual(len(set(item[1] for item in decision_calls)), 3)
        self.assertEqual(
            turns[0].decision_attempt_seeds,
            tuple(item[1] for item in decision_calls),
        )

    def test_vllm_training_raises_typed_failure_after_bounded_attempts(self):
        tokenizer = CharacterTokenizer()
        backend = object.__new__(VLLMBackend)
        backend.tokenizer = tokenizer
        backend.cfg = GenConfig(
            seed=19,
            decision_max_attempts=3,
            decision_retry_temperature=0.2,
        )

        def generate(prompt, stops, cap, temperature, *, seed, guided_regex=None):
            if guided_regex is not None:
                raise StructuredOutputMismatchError(
                    chunk("repeated", tokenizer, stop=False)
                )
            return chunk("reason</think>", tokenizer)

        backend._generate_until = generate
        with self.assertRaises(DecisionGenerationExhausted) as caught:
            backend.generate_turns_training(
                "prompt", self.skills, "put", count=1, seed_context=(4, 5, 0)
            )

        self.assertEqual(len(caught.exception.attempt_records), 3)
        self.assertEqual(caught.exception.thought.text, "reason</think>")
        self.assertEqual(
            [record["temperature"] for record in caught.exception.attempt_records],
            [0.0, 0.2, 0.2],
        )


if __name__ == "__main__":
    unittest.main()
