import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ctrlg_alfworld.backends import (
    BaseBackend,
    StateGenerationExhausted,
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


STATE = (
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

    def test_strict_head_builds_and_uses_hard_state_dfa(self):
        tokenizer = CharacterTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.vocab_size = len(tokenizer)
        backend.cfg = GenConfig(max_state_tokens=512)
        backend._generate_until = lambda *args, **kwargs: chunk("reason</think>", tokenizer)
        captured = {}

        def generate(prompt, graph, **kwargs):
            captured.update(kwargs)
            self.assertTrue(prompt.endswith("<state>"))
            ids = tokenizer.encode(STATE + "</state>")
            from ctrlg_alfworld.constraints import dfa_accepts
            self.assertTrue(dfa_accepts(graph, ids))
            self.assertFalse(dfa_accepts(graph, tokenizer.encode(STATE + "</decision>")))
            return chunk(STATE + "</state>", tokenizer)

        backend._generate_dfa_span = generate
        head = backend._strict_head("P", self.skills, "put", greedy=True)
        self.assertIn("</state>", head.chunk.text)
        self.assertEqual(captured["stop_string"], "</state>")
        self.assertEqual(captured["max_new_tokens"], 512)
        self.assertEqual(
            head.chunk.text, "reason</think><state>" + STATE + "</state><action>"
        )

    def test_strict_condition_uses_policy_dfa_and_hmm_without_oracle_actions(self):
        tokenizer = CharacterTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.vocab_size = len(tokenizer)
        backend.hmm_model = object()
        backend.device = "cpu"
        backend.cfg = GenConfig(max_action_tokens=32)
        thought = chunk("reason</think>", tokenizer)
        state = chunk(STATE + "</state>", tokenizer)
        head = BaseBackend._assemble_head(backend, thought, state, use_state=True)
        backend._strict_head = lambda *args, **kwargs: head
        seen = {}

        def hmm(prompt, graph, prefix):
            seen["graph"] = graph
            seen["prefix"] = prefix
            return chunk("inventory</action>", tokenizer)

        backend._generate_hmm_span = hmm
        turn = HFBackend.generate_turn(
            backend, "P", self.skills, "put", constrained=True,
            hmm_context=SimpleNamespace(token_ids=(), truncated=False)
        )
        self.assertTrue(turn.state_schema_valid)
        self.assertTrue(turn.action_grammar_valid)
        self.assertTrue(turn.hmm_applied)
        self.assertIn("preserve_target_lexical_type", turn.activated_policies)
        self.assertEqual(turn.parsed.action, "inventory")
        self.assertEqual(
            tokenizer.decode(seen["prefix"]), "<state>" + STATE + "</state><action>"
        )
        self.assertNotIn("admissible", repr(seen))

    def test_hmm_skips_reranking_when_only_one_policy_action_remains(self):
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

        with (
            patch("ctrlg.DFAModel") as dfa_class,
            patch("ctrlg.ConstraintLogitsProcessor"),
            patch("ctrlg.rank_generated_ids") as rank,
        ):
            dfa_class.return_value.to.return_value = object()
            chunk = backend._generate_hmm_span("P", graph, [])

        rank.assert_not_called()
        self.assertEqual(chunk.text, "look</action>")

    def test_action_dfa_cache_tracks_held_object_and_current_location(self):
        from ctrlg_alfworld.constraints import dfa_accepts

        tokenizer = CharacterTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer, backend.vocab_size, backend.device = tokenizer, len(tokenizer), "cpu"
        backend.hmm_model = object()
        backend.cfg = GenConfig(max_action_tokens=32)
        occupied = STATE.replace("held: none", "held: cup 1").replace("held_relation: none", "held_relation: unknown")
        states = iter([STATE, occupied, occupied.replace("at: countertop 1", "at: countertop 2"), STATE])

        def head(*args, **kwargs):
            return BaseBackend._assemble_head(
                backend, chunk("reason</think>", tokenizer),
                chunk(next(states) + "</state>", tokenizer), use_state=True,
            )

        backend._strict_head = head
        graphs = []

        def generate(prompt, graph, prefix):
            graphs.append(graph)
            return chunk("inventory</action>", tokenizer)

        backend._generate_hmm_span = generate
        for _ in range(4):
            HFBackend.generate_turn(backend, "P", self.skills, "put", constrained=True,
                                    hmm_context=SimpleNamespace(token_ids=(), truncated=False))
        take = tokenizer.encode("take cup 1 from countertop 1</action>")
        goto = tokenizer.encode("go to countertop 1</action>")
        self.assertTrue(dfa_accepts(graphs[0], take))
        self.assertFalse(dfa_accepts(graphs[1], take))
        self.assertFalse(dfa_accepts(graphs[1], goto))
        self.assertTrue(dfa_accepts(graphs[2], goto))
        self.assertIs(graphs[0], graphs[3])
        self.assertEqual(len(backend._action_dfa_cache), 3)

    def test_rejected_required_policy_is_preserved_in_generated_turn(self):
        tokenizer = CharacterTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer, backend.vocab_size, backend.device = tokenizer, len(tokenizer), "cpu"
        backend.hmm_model = object()
        backend.cfg = GenConfig(max_action_tokens=32)
        state = STATE.replace("previous_action: look", "previous_action: open countertop 1").replace(
            "arrived_receptacle_state: open", "arrived_receptacle_state: closed",
        )
        head = BaseBackend._assemble_head(backend, chunk("reason</think>", tokenizer),
                                         chunk(state + "</state>", tokenizer), use_state=True)
        backend._strict_head = lambda *args, **kwargs: head
        backend._generate_hmm_span = lambda *args: chunk("inventory</action>", tokenizer)
        turn = HFBackend.generate_turn(backend, "P", self.skills, "put", constrained=True,
                                       hmm_context=SimpleNamespace(token_ids=(), truncated=False))
        self.assertEqual(turn.rejected_required_policies,
                         {"open_declared_closed_receptacle": "forbidden_by_restrictions"})
        self.assertIn("do_not_repeat_declared_previous_action", turn.activated_policies)
        self.assertTrue(turn.policy_satisfied)

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

    def test_training_is_unmasked_and_attempt_seeds_are_distinct(self):
        backend = object.__new__(VLLMBackend)
        backend.tokenizer = CharacterTokenizer()
        backend.cfg = GenConfig(seed=17)
        calls = []
        def generate(prompt, stops, cap, temperature, *, seed, guided_regex=None):
            self.assertIsNone(guided_regex)
            calls.append((stops, temperature, seed))
            if stops == ["</think>"]:
                return chunk("reason</think>", backend.tokenizer)
            if stops == ["</state>"]:
                return chunk(STATE + "</state>", backend.tokenizer)
            return chunk("look</action>", backend.tokenizer)
        backend._generate_until = generate
        turns = backend.generate_turns_training("P", self.skills, "put", count=2, seed_context=(2,3,0))
        self.assertEqual(len(turns), 2)
        state_calls = [c for c in calls if c[0] == ["</state>"]]
        self.assertEqual([c[1] for c in state_calls], [0.7, 0.7])
        self.assertEqual(len({c[2] for c in state_calls}), 2)
        self.assertTrue(all(t.state_schema_valid for t in turns))
        self.assertTrue(all(t.state_attempts == 1 for t in turns))

    def test_joint_inference_uses_context_in_both_hmm_phases_with_matched_sampling(self):
        import torch
        from ctrlg import HMM
        from ctrlg_alfworld.hmm_context import build_hmm_context
        tokenizer = CharacterTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer, backend.vocab_size, backend.device = tokenizer, len(tokenizer), "cpu"
        backend.cfg = GenConfig(do_sample=True, temperature=.6, rollout_temperature=.4)
        backend.hmm_model = HMM(2, len(tokenizer), 0)
        calls = []
        class Model:
            def generate(self, **kwargs):
                prompt = tokenizer.decode(kwargs["input_ids"][0].tolist())
                text = (STATE + "</state>" if prompt.endswith("<state>") else
                        "inventory</action>" if prompt.endswith("<action>") else "reason</think>")
                generated = tokenizer.encode(text)
                if "logits_processor" in kwargs:
                    current = kwargs["input_ids"]
                    # Check real constraint scores at the beginning and boundary.
                    for index, token in enumerate(generated):
                        if index in (0, len(generated)-1):
                            scores = torch.zeros(1, len(tokenizer))
                            for processor in kwargs["logits_processor"]:
                                scores = processor(current, scores)
                            self.assert_allowed(scores, token)
                        current = torch.cat((current, torch.tensor([[token]])), dim=1)
                calls.append((prompt, kwargs))
                if prompt.endswith("<action>") and "logits_processor" in kwargs:
                    generated += [0]
                return torch.tensor([kwargs["input_ids"][0].tolist()+generated])
            @staticmethod
            def assert_allowed(scores, token):
                assert torch.isfinite(scores[0,token]) and scores[0,token] > -1e20
        backend.model = Model()
        context = build_hmm_context(tokenizer, "put cup", "room", "room", [])
        turn = backend.generate_turn("P", self.skills, "put", constrained=True,
                                     greedy_head=False, hmm_context=context)
        self.assertTrue(turn.state_hmm_applied and turn.hmm_applied)
        self.assertEqual(list(turn.hmm_prefix_token_ids[:len(context.token_ids)]), list(context.token_ids))
        self.assertEqual(calls[1][1]["temperature"], .4)
        self.assertEqual(calls[2][1]["temperature"], .6)
        self.assertEqual(calls[1][1]["num_beams"], 1)
        state_processor = calls[1][1]["logits_processor"][0]
        self.assertEqual(state_processor.prefix_ids, list(context.token_ids)+tokenizer.encode("<state>"))
        self.assertEqual(state_processor.suffix_ids, [])
        self.assertEqual(turn.parsed.action, "inventory")
        # The prompt arm gets exactly the same head/action sampling settings.
        calls.clear()
        baseline = backend.generate_turn("P", self.skills, "put", constrained=False,
                                         greedy_head=False, hmm_context=context)
        self.assertEqual([kw["temperature"] for _,kw in calls], [.4,.4,.6])
        self.assertEqual(turn.total_generated_tokens, baseline.total_generated_tokens + 1)

    def test_training_generates_action_after_invalid_state_and_preserves_spans(self):
        from ctrlg_alfworld.collection import observed_attempt_span
        for backend_type in (HFBackend, VLLMBackend):
            for state_text, state_closed in (("{task: put", False),
                                             ("malformed</state>", True), ("", False)):
                with self.subTest(backend=backend_type.__name__, state=state_text):
                    backend = object.__new__(backend_type)
                    backend.tokenizer = CharacterTokenizer()
                    backend.cfg = GenConfig(seed=19)
                    calls = []
                    def generate(prompt, stops, cap, temperature, *, seed=None, guided_regex=None):
                        self.assertIsNone(guided_regex)
                        calls.append((prompt, stops, temperature, seed))
                        if stops == ["</think>"]:
                            return chunk("reason</think>", backend.tokenizer)
                        if stops == ["</state>"]:
                            return chunk(state_text, backend.tokenizer, stop=state_closed)
                        return chunk("look</action>", backend.tokenizer)
                    backend._generate_until = generate
                    turn = backend.generate_turns_training("P", self.skills, "put", count=1)[0]
                    span = observed_attempt_span(turn, 0)
                    self.assertEqual([c[1] for c in calls], [["</think>"], ["</state>"], ["</action>"]])
                    self.assertEqual([c[2] for c in calls], [None, .7, .7])
                    self.assertTrue(calls[2][0].endswith("</state><action>"))
                    self.assertEqual(turn.state_token_ids, tuple(backend.tokenizer.encode(state_text)))
                    self.assertEqual(turn.parsed.action, "look")
                    self.assertEqual(turn.tail_token_ids, tuple(backend.tokenizer.encode("look</action>")))
                    self.assertFalse(turn.state_schema_valid)
                    self.assertIn("schema", turn.state_schema_error)
                    self.assertTrue(span["selected_for_execution"])
                    self.assertEqual(span["execution_rejection_reasons"], [])
                    self.assertEqual(span["distill_eligible"], bool(state_text))
                    self.assertEqual(turn.used_state_repair, not state_closed)
                    if not state_closed:
                        self.assertEqual(backend.tokenizer.decode(turn.hmm_target_token_ids),
                                         "<state>" + state_text)
                        self.assertEqual(backend.tokenizer.decode(turn.synthetic_state_close_token_ids), "</state>")
                        self.assertFalse(span["hmm_appended_eos"])
                        self.assertFalse(span["hmm_completed"])
                        self.assertIsNone(span["hmm_action_start_in_target"])
                        self.assertEqual(span["hmm_action_exclusion_reason"],
                                         "generated_after_synthetic_state_close")
                    else:
                        self.assertTrue(span["hmm_completed"])
                        self.assertEqual(turn.synthetic_state_close_token_ids, ())
                        self.assertIsNone(span["hmm_action_exclusion_reason"])
                    if backend_type is VLLMBackend:
                        self.assertIsNotNone(turn.tail_seed)
                        self.assertEqual(len({c[3] for c in calls}), 3)


if __name__ == "__main__":
    unittest.main()
