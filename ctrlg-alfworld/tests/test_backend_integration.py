import unittest
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

TEST_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_PACKAGE_ROOT / "src"))
sys.path.insert(0, str(TEST_PACKAGE_ROOT.parent))

import ctrlg
from ctrlg_alfworld.backends import GenConfig, HFBackend, VLLMBackend
from ctrlg_alfworld.generation import GeneratedChunk


class AsciiTokenizer:
    eos_token_id = 127

    def __len__(self):
        return 128

    def encode(
        self, text, add_special_tokens=False, return_tensors=None
    ):
        token_ids = [ord(character) for character in text]
        if any(token >= self.eos_token_id for token in token_ids):
            raise ValueError("test tokenizer only supports seven-bit ASCII")
        if return_tensors == "pt":
            return torch.tensor([token_ids], dtype=torch.long)
        return token_ids

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(
            chr(token) for token in token_ids if token != self.eos_token_id
        )


def make_backend():
    config = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_ctx=64,
        n_embd=16,
        n_layer=1,
        n_head=1,
        bos_token_id=None,
        eos_token_id=127,
        pad_token_id=127,
    )
    backend = object.__new__(HFBackend)
    backend.tokenizer = AsciiTokenizer()
    backend.model = GPT2LMHeadModel(config).eval()
    backend.device = "cpu"
    backend.cfg = GenConfig(
        max_action_tokens=16,
        beam_size=2,
        do_sample=False,
        temperature=1.0,
    )
    backend.vocab_size = 128
    backend.hmm_model = ctrlg.HMM(
        hidden_states=4, vocab_size=128, eos_token_id=127
    )
    return backend


class BackendIntegrationTests(unittest.TestCase):
    def test_head_generation_uses_independent_thought_and_decision_budgets(self):
        tokenizer = AsciiTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.cfg = GenConfig(max_thought_tokens=7, max_decision_tokens=5)
        calls = []

        def generate(prompt, stops, max_new_tokens, temperature):
            calls.append((prompt, list(stops), max_new_tokens, temperature))
            text = "reason</think>" if len(calls) == 1 else "go</decision>"
            return GeneratedChunk(
                text=text,
                token_ids=tuple(tokenizer.encode(text)),
                stop_found=True,
                truncated=False,
                latency_seconds=0.01,
            )

        backend._generate_until = generate
        head = backend._generate_head(
            "P<think>", use_decision=True, greedy=False
        )

        self.assertEqual(
            calls,
            [
                ("P<think>", ["</think>"], 7, 0.7),
                (
                    "P<think>reason</think><decision>",
                    ["</decision>"],
                    5,
                    0.7,
                ),
            ],
        )
        self.assertEqual(
            head.chunk.text,
            "reason</think><decision>go</decision><action>",
        )
        self.assertFalse(head.used_repair)

    def test_head_generation_repairs_each_phase_without_spending_next_budget(self):
        tokenizer = AsciiTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.cfg = GenConfig(max_thought_tokens=7, max_decision_tokens=5)
        outputs = iter(("reason", "go"))

        def generate(prompt, stops, max_new_tokens, temperature):
            text = next(outputs)
            return GeneratedChunk(
                text=text,
                token_ids=tuple(tokenizer.encode(text)),
                stop_found=False,
                truncated=True,
                latency_seconds=0.01,
            )

        backend._generate_until = generate
        head = backend._generate_head(
            "P<think>", use_decision=True, greedy=True
        )

        self.assertEqual(
            head.chunk.text,
            "reason</think><decision>go</decision><action>",
        )
        self.assertTrue(head.used_thought_repair)
        self.assertTrue(head.used_decision_repair)
        self.assertTrue(head.chunk.truncated)

    def _run_mocked_hmm_temperature_case(self, *, do_sample):
        tokenizer = AsciiTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.device = "cpu"
        backend.vocab_size = 128
        backend.cfg = GenConfig(
            do_sample=do_sample,
            temperature=0.35,
            beam_size=3,
        )
        backend.hmm_model = SimpleNamespace(eos_token_id=tokenizer.eos_token_id)
        captured = {}

        class FakeDFA:
            def __init__(self, graph, vocab_size):
                pass

            def to(self, device):
                return self

        class FakeProcessor:
            def __init__(self, *args, **kwargs):
                captured["processor_temperature"] = kwargs["temperature"]

        class FakeModel:
            def generate(self, **kwargs):
                captured["generate"] = kwargs
                return torch.cat(
                    (
                        kwargs["input_ids"],
                        torch.tensor([[ord("a")]], dtype=torch.long),
                    ),
                    dim=1,
                )

        backend.model = FakeModel()
        with (
            patch.object(ctrlg, "DFAModel", FakeDFA),
            patch.object(ctrlg, "ConstraintLogitsProcessor", FakeProcessor),
            patch.object(
                ctrlg, "extract_generated_ids", return_value=[[ord("a")]]
            ),
            patch.object(
                ctrlg, "rank_generated_ids", return_value=[[ord("a")]]
            ),
        ):
            backend._generate_hmm_action(
                "P</think><action>", ["a"], tokenizer.encode("<action>")
            )
        return captured

    def test_hmm_sampling_applies_temperature_once_in_generate(self):
        captured = self._run_mocked_hmm_temperature_case(do_sample=True)
        self.assertEqual(captured["processor_temperature"], 1.0)
        self.assertEqual(captured["generate"]["temperature"], 0.35)
        self.assertTrue(captured["generate"]["do_sample"])
        self.assertEqual(captured["generate"]["num_beams"], 1)

    def test_hmm_beam_search_does_not_apply_temperature(self):
        captured = self._run_mocked_hmm_temperature_case(do_sample=False)
        self.assertEqual(captured["processor_temperature"], 1.0)
        self.assertIsNone(captured["generate"]["temperature"])
        self.assertFalse(captured["generate"]["do_sample"])
        self.assertEqual(captured["generate"]["num_beams"], 3)

    def test_hmm_prefix_has_no_limit_by_default(self):
        tokenizer = AsciiTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.cfg = GenConfig()
        tail_text = "a</action>"
        thought = GeneratedChunk(
            text="reason</think>",
            token_ids=tuple(tokenizer.encode("reason</think>")),
            stop_found=True,
            truncated=False,
            latency_seconds=0.01,
        )
        head = backend._assemble_head(thought, None, use_decision=False)
        backend._generate_head = lambda prompt, use_decision, greedy: head
        backend._generate_dfa_action = lambda *args, **kwargs: self.fail(
            "DFA fallback should not run when the prefix limit is omitted"
        )
        backend._generate_hmm_action = lambda prompt, actions, prefix_ids: (
            "a",
            tuple(tokenizer.encode("a")),
            tuple(tokenizer.encode(tail_text)),
            0.02,
        )
        turn = HFBackend.generate_turn(
            backend,
            "P<think>",
            ["a"],
            use_decision=False,
            use_hmm=True,
        )
        self.assertTrue(turn.hmm_applied)
        self.assertIsNone(turn.hmm_skip_reason)
        self.assertGreater(len(turn.hmm_prefix_token_ids), 1)

    def test_truncated_thought_does_not_disable_hmm(self):
        tokenizer = AsciiTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.cfg = GenConfig()
        thought = GeneratedChunk(
            text="unfinished thought",
            token_ids=tuple(tokenizer.encode("unfinished thought")),
            stop_found=False,
            truncated=True,
            latency_seconds=0.01,
        )
        decision = GeneratedChunk(
            text="go</decision>",
            token_ids=tuple(tokenizer.encode("go</decision>")),
            stop_found=True,
            truncated=False,
            latency_seconds=0.01,
        )
        head = backend._assemble_head(thought, decision, use_decision=True)
        backend._generate_head = lambda prompt, use_decision, greedy: head
        backend._generate_dfa_action = lambda *args, **kwargs: self.fail(
            "DFA fallback should not run for thought-only truncation"
        )
        backend._generate_hmm_action = lambda prompt, actions, prefix_ids: (
            "a",
            tuple(tokenizer.encode("a")),
            tuple(tokenizer.encode("a</action>")),
            0.02,
        )

        turn = HFBackend.generate_turn(
            backend,
            "P<think>",
            ["a"],
            use_decision=True,
            use_hmm=True,
        )

        self.assertTrue(turn.thought_truncated)
        self.assertFalse(turn.decision_truncated)
        self.assertTrue(turn.hmm_applied)
        self.assertIsNone(turn.hmm_skip_reason)

    def test_truncated_decision_disables_hmm(self):
        tokenizer = AsciiTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.cfg = GenConfig()
        thought = GeneratedChunk(
            text="reason</think>",
            token_ids=tuple(tokenizer.encode("reason</think>")),
            stop_found=True,
            truncated=False,
            latency_seconds=0.01,
        )
        decision = GeneratedChunk(
            text="unfinished decision",
            token_ids=tuple(tokenizer.encode("unfinished decision")),
            stop_found=False,
            truncated=True,
            latency_seconds=0.01,
        )
        head = backend._assemble_head(thought, decision, use_decision=True)
        backend._generate_head = lambda prompt, use_decision, greedy: head
        backend._generate_hmm_action = lambda *args, **kwargs: self.fail(
            "HMM should not run for a truncated decision"
        )
        backend._generate_dfa_action = lambda prompt, actions: GeneratedChunk(
            text="a</action>",
            token_ids=tuple(tokenizer.encode("a</action>")),
            stop_found=True,
            truncated=False,
            latency_seconds=0.02,
        )

        turn = HFBackend.generate_turn(
            backend,
            "P<think>",
            ["a"],
            use_decision=True,
            use_hmm=True,
        )

        self.assertTrue(turn.decision_truncated)
        self.assertFalse(turn.hmm_applied)
        self.assertEqual(turn.hmm_skip_reason, "synthetic_decision_close")

    def test_overlong_hmm_prefix_is_measured_and_falls_back_to_dfa(self):
        tokenizer = AsciiTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.cfg = GenConfig(max_hmm_prefix_tokens=1)
        tail_text = "a</action>"
        thought = GeneratedChunk(
            text="reason</think>",
            token_ids=tuple(tokenizer.encode("reason</think>")),
            stop_found=True,
            truncated=False,
            latency_seconds=0.01,
        )
        decision = GeneratedChunk(
            text="x</decision>",
            token_ids=tuple(tokenizer.encode("x</decision>")),
            stop_found=True,
            truncated=False,
            latency_seconds=0.01,
        )
        head = backend._assemble_head(thought, decision, use_decision=True)
        backend._generate_head = lambda prompt, use_decision, greedy: head
        backend._generate_dfa_action = lambda prompt, actions: GeneratedChunk(
            text=tail_text,
            token_ids=tuple(tokenizer.encode(tail_text)),
            stop_found=True,
            truncated=False,
            latency_seconds=0.02,
        )
        backend._generate_hmm_action = lambda *args, **kwargs: self.fail(
            "HMM path should not run for an overlong prefix"
        )
        turn = HFBackend.generate_turn(
            backend,
            "P<think>",
            ["a"],
            use_decision=True,
            use_hmm=True,
        )
        self.assertFalse(turn.hmm_applied)
        self.assertEqual(turn.hmm_skip_reason, "hmm_prefix_too_long")
        self.assertEqual(turn.parsed.action, "a")

    def test_pure_dfa_generation_emits_only_an_allowed_action(self):
        torch.manual_seed(0)
        backend = make_backend()
        prompt = "P</think><action>"
        chunk = backend._generate_dfa_action(prompt, ["a", "b"])
        self.assertIn(chunk.text, {"a</action>", "b</action>"})
        self.assertTrue(chunk.stop_found)

    def test_real_ctrlg_processor_returns_an_allowed_action(self):
        torch.manual_seed(0)
        backend = make_backend()
        prompt = "P</think><action>"
        prefix_ids = backend.tokenizer.encode("<action>")
        action, action_ids, tail_ids, latency = backend._generate_hmm_action(
            prompt, ["a", "b"], prefix_ids
        )
        self.assertIn(action, {"a", "b"})
        self.assertEqual(
            backend.tokenizer.decode(action_ids), action
        )
        self.assertEqual(
            backend.tokenizer.decode(tail_ids), action + "</action>"
        )
        self.assertGreaterEqual(latency, 0.0)

    def test_vllm_path_requires_and_preserves_server_token_ids(self):
        tokenizer = AsciiTokenizer()
        text = "a</action>"
        choice = SimpleNamespace(text=text, token_ids=tokenizer.encode(text))

        class Completions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(choices=[choice])

        backend = object.__new__(VLLMBackend)
        backend.tokenizer = tokenizer
        backend.client = SimpleNamespace(completions=Completions())
        backend.model = "tiny"
        backend.cfg = GenConfig()
        chunk = backend._generate_until(
            "P<action>",
            ["</action>"],
            max_new_tokens=16,
            temperature=0.7,
            seed=123,
        )
        self.assertEqual(chunk.text, text)
        self.assertEqual(chunk.token_ids, tuple(tokenizer.encode(text)))
        self.assertTrue(
            backend.client.completions.kwargs["extra_body"]["return_token_ids"]
        )
        self.assertEqual(backend.client.completions.kwargs["extra_body"]["seed"], 123)

    def test_vllm_batches_three_phases_with_stable_candidate_seeds(self):
        tokenizer = AsciiTokenizer()
        backend = object.__new__(VLLMBackend)
        backend.tokenizer = tokenizer
        backend.cfg = GenConfig(
            max_thought_tokens=64,
            max_decision_tokens=32,
            max_action_tokens=16,
            rollout_temperature=0.7,
            seed=42,
        )
        calls = []
        thought_texts = [
            "reason 0</think>",
            "reason 1",
        ]
        decision_texts = ["d0</decision>", "d1"]
        tail_texts = ["look</action>", "inventory</action>"]

        def chunks(texts, *, stop_found=True):
            return [
                GeneratedChunk(
                    text=text,
                    token_ids=tuple(tokenizer.encode(text)),
                    stop_found=(
                        stop_found[index]
                        if isinstance(stop_found, list)
                        else stop_found
                    ),
                    truncated=not (
                        stop_found[index]
                        if isinstance(stop_found, list)
                        else stop_found
                    ),
                    latency_seconds=0.01,
                )
                for index, text in enumerate(texts)
            ]

        def generate_batch(prompts, stop_strings, max_new_tokens, temperature, seeds):
            calls.append(
                (
                    list(prompts),
                    list(stop_strings),
                    max_new_tokens,
                    temperature,
                    list(seeds),
                )
            )
            if len(calls) == 1:
                return chunks(thought_texts, stop_found=[True, False])
            if len(calls) == 2:
                return chunks(decision_texts, stop_found=[True, False])
            return chunks(tail_texts)

        backend._generate_batch_until = generate_batch
        turns = backend.generate_turns_unconstrained(
            "P<think>",
            count=2,
            use_decision=True,
            seed_context=(3, 4),
        )

        expected_thought_seeds = [
            VLLMBackend._stable_seed(42, (3, 4), index, "thought")
            for index in range(2)
        ]
        expected_decision_seeds = [
            VLLMBackend._stable_seed(42, (3, 4), index, "decision")
            for index in range(2)
        ]
        expected_action_seeds = [
            VLLMBackend._stable_seed(42, (3, 4), index, "action")
            for index in range(2)
        ]
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0], ["P<think>", "P<think>"])
        self.assertEqual(calls[0][1], ["</think>"])
        self.assertEqual(calls[0][2], 64)
        self.assertEqual(calls[0][4], expected_thought_seeds)
        self.assertEqual(
            calls[1][0],
            [
                "P<think>reason 0</think><decision>",
                "P<think>reason 1</think><decision>",
            ],
        )
        self.assertEqual(calls[1][1], ["</decision>"])
        self.assertEqual(calls[1][2], 32)
        self.assertEqual(calls[1][4], expected_decision_seeds)
        self.assertEqual(
            calls[2][0],
            [
                "P<think>reason 0</think><decision>d0</decision><action>",
                "P<think>reason 1</think><decision>d1</decision><action>",
            ],
        )
        self.assertEqual(calls[2][1], ["</action>"])
        self.assertEqual(calls[2][4], expected_action_seeds)
        self.assertEqual([turn.parsed.action for turn in turns], ["look", "inventory"])
        self.assertEqual([turn.used_head_repair for turn in turns], [False, True])
        self.assertEqual([turn.used_thought_repair for turn in turns], [False, True])
        self.assertEqual([turn.used_decision_repair for turn in turns], [False, True])
        self.assertEqual([turn.head_seed for turn in turns], expected_thought_seeds)
        self.assertEqual(
            [turn.decision_seed for turn in turns], expected_decision_seeds
        )
        self.assertEqual([turn.tail_seed for turn in turns], expected_action_seeds)
        all_seeds = (
            expected_thought_seeds + expected_decision_seeds + expected_action_seeds
        )
        self.assertEqual(len(set(all_seeds)), 6)

    def test_vllm_batch_requests_are_concurrent_and_results_stay_ordered(self):
        tokenizer = AsciiTokenizer()
        barrier = threading.Barrier(2)
        seeds_by_prompt = {}

        class Completions:
            def create(self, **kwargs):
                seeds_by_prompt[kwargs["prompt"]] = kwargs["extra_body"]["seed"]
                barrier.wait(timeout=1.0)
                text = kwargs["prompt"][-1] + "</action>"
                choice = SimpleNamespace(
                    text=text, token_ids=tokenizer.encode(text)
                )
                return SimpleNamespace(choices=[choice])

        backend = object.__new__(VLLMBackend)
        backend.tokenizer = tokenizer
        backend.client = SimpleNamespace(completions=Completions())
        backend.model = "tiny"
        backend.cfg = GenConfig()
        chunks = backend._generate_batch_until(
            ["prompt-a", "prompt-b"],
            ["</action>"],
            16,
            0.7,
            [101, 202],
        )

        self.assertEqual(
            [chunk.text for chunk in chunks], ["a</action>", "b</action>"]
        )
        self.assertEqual(seeds_by_prompt, {"prompt-a": 101, "prompt-b": 202})

    def test_vllm_candidate_seeds_are_reproducible_and_context_specific(self):
        first = [
            VLLMBackend._stable_seed(42, (1, 2), index, phase)
            for phase in ("thought", "decision", "action")
            for index in range(4)
        ]
        second = [
            VLLMBackend._stable_seed(42, (1, 2), index, phase)
            for phase in ("thought", "decision", "action")
            for index in range(4)
        ]
        other_step = [
            VLLMBackend._stable_seed(42, (1, 3), index, phase)
            for phase in ("thought", "decision", "action")
            for index in range(4)
        ]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))
        self.assertTrue(set(first).isdisjoint(other_step))


if __name__ == "__main__":
    unittest.main()
