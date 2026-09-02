import unittest
import sys
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
        head_text = "reason</think><action>"
        tail_text = "a</action>"
        backend._generate_head = lambda prompt, greedy: GeneratedChunk(
            text=head_text,
            token_ids=tuple(tokenizer.encode(head_text)),
            stop_found=True,
            truncated=False,
            latency_seconds=0.01,
        )
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

    def test_overlong_hmm_prefix_is_measured_and_falls_back_to_dfa(self):
        tokenizer = AsciiTokenizer()
        backend = object.__new__(HFBackend)
        backend.tokenizer = tokenizer
        backend.cfg = GenConfig(max_hmm_prefix_tokens=1)
        head_text = "reason</think><decision>x</decision><action>"
        tail_text = "a</action>"
        backend._generate_head = lambda prompt, greedy: GeneratedChunk(
            text=head_text,
            token_ids=tuple(tokenizer.encode(head_text)),
            stop_found=True,
            truncated=False,
            latency_seconds=0.01,
        )
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
        backend._request_index = 0
        chunk = backend._generate_until(
            "P<action>", ["</action>"], max_new_tokens=16, temperature=0.7
        )
        self.assertEqual(chunk.text, text)
        self.assertEqual(chunk.token_ids, tuple(tokenizer.encode(text)))
        self.assertTrue(
            backend.client.completions.kwargs["extra_body"]["return_token_ids"]
        )
        self.assertEqual(
            backend.client.completions.kwargs["extra_body"]["seed"], 42
        )


if __name__ == "__main__":
    unittest.main()
