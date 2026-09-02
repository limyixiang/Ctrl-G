import unittest

import torch

from ctrlg_alfworld.constraints import (
    FiniteActionLogitsProcessor,
    build_action_dfa,
    dfa_accepts,
    tokenize_fixed_suffix,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(item) for item in token_ids)


class ConstraintTests(unittest.TestCase):
    def test_action_dfa_accepts_exactly_admissible_commands(self):
        tokenizer = CharacterTokenizer()
        actions = ["look", "go to shelf 1"]
        dfa, tokenized = build_action_dfa(actions, tokenizer, 256, "<action>")
        self.assertTrue(all(dfa_accepts(dfa, tokenized[action]) for action in actions))
        self.assertFalse(dfa_accepts(dfa, tokenizer.encode("go to shelf 2")))
        self.assertFalse(dfa_accepts(dfa, tokenizer.encode("look extra")))

    def test_hard_mask_allows_only_valid_next_prefix_tokens(self):
        tokenizer = CharacterTokenizer()
        prompt = "<action>"
        prompt_ids = tokenizer.encode(prompt)
        processor = FiniteActionLogitsProcessor(
            tokenizer,
            prompt,
            ["look", "open drawer 1"],
            prompt_length=len(prompt_ids),
            eos_token_id=0,
        )
        scores = torch.zeros((1, 256))
        masked = processor(torch.tensor([prompt_ids]), scores)
        allowed = torch.isfinite(masked[0]).nonzero().flatten().tolist()
        self.assertEqual(allowed, sorted([ord("l"), ord("o")]))

    def test_hard_mask_follows_shared_prefix(self):
        tokenizer = CharacterTokenizer()
        prompt = "<action>"
        prompt_ids = tokenizer.encode(prompt)
        processor = FiniteActionLogitsProcessor(
            tokenizer,
            prompt,
            ["go to shelf 1", "go to shelf 2"],
            prompt_length=len(prompt_ids),
            eos_token_id=0,
        )
        generated = tokenizer.encode("go to shelf ")
        masked = processor(
            torch.tensor([prompt_ids + generated]), torch.zeros((1, 256))
        )
        allowed = torch.isfinite(masked[0]).nonzero().flatten().tolist()
        self.assertEqual(allowed, sorted([ord("1"), ord("2")]))

    def test_fixed_suffix_does_not_retokenize_action_body(self):
        tokenizer = CharacterTokenizer()
        prompt_ids = tokenizer.encode("<action>")
        body_ids = tokenizer.encode("look")
        suffix_ids = tokenize_fixed_suffix(
            tokenizer, prompt_ids + body_ids, "</action>"
        )
        self.assertEqual(tokenizer.decode(body_ids), "look")
        self.assertEqual(tokenizer.decode(suffix_ids), "</action>")
