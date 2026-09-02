import unittest
from types import SimpleNamespace

import torch

from ctrlg_alfworld.distillation import (
    extract_lvd_embeddings,
    pad_sequences,
    split_records,
    validate_tokenizer_contract,
    validate_record,
)


def make_record(use_decision, marker):
    prefix = [10, marker]
    tail = [20, 21]
    eos = 0
    return {
        "use_decision": use_decision,
        "prompt_token_ids": [1, 2],
        "head_token_ids": [3, 4] + prefix,
        "hmm_prefix_token_ids": prefix,
        "tail_token_ids": tail,
        "hmm_sequence_token_ids": prefix + tail + [eos],
        "raw_tail": "look</action>",
        "tail_span_exact": True,
        "head_truncated": False,
        "tail_truncated": False,
    }


class DistillationDataTests(unittest.TestCase):
    def test_validate_record_enforces_generated_prefix_alignment(self):
        record = make_record(False, 11)
        validate_record(record)
        record["head_token_ids"][-1] = 99
        with self.assertRaisesRegex(ValueError, "not a suffix"):
            validate_record(record)

    def test_validate_record_rejects_non_exact_tail(self):
        record = make_record(False, 11)
        record["raw_tail"] = "look</action>,"
        record["tail_span_exact"] = False
        with self.assertRaisesRegex(ValueError, "not exactly token-aligned"):
            validate_record(record)

    def test_validate_record_rejects_extra_tokens_after_eos(self):
        record = make_record(False, 11)
        record["hmm_sequence_token_ids"].insert(-1, 99)
        with self.assertRaisesRegex(ValueError, r"prefix \+ tail \+ one EOS"):
            validate_record(record)

    def test_split_is_nonempty_and_deterministic(self):
        records = [make_record(index % 2 == 0, 20 + index) for index in range(10)]
        first = split_records(records, dev_fraction=0.2, seed=5)
        second = split_records(records, dev_fraction=0.2, seed=5)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(len(first[1]), 2)

    def test_split_keeps_samples_from_one_state_together(self):
        records = []
        for episode in range(4):
            for use_decision in (False, True):
                record = make_record(use_decision, 20 + episode)
                record.update({"episode": episode, "step": 0})
                records.append(record)
        train, dev = split_records(records, dev_fraction=0.25, seed=5)
        train_states = {(item["episode"], item["step"]) for item in train}
        dev_states = {(item["episode"], item["step"]) for item in dev}
        self.assertFalse(train_states & dev_states)
        self.assertEqual(len(dev), 2)

    def test_padding_uses_hmm_eos(self):
        records = [make_record(False, 11), make_record(True, 12)]
        records[0]["hmm_sequence_token_ids"] = [10, 11, 20, 0]
        tensor = pad_sequences(records, eos_token_id=0)
        self.assertEqual(tensor.dtype, torch.long)
        self.assertEqual(tensor.shape, (2, 5))
        self.assertEqual(tensor[0, -1].item(), 0)

    def test_embeddings_are_aligned_to_causal_predecessors(self):
        class PositionEchoModel:
            device = "cpu"

            def __call__(self, input_ids, output_hidden_states, use_cache):
                # A one-dimensional hidden vector equal to the token at each
                # input position makes the causal offset directly observable.
                hidden = input_ids.to(torch.float32).unsqueeze(-1)
                return SimpleNamespace(hidden_states=(hidden,))

        record = make_record(False, 11)
        sequences, embeddings = extract_lvd_embeddings(
            PositionEchoModel(), [record], eos_token_id=0
        )
        self.assertEqual(sequences.tolist(), [[10, 11, 20, 21, 0]])
        # Targets [10,11,20,21,EOS] are predicted from predecessor tokens
        # [4,10,11,20,21] in prompt+head+tail+EOS.
        self.assertEqual(
            embeddings[0, :, 0].tolist(), [4.0, 10.0, 11.0, 20.0, 21.0]
        )

    def test_tokenizer_contract_rejects_added_delimiter(self):
        class Tokenizer:
            vocab_size = 128
            eos_token_id = 0

            def get_added_vocab(self):
                return {"<action>": 128}

        with self.assertRaisesRegex(ValueError, "added tokens"):
            validate_tokenizer_contract(Tokenizer(), [make_record(False, 11)])

    def test_tokenizer_contract_rejects_out_of_vocab_sample_id(self):
        class Tokenizer:
            vocab_size = 64
            eos_token_id = 0

            def get_added_vocab(self):
                return {}

        record = make_record(False, 70)
        with self.assertRaisesRegex(ValueError, "outside HMM vocabulary"):
            validate_tokenizer_contract(Tokenizer(), [record])
