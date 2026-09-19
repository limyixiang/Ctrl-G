import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ctrlg_alfworld.distillation import (
    extract_lvd_embeddings,
    load_advance_selections,
    load_eligible_records,
    pad_sequences,
    split_records,
    validate_prompt_regime,
    validate_tokenizer_contract,
    validate_record,
)


def make_record(use_decision, marker):
    prefix = [10, marker]
    tail = [20, 21]
    eos = 0
    return {
        "use_decision": use_decision,
        "schema_version": 1,
        "policy_version": 1,
        "constrained_decision_collection": True,
        "no_oracle_filtering": True,
        "skills_sha256": "skills",
        "model": "model",
        "tokenizer": "model",
        "prompt_format": "decision_with_persistent_history_no_oracle_v1",
        "prompt_token_ids": [1, 2],
        "head_token_ids": [3, 4] + prefix,
        "hmm_prefix_token_ids": prefix,
        "tail_token_ids": tail,
        "hmm_sequence_token_ids": prefix + tail + [eos],
        "raw_tail": "look</action>",
        "tail_span_exact": True,
        "head_truncated": False,
        "decision_truncated": False,
        "used_decision_repair": False,
        "tail_truncated": False,
    }


def make_rollout_record(*, episode, step, sample, action, marker, eligible=True):
    record = make_record(True, marker)
    record.update(
        {
            "episode": episode,
            "step": step,
            "sample": sample,
            "action": action,
            "parse_ok": True,
            "action_was_admissible": True,
            "distill_eligible": eligible,
        }
    )
    return record


def write_jsonl(directory, name, records):
    path = Path(directory) / name
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


class DistillationDataTests(unittest.TestCase):
    def test_selected_only_records_follow_episode_advance_trace(self):
        samples = [
            make_rollout_record(
                episode=0, step=0, sample=0, action="look", marker=11
            ),
            make_rollout_record(
                episode=0, step=0, sample=1, action="open fridge 1", marker=12
            ),
            make_rollout_record(
                episode=1, step=0, sample=0, action="inventory", marker=13
            ),
        ]
        episodes = [
            {
                "episode": 0,
                "advance_trace": [
                    {"step": 0, "sample": 1, "action": "open fridge 1"}
                ],
            },
            {
                "episode": 1,
                "advance_trace": [
                    {"step": 0, "sample": 0, "action": "inventory"}
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            samples_path = write_jsonl(directory, "samples.jsonl", samples)
            episodes_path = write_jsonl(directory, "episodes.jsonl", episodes)
            selected_actions, stats = load_advance_selections(episodes_path)
            records = load_eligible_records(
                samples_path, selected_actions=selected_actions
            )

        self.assertEqual(
            [
                (record["episode"], record["step"], record["sample"])
                for record in records
            ],
            [(0, 0, 1), (1, 0, 0)],
        )
        self.assertEqual(stats["advance_trace_steps"], 2)
        self.assertEqual(stats["sampled_advance_steps"], 2)
        self.assertEqual(stats["fallback_advance_steps"], 0)

    def test_selected_only_skips_fallback_and_ineligible_selected_turns(self):
        samples = [
            make_rollout_record(
                episode=0,
                step=1,
                sample=0,
                action="open fridge 1",
                marker=11,
                eligible=False,
            ),
            make_rollout_record(
                episode=0, step=2, sample=1, action="look", marker=12
            ),
        ]
        samples[0]["parse_ok"] = False
        samples[0]["action_was_admissible"] = False
        episodes = [
            {
                "episode": 0,
                "advance_trace": [
                    {"step": 0, "sample": None, "action": "look"},
                    {"step": 1, "sample": 0, "action": "open fridge 1"},
                    {"step": 2, "sample": 1, "action": "look"},
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            samples_path = write_jsonl(directory, "samples.jsonl", samples)
            episodes_path = write_jsonl(directory, "episodes.jsonl", episodes)
            selected_actions, stats = load_advance_selections(episodes_path)
            records = load_eligible_records(
                samples_path, selected_actions=selected_actions
            )

        self.assertEqual(
            [(record["step"], record["sample"]) for record in records],
            [(2, 1)],
        )
        self.assertEqual(stats["advance_trace_steps"], 3)
        self.assertEqual(stats["sampled_advance_steps"], 2)
        self.assertEqual(stats["fallback_advance_steps"], 1)

    def test_advance_trace_rejects_multiple_selections_for_one_state(self):
        episodes = [
            {
                "episode": 0,
                "advance_trace": [
                    {"step": 0, "sample": 0, "action": "look"},
                    {"step": 0, "sample": 1, "action": "inventory"},
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            episodes_path = write_jsonl(directory, "episodes.jsonl", episodes)
            with self.assertRaisesRegex(ValueError, "more than one turn"):
                load_advance_selections(episodes_path)

    def test_selected_only_rejects_action_mismatch(self):
        sample = make_rollout_record(
            episode=0, step=0, sample=0, action="look", marker=11
        )
        with tempfile.TemporaryDirectory() as directory:
            samples_path = write_jsonl(directory, "samples.jsonl", [sample])
            with self.assertRaisesRegex(ValueError, "action does not match"):
                load_eligible_records(
                    samples_path,
                    selected_actions={(0, 0, 0): "inventory"},
                )

    def test_selected_only_rejects_missing_sample_reference(self):
        sample = make_rollout_record(
            episode=0, step=0, sample=0, action="look", marker=11
        )
        with tempfile.TemporaryDirectory() as directory:
            samples_path = write_jsonl(directory, "samples.jsonl", [sample])
            with self.assertRaisesRegex(ValueError, "missing selected samples"):
                load_eligible_records(
                    samples_path,
                    selected_actions={(0, 0, 1): "look"},
                )

    def test_selected_only_keeps_inadmissible_selected_sample(self):
        sample = make_rollout_record(
            episode=0, step=0, sample=0, action="look", marker=11
        )
        sample["action_was_admissible"] = False
        with tempfile.TemporaryDirectory() as directory:
            samples_path = write_jsonl(directory, "samples.jsonl", [sample])
            records = load_eligible_records(
                samples_path,
                selected_actions={(0, 0, 0): "look"},
            )
            self.assertEqual(len(records), 1)
            self.assertFalse(records[0]["action_was_admissible"])

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

    def test_validate_record_allows_truncated_thought(self):
        record = make_record(True, 11)
        record["head_truncated"] = True
        record["thought_truncated"] = True
        validate_record(record)

    def test_validate_record_rejects_truncated_decision(self):
        record = make_record(True, 11)
        record["head_truncated"] = True
        record["decision_truncated"] = True
        record["used_decision_repair"] = True
        with self.assertRaisesRegex(ValueError, "decision span"):
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

    def test_collection_regime_must_be_uniform(self):
        records = [make_record(True, 11), make_record(True, 12)]
        self.assertTrue(validate_prompt_regime(records)["no_oracle_filtering"])
        records[1]["policy_version"] = 2
        with self.assertRaisesRegex(ValueError, "policy_version"):
            validate_prompt_regime(records)

    def test_missing_collection_provenance_is_rejected(self):
        record = make_record(True, 11)
        del record["no_oracle_filtering"]
        validate_record(record)
        with self.assertRaisesRegex(ValueError, "no_oracle_filtering"):
            validate_prompt_regime([record])

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
