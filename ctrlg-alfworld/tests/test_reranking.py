import unittest
from types import SimpleNamespace

import torch

from ctrlg import rank_generated_ids


class TinyCausalModel:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, *, use_cache):
        self.calls.append((tuple(input_ids.shape), use_cache))
        # The preferred next token is the previous token plus one.
        vocabulary = torch.arange(12)
        preferred = (input_ids + 1) % len(vocabulary)
        logits = torch.where(
            vocabulary == preferred.unsqueeze(-1), 2.0, 0.0
        )
        return SimpleNamespace(logits=logits)


class RerankingTests(unittest.TestCase):
    def test_scores_candidates_individually_and_keeps_length_normalized_order(self):
        model = TinyCausalModel()
        candidates = [(5,), (3,), (3, 8)]

        ranked = rank_generated_ids(model, candidates, [1, 2], [9])

        self.assertEqual(ranked, [(3, 8), (3,), (5,)])
        self.assertEqual(
            model.calls,
            [((1, 4), False), ((1, 4), False), ((1, 5), False)],
        )

    def test_suffix_only_ranking(self):
        model = TinyCausalModel()

        ranked = rank_generated_ids(
            model, [(3,), (3, 8)], [1, 2], [9],
            suffix_logits_only=True,
        )

        self.assertEqual(ranked, [(3, 8), (3,)])


if __name__ == "__main__":
    unittest.main()
