"""Check sparse HMM next-token scores against the former dense calculation."""

import unittest

import numpy as np
import torch

from ctrlg import DFAModel, HMM
from ctrlg.utils import (
    ConstraintLogitsProcessor,
    aggregate_edge_weights,
    distribute_state_weights,
    matmul_a_logb,
    matmul_log,
    matmul_loga_b,
)


def _edge(vocab_size, tokens):
    labels = np.zeros(vocab_size, dtype=bool)
    labels[list(tokens)] = True
    return labels


class SparseHMMScoringTests(unittest.TestCase):
    def _check_scores(self, graph, vocab_size, prefixes):
        torch.manual_seed(3)
        hmm = HMM(hidden_states=3, vocab_size=vocab_size, eos_token_id=0)
        dfa = DFAModel(graph, vocab_size, dense_token_mask=False)
        self.assertIsNone(dfa.T_mask)
        processor = ConstraintLogitsProcessor(
            hmm, dfa, min_new_tokens=1, max_new_tokens=2,
            prompt_ids=[], suffix_ids=[0],
        )
        dense_dfa = DFAModel(graph, vocab_size)
        edge_weights = matmul_a_logb(dense_dfa.T_mask, hmm.beta.T)
        edge_weights.nan_to_num_(neginf=-1e30)
        ending = torch.full((dense_dfa.num_states, hmm.hidden_states), -1e30)
        ending[list(dense_dfa.accept_states)] = hmm.beta[:, 0]
        future = matmul_loga_b(ending, hmm.alpha_exp.T)
        layers = [future]
        for _ in range(2):
            per_edge = distribute_state_weights(dense_dfa.E2Dst, future)
            per_state = aggregate_edge_weights(
                dense_dfa.E2Src, edge_weights + per_edge, dense_dfa.num_states
            )
            future = matmul_loga_b(per_state, hmm.alpha_exp.T)
            layers.append(future)
        live_states = {
            destination for edges in processor.live_edges.values()
            for destination, _ in edges
        }
        for high in (0, 1):
            dense_future = torch.logsumexp(torch.stack(layers[:high + 1]), dim=0)
            for state in live_states:
                torch.testing.assert_close(
                    processor.C_cache[(0, high)][state], dense_future[state],
                    atol=1e-4, rtol=1e-4,
                )
        for prefix in prefixes:
            constrained, plain = processor.compute_logits(
                [prefix], [(1, 2)], batch_size=1
            )
            forward = processor.A_cache[prefix]
            generated = len(prefix)
            future = processor.C_cache[(0, 1 - generated)]
            dense = matmul_log(forward[None, :] + future, hmm.beta)
            source = processor.D_cache[prefix]
            live_destinations = {
                destination for destination, _ in processor.live_edges[source]
            }
            for token in range(vocab_size):
                destination = next(
                    dest for dest, labels in dfa.G[source] if labels[token]
                )
                if destination in live_destinations:
                    torch.testing.assert_close(
                        constrained[0, token], dense[destination, token],
                        atol=1e-4, rtol=1e-4,
                    )
                else:
                    self.assertLess(float(constrained[0, token]), -1e29)
            torch.testing.assert_close(
                plain, matmul_log(forward[None, :], hmm.beta),
                atol=1e-4, rtol=1e-4,
            )

    def test_dead_trie_edges_are_skipped(self):
        vocab_size = 7
        graph = {
            "edges": [
                (0, 2, _edge(vocab_size, [1])),
                (0, 3, _edge(vocab_size, [3])),
                (0, 1, _edge(vocab_size, [0, 2, 4, 5, 6])),
                (2, 3, _edge(vocab_size, [2])),
                (2, 1, _edge(vocab_size, [0, 1, 3, 4, 5, 6])),
                (3, 1, _edge(vocab_size, range(vocab_size))),
                (1, 1, _edge(vocab_size, range(vocab_size))),
            ],
            "initial_state": 0,
            "accept_states": {3},
        }
        self._check_scores(graph, vocab_size, [(), (1,)])

    def test_large_live_edge_is_scored_in_chunks(self):
        vocab_size = 1100
        graph = {
            "edges": [
                (0, 1, _edge(vocab_size, range(vocab_size))),
                (1, 1, _edge(vocab_size, range(vocab_size))),
            ],
            "initial_state": 0,
            "accept_states": {1},
        }
        self._check_scores(graph, vocab_size, [()])


if __name__ == "__main__":
    unittest.main()
