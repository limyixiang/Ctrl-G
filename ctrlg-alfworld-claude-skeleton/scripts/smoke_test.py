"""CPU-only smoke test: no GPU, no AlfWorld install, no base model needed.

Checks:
  1. SKILLS.md parses into skills with preconditions
  2. state tracking on a scripted mini-episode
  3. grounding produces a sensible allowed-action set
  4. token-trie DFA accepts exactly the allowed actions (Qwen tokenizer)
  5. ReAct few-shot conversion to <tool> format

Run:  python scripts/smoke_test.py [--tokenizer Qwen/Qwen3-0.6B]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld import (  # noqa: E402
    AlfWorldState, SkillSet, build_step_dfa, dfa_accepts,
    convert_react_example, load_few_shot,
)

ROOT = Path(__file__).resolve().parents[1]


class MockTokenizer:
    """Whitespace-word tokenizer with a greedy-merge quirk at boundaries, for
    offline testing of the trie/boundary logic. Use a real tokenizer when the
    HF Hub is reachable (default: Qwen/Qwen3-0.6B)."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab))

    def encode(self, text: str, add_special_tokens: bool = False):
        return [self._id(p) for p in text.replace(">", "> ").replace("<", " <").split()]

    def get_vocab(self):
        return dict(self.vocab)

    def __len__(self):
        return max(len(self.vocab), 1)


INIT_OBS = (
    "You are in the middle of a room. Looking quickly around you, you see a "
    "cabinet 2, a cabinet 1, a countertop 1, a fridge 1, a garbagecan 1, a "
    "microwave 1, a sinkbasin 1, and a toaster 1."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    # 1. parse SKILLS.md ----------------------------------------------------
    skillset = SkillSet.from_file(ROOT / "SKILLS.md")
    names = [s.name for s in skillset.skills]
    print(f"[1] parsed {len(names)} skills: {names}")
    assert "take" in names and "goto" in names

    # 2. scripted state tracking -------------------------------------------
    st = AlfWorldState()
    st.reset(INIT_OBS)
    assert "countertop 1" in st.receptacles and len(st.receptacles) == 8
    st.update("go to countertop 1",
              "On the countertop 1, you see a apple 1, a bread 1, and a mug 2.")
    assert st.location == "countertop 1"
    assert st.contents["countertop 1"] == ["apple 1", "bread 1", "mug 2"]
    st.update("take apple 1 from countertop 1",
              "You pick up the apple 1 from the countertop 1.")
    assert st.holding == "apple 1"
    print(f"[2] tracker ok: at={st.location}, holding={st.holding}, "
          f"visible={st.visible_objects()}")

    # 3. grounding ----------------------------------------------------------
    allowed = skillset.ground_all(st.domains(), st.check)
    print(f"[3] {len(allowed)} allowed actions, e.g. {allowed[:6]}")
    assert "go to fridge 1" in allowed
    assert "put apple 1 in/on countertop 1" in allowed
    assert "go to countertop 1" not in allowed          # not_at fails
    assert "take apple 1 from countertop 1" not in allowed  # hand not empty
    assert "open countertop 1" not in allowed           # not openable
    assert "heat apple 1 with countertop 1" not in allowed  # wrong recep type
    assert not any(a.startswith("clean") or a.startswith("heat") or
                   a.startswith("cool") for a in allowed)  # not at sink/microwave/fridge

    # 4. token-trie DFA -----------------------------------------------------
    if args.tokenizer == "mock":
        tok = MockTokenizer()
        # pre-populate vocab from all strings we will tokenize
        for a in allowed + ["<tool>", "</tool>", "take apple 1 from countertop 1",
                            "go to countertop 1", "open fridge", "slice apple 1"]:
            tok.encode("<tool>" + a)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = max(len(tok), max(tok.get_vocab().values()) + 1)
    dfa_graph, seqs = build_step_dfa(allowed, tok, vocab_size, prompt_tail="<tool>")
    n_states = len({u for u, _, _ in dfa_graph["edges"]} |
                   {v for _, v, _ in dfa_graph["edges"]})
    print(f"[4] DFA: {n_states} states, {len(dfa_graph['edges'])} edges "
          f"for {len(allowed)} actions")
    for action, seq in seqs.items():
        assert dfa_accepts(dfa_graph, seq), f"should accept: {action}"
    for bad in ["take apple 1 from countertop 1", "go to countertop 1",
                "open fridge", "slice apple 1"]:
        bad_seq = tok.encode(bad, add_special_tokens=False)
        assert not dfa_accepts(dfa_graph, bad_seq), f"should reject: {bad}"
    # DFAModel compilation (CPU torch)
    try:
        import ctrlg
        dfa_model = ctrlg.DFAModel(dfa_graph, vocab_size)
        print(f"[4b] ctrlg.DFAModel compiled: {dfa_model.num_states} states")
    except ImportError:
        print("[4b] ctrlg not installed - skipped DFAModel compilation "
              "(pip install -e path/to/Ctrl-G)")

    # 5. ReAct example conversion ------------------------------------------
    examples = load_few_shot(ROOT / "few_shot/alfworld_3prompts.json", "put")
    assert "<tool>" in examples[0] and "<think>" in examples[0]
    print("[5] ReAct conversion ok; example head:\n---")
    print(examples[0][:400])
    print("---\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
