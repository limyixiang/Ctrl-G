import os
import unittest
from pathlib import Path

from ctrlg_alfworld.constraints import dfa_accepts, lift_character_fsm, regex_fsm


MODEL = Path(__file__).resolve().parents[2] / "models" / "Qwen3-0.6B-smoke"


@unittest.skipUnless(
    os.environ.get("CTRLG_RUN_QWEN_SMOKE") == "1" and MODEL.exists(),
    "set CTRLG_RUN_QWEN_SMOKE=1 when local Qwen tokenizer assets are available",
)
class QwenTokenizerSmokeTests(unittest.TestCase):
    def test_qwen_tokenizer_lift_accepts_closing_tag(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        graph = lift_character_fsm(regex_fsm(r"look</action>"), tokenizer)
        token_ids = tokenizer.encode("look</action>", add_special_tokens=False)
        self.assertTrue(dfa_accepts(graph, token_ids))
        decoded = tokenizer.decode(
            token_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self.assertEqual(decoded, "look</action>")


if __name__ == "__main__":
    unittest.main()
