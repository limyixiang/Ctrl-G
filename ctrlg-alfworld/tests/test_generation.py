import unittest

from ctrlg_alfworld.generation import (
    exact_action_tail_token_ids,
    parse_turn,
    token_boundary_for_char_offset,
)


class GenerationParsingTests(unittest.TestCase):
    def test_parse_action_only_turn_with_template_owned_think_open(self):
        parsed = parse_turn(
            "Find the target first.</think><action>",
            "go to countertop 1</action>",
            use_state=False,
        )
        self.assertTrue(parsed.parse_ok)
        self.assertEqual(parsed.thought, "Find the target first.")
        self.assertEqual(parsed.state, "")
        self.assertEqual(parsed.hmm_prefix_text, "<action>")
        self.assertEqual(parsed.action, "go to countertop 1")

    def test_parse_state_turn_and_hmm_prefix(self):
        parsed = parse_turn(
            (
                "<think>Inspect an unseen surface.</think>\n"
                "<state>Continue systematic search.</state>\n<action>"
            ),
            "go to shelf 2</action>",
            use_state=True,
        )
        self.assertTrue(parsed.parse_ok)
        self.assertEqual(parsed.state, "Continue systematic search.")
        self.assertEqual(
            parsed.hmm_prefix_text,
            "\n<state>Continue systematic search.</state>\n<action>",
        )
        self.assertEqual(parsed.action, "go to shelf 2")

    def test_tags_inside_native_thought_do_not_override_emitted_blocks(self):
        parsed = parse_turn(
            (
                "Consider <state>wait</state> and "
                "<action>inventory</action>.</think>"
                "<state>Continue searching.</state><action>"
            ),
            "look</action>",
            use_state=True,
        )
        self.assertTrue(parsed.parse_ok)
        self.assertEqual(parsed.state, "Continue searching.")
        self.assertEqual(parsed.action, "look")

    def test_legacy_and_mismatched_state_tags_are_rejected(self):
        for block in (
            "<decision>{task: put}</decision>",
            "<state>{task: put}</decision>",
            "<decision>{task: put}</state>",
            "<state> </state>",
        ):
            with self.subTest(block=block):
                parsed = parse_turn(
                    "reason</think>" + block + "<action>",
                    "look</action>",
                    use_state=True,
                )
                self.assertFalse(parsed.parse_ok)
                self.assertFalse(parsed.state_format_ok)
                self.assertIn("missing_or_empty_state", parsed.errors)

    def test_unclosed_action_tag_inside_native_thought_is_ignored(self):
        parsed = parse_turn(
            (
                "Do not emit <action>inventory yet.</think>"
                "<state>Continue searching.</state><action>"
            ),
            "look</action>",
            use_state=True,
        )
        self.assertTrue(parsed.parse_ok)
        self.assertEqual(parsed.state, "Continue searching.")
        self.assertEqual(parsed.action, "look")

    def test_unexpected_text_is_measured_not_silently_discarded(self):
        parsed = parse_turn(
            "reason</think>Maybe this one.<action>",
            "look</action>",
            use_state=False,
        )
        self.assertFalse(parsed.parse_ok)
        self.assertIn("unexpected_pre_action_text", parsed.errors)

    def test_missing_closer_is_a_parse_failure(self):
        parsed = parse_turn(
            "reason</think><action>", "look", use_state=False
        )
        self.assertFalse(parsed.parse_ok)
        self.assertIn("missing_action_close", parsed.errors)

    def test_post_action_text_is_a_parse_failure(self):
        parsed = parse_turn(
            "reason</think><action>", "look</action>,", use_state=False
        )
        self.assertFalse(parsed.parse_ok)
        self.assertIn("unexpected_post_action_text", parsed.errors)


class CharacterTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(item) for item in token_ids)


class TokenBoundaryTests(unittest.TestCase):
    def test_token_boundary_requires_exact_alignment(self):
        tokenizer = CharacterTokenizer()
        ids = [ord(char) for char in "abc"]
        self.assertEqual(token_boundary_for_char_offset(tokenizer, ids, 2), 2)

    def test_exact_action_tail_splits_body_and_closer(self):
        text = "look</action>"
        ids = [ord(character) for character in text]
        body, closer = exact_action_tail_token_ids(
            CharacterTokenizer(), ids, text
        )
        self.assertEqual("".join(chr(item) for item in body), "look")
        self.assertEqual("".join(chr(item) for item in closer), "</action>")


class TwoCharacterTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "".join({1: "ab", 2: "cd"}[item] for item in token_ids)


class MultiCharacterTokenBoundaryTests(unittest.TestCase):
    def test_token_boundary_rejects_inside_bpe_token(self):
        with self.assertRaisesRegex(ValueError, "not an exact token boundary"):
            token_boundary_for_char_offset(TwoCharacterTokenizer(), [1, 2], 1)

    def test_exact_action_tail_rejects_trailing_text_in_same_token(self):
        class TailTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                return "".join({1: "look", 2: "</action>,"}[item] for item in token_ids)

        with self.assertRaisesRegex(ValueError, "end exactly"):
            exact_action_tail_token_ids(
                TailTokenizer(), [1, 2], "look</action>,"
            )
