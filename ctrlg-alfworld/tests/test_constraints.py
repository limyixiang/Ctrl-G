import re
import unittest
from pathlib import Path

import torch

from ctrlg_alfworld.constraints import (
    HardDFALogitsProcessor,
    action_span_pattern,
    action_grammar_valid,
    compile_policy_language,
    decision_body_pattern,
    decision_span_pattern,
    dfa_accepts,
    lift_character_fsm,
    parse_decision,
    policy_satisfied,
    regex_fsm,
    tokenize_continuation,
)
from ctrlg_alfworld.skills import SkillSet


SKILLS = Path(__file__).resolve().parents[1] / "templates" / "SKILLS.md"


class CharacterTokenizer:
    name_or_path = "character-test"
    all_special_ids = [0]
    eos_token_id = 0

    def __len__(self):
        return 128

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return "".join(chr(item) for item in token_ids if item != 0)


class ConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skills = SkillSet.from_file(SKILLS)
        cls.tokenizer = CharacterTokenizer()

    def put_decision(self, **overrides):
        values = {
            "task": "put", "target_type": "cup", "target_id": "cup 1",
            "target_location": "countertop 1", "held": "none",
            "at": "countertop 1", "dest": "shelf 1",
            "previous_action": "look", "arrived_receptacle_state": "open",
            "held_relation": "none",
            "search": {"locations_searched": ["countertop 1"],
                       "locations_to_search": ["cabinet 1"]},
            "phase": "acquire",
        }
        values.update(overrides)
        return (
            "{task: %(task)s, target_type: %(target_type)s, target_id: %(target_id)s, "
            "target_location: %(target_location)s, held: %(held)s, at: %(at)s, "
            "dest: %(dest)s, previous_action: %(previous_action)s, "
            "arrived_receptacle_state: %(arrived_receptacle_state)s, "
            "held_relation: %(held_relation)s, search: {locations_searched: "
            "[countertop 1], locations_to_search: [cabinet 1]}, phase: %(phase)s}"
        ) % values

    def test_all_six_schemas_compile(self):
        self.assertEqual(set(self.skills.decision_schemas), {
            "put", "clean", "heat", "cool", "examine", "puttwo"
        })
        for schema in self.skills.decision_schemas.values():
            self.assertFalse(regex_fsm(decision_body_pattern(schema, self.skills)).empty())

    def test_generated_patterns_leave_literal_spaces_unescaped(self):
        patterns = [action_span_pattern(self.skills)] + [
            decision_span_pattern(schema, self.skills)
            for schema in self.skills.decision_schemas.values()
        ]
        for pattern in patterns:
            self.assertNotIn(r"\ ", pattern)
            self.assertFalse(regex_fsm(pattern).empty())

        decision = parse_decision(
            self.put_decision(arrived_receptacle_state="closed"),
            self.skills.decision_schemas["put"], self.skills,
        )
        required = compile_policy_language(decision, self.skills)
        self.assertNotIn(r"\ ", required.pattern)

    def test_fixed_order_and_categories_are_enforced(self):
        schema = self.skills.decision_schemas["put"]
        valid = self.put_decision()
        self.assertEqual(parse_decision(valid, schema, self.skills)["target_type"], "cup")
        reordered = valid.replace("target_type: cup, target_id: cup 1", "target_id: cup 1, target_type: cup")
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_decision(reordered, schema, self.skills)
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_decision(valid.replace("phase: acquire", "phase: impossible"), schema, self.skills)

    def test_puttwo_allows_two_targets_at_same_location(self):
        text = (
            "{task: puttwo, target_type: cup, required_count: 2, targets: "
            "[{id: cup 1, location: countertop 1, status: available}, "
            "{id: cup 2, location: countertop 1, status: available}], held: none, "
            "at: countertop 1, dest: shelf 1, previous_action: look, "
            "arrived_receptacle_state: open, held_relation: none, search: "
            "{locations_searched: [countertop 1], locations_to_search: []}, "
            "phase: acquire}"
        )
        parsed = parse_decision(text, self.skills.decision_schemas["puttwo"], self.skills)
        self.assertEqual(parsed["targets"][0]["location"], parsed["targets"][1]["location"])
        with self.assertRaises(ValueError):
            parse_decision(text.replace(", {id: cup 2, location: countertop 1, status: available}", ""), self.skills.decision_schemas["puttwo"], self.skills)

    def test_search_list_rejects_more_than_64_entries(self):
        text = self.put_decision().replace(
            "[countertop 1]",
            "[" + ", ".join(f"countertop {index}" for index in range(1, 66)) + "]",
            1,
        )
        with self.assertRaises(ValueError):
            parse_decision(text, self.skills.decision_schemas["put"], self.skills)

    def test_token_dfa_includes_closer_and_budget_reachability(self):
        fsm = regex_fsm(re.escape("look</action>"))
        graph = lift_character_fsm(fsm, self.tokenizer)
        ids = self.tokenizer.encode("look</action>")
        self.assertTrue(dfa_accepts(graph, ids))
        self.assertFalse(dfa_accepts(graph, self.tokenizer.encode("look")))
        processor = HardDFALogitsProcessor(graph, 1, len(ids), 0)
        masked = processor(torch.tensor([[ord("X")]]), torch.zeros((1, 128)))
        self.assertEqual(torch.isfinite(masked[0]).nonzero().flatten().tolist(), [ord("l")])
        with self.assertRaisesRegex(ValueError, "within the token budget"):
            HardDFALogitsProcessor(graph, 1, len(ids) - 1, 0)

    def test_prompt_boundary_continuation_is_exact(self):
        ids = tokenize_continuation(
            self.tokenizer, "prompt<action>", "look</action>"
        )
        self.assertEqual(self.tokenizer.decode(ids), "look</action>")

    def test_closed_receptacle_required_action_shadows_lower_rules(self):
        decision = parse_decision(
            self.put_decision(arrived_receptacle_state="closed"),
            self.skills.decision_schemas["put"], self.skills,
        )
        policy = compile_policy_language(decision, self.skills)
        self.assertEqual(policy.required_action, "open countertop 1")
        self.assertIn("preserve_target_lexical_type", policy.shadowed)
        self.assertTrue(policy_satisfied("open countertop 1", policy))
        self.assertFalse(policy_satisfied("look", policy))

    def test_non_target_held_requires_put_down(self):
        decision = parse_decision(
            self.put_decision(held="mug 1", held_relation="non_target"),
            self.skills.decision_schemas["put"], self.skills,
        )
        policy = compile_policy_language(decision, self.skills)
        self.assertEqual(policy.required_action, "move mug 1 to countertop 1")

    def test_take_type_preserves_cup_mug_distinction_and_repeat_is_removed(self):
        decision = parse_decision(self.put_decision(), self.skills.decision_schemas["put"], self.skills)
        policy = compile_policy_language(decision, self.skills)
        self.assertTrue(policy.fsm.accepts("take cup 2 from cabinet 1</action>"))
        self.assertFalse(policy.fsm.accepts("take mug 2 from cabinet 1</action>"))
        self.assertFalse(policy.fsm.accepts("look</action>"))
        self.assertTrue(policy.fsm.accepts("inventory</action>"))
        self.assertTrue(action_grammar_valid("take mug 2 from cabinet 1", self.skills))

    def test_unbound_restriction_falls_back_to_generic_grammar(self):
        decision = {"target_type": "cup 1", "previous_action": "none",
                    "arrived_receptacle_state": "unknown", "held_relation": "none"}
        policy = compile_policy_language(decision, self.skills)
        self.assertFalse(policy.evaluable)
        self.assertIn("unsatisfiable_policy_language", policy.fallback_reason)
        self.assertTrue(policy.fsm.accepts("look</action>"))


if __name__ == "__main__":
    unittest.main()
