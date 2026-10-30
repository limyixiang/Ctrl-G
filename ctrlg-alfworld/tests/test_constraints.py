import re
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ctrlg_alfworld.constraints import (
    HardDFALogitsProcessor,
    PolicyCompilationError,
    action_span_pattern,
    action_grammar_valid,
    compile_policy_language,
    state_body_pattern,
    state_span_pattern,
    dfa_accepts,
    lift_character_fsm,
    parse_state,
    policy_satisfied,
    regex_fsm,
    tokenize_continuation,
)
from ctrlg_alfworld.skills import PolicyRule, SkillSet


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

    def put_state(self, **overrides):
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
        self.assertEqual(set(self.skills.state_schemas), {
            "put", "clean", "heat", "cool", "examine", "puttwo"
        })
        for schema in self.skills.state_schemas.values():
            self.assertFalse(regex_fsm(state_body_pattern(schema, self.skills)).empty())

    def test_generated_patterns_leave_literal_spaces_unescaped(self):
        patterns = [action_span_pattern(self.skills)] + [
            state_span_pattern(schema, self.skills)
            for schema in self.skills.state_schemas.values()
        ]
        for pattern in patterns:
            self.assertNotIn(r"\ ", pattern)
            self.assertFalse(regex_fsm(pattern).empty())

        state = parse_state(
            self.put_state(arrived_receptacle_state="closed"),
            self.skills.state_schemas["put"], self.skills,
        )
        required = compile_policy_language(state, self.skills)
        self.assertNotIn(r"\ ", required.pattern)

    def test_fixed_order_and_categories_are_enforced(self):
        schema = self.skills.state_schemas["put"]
        valid = self.put_state()
        self.assertEqual(parse_state(valid, schema, self.skills)["target_type"], "cup")
        reordered = valid.replace("target_type: cup, target_id: cup 1", "target_id: cup 1, target_type: cup")
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_state(reordered, schema, self.skills)
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_state(valid.replace("phase: acquire", "phase: impossible"), schema, self.skills)

    def test_transformation_state_field_is_preserved_inside_state_summary(self):
        for task, status, tool in (
            ("clean", "clean", "sinkbasin"),
            ("heat", "hot", "microwave"),
            ("cool", "cool", "fridge"),
        ):
            with self.subTest(task=task):
                text = self.put_state().replace("task: put", f"task: {task}", 1)
                text = text.replace(
                    "dest: shelf 1", f"state: {status}, tool: {tool} 1, dest: shelf 1", 1
                )
                fields = parse_state(text, self.skills.state_schemas[task], self.skills)
                self.assertEqual(fields["state"], status)
                self.assertIsNotNone(compile_policy_language(fields, self.skills))

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
        parsed = parse_state(text, self.skills.state_schemas["puttwo"], self.skills)
        self.assertEqual(parsed["targets"][0]["location"], parsed["targets"][1]["location"])
        with self.assertRaises(ValueError):
            parse_state(text.replace(", {id: cup 2, location: countertop 1, status: available}", ""), self.skills.state_schemas["puttwo"], self.skills)

    def test_search_list_rejects_more_than_64_entries(self):
        text = self.put_state().replace(
            "[countertop 1]",
            "[" + ", ".join(f"countertop {index}" for index in range(1, 66)) + "]",
            1,
        )
        with self.assertRaises(ValueError):
            parse_state(text, self.skills.state_schemas["put"], self.skills)

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

    def test_dead_beam_does_not_abort_live_dfa_beam(self):
        graph = lift_character_fsm(regex_fsm(re.escape("look</action>")), self.tokenizer)
        ids = self.tokenizer.encode("look</action>")
        processor = HardDFALogitsProcessor(graph, 1, len(ids), 0)
        # A forced first token leaves beam search with one live beam and
        # placeholder beams whose impossible tokens already have -inf scores.
        beams = torch.tensor([[ord("X"), ids[0]], [ord("X"), ord("z")]])
        masked = processor(beams, torch.zeros((2, len(self.tokenizer))))
        self.assertEqual(
            torch.isfinite(masked[0]).nonzero().flatten().tolist(), [ids[1]]
        )
        self.assertFalse(torch.isfinite(masked[1]).any())

        with self.assertRaisesRegex(RuntimeError, "no accepting continuation"):
            processor(beams[1:], torch.zeros((1, len(self.tokenizer))))

    def test_prompt_boundary_continuation_is_exact(self):
        ids = tokenize_continuation(
            self.tokenizer, "prompt<action>", "look</action>"
        )
        self.assertEqual(self.tokenizer.decode(ids), "look</action>")

    def test_closed_receptacle_requirement_preserves_all_restrictions(self):
        state = parse_state(
            self.put_state(arrived_receptacle_state="closed"),
            self.skills.state_schemas["put"], self.skills,
        )
        policy = compile_policy_language(state, self.skills)
        self.assertEqual(policy.required_action, "open countertop 1")
        self.assertIn("preserve_target_lexical_type", policy.activated)
        self.assertNotIn("preserve_target_lexical_type", policy.shadowed)
        self.assertTrue(policy_satisfied("open countertop 1", policy))
        self.assertFalse(policy_satisfied("look", policy))

    def test_non_target_held_requires_put_down(self):
        state = parse_state(
            self.put_state(held="mug 1", held_relation="non_target"),
            self.skills.state_schemas["put"], self.skills,
        )
        policy = compile_policy_language(state, self.skills)
        self.assertEqual(policy.required_action, "move mug 1 to countertop 1")

    def test_take_type_preserves_cup_mug_distinction_and_repeat_is_removed(self):
        state = parse_state(self.put_state(), self.skills.state_schemas["put"], self.skills)
        policy = compile_policy_language(state, self.skills)
        self.assertTrue(policy.fsm.accepts("take cup 2 from cabinet 1</action>"))
        self.assertFalse(policy.fsm.accepts("take mug 2 from cabinet 1</action>"))
        self.assertFalse(policy.fsm.accepts("look</action>"))
        self.assertTrue(policy.fsm.accepts("inventory</action>"))
        self.assertTrue(action_grammar_valid("take mug 2 from cabinet 1", self.skills))

    def test_unbound_restriction_fails_without_unrestricted_fallback(self):
        state = {"target_type": "cup 1", "previous_action": "none",
                    "arrived_receptacle_state": "unknown", "held_relation": "none"}
        with self.assertRaisesRegex(PolicyCompilationError, "invalid_restriction"):
            compile_policy_language(state, self.skills)

    def fields(self, **overrides):
        return parse_state(self.put_state(**overrides), self.skills.state_schemas["put"], self.skills)

    def rule(self, name, effect, when=(), priority=90):
        return PolicyRule(name, 1, priority, tuple(when), effect)

    def test_occupied_hand_blocks_take_and_self_navigation(self):
        policy = compile_policy_language(self.fields(
            held="cup 1", held_relation="unknown", dest="unknown",
        ), self.skills)
        self.assertFalse(policy_satisfied("take cup 2 from countertop 1", policy))
        self.assertFalse(policy_satisfied("go to countertop 1", policy))
        self.assertTrue(policy_satisfied("go to countertop 2", policy))
        self.assertTrue(policy_satisfied("move cup 1 to countertop 1", policy))

    def test_delivery_ignores_incorrect_phase_and_opens_before_placing(self):
        cases = (
            ({"phase": "search", "arrived_receptacle_state": "closed"}, "go to shelf 1"),
            ({"phase": "acquire", "at": "shelf 1", "arrived_receptacle_state": "closed"}, "open shelf 1"),
            ({"phase": "acquire", "at": "shelf 1"}, "move cup 1 to shelf 1"),
            ({"phase": "search", "at": "shelf 1", "arrived_receptacle_state": "unknown"}, "move cup 1 to shelf 1"),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                policy = compile_policy_language(self.fields(
                    held="cup 1", held_relation="target", **overrides,
                ), self.skills)
                self.assertEqual(policy.required_action, expected)
                self.assertIn("no_take_with_occupied_hand", policy.activated)

    def test_delivery_uses_current_skills_without_implicit_transformation_guard(self):
        state = self.fields(held="cup 1", held_relation="target", phase="acquire")
        state.update(task="heat", state="unmodified", tool="microwave 1")
        self.assertEqual(compile_policy_language(state, self.skills).required_action, "go to shelf 1")

    def test_puttwo_delivery_does_not_need_target_record_selection(self):
        state = self.fields(held="cup 1", held_relation="target")
        state.update(task="puttwo", phase="acquire_second", required_count=2,
                     targets=[{"id": "cup 1", "location": "countertop 1", "status": "held"},
                              {"id": "cup 2", "location": "countertop 1", "status": "available"}])
        self.assertEqual(compile_policy_language(state, self.skills).required_action, "go to shelf 1")

    def test_repeated_required_actions_are_rejected_without_losing_restrictions(self):
        cases = (
            ({"arrived_receptacle_state": "closed", "previous_action": "open countertop 1"},
             "open_declared_closed_receptacle", "open countertop 1"),
            ({"held": "mug 1", "held_relation": "non_target",
              "previous_action": "move mug 1 to countertop 1"},
             "put_down_declared_non_target", "move mug 1 to countertop 1"),
        )
        for overrides, name, action in cases:
            with self.subTest(policy=name):
                policy = compile_policy_language(self.fields(**overrides), self.skills)
                self.assertIsNone(policy.required_action)
                self.assertEqual(policy.rejected_required_policies[name], "forbidden_by_restrictions")
                self.assertFalse(policy_satisfied(action, policy))
                self.assertFalse(policy_satisfied("go to countertop 1", policy))
                self.assertTrue(policy_satisfied("inventory", policy))
                self.assertTrue(policy.evaluable)
                self.assertIsNone(policy.fallback_reason)

    def test_required_action_selection_uses_highest_permitted_priority(self):
        rules = (
            self.rule("lower", {"require_action": {"action": "inventory", "bindings": {}}}, priority=10),
            self.rule("middle", {"require_action": {"action": "look", "bindings": {}}}, priority=40),
            self.rule("highest", {"require_action": {"action": "goto", "bindings": {"recep": "at"}}}, priority=95),
            self.rule("restriction", {"forbid_action": {"action": "goto", "bindings": {"recep": "at"}}}, priority=0),
        )
        policy = compile_policy_language(self.fields(), replace(self.skills, policies=rules))
        self.assertEqual(policy.required_action, "look")
        self.assertEqual(policy.rejected_required_policies, {"highest": "forbidden_by_restrictions"})
        self.assertEqual(policy.shadowed, ("lower",))
        self.assertIn("restriction", policy.activated)

    def test_unbound_requirement_is_skipped_but_restrictions_remain(self):
        required = self.rule("unbound", {"require_action": {"action": "goto", "bindings": {"recep": "dest"}}})
        skills = replace(self.skills, policies=self.skills.policies + (required,))
        policy = compile_policy_language(self.fields(dest="unknown"), skills)
        self.assertIsNone(policy.required_action)
        self.assertIn("unbound_required_action", policy.rejected_required_policies["unbound"])
        self.assertFalse(policy_satisfied("look", policy))
        self.assertTrue(policy_satisfied("inventory", policy))

    def test_field_comparisons_do_not_match_missing_or_unknown_operands(self):
        for operator in ("equals_field", "not_equals_field"):
            rule = self.rule("compare", {"require_action": {"action": "inventory", "bindings": {}}},
                             [{"field": "at", operator: "dest"}])
            skills = replace(self.skills, policies=(rule,))
            for values in ({}, {"at": "unknown", "dest": "shelf 1"},
                           {"at": "shelf 1", "dest": "unknown"}, {"at": "shelf 1"}):
                with self.subTest(operator=operator, values=values):
                    self.assertIsNone(compile_policy_language(values, skills).required_action)
            for at in ("shelf 1", "shelf 2"):
                policy = compile_policy_language({"at": at, "dest": "shelf 1"}, skills)
                expected = (at == "shelf 1") == (operator == "equals_field")
                self.assertEqual(policy.required_action is not None, expected)

    def test_existing_not_in_semantics_allow_missing_transformation_field(self):
        rule = self.rule("ready", {"require_action": {"action": "inventory", "bindings": {}}},
                         [{"field": "state", "not_in": ["unmodified", "unknown"]}])
        skills = replace(self.skills, policies=(rule,))
        self.assertEqual(compile_policy_language({}, skills).required_action, "inventory")
        for status in ("unmodified", "unknown"):
            self.assertIsNone(compile_policy_language({"state": status}, skills).required_action)

    def test_partial_forbidden_action_only_removes_matching_template_arguments(self):
        rule = self.rule("ban_source", {"forbid_action": {"action": "take", "bindings": {"recep": "at"}}})
        policy = compile_policy_language(self.fields(), replace(self.skills, policies=(rule,)))
        self.assertFalse(policy_satisfied("take cup 1 from countertop 1", policy))
        self.assertFalse(policy_satisfied("take mug 2 from countertop 1", policy))
        self.assertTrue(policy_satisfied("take cup 1 from countertop 2", policy))
        self.assertTrue(policy_satisfied("move cup 1 to countertop 1", policy))

    def test_unbound_forbidden_argument_does_not_become_wildcard(self):
        rule = self.rule("ban_source", {"forbid_action": {"action": "take", "bindings": {"recep": "at"}}})
        with self.assertRaisesRegex(PolicyCompilationError, "cannot bind"):
            compile_policy_language(self.fields(at="unknown"), replace(self.skills, policies=(rule,)))

    def test_move_binding_restrictions_handle_known_unknown_and_empty_hand(self):
        rule = self.rule("move_bindings", {"restrict_action_bindings": {
            "action": "move", "bindings": {"obj": "held", "recep": "at"},
        }})
        skills = replace(self.skills, policies=(rule,))
        cases = (
            ("cup 1", "countertop 1", True, False, False),
            ("none", "countertop 1", False, False, False),
            ("unknown", "countertop 1", True, True, False),
            ("cup 1", "unknown", True, False, True),
        )
        for held, at, same, other_obj, other_recep in cases:
            with self.subTest(held=held, at=at):
                policy = compile_policy_language(self.fields(held=held, at=at), skills)
                self.assertEqual(policy_satisfied("move cup 1 to countertop 1", policy), same)
                self.assertEqual(policy_satisfied("move cup 2 to countertop 1", policy), other_obj)
                self.assertEqual(policy_satisfied("move cup 1 to countertop 2", policy), other_recep)
                self.assertTrue(policy_satisfied("look", policy))
                self.assertTrue(policy_satisfied("take cup 1 from countertop 1", policy))

    def test_multiple_restrictions_accumulate(self):
        rules = (
            self.rule("first", {"forbid_exact_action": {"field": "previous_action"}}),
            self.rule("second", {"forbid_exact_action": {"field": "other_action"}}),
            self.rule("cup_only", {"restrict_take_type": {"field": "target_type"}}),
            self.rule("put_only", {"restrict_take_type": {"field": "task"}}),
        )
        state = dict(self.fields(), other_action="inventory")
        policy = compile_policy_language(state, replace(self.skills, policies=rules))
        for action in ("look", "inventory", "take cup 1 from countertop 1", "take put 1 from countertop 1"):
            self.assertFalse(policy_satisfied(action, policy), action)
        self.assertTrue(policy_satisfied("go to countertop 2", policy))

    def test_empty_restricted_language_raises(self):
        look = next(action for action in self.skills.actions if action.name == "look")
        rule = self.rule("ban_all", {"forbid_action": {"action": "look", "bindings": {}}})
        skills = replace(self.skills, actions=(look,), policies=(rule,))
        with self.assertRaisesRegex(PolicyCompilationError, "empty_restricted_language"):
            compile_policy_language({}, skills)

    def test_cache_key_includes_effective_restrictions_and_bindings(self):
        states = (
            self.fields(),
            self.fields(held="cup 1", held_relation="unknown"),
            self.fields(at="countertop 2"),
        )
        keys = [compile_policy_language(state, self.skills).cache_key for state in states]
        self.assertEqual(len(set(keys)), 3)
        self.assertEqual(keys[0], compile_policy_language(states[0], self.skills).cache_key)
        rule = self.rule("move_bindings", {"restrict_action_bindings": {
            "action": "move", "bindings": {"obj": "held", "recep": "at"},
        }})
        skills = replace(self.skills, policies=(rule,))
        keys = [compile_policy_language(self.fields(held=held), skills).cache_key
                for held in ("cup 1", "cup 2", "none", "unknown")]
        self.assertEqual(len(set(keys)), 4)


if __name__ == "__main__":
    unittest.main()
