import tempfile
import unittest
from pathlib import Path

from ctrlg_alfworld.skills import Action, SkillSet


ROOT = Path(__file__).resolve().parents[1]


class SkillParsingTests(unittest.TestCase):
    def test_repository_skill_has_six_schemas_and_eight_policies(self):
        skills = SkillSet.from_file(ROOT / "templates" / "SKILLS.md")
        self.assertTrue(all(isinstance(action, Action) for action in skills.actions))
        self.assertEqual(len(skills.state_schemas), 6)
        self.assertEqual(len(skills.policies), 8)
        self.assertEqual(skills.schema_version, 1)
        self.assertEqual(skills.policy_version, 1)

    def test_duplicate_schema_is_rejected(self):
        original = (ROOT / "templates" / "SKILLS.md").read_text()
        block_start = original.index("```state-schema")
        block_end = original.index("```", block_start + 3) + 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SKILLS.md"
            path.write_text(original + "\n" + original[block_start:block_end])
            with self.assertRaisesRegex(ValueError, "duplicate state schema"):
                SkillSet.from_file(path)

    def test_legacy_schema_blocks_are_rejected(self):
        original = (ROOT / "templates" / "SKILLS.md").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SKILLS.md"
            path.write_text(original.replace("```state-schema", "```decision-schema"))
            with self.assertRaisesRegex(ValueError, "No ```state-schema blocks"):
                SkillSet.from_file(path)

    def test_unknown_policy_action_template_is_rejected(self):
        original = (ROOT / "templates" / "SKILLS.md").read_text()
        broken = original.replace(
            "require_action: {action: open, bindings: {recep: at}}",
            "require_action: {action: teleport, bindings: {recep: at}}",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SKILLS.md"
            path.write_text(broken)
            with self.assertRaisesRegex(ValueError, "unknown action template"):
                SkillSet.from_file(path)

    def test_legacy_policy_skill_key_is_rejected(self):
        original = (ROOT / "templates" / "SKILLS.md").read_text()
        broken = original.replace(
            "require_action: {action: open, bindings: {recep: at}}",
            "require_action: {skill: open, bindings: {recep: at}}",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SKILLS.md"
            path.write_text(broken)
            with self.assertRaisesRegex(ValueError, "requires action and bindings"):
                SkillSet.from_file(path)

    def test_unsupported_operator_is_rejected(self):
        original = (ROOT / "templates" / "SKILLS.md").read_text()
        broken = original.replace("equals: closed", "contains: closed", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SKILLS.md"
            path.write_text(broken)
            with self.assertRaisesRegex(ValueError, "unsupported operator"):
                SkillSet.from_file(path)

    def test_policy_priorities_must_be_in_range(self):
        original = (ROOT / "templates" / "SKILLS.md").read_text()
        for value in ("-1", "101", "true"):
            with self.subTest(priority=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "SKILLS.md"
                path.write_text(original.replace("priority: 100", f"priority: {value}", 1))
                with self.assertRaisesRegex(ValueError, "between 0 and 100"):
                    SkillSet.from_file(path)

    def test_policy_field_references_are_validated(self):
        original = (ROOT / "templates" / "SKILLS.md").read_text()
        changes = (
            ("field: at, equals_field: dest", "field: missing, equals_field: dest"),
            ("equals_field: dest", "equals_field: missing"),
            ("not_equals_field: dest", "not_equals_field: [dest]"),
            ("recep: dest", "recep: missing"),
            ("restrict_take_type: {field: target_type}", "restrict_take_type: {field: missing}"),
        )
        for before, after in changes:
            with self.subTest(change=after), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "SKILLS.md"
                path.write_text(original.replace(before, after, 1))
                with self.assertRaisesRegex(ValueError, "unknown state field"):
                    SkillSet.from_file(path)

    def test_action_binding_shapes_are_validated(self):
        original = (ROOT / "templates" / "SKILLS.md").read_text()
        changes = (
            ("require_action: {action: open, bindings: {recep: at}}",
             "require_action: {action: open, bindings: {}}", "every action placeholder"),
            ("forbid_action: {action: take, bindings: {}}",
             "forbid_action: {action: take, bindings: {wrong: at}}", "unknown placeholder"),
            ("forbid_action: {action: take, bindings: {}}",
             "forbid_action: {action: teleport, bindings: {}}", "unknown action template"),
        )
        for before, after, error in changes:
            with self.subTest(change=after), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "SKILLS.md"
                path.write_text(original.replace(before, after, 1))
                with self.assertRaisesRegex(ValueError, error):
                    SkillSet.from_file(path)

    def test_optional_binding_effect_loads_without_changing_repository_policies(self):
        original = (ROOT / "templates" / "SKILLS.md").read_text()
        extra = """
```policy
version: 1
name: test_move_bindings
priority: 90
when: []
effect:
  restrict_action_bindings: {action: move, bindings: {obj: held, recep: at}}
```
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SKILLS.md"
            path.write_text(original + extra)
            skills = SkillSet.from_file(path)
            self.assertEqual(len(skills.policies), 9)


if __name__ == "__main__":
    unittest.main()
