import tempfile
import unittest
from pathlib import Path

from ctrlg_alfworld.skills import Action, SkillSet


ROOT = Path(__file__).resolve().parents[1]


class SkillParsingTests(unittest.TestCase):
    def test_repository_skill_has_six_schemas_and_four_policies(self):
        skills = SkillSet.from_file(ROOT / "templates" / "SKILLS.md")
        self.assertTrue(all(isinstance(action, Action) for action in skills.actions))
        self.assertEqual(len(skills.decision_schemas), 6)
        self.assertEqual(len(skills.policies), 4)
        self.assertEqual(skills.schema_version, 1)
        self.assertEqual(skills.policy_version, 1)

    def test_duplicate_schema_is_rejected(self):
        original = (ROOT / "templates" / "SKILLS.md").read_text()
        block_start = original.index("```decision-schema")
        block_end = original.index("```", block_start + 3) + 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SKILLS.md"
            path.write_text(original + "\n" + original[block_start:block_end])
            with self.assertRaisesRegex(ValueError, "duplicate decision schema"):
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


if __name__ == "__main__":
    unittest.main()
