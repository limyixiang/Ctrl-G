import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jsonl_to_json.py"
SPEC = importlib.util.spec_from_file_location("jsonl_to_json", SCRIPT)
jsonl_to_json = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jsonl_to_json)


class JsonlToJsonTests(unittest.TestCase):
    def test_converts_to_array_and_removes_all_token_id_fields(self):
        records = [
            {
                "prompt_text": "hello",
                "prompt_token_ids": [1, 2],
                "nested": {"token_ids": [3], "keep": True},
            },
            {"action": "look", "action_token_ids": [4]},
        ]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "samples.jsonl"
            destination = Path(directory) / "samples.json"
            source.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            written, removed = jsonl_to_json.convert_jsonl(source, destination)

            self.assertEqual(written, 2)
            self.assertEqual(removed, 3)
            output_text = destination.read_text(encoding="utf-8")
            self.assertIn('\n  {\n    "prompt_text": "hello",', output_text)
            self.assertEqual(
                json.loads(output_text),
                [
                    {"prompt_text": "hello", "nested": {"keep": True}},
                    {"action": "look"},
                ],
            )

    def test_reports_invalid_json_line_and_does_not_replace_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "samples.jsonl"
            destination = Path(directory) / "samples.json"
            source.write_text('{"ok": true}\nnot json\n', encoding="utf-8")
            destination.write_text("original", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"samples\.jsonl:2: invalid JSON"):
                jsonl_to_json.convert_jsonl(source, destination)

            self.assertEqual(destination.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
