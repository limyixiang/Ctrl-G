import tempfile
import unittest
from pathlib import Path

from ctrlg_alfworld.provenance import source_tree_sha256


class ProvenanceTests(unittest.TestCase):
    def test_source_hash_is_stable_and_changes_with_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code.py").write_text("x = 1\n")
            first = source_tree_sha256(root)
            self.assertEqual(first, source_tree_sha256(root))
            (root / "code.py").write_text("x = 2\n")
            self.assertNotEqual(first, source_tree_sha256(root))

    def test_source_hash_ignores_outputs_and_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code.py").write_text("x = 1\n")
            first = source_tree_sha256(root)
            (root / "out").mkdir()
            (root / "out" / "generated.py").write_text("x = 2\n")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "code.py").write_text("x = 3\n")
            self.assertEqual(first, source_tree_sha256(root))


if __name__ == "__main__":
    unittest.main()
