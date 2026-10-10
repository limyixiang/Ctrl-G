import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_hmm_fit.py"
SPEC = importlib.util.spec_from_file_location("evaluate_hmm_fit", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class HMMFitFormatTests(unittest.TestCase):
    def test_legacy_data_is_rejected_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "alfworld_actions.metadata.json"
            metadata.write_text(json.dumps({
                "prompt_format": "decision_with_persistent_history_no_oracle_v1"
            }), encoding="utf-8")
            arguments = [
                str(SCRIPT), "--hmm", "checkpoint", "--data-dir", directory,
                "--out", str(Path(directory) / "fit.json"),
            ]
            with patch("sys.argv", arguments), patch.object(module.HMM, "from_pretrained") as load:
                with self.assertRaisesRegex(ValueError, "prompt_format"):
                    module.main()
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
