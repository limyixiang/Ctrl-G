import tempfile
import unittest
from pathlib import Path

from ctrlg_alfworld.checkpoint_selection import (
    read_latest_log_likelihoods,
    select_best_checkpoint,
)


class CheckpointSelectionTests(unittest.TestCase):
    def make_checkpoint(self, root, step):
        path = root / f"checkpoint-{step}"
        path.mkdir()
        return path

    def test_selects_highest_dev_likelihood_among_saved_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (0, 8, 16, 24, 32):
                self.make_checkpoint(root, step)
            log = root / "train.log"
            log.write_text(
                "{'model_path': 'model', 'em_schedule': 'schedule'}\n"
                "0\t-1.0\t-140.0\n"
                "1\t-130.0\t-130.0\n"  # logged, but not saved
                "8\t-110.0\t-115.0\n"
                "16\t-105.0\t-101.0\n"
                "24\t-100.0\t-103.0\n"
                "32\t-98.0\t-106.0\n",
                encoding="utf-8",
            )

            selection = select_best_checkpoint(log, root)

            self.assertEqual(selection["best_checkpoint"], 16)
            self.assertEqual(
                [item["checkpoint"] for item in selection["candidates"]],
                [0, 8, 16, 24, 32],
            )

    def test_uses_only_most_recent_appended_training_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "train.log"
            log.write_text(
                "{'model_path': 'old', 'em_schedule': 'old'}\n"
                "8\t-1.0\t-1.0\n"
                "{'model_path': 'new', 'em_schedule': 'new'}\n"
                "8\t-20.0\t-30.0\n",
                encoding="utf-8",
            )

            metrics = read_latest_log_likelihoods(log)

            self.assertEqual(metrics[8]["dev_log_likelihood"], -30.0)

    def test_exact_tie_prefers_later_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root, 8)
            self.make_checkpoint(root, 16)
            log = root / "train.log"
            log.write_text(
                "8\t-10.0\t-20.0\n16\t-9.0\t-20.0\n",
                encoding="utf-8",
            )

            selection = select_best_checkpoint(log, root)

            self.assertEqual(selection["best_checkpoint"], 16)

    def test_score_definitions_are_scoped_to_latest_run(self):
        conditional = "mean_output_conditional_log_likelihood_per_attempt"
        joint = "mean_joint_log_likelihood_per_sequence"
        for latest_marker, expected_metric in (
            ("", "mean_dev_log_likelihood_per_sequence"),
            (f"# dev_score={joint}\n", "mean_joint_dev_log_likelihood_per_sequence"),
            (f"# dev_score={conditional}\n", "mean_output_conditional_dev_log_likelihood_per_attempt"),
        ):
            with self.subTest(latest_marker=latest_marker), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_checkpoint(root, 8)
                self.make_checkpoint(root, 16)
                log = root / "train.log"
                log.write_text(
                    "{'model_path': 'old', 'em_schedule': 'old'}\n"
                    f"# train_score={conditional}\n# dev_score={conditional}\n"
                    "8\t-1\t-1\n"
                    "{'model_path': 'new', 'em_schedule': 'new'}\n"
                    f"{latest_marker}"
                    "8\t-10\t-30\n16\t-20\t-25\n",
                    encoding="utf-8",
                )
                selection = select_best_checkpoint(log, root)
                self.assertEqual(selection['best_checkpoint'], 16)  # dev beats train
                self.assertEqual(selection['selection_metric'], expected_metric)
                # Earlier train markers cannot label a legacy/mixed-metric run.
                self.assertEqual(selection['score_definitions']['train'], 'unspecified_legacy_log_likelihood')

    def test_rejects_log_without_a_saved_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "train.log"
            log.write_text("8\t-10.0\t-20.0\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no saved checkpoints"):
                select_best_checkpoint(log, root)


if __name__ == "__main__":
    unittest.main()
