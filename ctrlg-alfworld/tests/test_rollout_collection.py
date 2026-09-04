import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rollouts.py"
SPEC = importlib.util.spec_from_file_location("run_rollouts", SCRIPT)
run_rollouts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_rollouts)


def make_turn(*, parse_ok, action):
    return SimpleNamespace(parsed=SimpleNamespace(parse_ok=parse_ok, action=action))


class RolloutCollectionTests(unittest.TestCase):
    def test_output_lock_allows_only_one_collector(self):
        with tempfile.TemporaryDirectory() as directory:
            first = run_rollouts.acquire_output_lock(directory)
            try:
                with self.assertRaisesRegex(RuntimeError, "another collector"):
                    run_rollouts.acquire_output_lock(directory)
            finally:
                run_rollouts._release_output_lock(first)

            second = run_rollouts.acquire_output_lock(directory)
            run_rollouts._release_output_lock(second)

    def test_output_paths_refuse_existing_artifacts_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            samples, episodes, metadata = run_rollouts.prepare_output_paths(
                directory, overwrite=False
            )
            self.assertEqual(samples.name, "samples.jsonl")
            self.assertEqual(episodes.name, "episodes.jsonl")
            self.assertEqual(metadata.name, "metadata.json")
            samples.write_text("existing sample\n")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_rollouts.prepare_output_paths(directory, overwrite=False)
            self.assertEqual(
                run_rollouts.prepare_output_paths(directory, overwrite=True),
                (samples, episodes, metadata),
            )

    def test_resume_requires_all_collection_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "artifacts are missing"):
                run_rollouts.prepare_output_paths(
                    directory, overwrite=False, resume=True
                )

            paths = run_rollouts.prepare_output_paths(directory, overwrite=False)
            for path in paths:
                path.touch()
            self.assertEqual(
                run_rollouts.prepare_output_paths(
                    directory, overwrite=False, resume=True
                ),
                paths,
            )
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                run_rollouts.prepare_output_paths(
                    directory, overwrite=True, resume=True
                )

    def test_resume_repairs_uncommitted_trailing_episode(self):
        episodes = [
            {
                "episode": 0,
                "num_steps": 1,
                "advance_sources": ["admissible_raw_model_sample"],
            },
            {
                "episode": 1,
                "num_steps": 1,
                "advance_sources": ["deterministic_admissible_fallback"],
            },
        ]
        committed_samples = [
            {
                "episode": 0,
                "step": 0,
                "sample": sample,
                "distill_eligible": sample == 0,
                "distill_exclusion_reasons": [] if sample == 0 else ["parse_failure"],
            }
            for sample in range(2)
        ]
        partial_sample = {
            "episode": 1,
            "step": 0,
            "sample": 0,
            "distill_eligible": True,
            "distill_exclusion_reasons": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            samples_path = directory / "samples.jsonl"
            episodes_path = directory / "episodes.jsonl"
            samples_path.write_bytes(
                b"".join(
                    json.dumps(record).encode() + b"\n"
                    for record in committed_samples + [partial_sample]
                )
                + b'{"episode": 1'
            )
            episodes_path.write_text(
                "".join(json.dumps(record) + "\n" for record in episodes),
                encoding="utf-8",
            )

            state = run_rollouts.recover_resume_state(
                samples_path,
                episodes_path,
                samples_per_state=2,
                num_episodes=2,
            )

            self.assertEqual(state[0:3], (1, 2, 1))
            self.assertEqual(
                state[3], {"admissible_raw_model_sample": 1}
            )
            self.assertEqual(state[4]["decision"]["exclusions"], {"parse_failure": 1})
            self.assertEqual(
                samples_path.read_text(encoding="utf-8"),
                "".join(json.dumps(record) + "\n" for record in committed_samples),
            )
            self.assertEqual(
                episodes_path.read_text(encoding="utf-8"),
                json.dumps(episodes[0]) + "\n",
            )

    def test_resume_rejects_interior_sample_corruption(self):
        episodes = [
            {"episode": episode, "num_steps": 1, "advance_sources": []}
            for episode in range(2)
        ]
        samples = [
            {
                "episode": 99,
                "step": 0,
                "sample": 0,
                "distill_eligible": True,
                "distill_exclusion_reasons": [],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            samples_path = directory / "samples.jsonl"
            episodes_path = directory / "episodes.jsonl"
            samples_path.write_text(
                "".join(json.dumps(record) + "\n" for record in samples),
                encoding="utf-8",
            )
            episodes_path.write_text(
                "".join(json.dumps(record) + "\n" for record in episodes),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "before the final episode"):
                run_rollouts.recover_resume_state(
                    samples_path,
                    episodes_path,
                    samples_per_state=1,
                    num_episodes=2,
                )

    def test_resume_metadata_rejects_changed_collection_setting(self):
        expected = {field: None for field in run_rollouts.RESUME_COMPATIBILITY_FIELDS}
        existing = dict(expected)
        existing["temperature"] = 0.7
        expected["temperature"] = 1.0
        with self.assertRaisesRegex(ValueError, "temperature"):
            run_rollouts.validate_resume_metadata(existing, expected)

    def test_selection_skips_malformed_admissible_turn(self):
        malformed = make_turn(parse_ok=False, action="look")
        well_formed = make_turn(parse_ok=True, action="look")
        self.assertEqual(
            run_rollouts.select_advance_turn(
                [(0, malformed), (1, well_formed)], ["look"]
            ),
            (1, well_formed),
        )

    def test_selection_returns_none_without_well_formed_admissible_turn(self):
        self.assertIsNone(
            run_rollouts.select_advance_turn(
                [
                    (0, make_turn(parse_ok=False, action="look")),
                    (1, make_turn(parse_ok=True, action="open fridge 1")),
                ],
                ["look"],
            )
        )


if __name__ == "__main__":
    unittest.main()
