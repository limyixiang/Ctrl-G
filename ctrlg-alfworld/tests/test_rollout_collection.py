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


def make_turn(
    *,
    parse_ok,
    action,
    used_head_repair=False,
    used_thought_repair=False,
    used_decision_repair=False,
    thought_truncated=False,
    decision_truncated=False,
    tail_truncated=False,
    action_close_found=True,
    tail_span_exact=True,
    hmm_prefix_token_ids=(10, 11),
):
    return SimpleNamespace(
        parsed=SimpleNamespace(
            parse_ok=parse_ok,
            action=action,
            action_close_found=action_close_found,
        ),
        used_head_repair=used_head_repair,
        used_thought_repair=used_thought_repair,
        used_decision_repair=used_decision_repair,
        thought_truncated=thought_truncated,
        decision_truncated=decision_truncated,
        tail_truncated=tail_truncated,
        tail_span_exact=tail_span_exact,
        hmm_prefix_token_ids=hmm_prefix_token_ids,
    )


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
            samples, episodes, metadata, history = run_rollouts.prepare_output_paths(
                directory, overwrite=False
            )
            self.assertEqual(samples.name, "samples.jsonl")
            self.assertEqual(episodes.name, "episodes.jsonl")
            self.assertEqual(metadata.name, "metadata.json")
            self.assertEqual(history.name, "history.jsonl")
            samples.write_text("existing sample\n")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_rollouts.prepare_output_paths(directory, overwrite=False)
            self.assertEqual(
                run_rollouts.prepare_output_paths(directory, overwrite=True),
                (samples, episodes, metadata, history),
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

    def test_resume_discards_episode_without_matching_history(self):
        episodes = []
        histories = []
        samples = []
        for episode in range(2):
            action = f"action {episode}"
            observation = f"observation {episode}"
            episodes.append(
                {
                    "episode": episode,
                    "gamefile": f"game-{episode}",
                    "task_key": "put",
                    "success": False,
                    "num_steps": 1,
                    "advance_sources": ["admissible_raw_model_sample"],
                    "advance_trace": [
                        {
                            "step": 0,
                            "action": action,
                            "decision": "continue",
                            "observation": observation,
                        }
                    ],
                }
            )
            histories.append(
                {
                    "episode": episode,
                    "gamefile": f"game-{episode}",
                    "task_key": "put",
                    "success": False,
                    "steps": [
                        {
                            "decision": "continue",
                            "action": action,
                            "observation": observation,
                        }
                    ],
                }
            )
            samples.append(
                {
                    "episode": episode,
                    "step": 0,
                    "sample": 0,
                    "distill_eligible": True,
                    "distill_exclusion_reasons": [],
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            samples_path = directory / "samples.jsonl"
            episodes_path = directory / "episodes.jsonl"
            history_path = directory / "history.jsonl"
            samples_path.write_text(
                "".join(json.dumps(record) + "\n" for record in samples),
                encoding="utf-8",
            )
            episodes_path.write_text(
                "".join(json.dumps(record) + "\n" for record in episodes),
                encoding="utf-8",
            )
            history_path.write_text(
                json.dumps(histories[0]) + "\n" + '{"episode": 1',
                encoding="utf-8",
            )

            state = run_rollouts.recover_resume_state(
                samples_path,
                episodes_path,
                history_path,
                samples_per_state=1,
                num_episodes=2,
            )

            self.assertEqual(state[0:3], (1, 1, 1))
            self.assertEqual(
                run_rollouts.read_history_records(history_path), histories[:1]
            )
            self.assertEqual(
                episodes_path.read_text(encoding="utf-8"),
                json.dumps(episodes[0]) + "\n",
            )
            self.assertEqual(
                samples_path.read_text(encoding="utf-8"),
                json.dumps(samples[0]) + "\n",
            )

    def test_resume_trims_history_ahead_of_episode_commit(self):
        episode = {
            "episode": 0,
            "gamefile": "game-0",
            "task_key": "put",
            "success": False,
            "num_steps": 0,
            "advance_sources": [],
            "advance_trace": [],
        }
        histories = [
            {
                "episode": index,
                "gamefile": f"game-{index}",
                "task_key": "put",
                "success": False,
                "steps": [],
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            samples_path = directory / "samples.jsonl"
            episodes_path = directory / "episodes.jsonl"
            history_path = directory / "history.jsonl"
            samples_path.touch()
            episodes_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
            history_path.write_text(
                "".join(json.dumps(record) + "\n" for record in histories),
                encoding="utf-8",
            )

            state = run_rollouts.recover_resume_state(
                samples_path,
                episodes_path,
                history_path,
                samples_per_state=1,
                num_episodes=2,
            )

            self.assertEqual(state[0], 1)
            self.assertEqual(
                run_rollouts.read_history_records(history_path), histories[:1]
            )

    def test_resume_metadata_rejects_changed_collection_setting(self):
        expected = {field: None for field in run_rollouts.RESUME_COMPATIBILITY_FIELDS}
        existing = dict(expected)
        existing["temperature"] = 0.7
        expected["temperature"] = 1.0
        with self.assertRaisesRegex(ValueError, "temperature"):
            run_rollouts.validate_resume_metadata(existing, expected)

    def test_v2_history_reconciles_fallback_reason_and_executed_action(self):
        history = {
            "episode": 0,
            "gamefile": "game-0",
            "task_key": "put",
            "success": False,
            "steps": [
                {
                    "selected_sample": None,
                    "decision": "",
                    "action": "look",
                    "action_taken": "look",
                    "fallback_reason": "no_sampled_action_was_admissible",
                    "observation": "Nothing happens.",
                }
            ],
        }
        episode = {
            "episode": 0,
            "gamefile": "game-0",
            "task_key": "put",
            "success": False,
            "num_steps": 1,
            "advance_trace": [
                {
                    "sample": None,
                    "decision": "",
                    "action": "look",
                    "action_taken": "look",
                    "fallback_reason": "no_sampled_action_was_admissible",
                    "observation": "Nothing happens.",
                }
            ],
        }

        self.assertTrue(run_rollouts._history_matches_episode(history, episode))
        history["steps"][0]["fallback_reason"] = "different_reason"
        self.assertFalse(run_rollouts._history_matches_episode(history, episode))

    def test_selection_uses_first_admissible_even_if_parse_is_malformed(self):
        malformed = make_turn(parse_ok=False, action="look")
        well_formed = make_turn(parse_ok=True, action="look")
        self.assertEqual(
            run_rollouts.select_advance_turn(
                [(0, malformed), (1, well_formed)], ["look"]
            ),
            (0, malformed),
        )

    def test_selection_uses_first_admissible_even_if_head_was_repaired(self):
        repaired = make_turn(
            parse_ok=True, action="look", used_head_repair=True
        )
        natural = make_turn(parse_ok=True, action="look")
        self.assertEqual(
            run_rollouts.select_advance_turn(
                [(0, repaired), (1, natural)], ["look"]
            ),
            (0, repaired),
        )

    def test_selection_skips_inadmissible_samples_and_uses_later_action(self):
        admissible = make_turn(parse_ok=False, action="look")
        self.assertEqual(
            run_rollouts.select_advance_turn(
                [
                    (0, make_turn(parse_ok=True, action="open fridge 1")),
                    (1, admissible),
                ],
                ["look"],
            ),
            (1, admissible),
        )

    def test_selection_returns_none_without_an_admissible_action(self):
        self.assertIsNone(
            run_rollouts.select_advance_turn(
                [
                    (0, make_turn(parse_ok=False, action="inventory")),
                    (1, make_turn(parse_ok=True, action="open fridge 1")),
                ],
                ["look"],
            )
        )

    def test_distillation_allows_truncated_thought(self):
        turn = make_turn(
            parse_ok=True,
            action="look",
            used_head_repair=True,
            used_thought_repair=True,
            thought_truncated=True,
        )
        self.assertEqual(
            run_rollouts.distill_exclusion_reasons(
                turn,
                [10, 11, 20, 0],
                max_hmm_prefix_tokens=None,
                max_hmm_sequence_tokens=128,
            ),
            [],
        )

    def test_distillation_excludes_truncated_decision(self):
        turn = make_turn(
            parse_ok=True,
            action="look",
            used_head_repair=True,
            used_decision_repair=True,
            decision_truncated=True,
        )
        self.assertEqual(
            run_rollouts.distill_exclusion_reasons(
                turn,
                [10, 11, 20, 0],
                max_hmm_prefix_tokens=None,
                max_hmm_sequence_tokens=128,
            ),
            ["synthetic_decision_close", "decision_truncated"],
        )


if __name__ == "__main__":
    unittest.main()
