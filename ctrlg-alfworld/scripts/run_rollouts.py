"""Collect prompt-conditioned raw model samples for ALFWorld HMM distillation.

Every sample uses the persistent-state prompt format that is evaluated in
both active conditions. One nonempty model action defines the trajectory at
each state, but all observed state/action attempts are HMM training examples, including truncations.
"""

import argparse
import atexit
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld.agent_loop import parse_initial_observation, process_ob
from ctrlg_alfworld.backends import (
    StateGenerationExhausted,
    GenConfig,
    HFBackend,
    VLLMBackend,
)
from ctrlg_alfworld.prompts import (
    PROMPT_FORMAT,
    SYSTEM_INSTRUCTION,
    Step,
    build_user_prompt,
    render_prompt,
    task_key_from_gamefile,
)
from ctrlg_alfworld.provenance import (
    file_sha256,
    git_revision,
    json_sha256,
    runtime_versions,
    source_tree_sha256,
)
from ctrlg_alfworld.skills import SkillSet
from ctrlg_alfworld.collection import executable_attempt, observed_attempt_span, ROLLOUT_SAMPLE_FORMAT
from ctrlg_alfworld.hmm_context import HMM_SEQUENCE_FORMAT

ROOT = Path(__file__).resolve().parents[1]

RESUME_COMPATIBILITY_FIELDS = (
    "model",
    "backend",
    "tokenizer",
    "split",
    "num_episodes",
    "sampling_policy",
    "rollout_sample_format", "collection_max_attempts",
    "prompt_format",
    "schema_version",
    "policy_version",
    "no_oracle_filtering",
    "constrained_state_collection",
    "max_steps",
    "max_thought_tokens",
    "max_state_tokens",
    "max_action_tokens",
    "temperature",
    "seed",
    "generation_schedule",
    "candidate_seed_scheme",
    "environment_seed_scheme",
    "game_files_sha256",
    "history_format",
    "config_sha256",
    "skills_sha256",
)


def _release_output_lock(lock_file) -> None:
    if lock_file.closed:
        return
    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def acquire_output_lock(output: str | Path):
    """Hold an OS-released single-writer lock for one collection directory."""

    output_directory = Path(output)
    output_directory.mkdir(parents=True, exist_ok=True)
    lock_path = output_directory / ".collect.lock"
    lock_file = open(lock_path, "a+b")
    if lock_path.stat().st_size == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock_file.close()
        raise RuntimeError(
            f"another collector is already using output directory {output_directory}"
        ) from exc

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        ).encode("utf-8")
    )
    lock_file.flush()
    atexit.register(_release_output_lock, lock_file)
    return lock_file


def _truncate_file(path: Path, size: int) -> None:
    with open(path, "r+b") as output_file:
        output_file.truncate(size)
        output_file.flush()
        os.fsync(output_file.fileno())


def write_json_atomic(path: Path, value) -> None:
    """Durably replace a small JSON checkpoint without exposing a partial file."""

    temporary_path = path.with_name(f".{path.name}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def validate_resume_metadata(existing: dict, expected: dict) -> None:
    mismatches = []
    for field in RESUME_COMPATIBILITY_FIELDS:
        if existing.get(field) != expected.get(field):
            mismatches.append(
                f"{field}: existing={existing.get(field)!r}, "
                f"requested={expected.get(field)!r}"
            )
    if mismatches:
        raise ValueError(
            "resume settings do not match the existing collection:\n  "
            + "\n  ".join(mismatches)
        )


def _read_episode_prefix(path: Path) -> tuple[list[dict], list[int]]:
    """Read committed-looking episode records, tolerating one torn final line."""

    records = []
    end_offsets = [0]
    with open(path, "rb") as input_file:
        while True:
            line = input_file.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"malformed complete line in {path} after episode "
                    f"{len(records) - 1}"
                ) from exc
            expected_episode = len(records)
            if (
                not isinstance(record, dict)
                or record.get("episode") != expected_episode
            ):
                actual_episode = (
                    record.get("episode") if isinstance(record, dict) else None
                )
                raise ValueError(
                    f"non-contiguous episode record in {path}: expected "
                    f"{expected_episode}, found {actual_episode!r}"
                )
            num_steps = record.get("num_steps")
            if not isinstance(num_steps, int) or num_steps < 0:
                raise ValueError(
                    f"episode {expected_episode} has invalid num_steps={num_steps!r}"
                )
            records.append(record)
            end_offsets.append(input_file.tell())
    return records, end_offsets


def _read_history_prefix(path: Path) -> tuple[list[dict], list[int]]:
    """Read complete episode history records, tolerating one torn final line."""

    records = []
    end_offsets = [0]
    with open(path, "rb") as history_file:
        while True:
            line = history_file.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"malformed complete history line in {path} after episode "
                    f"{len(records) - 1}"
                ) from exc
            expected_episode = len(records)
            if (
                not isinstance(record, dict)
                or record.get("episode") != expected_episode
            ):
                actual_episode = (
                    record.get("episode") if isinstance(record, dict) else None
                )
                raise ValueError(
                    f"non-contiguous history record in {path}: expected "
                    f"{expected_episode}, found {actual_episode!r}"
                )
            steps = record.get("steps")
            if not isinstance(steps, list) or not all(
                isinstance(step, dict) for step in steps
            ):
                raise ValueError(
                    f"history record {expected_episode} in {path} has invalid steps"
                )
            records.append(record)
            end_offsets.append(history_file.tell())
    return records, end_offsets


def read_history_records(path: Path) -> list[dict]:
    """Read the complete prefix of an episode-indexed history JSONL file."""

    return _read_history_prefix(path)[0]


def _history_matches_episode(history_record: dict, episode_record: dict) -> bool:
    """Return whether a history record describes the same completed episode."""

    steps = history_record["steps"]
    if len(steps) != episode_record["num_steps"]:
        return False
    for field in (
        "episode",
        "gamefile",
        "task_key",
        "success",
        "termination_reason",
    ):
        if history_record.get(field) != episode_record.get(field):
            return False

    advance_trace = episode_record.get("advance_trace")
    if (
        not isinstance(advance_trace, list)
        or len(advance_trace) != len(steps)
        or not all(isinstance(trace_step, dict) for trace_step in advance_trace)
    ):
        return False
    return all(
        history_step.get("action_taken", history_step.get("action"))
        == trace_step.get("action_taken", trace_step.get("action"))
        and history_step.get("state", "") == trace_step.get("state", "")
        and history_step.get("selected_sample") == trace_step.get("sample")
        and history_step.get("sample_attempts") == trace_step.get("sample_attempts")
        and history_step.get("fallback_reason")
        == trace_step.get("fallback_reason")
        and history_step.get("observation") == trace_step.get("observation")
        for history_step, trace_step in zip(steps, advance_trace)
    )


def update_state_retry_counts(
    counts: Counter, *, attempts: int, exhausted: bool
) -> None:
    """Accumulate one structured-state call for collection reporting."""

    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError(f"invalid state attempt count: {attempts!r}")
    counts["calls"] += 1
    counts["attempts"] += attempts
    if attempts > 1:
        counts["retried_calls"] += 1
        if not exhausted:
            counts["recovered_calls"] += 1
    if exhausted:
        counts["exhausted_calls"] += 1


def state_retry_metrics(counts: Counter, *, completed_episodes: int) -> dict:
    """Render retry, recovery, and exhausted-episode rates for one model run."""

    calls = counts["calls"]
    retried = counts["retried_calls"]
    recovered = counts["recovered_calls"]
    exhausted = counts["exhausted_calls"]
    return {
        "scope": "whole_turn_collection_attempts",
        "calls": calls,
        "attempts": counts["attempts"],
        "retried_calls": retried,
        "recovered_calls": recovered,
        "exhausted_calls": exhausted,
        "retry_rate": retried / max(calls, 1),
        "recovery_rate": recovered / max(retried, 1),
        "exhausted_episode_rate": exhausted / max(completed_episodes, 1),
    }


def recover_resume_state(
    samples_path: Path,
    episodes_path: Path,
    history_path: Path | None = None,
    *,
    num_episodes: int,
) -> tuple[int, int, int, Counter, dict]:
    """Repair an interrupted suffix and reconstruct counters at an episode boundary."""

    episode_records, episode_end_offsets = _read_episode_prefix(episodes_path)
    if history_path is None:
        history_records = None
        history_end_offsets = None
    else:
        history_records, history_end_offsets = _read_history_prefix(history_path)
    if len(episode_records) > num_episodes:
        raise ValueError(
            f"existing collection has {len(episode_records)} episodes, more than "
            f"requested --num_episodes={num_episodes}"
        )
    if history_records is not None and len(history_records) > num_episodes:
        raise ValueError(
            f"existing history has {len(history_records)} episodes, more than "
            f"requested --num_episodes={num_episodes}"
        )

    committed_episodes = 0
    committed_sample_offset = 0
    sample_count = 0
    eligible_count = 0
    exclusions = Counter()
    retry_counts = Counter()
    with open(samples_path, "rb") as samples_file:
        for episode_position, episode_record in enumerate(episode_records):
            episode_sample_offset = committed_sample_offset
            episode_eligible = 0
            episode_exclusions = Counter()
            episode_retry_counts = Counter()
            episode_complete = True
            incomplete_artifact = "sample"
            episode_samples = 0
            advance_trace = episode_record.get("advance_trace")
            if (
                not isinstance(advance_trace, list)
                or len(advance_trace) != episode_record["num_steps"]
            ):
                episode_complete = False
                incomplete_artifact = "advance trace"
            for step_index in range(episode_record["num_steps"]):
                if not episode_complete:
                    break
                trace_step = advance_trace[step_index]
                sample_attempts = (
                    trace_step.get("sample_attempts")
                    if isinstance(trace_step, dict)
                    else None
                )
                if not isinstance(sample_attempts, int) or sample_attempts < 1:
                    episode_complete = False
                    incomplete_artifact = "advance trace"
                    break
                terminal_state_failure = (
                    trace_step.get("termination_reason")
                    in ("state_generation_exhausted", "collection_attempts_exhausted")
                )
                if terminal_state_failure:
                    trace_position_ok = (
                        trace_step.get("sample") is None
                        and trace_step.get("failed_sample") == sample_attempts - 1
                    )
                else:
                    trace_position_ok = (
                        trace_step.get("sample") == sample_attempts - 1
                    )
                if not trace_position_ok:
                    episode_complete = False
                    incomplete_artifact = "advance trace"
                    break
                episode_samples += sample_attempts
                for sample_index in range(sample_attempts):
                    line = samples_file.readline()
                    if not line or not line.endswith(b"\n"):
                        episode_complete = False
                        break
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        episode_complete = False
                        break
                    if not isinstance(record, dict):
                        episode_complete = False
                        break
                    expected_position = (
                        episode_position,
                        step_index,
                        sample_index,
                    )
                    actual_position = (
                        record.get("episode"),
                        record.get("step"),
                        record.get("sample"),
                    )
                    if actual_position != expected_position:
                        episode_complete = False
                        break
                    action = record.get("action")
                    if (record.get("rollout_sample_format") == ROLLOUT_SAMPLE_FORMAT
                            or record.get("hmm_sequence_format") == HMM_SEQUENCE_FORMAT):
                        should_select = not terminal_state_failure and sample_index == sample_attempts - 1
                        action_ok = (isinstance(action, str)
                                     and record.get("selected_for_execution") is should_select
                                     and (not should_select or bool(action)))
                    elif terminal_state_failure:
                        action_ok = isinstance(action, str) and not action
                    else:
                        action_ok = isinstance(action, str) and (
                            not action if sample_index < sample_attempts - 1 else bool(action))
                    if not action_ok:
                        episode_complete = False
                        break
                    sample_exclusions = record.get(
                        "distill_exclusion_reasons", []
                    )
                    state_exhausted = (
                        "state_generation_exhausted" in sample_exclusions
                    )
                    try:
                        if (record.get("rollout_sample_format") != ROLLOUT_SAMPLE_FORMAT
                                and record.get("hmm_sequence_format") != HMM_SEQUENCE_FORMAT):
                            update_state_retry_counts(
                                episode_retry_counts,
                                attempts=record.get("state_attempts", 1),
                                exhausted=state_exhausted,
                            )
                    except ValueError:
                        episode_complete = False
                        incomplete_artifact = "sample"
                        break
                    episode_eligible += int(bool(record.get("distill_eligible")))
                    episode_exclusions.update(
                        sample_exclusions
                    )
                if not episode_complete:
                    break
                if (record.get("rollout_sample_format") == ROLLOUT_SAMPLE_FORMAT
                        or record.get("hmm_sequence_format") == HMM_SEQUENCE_FORMAT):
                    update_state_retry_counts(episode_retry_counts, attempts=sample_attempts,
                                              exhausted=terminal_state_failure)

            if episode_complete and history_records is not None and (
                episode_position >= len(history_records)
                or not _history_matches_episode(
                    history_records[episode_position], episode_record
                )
            ):
                episode_complete = False
                incomplete_artifact = "history"

            if not episode_complete:
                if episode_position != len(episode_records) - 1:
                    raise ValueError(
                        f"{incomplete_artifact} corruption occurs before the final "
                        "episode; refusing "
                        "to discard an interior portion of the collection"
                    )
                committed_sample_offset = episode_sample_offset
                break

            committed_episodes += 1
            committed_sample_offset = samples_file.tell()
            sample_count += episode_samples
            eligible_count += episode_eligible
            exclusions.update(episode_exclusions)
            retry_counts.update(episode_retry_counts)

    _truncate_file(samples_path, committed_sample_offset)
    _truncate_file(episodes_path, episode_end_offsets[committed_episodes])
    if history_path is not None:
        assert history_end_offsets is not None
        _truncate_file(history_path, history_end_offsets[committed_episodes])

    advance_source_counts = Counter()
    for episode_record in episode_records[:committed_episodes]:
        advance_source_counts.update(episode_record.get("advance_sources", []))
    successful_episodes = sum(
        episode_record.get("success") is True
        for episode_record in episode_records[:committed_episodes]
    )
    per_format = {
        "state": {
            "samples": sample_count,
            "eligible": eligible_count,
            "exclusions": exclusions,
        }
    }
    return (
        committed_episodes,
        sample_count,
        eligible_count,
        advance_source_counts,
        per_format,
        successful_episodes,
        retry_counts,
    )


def update_metadata_progress(
    metadata: dict,
    metadata_path: Path,
    *,
    completed_episodes: int,
    successful_episodes: int,
    sample_count: int,
    eligible_count: int,
    advance_source_counts: Counter,
    per_format: dict,
    state_retry_counts: Counter | None = None,
) -> None:
    metadata["completed_episodes"] = completed_episodes
    metadata["successful_episodes"] = successful_episodes
    metadata["success_rate"] = successful_episodes / max(completed_episodes, 1)
    metadata["collection_status"] = (
        "complete" if completed_episodes == metadata["num_episodes"] else "in_progress"
    )
    if state_retry_counts is not None:
        metadata["collection_retry_metrics"] = state_retry_metrics(state_retry_counts, completed_episodes=completed_episodes)
    metadata["counts"] = {
        "samples": sample_count,
        "eligible": eligible_count,
        "eligible_rate": eligible_count / max(sample_count, 1),
        "advance_sources": dict(sorted(advance_source_counts.items())),
        "per_format": {
            name: {
                "samples": counts["samples"],
                "eligible": counts["eligible"],
                "eligible_rate": counts["eligible"] / max(counts["samples"], 1),
                "exclusion_reasons": dict(sorted(counts["exclusions"].items())),
            }
            for name, counts in per_format.items()
        },
    }
    metadata["state_retries"] = state_retry_metrics(
        state_retry_counts or Counter(),
        completed_episodes=completed_episodes,
    )
    write_json_atomic(metadata_path, metadata)


def prepare_output_paths(
    output: str | Path, *, overwrite: bool, resume: bool = False
) -> tuple[Path, ...]:
    """Create the output directory and protect existing collection artifacts."""

    output_directory = Path(output)
    paths = (
        output_directory / "samples.jsonl",
        output_directory / "episodes.jsonl",
        output_directory / "metadata.json",
        output_directory / "history.jsonl",
    )
    existing = [path for path in paths if path.exists()]
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if resume:
        missing = [path for path in paths if not path.exists()]
        if missing:
            rendered = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"cannot resume because collection artifacts are missing: {rendered}"
            )
    elif existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing rollout artifacts: {rendered}; "
            "choose a new --out directory or pass --overwrite"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    return paths


def select_advance_turn(sampled_turns):
    """Return the first structurally executable sample, without oracle filtering."""

    return next(
        (
            (sample_index, turn)
            for sample_index, turn in sampled_turns
            if executable_attempt(turn)
        ),
        None,
    )


def distill_exclusion_reasons(
    turn,
    hmm_sequence,
    *,
    max_hmm_prefix_tokens,
    max_hmm_sequence_tokens,
):
    """Legacy context-free span diagnostics, retained for old artifact readers.

    Native thinking is outside the HMM sequence, so a repaired or truncated
    thought does not invalidate an otherwise exact state/action span.
    """

    reasons = []
    if not turn.parsed.parse_ok:
        reasons.append("parse_failure")
    if not turn.hmm_prefix_token_ids:
        reasons.append("missing_exact_hmm_prefix")
    if not turn.parsed.action_close_found:
        reasons.append("missing_action_close")
    if turn.used_state_repair:
        reasons.append("synthetic_state_close")
    if turn.state_truncated:
        reasons.append("state_truncated")
    if turn.tail_truncated:
        reasons.append("tail_truncated")
    if not turn.tail_span_exact:
        reasons.append("non_exact_action_tail_span")
    if (
        max_hmm_prefix_tokens is not None
        and len(turn.hmm_prefix_token_ids) > max_hmm_prefix_tokens
    ):
        reasons.append("hmm_prefix_too_long")
    if len(hmm_sequence) > max_hmm_sequence_tokens:
        reasons.append("hmm_sequence_too_long")
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs/config_tw.yaml"))
    parser.add_argument("--skills", default=str(ROOT / "templates/SKILLS.md"))
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
    )
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_thought_tokens", type=int, default=1024)
    parser.add_argument("--max_state_tokens", type=int, default=512)
    parser.add_argument("--max_action_tokens", type=int, default=32)
    parser.add_argument("--collection_max_attempts", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--out", default="out/alfworld_hmm_samples")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing rollout artifacts in --out instead of refusing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted collection in --out. The existing metadata "
            "must match, and any unfinished trailing episode is discarded."
        ),
    )
    args = parser.parse_args()

    if args.collection_max_attempts < 1:
        parser.error("collection attempts must be positive")
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if args.resume and args.backend != "vllm":
        parser.error(
            "--resume is only safe with --backend vllm because the HF backend "
            "uses a process-local RNG stream"
        )
    try:
        lock_file = acquire_output_lock(args.out)
    except RuntimeError as exc:
        parser.error(str(exc))
    try:
        (
            samples_path,
            episodes_path,
            metadata_path,
            history_path,
        ) = prepare_output_paths(
            args.out, overwrite=args.overwrite, resume=args.resume
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import alfworld.agents.environment as environment

    with open(args.config) as file:
        config = yaml.safe_load(file)
    if args.resume and config["env"].get("domain_randomization", False):
        parser.error("--resume requires env.domain_randomization=false")
    config["general"]["random_seed"] = args.seed
    env_factory = environment.get_environment(config["env"]["type"])(
        config, train_eval=args.split
    )
    env_factory.game_files = sorted(env_factory.game_files)
    env_factory.num_games = len(env_factory.game_files)
    env = env_factory.init_env(batch_size=1)
    env.seed(args.seed)

    generation_config = GenConfig(
        max_thought_tokens=args.max_thought_tokens,
        max_state_tokens=args.max_state_tokens,
        max_action_tokens=args.max_action_tokens,
        rollout_temperature=args.temperature,
        seed=args.seed,
        collection_max_attempts=args.collection_max_attempts,
    )
    if args.backend == "vllm":
        backend = VLLMBackend(
            args.model,
            base_url=args.base_url,
            tokenizer_path=args.tokenizer,
            gen_config=generation_config,
        )
    else:
        backend = HFBackend(
            args.model,
            device=args.device,
            dtype=getattr(torch, args.dtype),
            gen_config=generation_config,
        )
    skillset = SkillSet.from_file(args.skills)

    metadata = {
        "model": args.model,
        "backend": args.backend,
        "tokenizer": args.tokenizer or args.model,
        "split": args.split,
        "num_episodes": args.num_episodes,
        "sampling_policy": "unmasked_all_attempts_bounded_structural_retry",
        "prompt_format": PROMPT_FORMAT,
        "schema_version": skillset.schema_version,
        "policy_version": skillset.policy_version,
        "no_oracle_filtering": True,
        "constrained_state_collection": False,
        "max_steps": args.max_steps,
        "max_thought_tokens": args.max_thought_tokens,
        "max_state_tokens": args.max_state_tokens,
        "max_action_tokens": args.max_action_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "generation_schedule": "greedy_thought_unmasked_sampled_state_and_action",
        "rollout_sample_format": ROLLOUT_SAMPLE_FORMAT,
        "collection_max_attempts": args.collection_max_attempts,
        "candidate_seed_scheme": (
            "sha256(base_seed,episode,step,sample,candidate,phase_with_state_attempt)"
            if args.backend == "vllm"
            else "torch_rng_stream"
        ),
        "environment_seed_scheme": "sorted_gamefiles_then_textworld_seed",
        "game_files_sha256": json_sha256(env_factory.game_files),
        "history_format": (
            "episode_jsonl_v6_raw_unmasked_attempts"
        ),
        "device": args.device if args.backend == "hf" else "vllm_server",
        "dtype": args.dtype if args.backend == "hf" else "vllm_server",
        "config": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "skills": str(Path(args.skills).resolve()),
        "skills_sha256": skillset.content_sha256,
        "git_revision": git_revision(ROOT.parent),
        "source_tree_sha256": source_tree_sha256(ROOT),
        "runtime": runtime_versions(),
    }
    if args.resume:
        try:
            with open(metadata_path, encoding="utf-8") as metadata_file:
                existing_metadata = json.load(metadata_file)
            validate_resume_metadata(existing_metadata, metadata)
            (
                start_episode,
                sample_count,
                eligible_count,
                advance_source_counts,
                per_format,
                successful_episodes,
                state_retry_counts,
            ) = recover_resume_state(
                samples_path,
                episodes_path,
                history_path,
                num_episodes=args.num_episodes,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"cannot safely resume: {exc}")
        if (
            existing_metadata.get("source_tree_sha256")
            != metadata["source_tree_sha256"]
        ):
            print(
                "warning: source tree changed since the initial collection; "
                "generation settings, config, and skills still match",
                file=sys.stderr,
            )
        metadata = existing_metadata
        metadata["resume_count"] = int(metadata.get("resume_count", 0)) + 1
        metadata.setdefault("resume_history", []).append(
            {
                "from_episode": start_episode,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_revision": git_revision(ROOT.parent),
                "source_tree_sha256": source_tree_sha256(ROOT),
                "runtime": runtime_versions(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
        )
        if start_episode and start_episode < args.num_episodes:
            skip = getattr(env, "skip", None)
            if not callable(skip):
                parser.error(
                    "cannot safely resume because this TextWorld environment "
                    "does not expose env.skip()"
                )
            skip(start_episode)
        print(
            f"resuming at episode {start_episode}/{args.num_episodes}; "
            f"retained {successful_episodes} successful episodes and "
            f"{sample_count} committed samples"
        )
    else:
        start_episode = 0
        sample_count = 0
        eligible_count = 0
        successful_episodes = 0
        advance_source_counts = Counter()
        per_format = {
            "state": {
                "samples": 0,
                "eligible": 0,
                "exclusions": Counter(),
            },
        }
        state_retry_counts = Counter()
        metadata["resume_count"] = 0
        metadata["resume_history"] = []

    update_metadata_progress(
        metadata,
        metadata_path,
        completed_episodes=start_episode,
        successful_episodes=successful_episodes,
        sample_count=sample_count,
        eligible_count=eligible_count,
        advance_source_counts=advance_source_counts,
        per_format=per_format,
        state_retry_counts=state_retry_counts,
    )

    output_mode = "a" if args.resume else "w"
    with open(samples_path, output_mode, encoding="utf-8") as samples_file, open(
        episodes_path, output_mode, encoding="utf-8"
    ) as episodes_file, open(
        history_path, output_mode, encoding="utf-8"
    ) as history_file:
        for episode_index in range(start_episode, args.num_episodes):
            observations, info = env.reset()
            initial_observation, task_description = parse_initial_observation(
                observations[0]
            )
            gamefile = info["extra.gamefile"][0]
            task_key = task_key_from_gamefile(gamefile)
            observation = initial_observation
            history: list[Step] = []
            history_audit = []
            success = False
            episode_termination_reason = None
            advance_sources = []
            advance_trace = []

            for step_index in range(args.max_steps):
                admissible_actions = list(
                    info.get("admissible_commands", [[]])[0]
                )
                sampled_turns = []
                candidate_audit = []
                use_state = True
                user_prompt = build_user_prompt(
                    skill_content=skillset.raw_markdown,
                    task_description=task_description,
                    initial_observation=initial_observation,
                    current_observation=observation,
                    obs_history=history,
                    use_state=use_state,
                )
                prompt_text = render_prompt(
                    backend.tokenizer, SYSTEM_INSTRUCTION, user_prompt
                )
                prompt_token_ids = backend.tokenizer.encode(
                    prompt_text, add_special_tokens=False
                )

                selected = None
                sample_index = 0
                while selected is None and sample_index < args.collection_max_attempts:
                    turn = backend.generate_turns_training(
                        prompt_text, skillset, task_key, count=1,
                        seed_context=(episode_index, step_index, sample_index),
                    )[0]
                    sampled_turns.append((sample_index, turn))
                    span = observed_attempt_span(turn, backend.tokenizer.eos_token_id)
                    exclusion_reasons = span["distill_exclusion_reasons"]
                    distill_eligible = span["distill_eligible"]
                    action_was_admissible = (
                        turn.parsed.action in admissible_actions
                    )
                    candidate_audit.append(
                        {
                            "sample": sample_index,
                            "model_state": turn.parsed.state,
                            "model_action": turn.parsed.action,
                            "action_was_admissible": action_was_admissible,
                            "parse_ok": turn.parsed.parse_ok,
                            "parse_errors": list(turn.parsed.errors),
                            "state_attempts": turn.state_attempts,
                            "state_retry_used": turn.state_retry_used,
                            "state_failure_reasons": list(
                                turn.state_failure_reasons
                            ),
                            "distill_eligible": distill_eligible,
                            "distill_exclusion_reasons": exclusion_reasons,
                        }
                    )
                    record = {
                        "episode": episode_index,
                        "step": step_index,
                        "sample": sample_index,
                        "gamefile": gamefile,
                        "task_key": task_key,
                        "use_state": use_state,
                        "prompt_format": PROMPT_FORMAT,
                        "schema_version": skillset.schema_version,
                        "policy_version": skillset.policy_version,
                        "constrained_state_collection": False,
                        "no_oracle_filtering": True,
                        "skills_sha256": skillset.content_sha256,
                        "model": args.model,
                        "tokenizer": args.tokenizer or args.model,
                        "temperature": args.temperature,
                        "seed": args.seed,
                        "head_seed": turn.head_seed,
                        "state_seed": turn.state_seed,
                        "state_attempts": turn.state_attempts,
                        "state_retry_used": turn.state_retry_used,
                        "state_attempt_seeds": list(
                            turn.state_attempt_seeds
                        ),
                        "state_failure_reasons": list(
                            turn.state_failure_reasons
                        ),
                        "tail_seed": turn.tail_seed,
                        "prompt_text": prompt_text,
                        "prompt_token_ids": prompt_token_ids,
                        "raw_head": turn.parsed.raw_head,
                        "raw_tail": turn.parsed.raw_tail,
                        "head_token_ids": list(turn.head_token_ids),
                        "thought_token_ids": list(turn.thought_token_ids),
                        "state_token_ids": list(turn.state_token_ids),
                        "hmm_prefix_text": turn.parsed.hmm_prefix_text,
                        "hmm_prefix_token_ids": list(turn.hmm_prefix_token_ids),
                        "action": turn.parsed.action,
                        "action_token_ids": list(turn.action_token_ids),
                        "tail_token_ids": list(turn.tail_token_ids),
                        "parse_ok": turn.parsed.parse_ok,
                        "parse_errors": list(turn.parsed.errors),
                        "used_head_repair": turn.used_head_repair,
                        "used_thought_repair": turn.used_thought_repair,
                        "used_state_repair": turn.used_state_repair,
                        "head_stop_found": turn.head_stop_found,
                        "head_truncated": turn.head_truncated,
                        "thought_stop_found": turn.thought_stop_found,
                        "thought_truncated": turn.thought_truncated,
                        "state_stop_found": turn.state_stop_found,
                        "state_truncated": turn.state_truncated,
                        "tail_stop_found": turn.tail_stop_found,
                        "tail_truncated": turn.tail_truncated,
                        "tail_span_exact": turn.tail_span_exact,
                        "distill_eligible": distill_eligible,
                        "distill_exclusion_reasons": exclusion_reasons,
                        "action_was_admissible": action_was_admissible,
                        "admissible_gt": admissible_actions,
                        "head_latency_seconds": turn.head_latency_seconds,
                        "thought_latency_seconds": turn.thought_latency_seconds,
                        "state_latency_seconds": turn.state_latency_seconds,
                        "action_latency_seconds": turn.action_latency_seconds,
                    }
                    record.update(span)
                    record.update(collection_max_attempts=args.collection_max_attempts)
                    samples_file.write(json.dumps(record) + "\n")
                    sample_count += 1
                    eligible_count += int(distill_eligible)
                    format_key = "state"
                    per_format[format_key]["samples"] += 1
                    per_format[format_key]["eligible"] += int(distill_eligible)
                    per_format[format_key]["exclusions"].update(exclusion_reasons)

                    selected = select_advance_turn(sampled_turns)
                    sample_index += 1

                update_state_retry_counts(state_retry_counts, attempts=sample_index, exhausted=selected is None)
                if selected is None:
                    episode_termination_reason = "collection_attempts_exhausted"
                    failure_step = dict(step=step_index, sample=None, selected_sample=None,
                        failed_sample=sample_index-1, sample_attempts=sample_index,
                        empty_action_retries=sample_index-1, source=episode_termination_reason,
                        candidates=candidate_audit, selected_model_state="", selected_model_action="",
                        action="", action_taken="", state="", fallback_used=False,
                        fallback_reason=episode_termination_reason, fallback_policy=None,
                        termination_reason=episode_termination_reason, observation=observation)
                    history_audit.append(dict(failure_step))
                    advance_trace.append(dict(failure_step))
                    advance_sources.append(episode_termination_reason)
                    advance_source_counts[episode_termination_reason] += 1
                    break

                if episode_termination_reason is not None:
                    break

                advance_sample, selected_turn = selected
                advance_action = selected_turn.parsed.action
                action_was_admissible = advance_action in admissible_actions
                advance_source = (
                    "admissible_raw_model_sample"
                    if action_was_admissible
                    else "inadmissible_raw_model_sample"
                )
                advance_thought = selected_turn.parsed.thought
                advance_state = selected_turn.parsed.state
                selected_model_state = selected_turn.parsed.state
                selected_model_action = selected_turn.parsed.action
                fallback_reason = None
                fallback_policy = None

                next_observation, _, done, info = env.step([advance_action])
                observation = process_ob(next_observation[0])
                history.append(
                    Step(
                        thought=advance_thought,
                        state=advance_state,
                        action=advance_action,
                        observation=observation,
                    )
                )
                history_audit.append(
                    {
                        "step": step_index,
                        "candidates": candidate_audit,
                        "selected_sample": advance_sample,
                        "sample_attempts": sample_index,
                        "empty_action_retries": advance_sample,
                        "selected_model_state": selected_model_state,
                        "selected_model_action": selected_model_action,
                        "action_taken": advance_action,
                        "fallback_used": False,
                        "fallback_reason": fallback_reason,
                        "fallback_policy": fallback_policy,
                        # Retain the original fields for simple readers and
                        # completed-episode resume reconciliation.
                        "state": advance_state,
                        "action": advance_action,
                        "observation": observation,
                    }
                )
                advance_sources.append(advance_source)
                advance_source_counts[advance_source] += 1
                advance_trace.append(
                    {
                        "step": step_index,
                        "sample": advance_sample,
                        "sample_attempts": sample_index,
                        "empty_action_retries": advance_sample,
                        "source": advance_source,
                        "selected_model_state": selected_model_state,
                        "selected_model_action": selected_model_action,
                        "action": advance_action,
                        "action_taken": advance_action,
                        "state": advance_state,
                        "fallback_reason": fallback_reason,
                        "fallback_policy": fallback_policy,
                        "observation": observation,
                    }
                )
                if done[0]:
                    success = bool(info["won"][0])
                    break

            samples_file.flush()
            os.fsync(samples_file.fileno())
            episode_record = {
                "episode": episode_index,
                "gamefile": gamefile,
                "task_key": task_key,
                "success": success,
                "num_steps": len(advance_trace),
                "num_environment_steps": len(history),
                "termination_reason": episode_termination_reason,
                "advance_sources": advance_sources,
                "advance_trace": advance_trace,
            }
            history_record = {
                "episode": episode_index,
                "gamefile": gamefile,
                "task_key": task_key,
                "success": success,
                "termination_reason": episode_termination_reason,
                "initial_observation": initial_observation,
                "task_description": task_description,
                "steps": history_audit,
            }
            history_file.write(json.dumps(history_record) + "\n")
            history_file.flush()
            os.fsync(history_file.fileno())
            episodes_file.write(json.dumps(episode_record) + "\n")
            episodes_file.flush()
            os.fsync(episodes_file.fileno())
            successful_episodes += int(success)
            update_metadata_progress(
                metadata,
                metadata_path,
                completed_episodes=episode_index + 1,
                successful_episodes=successful_episodes,
                sample_count=sample_count,
                eligible_count=eligible_count,
                advance_source_counts=advance_source_counts,
                per_format=per_format,
                state_retry_counts=state_retry_counts,
            )
            print(
                f"[{episode_index + 1}/{args.num_episodes}] "
                f"successes={successful_episodes} "
                f"success_rate={successful_episodes / (episode_index + 1):.3f} "
                f"samples={sample_count} eligible={eligible_count} "
                f"eligible_rate={eligible_count / max(sample_count, 1):.3f} "
                f"state_retry_rate="
                f"{state_retry_counts['retried_calls'] / max(state_retry_counts['calls'], 1):.3f} "
                f"state_recovery_rate="
                f"{state_retry_counts['recovered_calls'] / max(state_retry_counts['retried_calls'], 1):.3f} "
                f"state_exhausted_episode_rate="
                f"{state_retry_counts['exhausted_calls'] / (episode_index + 1):.3f}"
            )

    update_metadata_progress(
        metadata,
        metadata_path,
        completed_episodes=args.num_episodes,
        successful_episodes=successful_episodes,
        sample_count=sample_count,
        eligible_count=eligible_count,
        advance_source_counts=advance_source_counts,
        per_format=per_format,
        state_retry_counts=state_retry_counts,
    )
    atexit.unregister(_release_output_lock)
    _release_output_lock(lock_file)


if __name__ == "__main__":
    main()
