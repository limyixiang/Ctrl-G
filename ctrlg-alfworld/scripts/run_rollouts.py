"""Collect prompt-conditioned raw model samples for ALFWorld HMM distillation.

Every sample uses the persistent-decision prompt format that is evaluated in
both active conditions. The environment may be advanced with one admissible
model sample, but only raw generations are written as HMM training examples.
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
from ctrlg_alfworld.backends import GenConfig, HFBackend, VLLMBackend
from ctrlg_alfworld.prompts import (
    SYSTEM_INSTRUCTION,
    Step,
    build_user_prompt,
    render_prompt,
    task_key_from_gamefile,
)
from ctrlg_alfworld.provenance import (
    file_sha256,
    git_revision,
    runtime_versions,
    source_tree_sha256,
)
from ctrlg_alfworld.skills import SkillSet

ROOT = Path(__file__).resolve().parents[1]

RESUME_COMPATIBILITY_FIELDS = (
    "model",
    "backend",
    "tokenizer",
    "split",
    "num_episodes",
    "samples_per_state",
    "prompt_format",
    "show_admissible_actions",
    "max_steps",
    "max_head_tokens",
    "max_action_tokens",
    "temperature",
    "max_hmm_prefix_tokens",
    "max_hmm_sequence_tokens",
    "seed",
    "generation_schedule",
    "candidate_seed_scheme",
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
            if record.get("episode") != expected_episode:
                raise ValueError(
                    f"non-contiguous episode record in {path}: expected "
                    f"{expected_episode}, found {record.get('episode')!r}"
                )
            num_steps = record.get("num_steps")
            if not isinstance(num_steps, int) or num_steps < 0:
                raise ValueError(
                    f"episode {expected_episode} has invalid num_steps={num_steps!r}"
                )
            records.append(record)
            end_offsets.append(input_file.tell())
    return records, end_offsets


def recover_resume_state(
    samples_path: Path,
    episodes_path: Path,
    *,
    samples_per_state: int,
    num_episodes: int,
) -> tuple[int, int, int, Counter, dict]:
    """Repair an interrupted suffix and reconstruct counters at an episode boundary."""

    episode_records, episode_end_offsets = _read_episode_prefix(episodes_path)
    if len(episode_records) > num_episodes:
        raise ValueError(
            f"existing collection has {len(episode_records)} episodes, more than "
            f"requested --num_episodes={num_episodes}"
        )

    committed_episodes = 0
    committed_sample_offset = 0
    sample_count = 0
    eligible_count = 0
    exclusions = Counter()
    with open(samples_path, "rb") as samples_file:
        for episode_position, episode_record in enumerate(episode_records):
            episode_sample_offset = committed_sample_offset
            episode_eligible = 0
            episode_exclusions = Counter()
            episode_complete = True
            for step_index in range(episode_record["num_steps"]):
                for sample_index in range(samples_per_state):
                    line = samples_file.readline()
                    if not line or not line.endswith(b"\n"):
                        episode_complete = False
                        break
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
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
                    episode_eligible += int(bool(record.get("distill_eligible")))
                    episode_exclusions.update(
                        record.get("distill_exclusion_reasons", [])
                    )
                if not episode_complete:
                    break

            if not episode_complete:
                if episode_position != len(episode_records) - 1:
                    raise ValueError(
                        "sample corruption occurs before the final episode; refusing "
                        "to discard an interior portion of the collection"
                    )
                committed_sample_offset = episode_sample_offset
                break

            committed_episodes += 1
            committed_sample_offset = samples_file.tell()
            episode_samples = episode_record["num_steps"] * samples_per_state
            sample_count += episode_samples
            eligible_count += episode_eligible
            exclusions.update(episode_exclusions)

    _truncate_file(samples_path, committed_sample_offset)
    _truncate_file(episodes_path, episode_end_offsets[committed_episodes])

    advance_source_counts = Counter()
    for episode_record in episode_records[:committed_episodes]:
        advance_source_counts.update(episode_record.get("advance_sources", []))
    per_format = {
        "decision": {
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
    )


def update_metadata_progress(
    metadata: dict,
    metadata_path: Path,
    *,
    completed_episodes: int,
    sample_count: int,
    eligible_count: int,
    advance_source_counts: Counter,
    per_format: dict,
) -> None:
    metadata["completed_episodes"] = completed_episodes
    metadata["collection_status"] = (
        "complete" if completed_episodes == metadata["num_episodes"] else "in_progress"
    )
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


def select_advance_turn(sampled_turns, admissible_actions):
    """Return the first well-formed admissible sample and its sample index."""

    return next(
        (
            (sample_index, turn)
            for sample_index, turn in sampled_turns
            if turn.parsed.parse_ok
            and turn.parsed.action in admissible_actions
        ),
        None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs/config_tw.yaml"))
    parser.add_argument("--skills", default=str(ROOT / "templates/SKILLS.md"))
    parser.add_argument(
        "--show_admissible_actions",
        action="store_true",
        help=(
            "Include the current admissible commands in the model-visible prompt. "
            "Disabled by default and must match evaluation."
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
    )
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--samples_per_state", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_head_tokens", type=int, default=512)
    parser.add_argument("--max_action_tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_hmm_prefix_tokens", type=int, default=None)
    parser.add_argument("--max_hmm_sequence_tokens", type=int, default=128)
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
        samples_path, episodes_path, metadata_path = prepare_output_paths(
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

    generation_config = GenConfig(
        max_head_tokens=args.max_head_tokens,
        max_action_tokens=args.max_action_tokens,
        rollout_temperature=args.temperature,
        seed=args.seed,
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
        "samples_per_state": args.samples_per_state,
        "prompt_format": "decision_with_persistent_history",
        "show_admissible_actions": args.show_admissible_actions,
        "max_steps": args.max_steps,
        "max_head_tokens": args.max_head_tokens,
        "max_action_tokens": args.max_action_tokens,
        "temperature": args.temperature,
        "max_hmm_prefix_tokens": args.max_hmm_prefix_tokens,
        "max_hmm_sequence_tokens": args.max_hmm_sequence_tokens,
        "seed": args.seed,
        "generation_schedule": (
            "two_phase_vllm_continuous_batch"
            if args.backend == "vllm"
            else "sequential_head_tail_pairs"
        ),
        "candidate_seed_scheme": (
            "sha256(base_seed,episode,step,candidate,phase)"
            if args.backend == "vllm"
            else "torch_rng_stream"
        ),
        "device": args.device if args.backend == "hf" else "vllm_server",
        "dtype": args.dtype if args.backend == "hf" else "vllm_server",
        "config": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "skills": str(Path(args.skills).resolve()),
        "skills_sha256": file_sha256(args.skills),
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
            ) = recover_resume_state(
                samples_path,
                episodes_path,
                samples_per_state=args.samples_per_state,
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
            f"retained {sample_count} committed samples"
        )
    else:
        start_episode = 0
        sample_count = 0
        eligible_count = 0
        advance_source_counts = Counter()
        per_format = {
            "decision": {
                "samples": 0,
                "eligible": 0,
                "exclusions": Counter(),
            },
        }
        metadata["resume_count"] = 0
        metadata["resume_history"] = []

    update_metadata_progress(
        metadata,
        metadata_path,
        completed_episodes=start_episode,
        sample_count=sample_count,
        eligible_count=eligible_count,
        advance_source_counts=advance_source_counts,
        per_format=per_format,
    )

    output_mode = "a" if args.resume else "w"
    with open(samples_path, output_mode, encoding="utf-8") as samples_file, open(
        episodes_path, output_mode, encoding="utf-8"
    ) as episodes_file:
        for episode_index in range(start_episode, args.num_episodes):
            observations, info = env.reset()
            initial_observation, task_description = parse_initial_observation(
                observations[0]
            )
            gamefile = info["extra.gamefile"][0]
            task_key = task_key_from_gamefile(gamefile)
            observation = initial_observation
            history: list[Step] = []
            success = False
            advance_sources = []
            advance_trace = []

            for step_index in range(args.max_steps):
                admissible_actions = list(
                    info.get("admissible_commands", [[]])[0]
                )
                if not admissible_actions:
                    raise RuntimeError("TextWorld returned no admissible commands")

                sampled_turns = []
                for use_decision in (True,):
                    user_prompt = build_user_prompt(
                        skill_content=skillset.raw_markdown,
                        task_description=task_description,
                        current_observation=observation,
                        obs_history=history,
                        use_decision=use_decision,
                        admissible_actions=admissible_actions,
                        show_admissible_actions=args.show_admissible_actions,
                    )
                    prompt_text = render_prompt(
                        backend.tokenizer, SYSTEM_INSTRUCTION, user_prompt
                    )
                    prompt_token_ids = backend.tokenizer.encode(
                        prompt_text, add_special_tokens=False
                    )

                    turns = backend.generate_turns_unconstrained(
                        prompt_text,
                        count=args.samples_per_state,
                        use_decision=use_decision,
                        greedy=False,
                        seed_context=(episode_index, step_index),
                    )
                    for sample_index, turn in enumerate(turns):
                        sampled_turns.append((sample_index, turn))
                        hmm_sequence = (
                            list(turn.hmm_prefix_token_ids)
                            + list(turn.tail_token_ids)
                            + [backend.tokenizer.eos_token_id]
                        )
                        exclusion_reasons = []
                        if not turn.parsed.parse_ok:
                            exclusion_reasons.append("parse_failure")
                        if not turn.hmm_prefix_token_ids:
                            exclusion_reasons.append("missing_exact_hmm_prefix")
                        if not turn.parsed.action_close_found:
                            exclusion_reasons.append("missing_action_close")
                        if turn.used_head_repair:
                            exclusion_reasons.append("synthetic_action_open")
                        if turn.head_truncated:
                            exclusion_reasons.append("head_truncated")
                        if turn.tail_truncated:
                            exclusion_reasons.append("tail_truncated")
                        if not turn.tail_span_exact:
                            exclusion_reasons.append("non_exact_action_tail_span")
                        if (
                            args.max_hmm_prefix_tokens is not None
                            and len(turn.hmm_prefix_token_ids)
                            > args.max_hmm_prefix_tokens
                        ):
                            exclusion_reasons.append("hmm_prefix_too_long")
                        if len(hmm_sequence) > args.max_hmm_sequence_tokens:
                            exclusion_reasons.append("hmm_sequence_too_long")
                        distill_eligible = not exclusion_reasons
                        record = {
                            "episode": episode_index,
                            "step": step_index,
                            "sample": sample_index,
                            "gamefile": gamefile,
                            "task_key": task_key,
                            "use_decision": use_decision,
                            "show_admissible_actions": args.show_admissible_actions,
                            "temperature": args.temperature,
                            "seed": args.seed,
                            "head_seed": turn.head_seed,
                            "tail_seed": turn.tail_seed,
                            "prompt_text": prompt_text,
                            "prompt_token_ids": prompt_token_ids,
                            "raw_head": turn.parsed.raw_head,
                            "raw_tail": turn.parsed.raw_tail,
                            "head_token_ids": list(turn.head_token_ids),
                            "hmm_prefix_text": turn.parsed.hmm_prefix_text,
                            "hmm_prefix_token_ids": list(
                                turn.hmm_prefix_token_ids
                            ),
                            "action": turn.parsed.action,
                            "action_token_ids": list(turn.action_token_ids),
                            "tail_token_ids": list(turn.tail_token_ids),
                            "hmm_sequence_token_ids": hmm_sequence,
                            "parse_ok": turn.parsed.parse_ok,
                            "parse_errors": list(turn.parsed.errors),
                            "used_head_repair": turn.used_head_repair,
                            "head_stop_found": turn.head_stop_found,
                            "head_truncated": turn.head_truncated,
                            "tail_stop_found": turn.tail_stop_found,
                            "tail_truncated": turn.tail_truncated,
                            "tail_span_exact": turn.tail_span_exact,
                            "distill_eligible": distill_eligible,
                            "distill_exclusion_reasons": exclusion_reasons,
                            "action_was_admissible": (
                                turn.parsed.action in admissible_actions
                            ),
                            "admissible_gt": admissible_actions,
                            "head_latency_seconds": turn.head_latency_seconds,
                            "action_latency_seconds": turn.action_latency_seconds,
                        }
                        samples_file.write(json.dumps(record) + "\n")
                        sample_count += 1
                        eligible_count += int(distill_eligible)
                        format_key = "decision"
                        per_format[format_key]["samples"] += 1
                        per_format[format_key]["eligible"] += int(distill_eligible)
                        per_format[format_key]["exclusions"].update(exclusion_reasons)

                selected = select_advance_turn(sampled_turns, admissible_actions)
                if selected is None:
                    advance_sample = None
                    advance_action = (
                        "look" if "look" in admissible_actions else admissible_actions[0]
                    )
                    advance_source = "deterministic_admissible_fallback"
                    advance_thought = ""
                    advance_decision = ""
                else:
                    advance_sample, selected_turn = selected
                    advance_action = selected_turn.parsed.action
                    advance_source = "admissible_raw_model_sample"
                    advance_thought = selected_turn.parsed.thought
                    advance_decision = selected_turn.parsed.decision

                next_observation, _, done, info = env.step([advance_action])
                observation = process_ob(next_observation[0])
                history.append(
                    Step(
                        thought=advance_thought,
                        decision=advance_decision,
                        action=advance_action,
                        observation=observation,
                    )
                )
                advance_sources.append(advance_source)
                advance_source_counts[advance_source] += 1
                advance_trace.append(
                    {
                        "step": step_index,
                        "sample": advance_sample,
                        "source": advance_source,
                        "action": advance_action,
                        "decision": advance_decision,
                        "observation": observation,
                    }
                )
                if done[0]:
                    success = bool(info["won"][0])
                    break

            samples_file.flush()
            os.fsync(samples_file.fileno())
            episodes_file.write(
                json.dumps(
                    {
                        "episode": episode_index,
                        "gamefile": gamefile,
                        "task_key": task_key,
                        "success": success,
                        "num_steps": len(history),
                        "advance_sources": advance_sources,
                        "advance_trace": advance_trace,
                    }
                )
                + "\n"
            )
            episodes_file.flush()
            os.fsync(episodes_file.fileno())
            update_metadata_progress(
                metadata,
                metadata_path,
                completed_episodes=episode_index + 1,
                sample_count=sample_count,
                eligible_count=eligible_count,
                advance_source_counts=advance_source_counts,
                per_format=per_format,
            )
            print(
                f"[{episode_index + 1}/{args.num_episodes}] "
                f"samples={sample_count} eligible={eligible_count} "
                f"eligible_rate={eligible_count / max(sample_count, 1):.3f}"
            )

    update_metadata_progress(
        metadata,
        metadata_path,
        completed_episodes=args.num_episodes,
        sample_count=sample_count,
        eligible_count=eligible_count,
        advance_source_counts=advance_source_counts,
        per_format=per_format,
    )
    atexit.unregister(_release_output_lock)
    _release_output_lock(lock_file)


if __name__ == "__main__":
    main()
