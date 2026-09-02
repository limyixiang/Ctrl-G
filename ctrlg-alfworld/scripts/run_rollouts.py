"""Collect prompt-conditioned raw model samples for ALFWorld HMM distillation.

Every sample uses the persistent-decision prompt format that is evaluated in
both active conditions. The environment may be advanced with one admissible
model sample, but only raw generations are written as HMM training examples.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld.agent_loop import process_ob
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
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import alfworld.agents.environment as environment

    with open(args.config) as file:
        config = yaml.safe_load(file)
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

    output_directory = Path(args.out)
    output_directory.mkdir(parents=True, exist_ok=True)
    samples_path = output_directory / "samples.jsonl"
    episodes_path = output_directory / "episodes.jsonl"
    metadata_path = output_directory / "metadata.json"

    metadata = {
        "model": args.model,
        "backend": args.backend,
        "tokenizer": args.tokenizer or args.model,
        "split": args.split,
        "num_episodes": args.num_episodes,
        "samples_per_state": args.samples_per_state,
        "prompt_format": "decision_with_persistent_history",
        "max_steps": args.max_steps,
        "max_head_tokens": args.max_head_tokens,
        "max_action_tokens": args.max_action_tokens,
        "temperature": args.temperature,
        "max_hmm_prefix_tokens": args.max_hmm_prefix_tokens,
        "max_hmm_sequence_tokens": args.max_hmm_sequence_tokens,
        "seed": args.seed,
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
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    sample_count = 0
    eligible_count = 0
    per_format = {
        "decision": {"samples": 0, "eligible": 0, "exclusions": Counter()},
    }
    with open(samples_path, "w") as samples_file, open(
        episodes_path, "w"
    ) as episodes_file:
        for episode_index in range(args.num_episodes):
            observations, info = env.reset()
            observation_parts = observations[0].split("\n\n")
            initial_observation = "\n".join(observation_parts[1:])
            task_description = observation_parts[2]
            gamefile = info["extra.gamefile"][0]
            task_key = task_key_from_gamefile(gamefile)
            observation = initial_observation
            history: list[Step] = []
            success = False
            advance_sources = []

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
                        show_admissible_actions=False,
                    )
                    prompt_text = render_prompt(
                        backend.tokenizer, SYSTEM_INSTRUCTION, user_prompt
                    )
                    prompt_token_ids = backend.tokenizer.encode(
                        prompt_text, add_special_tokens=False
                    )

                    for sample_index in range(args.samples_per_state):
                        turn = backend.generate_turn_unconstrained(
                            prompt_text,
                            use_decision=use_decision,
                            greedy=False,
                        )
                        sampled_turns.append(turn)
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
                            "temperature": args.temperature,
                            "seed": args.seed,
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

                selected = next(
                    (
                        turn
                        for turn in sampled_turns
                        if turn.parsed.action in admissible_actions
                    ),
                    None,
                )
                if selected is None:
                    advance_action = (
                        "look" if "look" in admissible_actions else admissible_actions[0]
                    )
                    advance_source = "deterministic_admissible_fallback"
                    advance_thought = ""
                    advance_decision = ""
                else:
                    advance_action = selected.parsed.action
                    advance_source = "admissible_raw_model_sample"
                    advance_thought = selected.parsed.thought
                    advance_decision = selected.parsed.decision

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
                if done[0]:
                    success = bool(info["won"][0])
                    break

            episodes_file.write(
                json.dumps(
                    {
                        "episode": episode_index,
                        "gamefile": gamefile,
                        "task_key": task_key,
                        "success": success,
                        "num_steps": len(history),
                        "advance_sources": advance_sources,
                    }
                )
                + "\n"
            )
            samples_file.flush()
            episodes_file.flush()
            print(
                f"[{episode_index + 1}/{args.num_episodes}] "
                f"samples={sample_count} eligible={eligible_count} "
                f"eligible_rate={eligible_count / max(sample_count, 1):.3f}"
            )

    metadata["counts"] = {
        "samples": sample_count,
        "eligible": eligible_count,
        "eligible_rate": eligible_count / max(sample_count, 1),
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
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


if __name__ == "__main__":
    main()
