"""Evaluate on AlfWorld eval_out_of_distribution (134 games), ReAct-style
per-task-type success table.

Modes: unconstrained | constrained | constrained-oracle

Usage:
  python scripts/run_eval.py --model Qwen/Qwen3.5-9B --mode unconstrained
  python scripts/run_eval.py --model Qwen/Qwen3.5-9B --mode constrained \
      --hmm path/to/hmm_checkpoint
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld import SkillSet, run_episode  # noqa: E402
from ctrlg_alfworld.backends import HFBackend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hmm", default=None, help="HMM checkpoint (required for constrained modes)")
    ap.add_argument("--mode", default="unconstrained",
                    choices=["unconstrained", "constrained", "constrained-oracle"])
    ap.add_argument("--config", default=str(ROOT / "configs/base_config.yaml"))
    ap.add_argument("--skills", default=str(ROOT / "SKILLS.md"))
    ap.add_argument("--few_shot", default=str(ROOT / "few_shot/alfworld_3prompts.json"))
    ap.add_argument("--num_episodes", type=int, default=134)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--out", default="out/eval")
    args = ap.parse_args()

    if args.mode.startswith("constrained") and args.hmm is None:
        ap.error("--hmm is required for constrained modes")

    import alfworld.agents.environment as environment

    with open(args.config) as f:
        config = yaml.safe_load(f)
    env = getattr(environment, config["env"]["type"])(
        config, train_eval="eval_out_of_distribution"
    )
    env = env.init_env(batch_size=1)

    skillset = SkillSet.from_file(args.skills)
    backend = HFBackend(args.model, hmm_path=args.hmm)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_type = defaultdict(lambda: [0, 0])  # task_key -> [successes, count]
    n_admissible, n_actions = 0, 0

    with open(out_dir / f"eval_{args.mode}.jsonl", "w") as fout:
        for ep in range(args.num_episodes):
            record = run_episode(
                env, backend, skillset, args.few_shot,
                mode=args.mode, max_steps=args.max_steps, verbose=True,
            )
            fout.write(json.dumps(record.to_dict()) + "\n")
            fout.flush()
            per_type[record.task_key][0] += record.success
            per_type[record.task_key][1] += 1
            for s in record.steps:
                n_actions += 1
                n_admissible += s.action_was_admissible

            total_s = sum(v[0] for v in per_type.values())
            total_n = sum(v[1] for v in per_type.values())
            print(f"[{ep + 1}/{args.num_episodes}] running SR {total_s}/{total_n} "
                  f"= {total_s / total_n:.3f} | admissible-action rate "
                  f"{n_admissible / max(n_actions, 1):.3f}")

    print("\n=== per task type ===")
    for k, (s, n) in sorted(per_type.items()):
        print(f"  {k:10s}: {s}/{n} = {s / n:.3f}")


if __name__ == "__main__":
    main()
