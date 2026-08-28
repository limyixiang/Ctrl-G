"""Unconstrained rollouts on AlfWorld train games.

Produces:
  <out>/rollouts.jsonl        - full episode logs (baseline stats, tracker eval)
  <out>/distill_prompts.json  - JSON list of prompt strings for Ctrl-G's
                                distillation_vllm/sample_data_vllm.sh
                                (prompt = context + thought + '<tool>')

Usage:
  python scripts/run_rollouts.py --model Qwen/Qwen3.5-9B \
      --num_episodes 500 --out out/rollouts_train
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld import SkillSet, run_episode  # noqa: E402
from ctrlg_alfworld.backends import HFBackend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/base_config.yaml"))
    ap.add_argument("--skills", default=str(ROOT / "SKILLS.md"))
    ap.add_argument("--few_shot", default=str(ROOT / "few_shot/alfworld_3prompts.json"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--num_episodes", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--out", default="out/rollouts")
    args = ap.parse_args()

    import alfworld.agents.environment as environment

    with open(args.config) as f:
        config = yaml.safe_load(f)
    env = getattr(environment, config["env"]["type"])(config, train_eval=args.split)
    env = env.init_env(batch_size=1)

    skillset = SkillSet.from_file(args.skills)
    backend = HFBackend(args.model)  # no HMM needed for unconstrained rollouts

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    distill_prompts, n_success = [], 0

    with open(out_dir / "rollouts.jsonl", "w") as fout:
        for ep in range(args.num_episodes):
            record = run_episode(
                env, backend, skillset, args.few_shot,
                mode="unconstrained", max_steps=args.max_steps, verbose=True,
            )
            fout.write(json.dumps(record.to_dict()) + "\n")
            fout.flush()
            n_success += record.success
            for s in record.steps:
                if s.prefix_to_tool:
                    distill_prompts.append(s.prefix_to_tool)
            print(f"episode {ep}: success={record.success} "
                  f"steps={record.num_steps} (running SR {n_success/(ep+1):.3f})")

    with open(out_dir / "distill_prompts.json", "w") as f:
        json.dump(distill_prompts, f)
    print(f"wrote {len(distill_prompts)} distillation prompts to "
          f"{out_dir / 'distill_prompts.json'}")


if __name__ == "__main__":
    main()
