import argparse
import sys
import yaml
from pathlib import Path

import alfworld.agents.environment as environment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-9B")
    ap.add_argument("--config", default=str(ROOT / "configs/config_tw.yaml"))
    ap.add_argument("--skills", default=str(ROOT / "templates/SKILLS.md"))
    ap.add_argument("--split", default="train", choices=["train", "eval_in_distribution", "eval_out_of_distribution"])
    ap.add_argument("--num_episodes", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--out", default="out/rollouts")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    env = environment.get_environment(config["env"]["type"])(config, train_eval=args.split)
    env = env.init_env(batch_size=1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    distill_prompts, n_success = [], 0

    with open(out_dir / "rollouts.jsonl", "w") as fout:
        for ep in range(args.num_episodes):
            record = 

if __name__ == "__main__":
    main()