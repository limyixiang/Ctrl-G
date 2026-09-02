"""Validate ALFWorld data and inspect one real TextWorld state."""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs/config_tw.yaml")
    )
    parser.add_argument(
        "--split",
        default="eval_out_of_distribution",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.environ.get("ALFWORLD_DATA"):
        raise EnvironmentError("set ALFWORLD_DATA to the downloaded data directory")

    import alfworld.agents.environment as environment

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    config["general"]["random_seed"] = args.seed
    factory = environment.get_environment(config["env"]["type"])(
        config, train_eval=args.split
    )
    factory.game_files = sorted(factory.game_files)
    factory.num_games = len(factory.game_files)
    if not factory.game_files:
        raise RuntimeError(f"ALFWorld returned no games for split {args.split}")

    env = factory.init_env(batch_size=1)
    observations, info = env.reset()
    admissible = list(info.get("admissible_commands", [[]])[0])
    if not admissible:
        raise RuntimeError("TextWorld returned no admissible commands")

    print(
        json.dumps(
            {
                "split": args.split,
                "seed": args.seed,
                "num_games": factory.num_games,
                "gamefile": info["extra.gamefile"][0],
                "observation_prefix": observations[0][:240],
                "admissible_count": len(admissible),
                "admissible_commands": admissible,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
