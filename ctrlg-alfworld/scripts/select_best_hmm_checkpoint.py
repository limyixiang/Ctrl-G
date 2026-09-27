"""Select the saved HMM checkpoint with the best development likelihood."""

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld.checkpoint_selection import select_best_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    selection = select_best_checkpoint(args.log, args.model_path)
    rendered = json.dumps(selection, indent=2)
    Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    # stdout is intentionally machine-readable for command substitution in the
    # Slurm training wrapper.
    print(selection["best_checkpoint"])


if __name__ == "__main__":
    main()
