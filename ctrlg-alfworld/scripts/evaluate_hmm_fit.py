"""Report the matched HMM checkpoint's held-out decision-format fit."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg import HMM
from ctrlg_alfworld.provenance import artifact_sha256, file_sha256, runtime_versions


def score(model, tensor, batch_size):
    count = int(tensor.shape[0])
    total = float(model.loglikelihood(tensor, batch_size).item())
    return {
        "sequences": count,
        "sequence_length": int(tensor.shape[1]),
        "total_log_likelihood": total,
        "mean_log_likelihood_per_sequence": total / count,
        "mean_negative_log_likelihood_per_sequence": -total / count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmm", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", default="alfworld_actions")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = HMM.from_pretrained(args.hmm, map_location="cpu").to(args.device)
    path = Path(args.data_dir) / f"{args.dataset}.dev"
    metrics = score(model, torch.load(path), args.batch_size)
    output = {
        "hmm": str(Path(args.hmm).resolve()),
        "hmm_sha256": artifact_sha256(args.hmm),
        "prompt_format": "decision_with_persistent_history",
        "architecture": {
            "hidden_states": model.hidden_states,
            "vocab_size": model.vocab_size,
            "eos_token_id": model.eos_token_id,
        },
        "dataset_base": args.dataset,
        "data_metadata": str(Path(args.data_dir) / f"{args.dataset}.metadata.json"),
        "data_metadata_sha256": file_sha256(
            Path(args.data_dir) / f"{args.dataset}.metadata.json"
        ),
        "runtime": runtime_versions(),
        "held_out_fit": {"data": str(path), **metrics},
    }
    rendered = json.dumps(output, indent=2)
    Path(args.out).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
