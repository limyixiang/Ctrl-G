"""Report the matched HMM checkpoint's held-out state-format fit."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg import HMM
from ctrlg_alfworld.prompts import PROMPT_FORMAT
from ctrlg_alfworld.hmm_context import HMM_SEQUENCE_FORMAT
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


def score_outputs(model, tensor, spans, batch_size):
    if len(tensor) != len(spans):
        raise ValueError("span metadata does not match held-out rows")
    totals = dict(output=0.0, state=0.0, action=0.0)
    tokens = dict(output=0, state=0, action=0)
    action_sequences = 0
    for row, span in zip(tensor, spans):
        length, context = span["length"], span["context_length"]
        action = span["action_start"]
        action = length if action is None else min(action, length)
        observed = row[:length][None, :].to(model.alpha_exp.device)
        with torch.no_grad():
            full = float(model.forward(observed)[-1].item())
            context_ll = float(model.forward(observed[:, :context])[-1].item())
            state_ll = float(model.forward(observed[:, :action])[-1].item())
        totals["output"] += full-context_ll
        totals["state"] += state_ll-context_ll
        totals["action"] += full-state_ll
        tokens["output"] += length-context
        tokens["state"] += action-context
        tokens["action"] += length-action
        action_sequences += int(length > action)
    return {"sequences": len(spans), "action_sequences": action_sequences,
        "completed_sequences": sum(s["completed"] for s in spans),
        "censored_sequences": sum(not s["completed"] for s in spans),
        **{f"{phase}_conditional_nll_per_sequence":
           -total/(action_sequences if phase == "action" else len(spans))
           if (action_sequences if phase == "action" else len(spans)) else None
           for phase, total in totals.items()},
        **{f"{phase}_conditional_nll_per_token": -totals[phase]/count if count else None
           for phase, count in tokens.items()}, "observed_tokens": tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmm", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", default="alfworld_actions")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metadata_path = Path(args.data_dir) / f"{args.dataset}.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("prompt_format") != PROMPT_FORMAT:
        raise ValueError(f"held-out data must use prompt_format {PROMPT_FORMAT!r}")

    if metadata.get("hmm_sequence_format") != HMM_SEQUENCE_FORMAT:
        raise ValueError("held-out data uses an obsolete HMM format")

    model = HMM.from_pretrained(args.hmm, map_location="cpu").to(args.device)
    path = Path(args.data_dir) / f"{args.dataset}.dev"
    tensor = torch.load(path)
    metrics = score(model, tensor, args.batch_size)
    spans = json.loads((Path(args.data_dir)/f"{args.dataset}.dev.spans.json").read_text())
    metrics.update(score_outputs(model, tensor, spans, args.batch_size))
    output = {
        "hmm": str(Path(args.hmm).resolve()),
        "hmm_sha256": artifact_sha256(args.hmm),
        "prompt_format": PROMPT_FORMAT,
        "architecture": {
            "hidden_states": model.hidden_states,
            "vocab_size": model.vocab_size,
            "eos_token_id": model.eos_token_id,
        },
        "dataset_base": args.dataset,
        "data_metadata": str(metadata_path),
        "data_metadata_sha256": file_sha256(metadata_path),
        "runtime": runtime_versions(),
        "held_out_fit": {"data": str(path), **metrics},
    }
    rendered = json.dumps(output, indent=2)
    Path(args.out).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
