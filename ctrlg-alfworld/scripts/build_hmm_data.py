"""Build the decision-format Ctrl-G HMM train/dev/LVD tensors."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld.distillation import (
    extract_lvd_embeddings,
    load_advance_selections,
    load_eligible_records,
    pad_sequences,
    split_records,
    validate_prompt_regime,
    validate_tokenizer_contract,
)
from ctrlg_alfworld.provenance import (
    file_sha256,
    git_revision,
    runtime_versions,
    source_tree_sha256,
)


def pad_lvd(sequences, embeddings, *, length, eos_token_id):
    if sequences.shape[1] == length:
        return sequences, embeddings
    padding_length = length - sequences.shape[1]
    if padding_length < 0:
        raise ValueError("common sequence length truncates LVD data")
    return (
        torch.cat(
            (
                sequences,
                torch.full(
                    (sequences.shape[0], padding_length),
                    eos_token_id,
                    dtype=sequences.dtype,
                ),
            ),
            dim=1,
        ),
        torch.cat(
            (
                embeddings,
                torch.zeros(
                    (embeddings.shape[0], padding_length, embeddings.shape[2]),
                    dtype=embeddings.dtype,
                ),
            ),
            dim=1,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument(
        "--episodes",
        help="Rollout episodes.jsonl containing advance_trace selections",
    )
    parser.add_argument(
        "--selected_only",
        action="store_true",
        help="Build from executed sampled turns referenced by --episodes only",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="alfworld_actions")
    parser.add_argument("--train_chunks", type=int, default=8)
    parser.add_argument("--dev_fraction", type=float, default=0.1)
    parser.add_argument("--lvd_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    args = parser.parse_args()
    if args.save_embeddings and args.model is None:
        parser.error("--save_embeddings requires --model")
    if args.selected_only and not args.episodes:
        parser.error("--selected_only requires --episodes")
    if args.episodes and not args.selected_only:
        parser.error("--episodes is only used with --selected_only")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    selected_actions = None
    selection_stats = None
    if args.selected_only:
        selected_actions, selection_stats = load_advance_selections(args.episodes)
    all_eligible = load_eligible_records(
        args.samples, selected_actions=selected_actions
    )
    records = [item for item in all_eligible if bool(item["use_decision"])]
    if not records:
        raise ValueError("no eligible decision-format records")
    collection_regime = validate_prompt_regime(records)
    if args.tokenizer != collection_regime["tokenizer"]:
        raise ValueError(
            "--tokenizer does not match collection provenance: "
            f"{args.tokenizer!r} != {collection_regime['tokenizer']!r}"
        )
    if args.model is not None and args.model != collection_regime["model"]:
        raise ValueError(
            "--model does not match collection provenance: "
            f"{args.model!r} != {collection_regime['model']!r}"
        )
    validate_tokenizer_contract(tokenizer, records)
    train_records, dev_records = split_records(
        records, dev_fraction=args.dev_fraction, seed=args.seed
    )
    sequence_length = max(len(item["hmm_sequence_token_ids"]) for item in records)

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    prefix = output_directory / args.dataset
    torch.save(
        pad_sequences(
            dev_records,
            eos_token_id=tokenizer.eos_token_id,
            length=sequence_length,
        ),
        f"{prefix}.dev",
    )
    chunk_count = min(args.train_chunks, len(train_records))
    chunks = torch.tensor_split(
        pad_sequences(
            train_records,
            eos_token_id=tokenizer.eos_token_id,
            length=sequence_length,
        ),
        chunk_count,
    )
    for index, chunk in enumerate(chunks):
        torch.save(chunk.contiguous(), f"{prefix}.train.{index}")

    lvd_records = train_records[: min(args.lvd_samples, len(train_records))]
    if args.save_embeddings:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype)
        ).to("cuda")
        model.eval()
        lvd_sequences, lvd_embeddings = extract_lvd_embeddings(
            model, lvd_records, eos_token_id=tokenizer.eos_token_id
        )
        lvd_sequences, lvd_embeddings = pad_lvd(
            lvd_sequences,
            lvd_embeddings,
            length=sequence_length,
            eos_token_id=tokenizer.eos_token_id,
        )
        torch.save(lvd_sequences, f"{prefix}.lvd")
        torch.save(lvd_embeddings, f"{prefix}.lvd.embeddings")
    else:
        torch.save(
            pad_sequences(
                lvd_records,
                eos_token_id=tokenizer.eos_token_id,
                length=sequence_length,
            ),
            f"{prefix}.lvd",
        )

    train_states = {(item.get("episode"), item.get("step")) for item in train_records}
    dev_states = {(item.get("episode"), item.get("step")) for item in dev_records}
    if train_states & dev_states:
        raise RuntimeError("environment-state leakage between train and dev")
    metadata = {
        "source": args.samples,
        "source_sha256": file_sha256(args.samples),
        "selection_policy": (
            "executed_sample_only" if args.selected_only else "all_eligible"
        ),
        "dataset": args.dataset,
        **collection_regime,
        "no_oracle_filtering": True,
        "skills_sha256": records[0].get("skills_sha256"),
        "eligible_records_in_source": len(all_eligible),
        "eligible_decision_records": len(records),
        "ignored_non_decision_records": len(all_eligible) - len(records),
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "train_state_groups": len(train_states),
        "dev_state_groups": len(dev_states),
        "lvd_records": len(lvd_records),
        "train_chunks": chunk_count,
        "sequence_length": sequence_length,
        "eos_token_id": tokenizer.eos_token_id,
        "save_embeddings": args.save_embeddings,
        "model": collection_regime["model"],
        "tokenizer": collection_regime["tokenizer"],
        "seed": args.seed,
        "split_policy": "state-group train/dev split",
        "git_revision": git_revision(Path(__file__).resolve().parents[2]),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[1]),
        "runtime": runtime_versions(),
    }
    if args.selected_only:
        metadata.update(
            {
                "episodes_source": args.episodes,
                "episodes_source_sha256": file_sha256(args.episodes),
                **selection_stats,
                "selected_eligible_records": len(all_eligible),
                "selected_ineligible_records": (
                    selection_stats["sampled_advance_steps"] - len(all_eligible)
                ),
            }
        )
    with open(f"{prefix}.metadata.json", "w") as output_file:
        json.dump(metadata, output_file, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
