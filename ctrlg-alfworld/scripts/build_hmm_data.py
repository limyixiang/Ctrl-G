"""Build the state-format Ctrl-G HMM train/dev/LVD tensors."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg_alfworld.hmm_context import PADDING_TOKEN_ID, HMM_SEQUENCE_FORMAT, HMM_CONTEXT_FORMAT
from ctrlg_alfworld.distillation import (
    extract_lvd_embeddings,
    load_eligible_records,
    build_contextual_records,
    summarize_context_usage,
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
                    PADDING_TOKEN_ID,
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
        "--history",
        help="Observable episode history; defaults to history.jsonl beside --samples",
    )
    parser.add_argument(
        "--selected_only",
        action="store_true",
        help="Obsolete: all observed attempts are required",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="alfworld_actions")
    parser.add_argument("--train_chunks", type=int, default=8)
    parser.add_argument("--dev_fraction", type=float, default=0.1)
    parser.add_argument("--lvd_samples", type=int, default=10000)
    parser.add_argument("--lvd_embedding_budget_gib", type=float, default=32.0)
    parser.add_argument("--hmm_context_steps", type=int, default=1)
    parser.add_argument("--max_hmm_context_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    args = parser.parse_args()
    if args.lvd_embedding_budget_gib <= 0 or args.lvd_samples < 1:
        parser.error("LVD embedding budget and sample count must be positive")
    if args.save_embeddings and args.model is None:
        parser.error("--save_embeddings requires --model")
    if args.selected_only:
        parser.error("the main dataset includes all attempts; --selected_only is obsolete")
    if args.hmm_context_steps < 0 or args.max_hmm_context_tokens < 1:
        parser.error("context steps must be nonnegative and context budget positive")
    history_path = Path(args.history) if args.history else Path(args.samples).with_name("history.jsonl")
    if not history_path.is_file():
        parser.error(f"observable history is required to rebuild context: {history_path}; use --history")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    all_eligible = load_eligible_records(args.samples)
    records = [item for item in all_eligible if bool(item["use_state"])]
    if not records:
        raise ValueError("no eligible state-format records")
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
    records = build_contextual_records(tokenizer, records, history_path,
        k=args.hmm_context_steps, max_tokens=args.max_hmm_context_tokens)
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
        # Qwen3.5 attention can otherwise select cuDNN SDPA, which is unusable
        # on clusters whose loaded cuDNN sublibraries do not have matching
        # versions. Keep Flash/Efficient/Math SDPA available, as the rollout
        # backend does, and disable only the failing cuDNN implementation.
        torch.backends.cuda.enable_cudnn_sdp(False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=getattr(torch, args.dtype)
        ).to("cuda")
        model.eval()
        text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        hidden_size = text_config.hidden_size
        bytes_per_row = sequence_length * hidden_size * torch.empty((), dtype=getattr(torch, args.dtype)).element_size()
        capacity = int(args.lvd_embedding_budget_gib * 1024**3) // bytes_per_row
        if capacity < 1:
            raise ValueError("LVD embedding budget cannot hold one padded sequence")
        lvd_records = lvd_records[:capacity]
        lvd_sequences, lvd_embeddings = extract_lvd_embeddings(
            model, lvd_records, eos_token_id=tokenizer.eos_token_id, sequence_length=sequence_length
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
    train_games = {item["gamefile"] for item in train_records}
    dev_games = {item["gamefile"] for item in dev_records}
    if train_games & dev_games:
        raise RuntimeError("game-instance leakage between train and dev")
    # Explicit observed lengths and phase boundaries support output-only scoring.
    for name, rows in (("train", train_records), ("dev", dev_records), ("lvd", lvd_records)):
        with open(f"{prefix}.{name}.spans.json", "w") as output_file:
            json.dump([dict(length=len(r["hmm_sequence_token_ids"]),
                context_length=len(r["hmm_context_token_ids"]),
                context_truncated=r["hmm_context_truncated"],
                context_history_turns=r["hmm_context_history_turns"],
                action_start=r["hmm_action_start"], completed=r["hmm_completed"],
                gamefile=r["gamefile"], episode=r["episode"], step=r["step"], sample=r["sample"])
                for r in rows], output_file)
    metadata = {
        "source": args.samples,
        "source_sha256": file_sha256(args.samples),
        "history_source": str(history_path),
        "history_source_sha256": file_sha256(history_path),
        "selection_policy": (
            "all_observed_attempts"
        ),
        "dataset": args.dataset,
        **collection_regime,
        "hmm_sequence_format": HMM_SEQUENCE_FORMAT,
        "hmm_context_format": HMM_CONTEXT_FORMAT,
        "hmm_context_steps": args.hmm_context_steps,
        "max_hmm_context_tokens": args.max_hmm_context_tokens,
        "context_construction_stage": "dataset_build",
        "hmm_context_usage": summarize_context_usage(records, args.max_hmm_context_tokens),
        "no_oracle_filtering": True,
        "skills_sha256": records[0].get("skills_sha256"),
        "eligible_records_in_source": len(all_eligible),
        "eligible_state_records": len(records),
        "ignored_non_state_records": len(all_eligible) - len(records),
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "train_state_groups": len(train_states),
        "dev_state_groups": len(dev_states),
        "train_game_groups": len(train_games),
        "dev_game_groups": len(dev_games),
        "lvd_records": len(lvd_records),
        "requested_lvd_samples": args.lvd_samples,
        "lvd_embedding_budget_gib": args.lvd_embedding_budget_gib,
        "train_chunks": chunk_count,
        "sequence_length": sequence_length,
        "eos_token_id": tokenizer.eos_token_id,
        "save_embeddings": args.save_embeddings,
        "model": collection_regime["model"],
        "tokenizer": collection_regime["tokenizer"],
        "seed": args.seed,
        "split_policy": "game-instance train/dev split",
        "padding_token_id": PADDING_TOKEN_ID,
        "likelihood_policy": "one sequence per attempt; incomplete targets are censored prefixes",
        "completed_records": sum(item["hmm_completed"] for item in records),
        "truncated_records": sum(not item["hmm_completed"] for item in records),
        "git_revision": git_revision(Path(__file__).resolve().parents[2]),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[1]),
        "runtime": runtime_versions(),
    }
    with open(f"{prefix}.metadata.json", "w") as output_file:
        json.dump(metadata, output_file, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
