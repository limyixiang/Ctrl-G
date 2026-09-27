"""Select saved HMM checkpoints using development-set log likelihood."""

from __future__ import annotations

import math
import re
from pathlib import Path


CHECKPOINT_DIRECTORY = re.compile(r"checkpoint-(\d+)$")


def read_latest_log_likelihoods(path: str | Path) -> dict[int, dict[str, float]]:
    """Read checkpoint metrics from the most recent run in a training log."""

    metrics: dict[int, dict[str, float]] = {}
    with open(path, encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            # train_hmm.py appends its argument dictionary before every run. Reset
            # here so reusing an output directory cannot select a stale checkpoint.
            if (
                line.startswith("{")
                and "'em_schedule'" in line
                and "'model_path'" in line
            ):
                metrics = {}
                continue

            columns = line.split("\t")
            if len(columns) != 3:
                continue
            try:
                checkpoint = int(columns[0])
                train_log_likelihood = float(columns[1])
                dev_log_likelihood = float(columns[2])
            except ValueError:
                continue
            if not math.isfinite(dev_log_likelihood):
                continue
            metrics[checkpoint] = {
                "train_log_likelihood": train_log_likelihood,
                "dev_log_likelihood": dev_log_likelihood,
            }

    if not metrics:
        raise ValueError(f"no checkpoint likelihood records found in {path}")
    return metrics


def saved_checkpoint_steps(model_path: str | Path) -> set[int]:
    """Return checkpoint numbers that have materialized directories."""

    root = Path(model_path)
    steps = set()
    for path in root.glob("checkpoint-*"):
        match = CHECKPOINT_DIRECTORY.fullmatch(path.name)
        if path.is_dir() and match:
            steps.add(int(match.group(1)))
    return steps


def select_best_checkpoint(log_path: str | Path, model_path: str | Path) -> dict:
    """Select the saved checkpoint with maximum development log likelihood."""

    log_path = Path(log_path)
    model_path = Path(model_path)
    metrics = read_latest_log_likelihoods(log_path)
    saved_steps = saved_checkpoint_steps(model_path)
    candidates = [
        {
            "checkpoint": checkpoint,
            "path": str((model_path / f"checkpoint-{checkpoint}").resolve()),
            **values,
        }
        for checkpoint, values in sorted(metrics.items())
        if checkpoint in saved_steps
    ]
    if not candidates:
        raise ValueError(
            f"no saved checkpoints in {model_path} have likelihood records in {log_path}"
        )

    # Prefer the later checkpoint when likelihoods are exactly tied.
    best = max(
        candidates,
        key=lambda item: (item["dev_log_likelihood"], item["checkpoint"]),
    )
    return {
        "selection_metric": "mean_dev_log_likelihood_per_sequence",
        "selection_rule": "maximum; latest checkpoint breaks exact ties",
        "training_log": str(log_path.resolve()),
        "candidates": candidates,
        "best_checkpoint": best["checkpoint"],
        "best_checkpoint_path": best["path"],
        "best_dev_log_likelihood": best["dev_log_likelihood"],
    }
