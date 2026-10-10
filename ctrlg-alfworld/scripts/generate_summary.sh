#!/usr/bin/env bash
set -euo pipefail

python ctrlg-alfworld/scripts/summarize_results.py --results results/alfworld/pair_4b --out results/alfworld/pair_4b/paired_summary.json

python ctrlg-alfworld/scripts/format_failed_episodes.py "results/alfworld/pair_4b/eval_decision_ctrlg.jsonl" --full --out results/alfworld/pair_4b/failed_episodes_ctrlg_full.md

python ctrlg-alfworld/scripts/format_failed_episodes.py "results/alfworld/pair_4b/eval_decision_ctrlg.jsonl" --out results/alfworld/pair_4b/failed_episodes_ctrlg.md

python ctrlg-alfworld/scripts/format_failed_episodes.py "results/alfworld/pair_4b/eval_decision_prompt.jsonl" --full --out results/alfworld/pair_4b/failed_episodes_prompt_full.md

python ctrlg-alfworld/scripts/format_failed_episodes.py "results/alfworld/pair_4b/eval_decision_prompt.jsonl" --out results/alfworld/pair_4b/failed_episodes_prompt.md
