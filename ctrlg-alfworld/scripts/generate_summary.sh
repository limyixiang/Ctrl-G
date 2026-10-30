#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 {4b|9b} [k4 k8 prompt-only]"
    echo "If no arms are specified, all three are generated."
    echo "bash ctrlg-alfworld/scripts/generate_summary.sh 9b k4 k8"
    echo "bash ctrlg-alfworld/scripts/generate_summary.sh 4b prompt-only"
    echo "bash ctrlg-alfworld/scripts/generate_summary.sh 9b  # All arms"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model="${1:-}"
case "$model" in
    4b|9b) shift ;;
    *)
        usage >&2
        exit 1
        ;;
esac

arms=("$@")
if [[ ${#arms[@]} -eq 0 ]]; then
    arms=(k4 k8 prompt-only)
fi

# Validate every arm before generating files.
for arm in "${arms[@]}"; do
    case "$arm" in
        k4|k8|prompt-only|prompt_only) ;;
        *)
            echo "Error: unknown arm '$arm'. Use k4, k8, or prompt-only." >&2
            exit 1
            ;;
    esac
done

formatter="ctrlg-alfworld/scripts/format_failed_episodes.py"
results_root="results/alfworld/grid_${model}"

for arm in "${arms[@]}"; do
    case "$arm" in
        k4|k8)
            directory="${results_root}/${arm}"
            mode="ctrlg"
            ;;
        prompt-only|prompt_only)
            directory="${results_root}/prompt_only"
            mode="prompt"
            ;;
    esac

    input="${directory}/eval_state_${mode}.jsonl"
    output="${directory}/failed_episodes_${mode}"

    if [[ ! -f "$input" ]]; then
        echo "Error: input file not found: $input" >&2
        exit 1
    fi

    echo "Generating reports for ${model}/${arm}..."

    python "$formatter" "$input" --full --out "${output}_full.md"
    python "$formatter" "$input" --out "${output}.md"
done