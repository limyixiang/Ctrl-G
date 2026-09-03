# Ctrl-G–ALFWorld matched two-condition experiment

This package answers one question:

> Within the same persistent-decision-memory agent, what is the effect of
> adding a matched Ctrl-G HMM to the hard admissible-action DFA?

| Condition | Persistent decision memory | Decoder |
|---|---:|---|
| `decision_dfa` | yes | hard DFA |
| `decision_dfa_hmm` | yes | hard DFA + decision-format HMM |

Both conditions generate a `<decision>...</decision>` before each action and
replay each nonempty prior decision immediately before its corresponding prior
`<action>...</action>` and observation. Native/hidden thinking is never
replayed. Prompts, DFA, base model, generation settings, seed, and ordered
episode manifest are held fixed; only HMM use changes. This experiment does not
estimate whether adding decision blocks improves performance.

Every DFA is constructed directly from the current TextWorld
`info["admissible_commands"]`. The admissible list is hidden from the model by
default. A matched prompt-visible experiment can be enabled explicitly during
both sample collection and evaluation. No symbolic `state.py` tracker is used.

## Distillation boundary

The HMM is trained only on matching persistent-decision-format rollouts:

```text
base-LLM-only context: system + user prompt + native thinking
HMM prefix:           generated post-think text through <action>
DFA span:             action body
HMM suffix:           </action> + EOS
```

Original token IDs are retained. Malformed, truncated, or non-token-aligned
samples are logged but excluded. The state-group train/dev split keeps all
samples from one `(episode, step)` on the same side of the split.

## 1. Collect decision-format samples

```bash
export ALFWORLD_DATA=/path/to/alfworld_data
python ctrlg-alfworld/scripts/smoke_environment.py

python ctrlg-alfworld/scripts/run_rollouts.py \
  --backend vllm \
  --model Qwen/Qwen3.5-9B \
  --num_episodes 100 \
  --samples_per_state 4 \
  --temperature 0.7 \
  --out out/alfworld_hmm_samples
```

Add `--show_admissible_actions` to collect a separate prompt-visible dataset.
The setting is written to every sample and to collection metadata; the dataset
builder rejects mixtures of prompt-hidden and prompt-visible samples.

Only decision-format samples are collected. The metadata reports eligible
counts and exclusion reasons. The vLLM backend requires exact returned token
IDs and fails rather than retokenizing generated text.

## 2. Build one HMM dataset

```bash
python ctrlg-alfworld/scripts/build_hmm_data.py \
  --samples out/alfworld_hmm_samples/samples.jsonl \
  --tokenizer Qwen/Qwen3.5-9B \
  --model Qwen/Qwen3.5-9B \
  --output_dir out/alfworld_hmm_data \
  --dataset alfworld_actions \
  --save_embeddings
```

This produces `alfworld_actions.lvd`, `.lvd.embeddings`, `.train.*`, `.dev`,
and `.metadata.json` for the one matched HMM.

## 3. Train the matched HMM

```bash
DATA_DIR=out/alfworld_hmm_data \
OUTPUT=out/alfworld_hmm_model \
sbatch ctrlg-alfworld/slurm/train_hmm.sh
```

The output includes `train.log`, checkpoints, and `held_out_fit.json` for the
decision-format dev set. LVD and EM initialization use the configured seed.

## 4. Run the matched evaluation pair

```bash
HMM=out/alfworld_hmm_model/checkpoint-400 \
sbatch ctrlg-alfworld/slurm/eval_grid.sh
```

Or run the HMM condition directly:

```bash
python ctrlg-alfworld/scripts/run_eval.py \
  --model Qwen/Qwen3.5-9B \
  --hmm out/alfworld_hmm_model/checkpoint-400 \
  --condition decision_dfa_hmm \
  --num_episodes 134 \
  --seed 42 \
  --out out/alfworld_pair
```

Omit `--hmm` for `decision_dfa`. After both runs:

```bash
python ctrlg-alfworld/scripts/summarize_results.py \
  --results out/alfworld_pair \
  --out out/alfworld_pair/effects.json
```

The summarizer rejects differences in prompt/generation settings, source hash,
seed, ordered episode manifest, or other paired controls. It reports
`decision_dfa_hmm - decision_dfa` for every metric.

### Prompt-visible matched experiment

Train a separate HMM from samples collected with
`--show_admissible_actions`, then pass the same flag to both evaluation cells.
The Slurm launchers expose this as an environment toggle:

```bash
SHOW_ADMISSIBLE_ACTIONS=1 \
OUTPUT=results/alfworld/actions_shown/hmm_samples \
sbatch ctrlg-alfworld/slurm/collect_hmm_samples.sh

SHOW_ADMISSIBLE_ACTIONS=1 \
HMM=results/alfworld/actions_shown/hmm_model/checkpoint-N \
OUTPUT=results/alfworld/actions_shown/eval \
sbatch ctrlg-alfworld/slurm/eval_grid.sh
```

Keep prompt-hidden and prompt-visible samples, HMM checkpoints, and evaluation
outputs in separate directories. New locally trained checkpoints carry dataset
metadata, and evaluation rejects a checkpoint whose prompt regime conflicts
with the requested flag. Legacy checkpoints without this metadata remain
supported.

## Focused tests

```bash
cd ctrlg-alfworld
PYTHONPATH=src ../.venv/bin/python -m unittest \
  tests.test_experiment tests.test_eval_routing tests.test_prompts \
  tests.test_distillation tests.test_summarize_results tests.test_agent_loop
```
