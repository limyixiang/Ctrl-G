# ctrlg-alfworld

Constrained tool generation on the AlfWorld benchmark with
[Ctrl-G](https://github.com/joshuacnf/Ctrl-G) (Zhang et al., NeurIPS 2024).

The agent thinks freely over past actions/observations, then emits one action
as `<tool>action</tool>`. Ctrl-G constrains ONLY the tool span: at every step,
SKILLS.md templates + preconditions are grounded against a tracked symbolic
state, and the resulting finite set of admissible actions is compiled into a
token-trie DFA for Ctrl-G's `ConstraintLogitsProcessor` (HMM x DFA).

## Layout

```
SKILLS.md                    domain skills: prompt content AND constraint source
configs/base_config.yaml     AlfWorld env config (from ReAct)
few_shot/alfworld_3prompts.json  ReAct few-shot prompts (converted at load time)
src/ctrlg_alfworld/
  skills.py                  SKILLS.md -> Skill specs (template + preconditions)
  state.py                   symbolic state tracker (sound-but-incomplete pruning)
  constraints.py             finite action set -> token-trie DFA (Ctrl-G format)
  prompts.py                 system prompt / transcript / ReAct conversion
  backends.py                HF generation: unconstrained + 2-phase constrained
  agent_loop.py              episode loop (rollouts + eval share it)
scripts/
  smoke_test.py              CPU-only sanity check (run this first)
  run_rollouts.py            unconstrained train rollouts -> distill prompt dump
  run_eval.py                eval_out_of_distribution, per-task-type table
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # read its comments (transformers@main for Qwen3.5)
git clone https://github.com/joshuacnf/Ctrl-G.git && pip install -e Ctrl-G
pip install alfworld && alfworld-download # game data -> $ALFWORLD_DATA
python scripts/smoke_test.py             # no GPU needed
```

## Pipeline

1. **Baseline + distillation data** (unconstrained, needs GPU):

   ```bash
   python scripts/run_rollouts.py --model Qwen/Qwen3.5-9B \
       --num_episodes 500 --out out/rollouts_train
   ```

   Gives the unconstrained success rate (baseline #1) and
   `out/rollouts_train/distill_prompts.json` - prompt strings of the form
   `context + thought + '<tool>'`, matching the constrained call exactly.

2. **Distill the HMM** with Ctrl-G's `distillation_vllm/` pipeline, feeding it
   `distill_prompts.json`:

   ```bash
   cd Ctrl-G/distillation_vllm
   ./launch_vllm_model.sh 0 Qwen/Qwen3.5-9B            # own terminal
   ./sample_data_vllm.sh Qwen/Qwen3.5-9B alfworld \
       ../../out/rollouts_train/distill_prompts.json ./out False
   # then LVD init + EM training - follow train_hmm.ipynb Steps 3-4
   ```

   Sampling params: cap `max_new_tokens` low (~32) - responses are
   `action</tool><eos>`, not free text. Start with **4096 hidden states**:
   Qwen3.5's 248,320-token vocab makes the emission matrix `states x vocab`
   fp32; at 32,768 states EM will not fit on a single 80GB H100.

3. **Constrained evaluation**:

   ```bash
   python scripts/run_eval.py --model Qwen/Qwen3.5-9B --mode unconstrained
   python scripts/run_eval.py --model Qwen/Qwen3.5-9B --mode constrained \
       --hmm path/to/hmm_checkpoint
   python scripts/run_eval.py --model Qwen/Qwen3.5-9B --mode constrained-oracle \
       --hmm path/to/hmm_checkpoint   # env admissible_commands as allowed set
   ```

   The eval also reports the **admissible-action rate** (fraction of emitted
   actions the env considered admissible) - the direct measure of what
   constraining buys, alongside task success.

## Experimental conditions worth running

- unconstrained (baseline)
- naive DFA masking without the HMM (ablation: does Ctrl-G's HMM lookahead
  pick better actions than plain masking? implement by zeroing the HMM term
  or via any structured-output engine)
- constrained (SKILLS.md preconditions + tracker)
- constrained-oracle (admissible_commands; upper bound, isolates tracker error)

## Design notes / gotchas

- **Soundness convention**: the tracker only prunes provably-illegal actions;
  unknown facts never exclude anything. `StepRecord.allowed_actions` vs
  `admissible_gt` in the logs quantifies tracker precision/recall per step.
- **Tokenization boundaries**: action spans are tokenized as continuations of
  the exact seam text (`tokenize_continuation`), not standalone - BPE merges
  across `<tool>` would otherwise corrupt the trie.
- **HMM/model coupling**: the HMM must be distilled from the same base model
  (same vocab). Suffix is `</tool>` + EOS; distillation targets must end the
  same way.
- **Qwen3.5-9B risks**: hybrid DeltaNet attention + transformers@main; beam
  search cache reordering is the least-tested path - if it misbehaves, set
  `GenConfig.do_sample=True` (sampling avoids beam cache reordering), or fall
  back to Qwen3-8B (standard transformer). Validate the whole pipeline on
  Qwen3-1.7B-Base first (the Ctrl-G distillation tutorial's own example).
- Native Qwen thinking mode is disabled by default (`enable_thinking=False`);
  the harness manages reasoning via its own plain-text `<think>` convention so
  prompt strings match between HF generation and vLLM distillation sampling.
