# Ctrl-G on ALFWorld

This package evaluates one matched pair using the same model prompt, complete
trajectory, and byte-identical `templates/SKILLS.md`:

| Condition | Decision | Action |
|---|---|---|
| `decision_prompt` | unconstrained | unconstrained |
| `decision_ctrlg` | hard schema DFA | policy DFA + action HMM |

`info["admissible_commands"]` is never passed to either decoder and is never
shown in the prompt. It is retained only as `admissible_gt` after generation,
to score `action_was_admissible`. There is no symbolic state tracker and no
fact extraction from observations or prompt text.

## Skill contract

`SKILLS.md` contains three fenced YAML block types:

- `action`: named action templates.
- `decision-schema`: versioned ordered schemas for the six ALFWorld task types.
- `policy`: versioned declarative conditions and effects.

Loading fails on malformed YAML, duplicate names, unknown action templates,
unsupported schema types/operators, invalid bounds, or unsupported effects.
Both conditions receive the raw skill file; only `decision_ctrlg` compiles its
machine-readable blocks.

Decisions are canonical one-line flow-style YAML. The schemas retain the task
fields and search ledger, require exactly two independent `targets` records for
`puttwo`, and add:

```text
previous_action: ACTION | none | unknown
arrived_receptacle_state: open | closed | not_applicable | unknown
held_relation: none | target | non_target | unknown
```

These values are model claims. They are not checked against the real
trajectory. Search-ledger syntax is guaranteed in `decision_ctrlg`; factual
correctness, overlap, and revisit tracking are not.

The initial policy set is:

1. Priority 100: declared closed arrival at an entity requires `open {at}`.
2. Priority 90: declared non-target held object requires `move {held} to {at}`.
3. Priority 20: every generated `take` uses the lexical type in `target_type`.
   Thus `cup` and `mug` remain distinct, although object presence is not known.
4. Priority 10: the action cannot exactly equal declared `previous_action`,
   except `none` and `unknown`.

The highest-priority matching required-action rule shadows lower-priority
rules. Otherwise restrictions intersect. An unbound or empty language falls
back to the generic action grammar and records `policy_fallback_reason`; it
never falls back to environment admissible commands.

## Generation

Strict generation has five phases:

1. Native thought generation.
2. Decision body plus `</decision>` under a hard tokenizer DFA.
3. Guaranteed-valid YAML parsing.
4. Per-turn policy-language compilation from declared fields.
5. Action body plus `</action>` under the policy DFA and action HMM.

The default decision/action caps are 512/32 tokens. Both closing tags are
inside their DFA spans; the action HMM has EOS as its fixed suffix. The hard
logits processor tracks DFA state and masks paths that cannot reach acceptance
within the remaining budget.

## HMM data collection

Training samples match the strict evaluator's prefix distribution: a hard-DFA
decision is followed by an unconstrained sampled action. Both the in-process HF
backend and vLLM 0.10.2 are supported; vLLM receives the decision language as
`guided_regex`.

```bash
python ctrlg-alfworld/scripts/run_rollouts.py \
  --backend vllm \
  --model Qwen/Qwen3.5-9B \
  --tokenizer Qwen/Qwen3.5-9B \
  --num_episodes 100 \
  --max_decision_tokens 512 \
  --max_action_tokens 32 \
  --max_hmm_sequence_tokens 640 \
  --out results/alfworld/9b_policy_hmm
```

Every structurally valid, token-exact sample is eligible regardless of action
admissibility. Dataset/checkpoint metadata records the skill hash, schema and
policy versions, model/tokenizer, hard-decision collection regime, and
no-oracle filtering flag.

Prepare Ctrl-G data with:

```bash
python ctrlg-alfworld/scripts/build_hmm_data.py \
  --samples results/alfworld/9b_policy_hmm/samples.jsonl \
  --tokenizer Qwen/Qwen3.5-9B \
  --output_dir distillation/alfworld
```

## Evaluation

Run both cells with the same model, skills, seed, split, and episode manifest:

```bash
python ctrlg-alfworld/scripts/run_eval.py \
  --model Qwen/Qwen3.5-9B \
  --condition decision_prompt \
  --out results/alfworld/pair

python ctrlg-alfworld/scripts/run_eval.py \
  --model Qwen/Qwen3.5-9B \
  --condition decision_ctrlg \
  --hmm distillation/alfworld/checkpoint \
  --out results/alfworld/pair

python ctrlg-alfworld/scripts/summarize_results.py \
  --results results/alfworld/pair
```

The paired summary reports Ctrl-G minus prompt-only deltas for success,
post-hoc admissibility, decision-schema validity, action-grammar validity,
policy adherence/fallback, latency, and tokens. Step records include parsed
decision fields, activated/shadowed rules, evaluability/satisfaction, HMM
application/skip reason, and post-hoc admissibility. Summaries reject mismatched
model, prompt settings, skills, episode manifest, policy version, or HMM
provenance.

## Tests

```bash
pytest -q ctrlg-alfworld/tests
```

The CPU suite covers DSL rejection, all six schemas, exact two-target
`puttwo` decisions including shared locations, character/token DFA behavior,
closing tags, token-budget completion, policy priority and fallback, cup/mug
separation, prompt identity, no-oracle decoder routing, collection eligibility,
vLLM guided regex, and paired-summary provenance.
