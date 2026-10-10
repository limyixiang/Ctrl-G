# Ctrl-G on ALFWorld

This package evaluates one matched pair using the same model prompt, complete
trajectory, and byte-identical `templates/SKILLS.md`:

| Condition | State | Action |
|---|---|---|
| `state_prompt` | unconstrained | unconstrained |
| `state_ctrlg` | hard schema DFA | policy DFA + action HMM |

`info["admissible_commands"]` is never passed to either decoder and is never
shown in the prompt. It is retained only as `admissible_gt` after generation,
to score `action_was_admissible`. There is no symbolic state tracker and no
fact extraction from observations or prompt text.

## Skill contract

`SKILLS.md` contains three fenced YAML block types:

- `action`: named action templates.
- `state-schema`: versioned ordered schemas for the six ALFWorld task types.
- `policy`: versioned declarative conditions and effects.

Loading fails on malformed YAML, duplicate names, unknown action templates,
unsupported schema types/operators, invalid bounds, or unsupported effects.
Both conditions receive the raw skill file; only `state_ctrlg` compiles its
machine-readable blocks.

State summaries are canonical one-line flow-style YAML. The schemas retain the task
fields and search ledger, require exactly two independent `targets` records for
`puttwo`, and add:

```text
previous_action: ACTION | none | unknown
arrived_receptacle_state: open | closed | not_applicable | unknown
held_relation: none | target | non_target | unknown
```

These values are model claims. They are not checked against the real
trajectory. Search-ledger syntax is guaranteed in `state_ctrlg`; factual
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
2. State body plus `</state>` under a hard tokenizer DFA.
3. Guaranteed-valid YAML parsing.
4. Per-turn policy-language compilation from declared fields.
5. Action body plus `</action>` under the policy DFA and action HMM.

The default state/action caps are 512/32 tokens. Both closing tags are
inside their DFA spans; the action HMM has EOS as its fixed suffix. The hard
logits processor tracks DFA state and masks paths that cannot reach acceptance
within the remaining budget.

## HMM data collection

Training samples use a hard-DFA state followed by an unconstrained sampled
action. Both the in-process HF backend and vLLM 0.12.0 or newer are supported;
vLLM receives the state language through `structured_outputs.regex`.

```bash
python ctrlg-alfworld/scripts/run_rollouts.py \
  --backend vllm \
  --model Qwen/Qwen3.5-9B \
  --tokenizer Qwen/Qwen3.5-9B \
  --num_episodes 100 \
  --max_state_tokens 512 \
  --state_max_attempts 5 \
  --state_retry_temperature 0.2 \
  --max_action_tokens 32 \
  --max_hmm_sequence_tokens 640 \
  --out results/alfworld/9b_policy_hmm
```

Every structurally valid, token-exact sample is eligible regardless of action
admissibility. Dataset/checkpoint metadata records the skill hash, schema and
policy versions, model/tokenizer, hard-state collection regime, and
no-oracle filtering flag.

vLLM state collection makes one greedy structured-output attempt, then up
to four deterministic low-temperature attempts with distinct seeds. If all
five attempts fail to complete the schema, the terminal sample is recorded as
non-distillable and that episode ends; valid samples from its earlier steps are
retained. `metadata.json` reports state retry, recovery, and exhausted-
episode rates for the model run.

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
  --condition state_prompt \
  --out results/alfworld/pair

python ctrlg-alfworld/scripts/run_eval.py \
  --model Qwen/Qwen3.5-9B \
  --condition state_ctrlg \
  --hmm distillation/alfworld/checkpoint \
  --out results/alfworld/pair

python ctrlg-alfworld/scripts/summarize_results.py \
  --results results/alfworld/pair
```

The paired summary reports Ctrl-G minus prompt-only deltas for success,
post-hoc admissibility, state-schema validity, action-grammar validity,
policy adherence/fallback, latency, and tokens. Step records include parsed
state fields, activated/shadowed rules, evaluability/satisfaction, HMM
application/skip reason, and post-hoc admissibility. Summaries reject mismatched
model, prompt settings, skills, episode manifest, policy version, or HMM
provenance.

## Migration from decision blocks

This is a breaking rename: model output now uses `<state>...</state>`,
skill schemas use `state-schema`, and Python APIs and saved-data keys use
`state` in place of `decision`. The YAML body, field order, policies, and token
budgets retain their existing behavior. The inner `state` field in clean,
heat, and cool tasks still describes the object's transformation status.

Update launch commands and external consumers using these names:

| Previous name | Current name |
|---|---|
| `decision_prompt`, `decision_ctrlg` | `state_prompt`, `state_ctrlg` |
| `--max_decision_tokens` | `--max_state_tokens` |
| `--decision_max_attempts` | `--state_max_attempts` |
| `--decision_retry_temperature` | `--state_retry_temperature` |
| `DECISION_MAX_ATTEMPTS`, `DECISION_RETRY_TEMPERATURE` | `STATE_MAX_ATTEMPTS`, `STATE_RETRY_TEMPERATURE` |
| `decision`, `model_decision`, `selected_model_decision` | `state`, `model_state`, `selected_model_state` |
| `decision_fields`, `decision_schema_valid` | `state_fields`, `state_schema_valid` |

New artifacts identify their prompt format as
`state_with_persistent_history_no_oracle_v1`. Use a new collection output
directory, recollect samples, and retrain the action HMM before evaluating
`state_ctrlg`. Decision-format runs cannot be resumed with this code, and
their HMM checkpoints are rejected. Renaming metadata cannot convert the
old token sequences to the new tags.

`scripts/submit_all.sh` starts fresh collections in `9b_state_rollout` and
`4b_state_rollout` under `results/alfworld/`. Set `RESUME=1` only when continuing
an existing state-format collection in those directories.

## Tests

```bash
pytest -q ctrlg-alfworld/tests
```

The CPU suite covers DSL rejection, all six schemas, exact two-target
`puttwo` states including shared locations, character/token DFA behavior,
closing tags, token-budget completion, policy priority and fallback, cup/mug
separation, prompt identity, no-oracle decoder routing, collection eligibility,
vLLM guided regex, and paired-summary provenance.
