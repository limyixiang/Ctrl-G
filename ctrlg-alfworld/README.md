# Skill2Act: ALFWorld TextWorld Prototype

This package implements the ALFWorld prototype for
[Skill2Act: User-Guided Robot Decision-Making through Constrained State and Action Generation](../README.md).
The project investigates how users or programmers can express supported
constraints in a `SKILLS.md` document and enforce them during language-model-based
state and action generation using Ctrl-G.

## Objective and scope

The objective is to connect user-editable state schemas, action templates, and
state-dependent behavioural rules to generation constraints. In the constrained
condition, each decision follows this sequence:

1. Read the task, skill document, and observation/action history.
2. Generate a structured state representation under the declared state schema.
3. Compile action constraints from the generated state and applicable policy rules.
4. Generate an action under those constraints and execute it in the environment.

Evaluation uses **ALFWorld's text-based (TextWorld) environment**, configured as
`AlfredTWEnv`, with textual observations and action commands. It investigates
high-level decision-making relevant to robotics. Physical robot deployment,
visual perception, and low-level motion control are outside the current scope.

Enforcement applies to supported machine-readable specifications; free-form
instructions are also provided to the model as prompt guidance. A schema-valid
state can still describe the environment incorrectly, so an action that follows
the rules for that generated state can still fail in the environment. Evaluation
therefore reports schema validity, policy adherence, action admissibility, and
task success separately. Policy priorities and fallback behaviour are documented
in the [skill contract](#skill-contract).

## Evaluation design

This package evaluates one matched pair using the same prompt construction,
complete action/observation history, and byte-identical `templates/SKILLS.md`:

| Condition | State | Action |
|---|---|---|
| `state_prompt` | unconstrained | unconstrained |
| `state_ctrlg` | schema DFA + joint HMM | policy DFA + same joint HMM |

`info["admissible_commands"]` is never passed to either decoder and is never
shown in the prompt. It is retained only as `admissible_gt` after generation,
to score `action_was_admissible`. There is no symbolic state tracker and no
fact extraction from observations or prompt text.

The LLM prompt includes the task description, skill document, initial
observation, all previous executed actions and their resulting observations,
and the current observation. Previous generated state blocks and native
thinking are logged but are not replayed in the prompt. Each turn's state is
generated from the action/observation history. Both conditions use the same
history format and receive identical prompt content for an identical trajectory.

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

The current policy set contains eight rules:

1. Priority 100: forbid the declared previous action, except `none` and `unknown`.
2. Priority 90: restrict pickups to `target_type`, forbid all pickups when an
   object is declared held, and forbid navigation to the declared current location.
3. Priority 80: a declared held target with known `at != dest` requires `go to {dest}`.
4. Priority 70: declared closed arrival at an entity requires `open {at}`.
5. Priority 60: a declared non-target held object requires `move {held} to {at}`.
6. Priority 50: a declared held target with known `at == dest` requires placement.

The compiler follows the supplied blocks exactly. Delivery currently has no
task, transformation-state, or destination-access guard beyond the other rules;
it is independent of the declared phase.

Every matching restriction intersects before required actions are considered,
regardless of priority. Required actions are tried in descending priority order
(ties use policy name); forbidden or unbound requirements are rejected. The
highest-priority permitted requirement selects a singleton language. If none
qualifies, generation uses the remaining restricted grammar. Invalid restrictions
or an empty restricted language raise `PolicyCompilationError`, without an
unrestricted fallback or environment admissible commands.

`equals_field` and `not_equals_field` compare declared fields and do not match
missing/unknown operands. Existing `not_in` conditions retain their behavior for
missing fields. Field references are validated against the union of task schemas.
Policy priorities must be integers in 0..100.

`forbid_action` accepts partial bindings; omitted placeholders are wildcards,
but explicitly unbound arguments are errors. The compiler also supports optional
`restrict_action_bindings`: numbered entities fix arguments, `none` excludes the
action family, and unknown/missing values leave only that argument unconstrained.
This effect is not present in the current skill file. Rejected required policies
and their reasons are recorded separately from active restrictions and shadowed
lower-priority requirements. DFA caching includes all resolved restrictions and
the selected required action.

## Generation

Strict generation has five phases:

1. Native thought generation.
2. State body plus `</state>` under the schema DFA and context-conditioned HMM, including a bounded generic action continuation in its lookahead.
3. Guaranteed-valid YAML parsing.
4. Per-turn policy-language compilation from declared fields.
5. Action body plus `</action>` under the policy DFA and the same HMM, followed by EOS.

The default state/action caps are 512/32 tokens. Head sampling and action
sampling are independent controls with the same temperatures and beam
settings across arms. Reported generated tokens exclude controller-inserted
phase delimiters and count the treatment's sampled terminal EOS. State/action
latencies include DFA compilation and HMM lookahead setup. Both closing tags are
inside their DFA spans; the action HMM has EOS as its fixed suffix. The hard
logits processor tracks DFA state and masks paths that cannot reach acceptance
within the remaining budget.

## HMM data collection

Collection uses natural, unmasked state and action draws in both HF and vLLM.
Native thinking is greedy; states/actions use the collection temperature (0.7
by default). Every attempt is saved, including invalid YAML, grammar failures,
inadmissible actions, unsuccessful episodes and truncated outputs.

At a step, collection makes at most `--collection_max_attempts` independent
whole-turn draws (default 5) from the same prompt and environment state. It
executes the first nonempty, complete action that matches the public action
grammar and has an exact token span ending in `</action>`. State-schema validity
and natural state completion are not execution requirements. If a state fails
to close, the collector inserts only `</state>` and proceeds to unmasked action
generation; it does not invent or fix state fields. Environment admissibility
and actual progress never determine selection. If that action fails in the
environment, its feedback enters history. If no attempt is executable, the episode ends
and all its observed attempts remain available for training. Evaluation uses
one turn per step in both arms, without collection retries.

Attempt records and history candidate audits log state-schema validity/errors,
state repair/truncation, action-grammar validity, execution selection and explicit
action rejection reasons. Original state tokens, synthetic state-closer tokens
and generated action tokens are stored separately in each attempt record.

Collection saves exact observed targets in `samples.jsonl` and full observable
episode history in `history.jsonl`. It does not choose an HMM history window or
context token budget. Keep both files to reuse the rollouts for multiple builds.

At dataset-build time, each attempt becomes a separate HMM example with the
same step context:

```text
C_t <state>STATE</state><action>ACTION</action> EOS
```

`C_t` is compact deterministic JSON containing the task, initial observation,
the last `--hmm_context_steps` executed action/observation turns (default 1),
and the current observation. Current feedback is deduplicated against the last
turn. It contains no old state summaries, native thinking, oracle commands or
rejected attempts. Its budget is `--max_hmm_context_tokens` (default 1024):
oldest complete turns are dropped first, then the initial observation is
shortened. These options belong to `build_hmm_data.py`, exposed by
`HMM_CONTEXT_STEPS` and `MAX_HMM_CONTEXT_TOKENS` in `slurm/build_hmm_data.sh`.
The builder joins each attempt to its earlier executed turns from `--history`
(default: `history.jsonl` beside `--samples`), never to later feedback or
rejected attempts. Dataset building and evaluation must use the same settings.
A budget too small for the task/current observation is rejected.

Build metadata reports used context token mean, median, p95, p99, maximum,
mean budget utilization and truncation rate, counting each step once. Span
files also record retained history-turn counts and truncation flags. Evaluation
logs retained token counts and truncation per step. Original token counts
before truncation are not recorded.

This serialization is **HMM input only**. Both LLM arms keep the same full
prompt described above. At state inference the HMM consumes `C_t + <state>`;
at action inference it consumes `C_t + generated state + <action>`. No extra
context text is added to either LLM prompt. State generation stops at its
closing tag, while the HMM looks ahead through a generic action and EOS.
The declared state then determines the actual policy action DFA.

Truncated examples contain only their observed target prefix. They receive
no invented closers or EOS; an actual model EOS is retained even if the
protocol is incomplete. For a synthetically closed state, the HMM target ends
before the inserted closer: the subsequent action remains in the raw record
and may be executed, but is excluded from that joint HMM sequence. Its
`hmm_action_exclusion_reason` is `generated_after_synthetic_state_close`.
Only a naturally completed state/action protocol gets an appended EOS.
`-2` is right padding with no evidence or EM counts, distinct from `-1`
token dropout. Attempts with no sampled tokens before the HMM target cutoff
are saved but have no HMM training target. This includes an empty repaired
state even if its later action is executable. Training gives each observed attempt one sequence
likelihood term, so steps requiring more retries contribute more examples.
Retries and all steps from the same game instance stay in one train/dev split.

LVD extracts context embeddings from the explicit serialization and target
embeddings from the original LLM prompt/thinking. The deployed HMM conditions
on explicit tokens, never on the teacher's hidden states. The default HMM size
remains 4096; longer context does not automatically require more hidden states.
Compare capacity or context-window changes using held-out output-conditional
state/action NLL and paired task success. Checkpoint selection uses mean
`log P(observed target | C_t)` per attempt; joint sequence likelihood is also
reported as a diagnostic.

The three-column training log (`checkpoint`, `train`, `dev`) now uses this same
output-conditional score for both splits, including the initial checkpoint.
Training evaluation uses the first `min(dev rows, train rows)` attempts in numeric
chunk order, fixed across EM updates, resumes, seeds, and process counts for the
same dataset. The header records the metric and actual evaluation counts. EM still
fits joint sequence likelihood; joint held-out fit is reported by
`evaluate_hmm_fit.py`. Both train and dev span files are required for conditional
scoring. Legacy datasets without either file use explicitly labelled joint scores
on both splits; old logs without metric headers retain their legacy labels.

Start fresh data and checkpoint directories:

```bash
MODEL=Qwen/Qwen3.5-9B OUTPUT=results/alfworld/9b_context_rollout_v1 \
  sbatch ctrlg-alfworld/slurm/collect_hmm_samples.sh

bash ctrlg-alfworld/slurm/build_hmm_data.sh 9B 1

# Reuse the same raw rollouts with a different history window.
bash ctrlg-alfworld/slurm/build_hmm_data.sh 9B 3

MODEL=Qwen/Qwen3.5-9B DATA_DIR=results/alfworld/hmm_context_data_9b_k1_b1024_v1 \
  OUTPUT=results/alfworld/hmm_context_model_9b_v1 \
  sbatch ctrlg-alfworld/slurm/train_hmm.sh
```

The build script takes the model size (case insensitive, default `9B`) and
HMM context steps (default `1`), then submits itself with `sbatch`. For example,
`bash ctrlg-alfworld/slurm/build_hmm_data.sh 4B 8` uses `Qwen/Qwen3.5-4B`,
reads `results/alfworld/4b_context_rollout_v1/samples.jsonl` and its sibling
`history.jsonl`, and writes `results/alfworld/hmm_context_data_4b_k8_b1024_v1`.
Its job name is `alfworld-lvd-4b-k8-b1024`, with logs in
`logs/alfworld-lvd-4b-k8-b1024_JOB_ID.{out,err}`. Use `bash` for automatic
job/log naming. Environment overrides remain available for `MODEL`,
`MODEL_SIZE`, `HMM_CONTEXT_STEPS`, `SAMPLES`, `HISTORY`, `OUTPUT`, `JOB_NAME`,
and `MAX_HMM_CONTEXT_TOKENS`; the token budget also appears in output and job names.
An explicit `MODEL` takes precedence over the model size, and names follow its
trailing size (for example, `Qwen/Qwen3.5-4B` gives `4b`).

Submit each stage after the prior stage completes, or use a Slurm
`--dependency=afterok:JOB_ID`. The build stage keeps all observed attempts and
writes explicit lengths/completion flags and phase boundaries alongside
train/dev/LVD tensors. LVD embedding storage is preallocated and bounded by
`--lvd_embedding_budget_gib` (default 32 GiB); the effective initialization
subset size is recorded. All attempts remain in the EM dataset. The training
job copies data provenance to checkpoints.
Each build uses the same game split for a fixed seed and keeps observed target
tokens unchanged. Context tensors and LVD embeddings are rebuilt for the chosen
window. Older unmasked `context_state_action_censored_v1` collections can also
be rebuilt when their original `history.jsonl` is available; their saved context
is replaced. Old masked or incompatible samples/checkpoints are rejected;
editing metadata cannot convert their token sequences.

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
  --hmm results/alfworld/hmm_context_model_9b_v1/checkpoint-112 \
  --out results/alfworld/pair

python ctrlg-alfworld/scripts/summarize_results.py \
  --results results/alfworld/pair
```

The summary supports both a flat matched pair and the `k8/`, `k4/`, and
`prompt_only/` grid. Each condition reports episode/success/failure counts,
total actions and inadmissible actions, actions per episode, token totals and
per-episode costs, generation latency, and constraint diagnostics. Ctrl-G minus
prompt-only deltas remain in `ctrlg_effect`.

When sibling `eval_<condition>.jsonl` files are available, the report also includes:

- Mean, median, nearest-rank p90, minimum, and maximum action counts, separately
  for all, successful, and failed episodes.
- Success within 10, 20, 30, and `max_steps` actions (budgets above `max_steps`
  are omitted). Each success rate uses all evaluated episodes as its denominator.
- Failed episodes that reached the action limit and episodes containing an
  inadmissible action. Reaching the limit is not treated as a recorded timeout.
- Per-task success, action usage, and token/generation costs.
- Paired Ctrl-G-only successes, baseline-only successes, joint successes/failures,
  and action differences on episodes both methods solve.

Generation latency is head plus action latency; head already includes thought
and state generation. It excludes environment time and other overhead.
Undefined averages/rates are `null`. Missing trace files leave
`episode_details_available` false, `episode_statistics` and the affected
`paired_comparisons` entries null, while retaining available aggregate metrics
and per-task success counts. Present traces must match the summary's ordered
manifest, episode/action/success counts, and success rate. The output retains
the manifest hash but omits gamefile lists and full trajectories.

Step records include parsed
state fields, activated/shadowed rules, evaluability/satisfaction, HMM
application/skip reason separately for state and action, context token count/shortening, and post-hoc admissibility. Summaries reject mismatched
model, prompt settings, skills, episode manifest, policy version, or HMM
provenance.

To test policy changes with a frozen HMM, pass `--allow-skills-mismatch` to
`run_eval.py` for `state_ctrlg` and to `summarize_results.py`. This permits only
the HMM training skill hash to differ from the evaluation skill hash. Both
evaluation conditions must still use byte-identical updated skills, and all
other provenance checks remain active. Evaluation summaries retain the original
`hmm_provenance`, the current `skills_sha256`, and the override setting; the
combined report adds `skills_validation` with the evaluation hash, training
hash per HMM arm, and whether each matches. Checkpoint metadata is never rewritten.

```bash
python ctrlg-alfworld/scripts/run_eval.py \
  --model Qwen/Qwen3.5-9B --condition state_prompt \
  --skills ctrlg-alfworld/templates/SKILLS.md --out results/alfworld/policy_trial

python ctrlg-alfworld/scripts/run_eval.py \
  --model Qwen/Qwen3.5-9B --condition state_ctrlg \
  --hmm results/alfworld/hmm_context_model_9b_v1/checkpoint-112 \
  --skills ctrlg-alfworld/templates/SKILLS.md --allow-skills-mismatch \
  --out results/alfworld/policy_trial

python ctrlg-alfworld/scripts/summarize_results.py \
  --results results/alfworld/policy_trial --allow-skills-mismatch
```

The Slurm evaluation launchers accept `ALLOW_SKILLS_MISMATCH=1` to forward the
override and `SKILLS=/path/to/SKILLS.md` to select a skill file. Use this for
policy edits over the existing state schemas and action templates. The override
does not verify structural equivalence of edited schemas or action templates;
schema/policy version checks remain active. Schema, action-format, or serialization
changes should use an appropriately rebuilt dataset and HMM.

## Format and migration

The LLM prompt identifier remains `state_with_persistent_history_no_oracle_v1`:
`persistent_history` means executed action/observation history, with no replay
of generated states or native thinking. The independent HMM identifiers are
`context_state_action_censored_v1` and `observable_trajectory_json_v1`.

`scripts/submit_all.sh` starts fresh collections in `9b_context_rollout_v1`
and `4b_context_rollout_v1`. New raw samples use
`unmasked_state_action_attempt_v1`. Resume checks enforce the raw serialization,
sampling policy, `collection_execution_policy` and retry budget; HMM context
settings are independent of collection. Collections from older serialization versions cannot be resumed
under the new raw collector, but the unmasked context-format records above can
be reused for dataset building.
Start a fresh collection directory for the action-based execution policy
`complete_action_allow_state_repair_v1`; collections using the older strict
state gate cannot be resumed with the new policy. Their unmasked samples and
original histories remain reusable for dataset building. New builds carry the
execution-policy marker into dataset/checkpoint provenance.
`--state_max_attempts` and `--state_retry_temperature` are replaced by
`--collection_max_attempts`, which bounds whole unmasked turn attempts.

## Tests

```bash
pytest -q ctrlg-alfworld/tests
```

For Windows CPU-only environments, set `TORCHDYNAMO_DISABLE=1` to test the
math without a native compiler. Set `CTRLG_RUN_QWEN_SMOKE=1` to enable the
optional local tokenizer check. `scripts/profile_joint_hmm.py` profiles real
tokenizer DFAs with an untrained small HMM; it does not measure task quality.
At the default token budgets, the local Qwen tokenizer profile estimates
about 12.3 GB for the state lookahead cache at 4096 hidden states, excluding
model weights and temporary buffers.
Lookahead construction accumulates length ranges incrementally into that final
cache, retaining only the current backward layer and a running sum per active
lower bound (two running sums at the default minimum length of one). It avoids
the additional full layer stack and normalized copy previously needed by the
range-reduction matrix multiplication.

The CPU suite covers DSL rejection, all six schemas, exact two-target
`puttwo` states including shared locations, character/token DFA behavior,
closing tags, token-budget completion, policy priority and fallback, cup/mug
separation, prompt identity, no-oracle decoder routing, collection eligibility,
unmasked HF/vLLM draws, durable retry collection/resume, censored-prefix EM counts, context/teacher embedding alignment, exact state/action lookahead probabilities, and paired-summary provenance.
