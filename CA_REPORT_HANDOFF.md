# CA report drafting handoff

Prepared on 24 October 2026 for continuing the report in a fresh chat.
Workspace: D:\repos\Ctrl-G

## Purpose and immediate next step

The user is writing a CA report for this project. The introduction/project objectives and literature review were already written. This conversation planned the Project Progress section and drafted its first two subsections.

Continue helping the user draft the remaining subsections one at a time. The next subsection in the agreed sequence is **HMM Data Collection and Training Pipeline**. Ask focused clarifying questions when experimental facts are missing. Do not write all the remaining subsections in one response unless requested.

The latest user-pasted Framework Design and Implementation subsection is included verbatim below. It is the authoritative current version, superseding earlier assistant drafts. The user-edited Overview of Progress and original Project Objectives are also preserved below, together with the supplied literature review.

## Agreed writing preferences and scope

- Provide copyable LaTeX prose, matching the existing first-person academic style and British spelling, such as "behavioural" and "summarises".
- Keep the general framework explanation brief and applicable across domains.
- The user explicitly rejected an earlier version because the Skill Document Specification, Constrained State Generation, and State-Dependent Action Generation components contained too many ALFWorld details.
- Put concrete ALFWorld task categories, state fields, command examples, policies, numerical priorities, and evaluation settings in **ALFWorld Integration and Evaluation Setup**.
- Explain the HMM pipeline clearly, with detailed run configurations placed where they fit the experiment description. Avoid repeating the completed Ctrl-G explanation.
- Use the exact notation of the Ctrl-G paper for its decoding equations: x_t, x_{<t}, x_{>t}, x_{1:n}, alpha, mathcal M, p_lm, p_hmm, and p_ctrl-g. The preserved LaTeX below gives the exact forms.
- In the paper's equations, t indexes tokens. Keep that distinct from an environment decision-step index if introducing further notation.
- Keep the distinction between output-format validity, policy compliance relative to a generated state, environmental action admissibility, and task success.
- A schema-valid state can be factually wrong. HMM inference is exact under the HMM, while its guidance approximates the LLM's future-satisfaction probabilities.
- Do not claim arbitrary natural-language rules or arbitrary action preconditions are compiled. Hard enforcement is limited to the supported machine-readable specifications.
- Do not invent numerical findings, training completion details, model choices, dataset sizes, compute settings, or benchmark splits.
- The user has asked for drafting help, not edits to implementation code or automatic creation of a complete report document.

## Agreed Project Progress structure

~~~latex
\section{Project Progress}
\label{sec:progress}

\subsection{Overview of Progress}

\subsection{Framework Design and Implementation}
\subsubsection{Ctrl-G Decoding Mechanism}
\subsubsection{Skill Document Specification}
\subsubsection{Constrained State Generation}
\subsubsection{State-Dependent Action Generation}

\subsection{HMM Data Collection and Training Pipeline}

\subsection{ALFWorld Integration and Evaluation Setup}

\subsection{Preliminary Validation and Results}

\subsection{Current Challenges and Limitations}

\subsection{Remaining Work and Next Milestones}
~~~

The final subsection can become a separate Future Work section if the report format requires it; no such requirement has been supplied.

## Confirmed progress and facts still to establish

The user explicitly confirmed that the ALFWorld experiments are **preliminary runs**. Do not describe them as the completed main evaluation.

The user's Overview of Progress says that the prototype, ALFWorld integration, HMM data/training/checkpoint-selection pipeline, and evaluation setup have been implemented, and that preliminary runs and trajectory analyses have been conducted.

The intended comparison is constrained state/action generation versus a prompt-only baseline using the same LLM and skills document. The principal metrics stated in the objectives are task success, policy adherence relative to the generated state, and action admissibility. Other diagnostic metrics can be discussed when supported.

The user plans to evaluate additional agentic benchmarks by the end of the project. No additional benchmark names or completed cross-domain experiments have been confirmed.

Before presenting specific experimental settings or numerical results, establish the actual model/checkpoint, collection volume, task split, HMM size, history window, training configuration, selected checkpoint, evaluation sample size, hardware, and measured outcomes as needed. Repository examples and defaults are evidence of available implementation, not confirmation of settings used in a particular run.

## Implementation context for the remaining subsections

These notes come from reading the repository. They are reference context for drafting, not additional user-approved report paragraphs. Recheck current sources before detailed claims.

### General architecture

- The repository calls the project/prototype Skill2Act. The user has not selected a final report title in this chat.
- The skills document contains Markdown prose and fenced YAML blocks of three types: action, state-schema, and policy.
- Natural-language content is prompt guidance. Supported blocks are parsed and validated.
- State schemas are converted into regular-language specifications, character automata, and then DFAs over model vocabulary tokens.
- State generation is followed by parsing, policy evaluation on the model-declared fields, construction of the action DFA, and action generation.
- One trained joint state/action HMM provides probabilistic guidance in both phases.
- HMM hidden states and DFA states are internal decoding variables, distinct from the task-state representation generated by the LLM.
- Action-template precondition annotations are parsed and supplied in the prompt; general symbolic precondition predicates are not independently enforced by the action compiler.
- In the current policy set, the highest-priority matching required-action rule selects the command and shadows lower-priority rules. If no required-action rule applies, the supported argument restriction and exact-action exclusion are combined.
- A binding failure or empty policy language falls back to the generic action grammar, with the fallback reason recorded. This relaxes policy enforcement.
- The latest user draft retains a brief priority explanation and omits the longer fallback discussion. Preserve that editorial choice unless further detail is requested or needed later.

### HMM pipeline

- Collection draws unmasked state and action samples. Invalid states, grammar failures, inadmissible actions, unsuccessful episodes, and truncated attempts are retained.
- Collection can retry whole-turn draws from the same observable context. Execution selection uses public command-format/completion checks, not oracle admissibility or actual task progress.
- Raw samples and observable episode history are saved separately so context windows can be rebuilt later.
- Dataset building constructs a joint sequence of observable context, generated state, generated action, and a terminal marker when justified by actual completion.
- The HMM's observable context is a compact deterministic serialization of the task, initial observation, recent executed action/observation history, and current observation.
- This HMM context is distinct from the LLM prompt. Both evaluation arms retain the same full LLM prompt construction.
- Context construction avoids future feedback, rejected attempts, oracle command lists, old generated state summaries, and native thinking.
- Incomplete outputs are treated as observed prefixes; synthetic closers must not be presented as sampled training evidence. See the README and builder for exact censoring rules.
- Retries and steps from the same game instance remain within one training/development split.
- The implementation includes LVD-related initialization/embedding extraction and EM-based HMM training. Check the scripts/source for the exact procedure before drafting it in detail.
- Checkpoint selection uses held-out output-conditional likelihood, with joint sequence likelihood available as a diagnostic.
- During inference, the HMM consumes observable context plus the state opening delimiter for state generation, and observable context plus the generated state and action opening delimiter for action generation.
- State-phase lookahead includes a bounded generic action continuation and EOS. The actual state-dependent policy DFA is constructed only after the state is complete.

### ALFWorld integration and evaluation

- The prototype uses ALFWorld's text-based TextWorld environment, configured as AlfredTWEnv.
- Its current schemas cover six ALFWorld task categories. Concrete field definitions and task-specific examples belong in the integration subsection.
- Each generated state is reconstructed from task/observation/action history. Previous generated states and native thinking are logged but not replayed as history.
- There is no symbolic ground-truth state tracker or independent fact extractor.
- Environment-provided admissible commands are withheld from prompts and constraint compilation, then used after generation to score action admissibility.
- Evaluation conditions are named state_prompt and state_ctrlg in the code.
- The matched comparison controls the model, skills document, prompt construction, settings, and episode manifest. Actual run details must still be confirmed.
- Diagnostics include schema validity, action grammar validity, policy evaluability/adherence/fallback, latency, and generated tokens, alongside task success and admissibility.
- Schema constraints do not establish factual consistency of state fields or correctness of a search ledger.
- Preliminary analyses have examined inaccurate generated states and repeated actions. Do not promote preliminary observations to established general findings.

## Suggested content of the remaining subsections

1. **HMM Data Collection and Training Pipeline:** explain how samples are collected, serialized with observable context, split, used for HMM initialization/training, and assessed for checkpoint selection. Separate implemented infrastructure from completed experimental runs.
2. **ALFWorld Integration and Evaluation Setup:** explain environment/task categories, concrete schemas and policies, prompting/history, model and generation settings, matched baseline, evaluation instances, and metrics. Include the benchmark-specific details intentionally omitted from the general framework description.
3. **Preliminary Validation and Results:** report validation actually completed, preliminary quantitative results with sample sizes/settings, and illustrative success/failure trajectories. Confirm figures and run provenance before reporting them.
4. **Current Challenges and Limitations:** relate observed problems to consequences and possible responses, including state inaccuracies, policy expressiveness, fallback, repeated actions, HMM approximation quality, and compute overhead where evidenced.
5. **Remaining Work and Next Milestones:** give concrete next deliverables, including further HMM/evaluation work, failure analysis, refinement, and eventual additional benchmarks, consistent with the user's actual schedule.

## Useful local sources

- Project overview: D:\repos\Ctrl-G\README.md
- Main prototype documentation: D:\repos\Ctrl-G\ctrlg-alfworld\README.md
- Skill document and concrete policies: D:\repos\Ctrl-G\ctrlg-alfworld\templates\SKILLS.md
- Specification parser: D:\repos\Ctrl-G\ctrlg-alfworld\src\ctrlg_alfworld\skills.py
- Constraint and policy compilation: D:\repos\Ctrl-G\ctrlg-alfworld\src\ctrlg_alfworld\constraints.py
- Generation implementation: D:\repos\Ctrl-G\ctrlg-alfworld\src\ctrlg_alfworld\backends.py
- Generation parsing/records: D:\repos\Ctrl-G\ctrlg-alfworld\src\ctrlg_alfworld\generation.py
- Observable HMM context: D:\repos\Ctrl-G\ctrlg-alfworld\src\ctrlg_alfworld\hmm_context.py
- Agent loop: D:\repos\Ctrl-G\ctrlg-alfworld\src\ctrlg_alfworld\agent_loop.py
- Collection and distillation: D:\repos\Ctrl-G\ctrlg-alfworld\src\ctrlg_alfworld\collection.py and distillation.py in the same directory
- Checkpoint selection: D:\repos\Ctrl-G\ctrlg-alfworld\src\ctrlg_alfworld\checkpoint_selection.py
- Scripts: D:\repos\Ctrl-G\ctrlg-alfworld\scripts (especially run_rollouts.py, build_hmm_data.py, evaluate_hmm_fit.py, select_best_hmm_checkpoint.py, run_eval.py, and summarize_results.py)
- Job scripts: D:\repos\Ctrl-G\ctrlg-alfworld\slurm
- Configuration examples: D:\repos\Ctrl-G\ctrlg-alfworld\configs\experiment.yaml
- HMM implementation and decoding: D:\repos\Ctrl-G\ctrlg\hmm.py and D:\repos\Ctrl-G\ctrlg\utils.py
- Latest user framework attachment: C:\Users\limyi\.codex\attachments\5e6fec28-9c14-4cfb-85a3-69ca066c7a1c\Pasted text.txt
- Original literature-review attachment: C:\Users\limyi\.codex\attachments\02cf704e-5939-40f3-91be-3cb0267fb4c7\Pasted text.txt
- Ctrl-G paper, Section 3: https://arxiv.org/html/2406.13892v1#S3

## Suggested opening message for the fresh chat

> Read CA_REPORT_HANDOFF.md for the current report context and completed drafts. Continue helping me draft the remaining Project Progress subsections one at a time, starting with HMM Data Collection and Training Pipeline. Keep the framework discussion general and reserve concrete ALFWorld details for the integration/evaluation subsection. Ask me focused questions where actual experiment settings or progress are not established.

## User-edited Overview of Progress

This is the user's latest supplied version, preserved as written.

~~~latex
\subsection{Overview of Progress}
\label{subsec:progress-overview}

During the first semester of this project, I have developed an agentic framework that connects specifications in a skills document to constrained state and action generation using Ctrl-G. The prototype supports machine-readable state schemas, action templates, and state-dependent behavioural policies. At each decision step, the framework first constrains the generation of a structured state representation, then uses the generated state and applicable policies to construct constraints for action generation.

As a preliminary investigation, I have integrated the prototype with ALFWorld's text-based environment. ALFWorld was selected as the first benchmark because its textual observations and action commands provide a suitable setting for examining state and action generation in multi-step interactive tasks. The initial experiments assess the feasibility and effectiveness of applying skill-document constraints within an agent's decision-making loop. To support these experiments, I have also implemented a pipeline for collecting model-generated state and action samples, constructing training datasets, and training and selecting HMM checkpoints for probabilistic guidance. By the end of the project, I plan to extend the evaluation to additional agentic benchmarks to assess the framework's applicability across different environments and tasks.

An evaluation setup has been implemented to compare constrained generation with a prompt-only baseline using the same LLM and skills document. Preliminary runs have been conducted, alongside trajectory analyses to identify failure modes such as inaccurate state representations and repeated actions. These preliminary observations will guide further refinement of the framework and subsequent evaluation.
~~~

## Original user-supplied Project Objectives

Preserved as originally supplied; no subsequent replacement was provided in this conversation.

~~~latex
\section{Project Objectives}
\label{sec:objectives}

\subsection{Background and Problem}
Large language models (LLMs) can be used as decision-making components in agentic systems, where they interpret observations, select actions, and interact with external tools or environments \citep{yao2023react}. However, in practical applications, users and developers may also wish to constrain how the agent completes a task, including which tools it may invoke, which arguments it may supply, and under what conditions an action is permitted or required.

Classical symbolic planners offer a formal approach to constrained action selection in agentic systems, including software workflows and tool orchestration. These planners typically operate on an explicit domain model describing action preconditions and effects. Such planners can guarantee constraint satisfaction, but constructing and maintaining them requires substantial effort especially when tool behaviour is complex or changes over time.

LLMs offer a convenient interface for expressing constraints through natural-language instructions. Users can place constraints alongside task instructions and maintain them in a document such as \texttt{SKILL.md}. However, including a rule in the model's context does not guarantee that the constraint will be satisfied. Compliance remains dependent on the model's instruction-following behaviour. Natural-language guidance and symbolic planning address different needs and can be combined, with instructions conveying user intent and explicit models supporting reasoning about actions \citep{dagan2023dynamicplanningllm}.

Constrained decoding provides a complementary mechanism that restricts the outputs a language model can generate. For example, token masking can prevent the generation of strings that violate a specified regular expression or grammar. Such mechanisms can produce outputs with guaranteed satisfaction of the encoded constraint, but a formally valid tool call can still contain inappropriate arguments. Moreover, local token masking can distort the relative probabilities of valid complete outputs, potentially reducing generation quality \citep{park2024grammar}.

Ctrl-G provides a foundation for investigating this problem by combining constraints represented as deterministic finite automata with probabilistic guidance from a Hidden Markov Model (HMM). This approach supports constrained language generation while accounting for the estimated likelihood of valid future continuations \citep{zhang2024ctrlg}.

\subsection{Objectives and Scope}
The project aims to develop and evaluate a framework that enables users or programmers to guide an LLM agent's decisions through specifications in a skills document. Supported state schemas, action templates, and behavioural rules will constrain state and action generation without requiring a perfect model of the environment.

The project assumes that users have sufficient domain knowledge to provide meaningful specifications. The framework is intended for situations in which an agent does not reliably select appropriate actions or tools given its current state representation. Users can encode domain knowledge as state-dependent behavioural policies to guide the agent towards task completion. For example, an agent searching for an object may overlook closed drawers. A user-defined policy can require the agent to open a closed drawer upon reaching it, allowing its contents to be inspected before the search continues.

Preliminary evaluation will use ALFWorld's text-based environment and compare constrained generation with a prompt-only baseline using the same LLM and specifications. To measure success, I will primarily use task success rate, policy adherence relative to the generated state, and action admissibility rate as the evaluation metrics. Other relevant metrics may also be used for analysis. By the end of the project, I aim to evaluate this framework on more agentic benchmarks, to test the applicability of this framework on various domains.
~~~


## Latest user-supplied Framework Design and Implementation

Copied verbatim from the latest attachment. This is the current version to continue from.

~~~latex
\subsection{Framework Design and Implementation}

The framework consists of three main components: a skills document containing supported specifications, a constraint compiler that translates these specifications into DFAs, and a decoding procedure that applies the resulting constraints during generation. These components are integrated into an agent loop in which the LLM generates a task-relevant state representation, selects an action based on the generated state, and receives feedback from the environment.

\subsubsection{Ctrl-G Decoding Mechanism}
\label{subsubsec:ctrlg-mechanism}

Ctrl-G combines an LLM with a distilled HMM and a DFA encoding the desired logical constraint \citep{zhang2024ctrlg}. Let $x_{1:n}$ denote a complete token sequence and $\alpha$ the constraint, represented by a DFA $\mathcal{M}$. The automaton accepts $x_{1:n}$ if and only if the sequence satisfies $\alpha$. At token position $t$, $x_{<t}$ denotes the generated prefix, $x_t$ the next token, and $x_{>t}$ the remaining continuation.

During inference, Ctrl-G reweights the LLM's next-token probabilities according to
\begin{equation}
    p_{\text{ctrl-g}}(x_t \mid x_{<t}, \alpha)
    \propto
    p_{\text{lm}}(x_t \mid x_{<t})
    \cdot
    p_{\text{hmm}}(\alpha \mid x_t, x_{<t}).
    \label{eq:ctrlg-guidance}
\end{equation}
The first factor is the LLM's probability of selecting $x_t$ given the generated prefix. The second factor is the probability, under the HMM, that the completed sequence will satisfy $\alpha$ after selecting $x_t$, which approximates the future-satisfaction probability under the LLM, $p_{\text{lm}}(\alpha \mid x_t, x_{<t})$.

Given the DFA $\mathcal{M}$, the HMM term is computed by summing over all future token sequences that produce an accepted output:
\begin{equation}
    p_{\text{hmm}}(\alpha \mid x_t, x_{<t})
    =
    \sum_{\substack{
        x_{>t}\ \text{s.t.}\\
        \mathcal{M}\ \text{accepts}\ x_{1:n}
    }}
    p_{\text{hmm}}(x_{>t} \mid x_t, x_{<t}).
    \label{eq:ctrlg-future-satisfaction}
\end{equation}
Ctrl-G computes this sum using dynamic programming over combinations of HMM hidden states and DFA states, avoiding explicit enumeration of complete continuations. The computation is exact under the HMM, while its guidance remains an approximation to the LLM's future-satisfaction probabilities.

In this project, Ctrl-G is applied at two stages of each agent decision. During state generation, a schema DFA restricts the structure and permitted field values of the generated state representation. Once this representation is complete, the framework evaluates the behavioural policies against its declared values and constructs an action DFA. The action DFA then constrains the subsequent action generation. The same trained HMM provides probabilistic guidance in both stages, with its conditioning context extended to include the generated state during action generation.

\subsubsection{Skill Document Specification}
\label{subsubsec:skill-specification}

The skills document combines natural-language guidance with structured specifications that can be interpreted by the constraint compiler. These specifications define three elements: state schemas describing the structure and permitted values of state representations, action templates describing command formats and argument placeholders, and behavioural policies specifying when actions are required or restricted.

The compiler parses and validates the supported declarations before using them to construct generation constraints. Natural-language instructions remain available to the LLM as prompt guidance, while enforcement applies to the supported machine-readable specifications. This provides an interface through which users can modify the agent's constraints without changing the underlying LLM's parameters.

\subsubsection{Constrained State Generation}
\label{subsubsec:constrained-state-generation}

At each decision step, the LLM generates a structured representation of the task-relevant state from the task instructions and available observation and interaction history. The declared state schema is compiled into a DFA that restricts the representation's structure and permitted field values. Ctrl-G applies this DFA together with HMM guidance during generation.

The completed state representation is parsed and passed to the policy compiler, providing an explicit interface between the LLM's interpretation of the situation and the constraints imposed on its next action. Although schema enforcement establishes structural validity, the accuracy of the generated state remains dependent on the model's interpretation of the available information.

\subsubsection{State-Dependent Action Generation}
\label{subsubsec:state-dependent-actions}

Once the state representation has been generated, the framework evaluates the behavioural policies against its declared values. Users can design rules that determine the permitted action language, such as requiring an action to be taken, restricting its arguments, or excluding a specific action, based on the generated state fields. Each policy is assigned a numerical priority, with larger values indicating higher precedence. In the case of conflicting policies, i.e. both policies require a specific action to be taken, the higher priority policy will be taken.

The resulting action language is compiled into a DFA and used by Ctrl-G to constrain action generation. The selected action is executed in the environment, and the resulting feedback informs the next decision step.
~~~

## Existing user-supplied Literature Review

Copied verbatim from the original literature-review attachment for context and citation consistency.

~~~latex
\section{Literature Review}
\label{sec:literature}

Constrained decoding modifies a language model's generation procedure to enforce requirements on its outputs. Relevant methods differ in the type of constraints they support and the computational cost of enforcement. This review focuses on lexical constraints, grammar-based decoding, probabilistic guidance, and semantic constraints. These directions provide the foundations for constraining an agent's generated states and actions.

\subsection{Lexically Constrained Decoding}
\label{subsec:lexical-decoding}

Early constrained decoding methods focused on requiring specified words or phrases to appear in an output. Grid Beam Search extends beam search by tracking progress through lexical constraints coverage \citep{hokamp2017grid}. Dynamic Beam Allocation improves the practicality of this approach by distributing a fixed beam across hypotheses with different levels of constraint satisfaction \citep{post2018fast}. These methods demonstrate that requirements can be imposed during inference without modifying the underlying model's parameters.

NeuroLogic Decoding extends lexical control to logical combinations of requirements, supporting richer inclusion and exclusion conditions \citep{lu2021neurologic}. NeuroLogic A*esque Decoding subsequently incorporates lookahead heuristic estimates of future constraint satisfaction to address a limitation that a locally promising choice may make later satisfaction difficult \citep{lu2022neurologic}.

These methods establish search-based approaches to constrained generation. However, lexical requirements provide limited control over structured state records or action arguments. 

\subsection{Grammar-Constrained Decoding}
\label{subsec:grammar-decoding}

Grammar-constrained decoding restricts generation to a formal language by using a parser or automaton to filter candidate tokens. PICARD and subsequent work demonstrate this approach in text-to-SQL and broader structured NLP tasks \citep{scholak2021picard,geng2023grammar}. Input-dependent grammars additionally allow permitted outputs to vary with the input, providing a mechanism for incorporating changing identifiers and arguments \citep{geng2023grammar}.

Correct enforcement must account for the mismatch between model tokens and grammar terminals. Koo et al.\ formalize this relationship through automata composition for regular and deterministic context-free languages \citep{koo2024automata}. DOMINO addresses tokens spanning grammar boundaries, while SynCode combines incremental parsing with precomputed automaton-based masks \citep{beurerkellner2024guiding,ugare2024syncode}. These methods distinguish rejecting invalid continuations from preserving valid ones; the latter can depend on lookahead settings.

Efficiency depends on both preprocessing and runtime checks. Efficient Guided Generation indexes vocabulary tokens by automaton state \citep{willard2023efficient}, while XGrammar caches token checks that are independent of the deeper parser stack and overlaps grammar processing with inference \citep{dong2025xgrammar}. GreatGramma reduces preprocessing costs by considering only terminal sequences realizable by the LLM’s vocabulary, rather than enumerating all terminal combinations up to a fixed length \citep{park2025flexible}.

With correct token handling and acceptance checks, grammar enforcement can guarantee that completed outputs belong to the encoded language. However, this does not establish completion within a budget, exact conditional sampling, or task correctness.

\subsection{Probabilistic Guidance and Distribution Alignment}
\label{subsec:probabilistic-guidance}

Validity alone does not determine how probability should be distributed among valid outputs. Token masking generally differs from conditioning the LLM on the complete constraint as correct conditioning must account for the probability of future satisfaction after each candidate token. Computing this quantity directly requires summing over potentially exponentially many continuations.

Existing methods address this difficulty through different approximations. FUDGE and NADO learn future-satisfaction predictors \citep{yang2021fudge,meng2022nado}, while Sequential Monte Carlo steering approximates the constrained distribution through weighted candidate trajectories \citep{lew2023smc}. Grammar-Aligned Decoding introduces ASAp, which progressively refines future-validity estimates using previous samples \citep{park2024grammar}. Their guidance or distribution alignment therefore depends on prediction accuracy or sampling effort.

GeLaTo instead makes future-satisfaction inference tractable within an auxiliary probabilistic model \citep{zhang2023gelato}. It distills an HMM from the language model and uses dynamic programming to compute lexical-constraint probabilities. Ctrl-G extends this approach to DFA-represented constraints \citep{zhang2024ctrlg}. By combining HMM hidden states with DFA states, Ctrl-G sums over valid futures without enumerating individual continuations. The same distilled HMM can support new constraints without retraining.

Ctrl-G's main advantage is combining DFA-based constraint enforcement with tractable probabilistic lookahead, using a reusable HMM to guide generation under new constraints without retraining. This project adopts Ctrl-G as its decoding backbone, extending it to constrain generated states and actions using supported specifications derived from a skill document. 

\subsection{Extending Ctrl-G to Agent Planning}
\label{subsec:ctrlg-agent-planning}

Meta-Ctrl extends Ctrl-G to plan generation by separating token-level syntax from action-level semantics \citep{yidouweng2026metactrl}. A syntactic DFA enforces valid action formats, while a semantic DFA tracks preconditions, goal progress, and ordering over \emph{meta-tokens}, each representing a grounded action. Separate HMM--DFA computations provide probabilistic lookahead at both levels, connected through a parser mapping token sequences to actions. This avoids constructing a single large automaton combining formatting and task state, making constraint-aware planning substantially more tractable.

Meta-Ctrl provides closely related work by applying Ctrl-G to state-dependent action constraints. Its semantic automaton tracks symbolic state across a generated plan using structured domain specifications. This project instead uses a skill document to specify supported constraints on generated states and their associated actions.
~~~
