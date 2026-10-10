from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import torch

from .constraints import (
    HardDFALogitsProcessor,
    action_grammar_valid,
    compile_policy_language,
    state_span_pattern,
    action_span_pattern,
    dfa_accepts,
    lift_character_fsm,
    parse_state,
    policy_satisfied,
    regex_fsm,
    tokenize_fixed_suffix,
)
from .generation import (
    GeneratedChunk,
    HeadGeneration,
    TurnGeneration,
    exact_action_tail_token_ids,
    hmm_prefix_token_ids,
    parse_turn,
    token_boundary_for_char_offset,
)
from .prompts import (
    ACTION_CLOSE,
    ACTION_OPEN,
    STATE_CLOSE,
    STATE_OPEN,
    THINK_CLOSE,
)
from .skills import SkillSet

torch.backends.cuda.enable_cudnn_sdp(False)

ACTION_DFA_CACHE_SIZE = 4


def _synchronize_if_cuda(device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class StopOnStrings:
    def __init__(self, stop_strings, tokenizer, prompt_len: int):
        self.stop_strings = tuple(stop_strings)
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Stop only when every live beam/sample contains a stop string. This
        # avoids terminating a batch because beam zero happened to finish first.
        return all(
            any(
                stop in self.tokenizer.decode(row[self.prompt_len :])
                for stop in self.stop_strings
            )
            for row in input_ids
        )


@dataclass
class GenConfig:
    max_thought_tokens: int = 1024
    max_state_tokens: int = 512
    max_action_tokens: int = 32
    min_action_tokens: int = 1
    beam_size: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    rollout_temperature: float = 0.7
    seed: int = 42
    max_hmm_prefix_tokens: int | None = None
    collection_max_attempts: int = 5
    hmm_context_steps: int = 1
    max_hmm_context_tokens: int = 1024


class StructuredOutputMismatchError(RuntimeError):
    """A vLLM structured-output request ended outside its requested language."""

    def __init__(self, chunk: GeneratedChunk):
        self.chunk = chunk
        reason = (
            "max_tokens_before_accept"
            if chunk.truncated
            else "structured_output_mismatch"
        )
        self.reason = reason
        super().__init__(
            "vLLM returned text outside the requested structured-output regex: "
            f"{chunk.text!r}"
        )


class StateGenerationExhausted(RuntimeError):
    """All bounded vLLM state attempts failed structural validation."""

    def __init__(
        self,
        *,
        thought: GeneratedChunk,
        thought_seed: int,
        attempt_records: tuple[dict, ...],
    ):
        self.thought = thought
        self.thought_seed = thought_seed
        self.attempt_records = attempt_records
        super().__init__(
            f"state generation exhausted {len(attempt_records)} attempts"
        )


def _crop_chunk_at_stop(tokenizer, token_ids: list[int], stop_strings) -> tuple[str, list[int], bool]:
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    matches = [
        (text.find(stop) + len(stop), stop)
        for stop in stop_strings
        if stop in text
    ]
    if not matches:
        return text, token_ids, False

    end_char, _ = min(matches, key=lambda item: item[0])
    try:
        end_token = token_boundary_for_char_offset(tokenizer, token_ids, end_char)
    except ValueError:
        # Preserve exact token/text alignment. The parser can tolerate trailing
        # characters, while distillation can reject a non-aligned boundary.
        return text, token_ids, True
    cropped_ids = token_ids[:end_token]
    return (
        tokenizer.decode(cropped_ids, skip_special_tokens=False),
        cropped_ids,
        True,
    )


class BaseBackend:
    """Shared unconstrained sampling and turn parsing.

    HF and vLLM both expose the same generated-token record. Constrained
    generation is intentionally implemented only by :class:`HFBackend`, where
    Ctrl-G can access logits in-process.
    """

    def _generate_until(
        self,
        prompt_text: str,
        stop_strings,
        max_new_tokens: int,
        temperature: float | None = None,
    ) -> GeneratedChunk:
        raise NotImplementedError

    def _assemble_head(
        self,
        thought: GeneratedChunk,
        state: GeneratedChunk | None,
        *,
        use_state: bool,
    ) -> HeadGeneration:
        """Add fixed phase delimiters while preserving exact continuation IDs."""

        text = thought.text
        token_ids = list(thought.token_ids)

        def append_fixed(value: str) -> None:
            nonlocal text
            token_ids.extend(tokenize_fixed_suffix(self.tokenizer, token_ids, value))
            text += value

        used_thought_repair = not thought.stop_found
        if used_thought_repair:
            append_fixed(THINK_CLOSE)

        used_state_repair = False
        hmm_target_start = len(token_ids)
        hmm_observed_head_end = len(token_ids)
        if use_state:
            if state is None:
                raise ValueError("state generation is required in state mode")
            append_fixed(STATE_OPEN)
            before_state = self.tokenizer.decode(
                token_ids, skip_special_tokens=False
            )
            combined_ids = token_ids + list(state.token_ids)
            combined_text = self.tokenizer.decode(
                combined_ids, skip_special_tokens=False
            )
            if combined_text != before_state + state.text:
                raise ValueError(
                    "state token IDs are not compositional with the head"
                )
            token_ids = combined_ids
            text += state.text
            hmm_observed_head_end = len(token_ids)
            used_state_repair = not state.stop_found
            if used_state_repair:
                append_fixed(STATE_CLOSE)

        append_fixed(ACTION_OPEN)
        if not used_state_repair:
            hmm_observed_head_end = len(token_ids)
        expected = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        if expected != text:
            raise ValueError("assembled head text does not match its exact token IDs")

        state_stop_found = state.stop_found if state is not None else True
        state_truncated = state.truncated if state is not None else False
        return HeadGeneration(
            chunk=GeneratedChunk(
                text=text,
                token_ids=tuple(token_ids),
                stop_found=thought.stop_found and state_stop_found,
                truncated=thought.truncated or state_truncated,
                latency_seconds=(
                    thought.latency_seconds
                    + (state.latency_seconds if state is not None else 0.0)
                ),
            ),
            thought_chunk=thought,
            state_chunk=state,
            used_thought_repair=used_thought_repair,
            used_state_repair=used_state_repair,
            hmm_target_start=hmm_target_start,
            hmm_observed_head_end=hmm_observed_head_end,
        )

    def _generate_head(
        self, prompt_text: str, *, use_state: bool, greedy: bool
    ) -> HeadGeneration:
        temperature = None if greedy else self.cfg.rollout_temperature
        thought = self._generate_until(
            prompt_text,
            [THINK_CLOSE],
            self.cfg.max_thought_tokens,
            temperature,
        )
        thought_text = thought.text + (THINK_CLOSE if not thought.stop_found else "")
        state = None
        if use_state:
            state = self._generate_until(
                prompt_text + thought_text + STATE_OPEN,
                [STATE_CLOSE],
                self.cfg.max_state_tokens,
                temperature,
            )
        return self._assemble_head(thought, state, use_state=use_state)

    def generate_turn_unconstrained(
        self, prompt_text: str, *, use_state: bool, greedy: bool = False, evaluation: bool = False
    ) -> TurnGeneration:
        """Sample raw model data for evaluation controls or HMM distillation."""

        head = self._generate_head(
            prompt_text, use_state=use_state, greedy=greedy
        )
        action_prompt = prompt_text + head.chunk.text

        temperature = None if greedy else self.cfg.rollout_temperature
        tail = (self._generate_evaluation_action(action_prompt) if evaluation else
                self._generate_until(action_prompt, [ACTION_CLOSE], self.cfg.max_action_tokens, temperature))
        return self._assemble_unconstrained_turn(
            head,
            tail,
            use_state=use_state,
        )

    def generate_turns_unconstrained(
        self,
        prompt_text: str,
        *,
        count: int,
        use_state: bool,
        greedy: bool = False,
        seed_context: tuple[int, ...] = (),
    ) -> list[TurnGeneration]:
        """Generate multiple candidates, sequentially unless overridden."""

        if count < 1:
            raise ValueError("count must be at least one")
        return [
            self.generate_turn_unconstrained(
                prompt_text, use_state=use_state, greedy=greedy
            )
            for _ in range(count)
        ]

    def _assemble_unconstrained_turn(
        self,
        head: HeadGeneration,
        tail: GeneratedChunk,
        *,
        use_state: bool,
        head_seed: int | None = None,
        state_seed: int | None = None,
        tail_seed: int | None = None,
    ) -> TurnGeneration:
        """Parse and retain the exact token spans from a head/tail pair."""

        parsed = parse_turn(
            head.chunk.text, tail.text, use_state=use_state
        )

        try:
            prefix_ids = tuple(
                hmm_prefix_token_ids(
                    self.tokenizer,
                    list(head.chunk.token_ids),
                    head.chunk.text,
                )
            )
        except ValueError:
            prefix_ids = ()

        tail_span_exact = False
        action_ids = ()
        try:
            exact_action_ids, _ = exact_action_tail_token_ids(
                self.tokenizer, list(tail.token_ids), tail.text
            )
            action_ids = tuple(exact_action_ids)
            tail_span_exact = True
        except ValueError:
            pass
        return TurnGeneration(
            parsed=parsed,
            head_token_ids=head.chunk.token_ids,
            action_token_ids=action_ids,
            tail_token_ids=tail.token_ids,
            hmm_prefix_token_ids=prefix_ids,
            head_latency_seconds=head.chunk.latency_seconds,
            action_latency_seconds=tail.latency_seconds,
            used_head_repair=head.used_repair,
            hmm_applied=False,
            head_stop_found=head.chunk.stop_found,
            head_truncated=head.chunk.truncated,
            tail_stop_found=tail.stop_found,
            tail_truncated=tail.truncated,
            tail_span_exact=tail_span_exact,
            head_seed=head_seed,
            state_seed=state_seed,
            tail_seed=tail_seed,
            thought_token_ids=head.thought_chunk.token_ids,
            state_token_ids=(
                head.state_chunk.token_ids
                if head.state_chunk is not None
                else ()
            ),
            thought_latency_seconds=head.thought_chunk.latency_seconds,
            state_latency_seconds=(
                head.state_chunk.latency_seconds
                if head.state_chunk is not None
                else 0.0
            ),
            thought_stop_found=head.thought_chunk.stop_found,
            thought_truncated=head.thought_chunk.truncated,
            state_stop_found=(
                head.state_chunk.stop_found
                if head.state_chunk is not None
                else True
            ),
            state_truncated=(
                head.state_chunk.truncated
                if head.state_chunk is not None
                else False
            ),
            used_thought_repair=head.used_thought_repair,
            used_state_repair=head.used_state_repair,
            hmm_target_token_ids=tuple(
                head.chunk.token_ids[head.hmm_target_start:head.hmm_observed_head_end]
                + (tail.token_ids if not head.used_state_repair else ())
            ),
            hmm_target_start_in_head=head.hmm_target_start,
            sampled_head_tokens=len(head.thought_chunk.token_ids) + (len(head.state_chunk.token_ids) if head.state_chunk else 0),
        )

    def generate_turns_training(self, prompt_text, skillset, task_key, *, count,
                                seed_context=()):
        """One unmasked draw per attempt; structural retries belong to collection."""
        if count < 1:
            raise ValueError("count must be at least one")
        turns = []
        for index in range(count):
            seeds = {}
            def sample(text, stops, budget, temperature, phase):
                kwargs = {}
                if hasattr(self, "_stable_seed"):
                    seed = self._stable_seed(self.cfg.seed, seed_context, index, phase)
                    seeds[phase] = seed
                    kwargs["seed"] = seed
                return self._generate_until(text, stops, budget, temperature, **kwargs)
            thought = sample(prompt_text, [THINK_CLOSE], self.cfg.max_thought_tokens,
                             None, "thought")
            thought_text = thought.text + (THINK_CLOSE if not thought.stop_found else "")
            state = sample(prompt_text + thought_text + STATE_OPEN, [STATE_CLOSE],
                           self.cfg.max_state_tokens, self.cfg.rollout_temperature, "state")
            head = self._assemble_head(thought, state, use_state=True)
            tail = (sample(prompt_text + head.chunk.text, [ACTION_CLOSE],
                           self.cfg.max_action_tokens, self.cfg.rollout_temperature, "action")
                    if state.stop_found else GeneratedChunk("", (), False, False, 0.0))
            turn = self._assemble_unconstrained_turn(
                head, tail, use_state=True, head_seed=seeds.get("thought"),
                state_seed=seeds.get("state"), tail_seed=seeds.get("action"))
            try:
                fields = parse_state(turn.parsed.state, skillset.state_schemas[task_key], skillset)
            except ValueError:
                fields = {}
            turns.append(replace(turn, state_schema_valid=bool(fields), state_fields=fields,
                                 action_grammar_valid=action_grammar_valid(turn.parsed.action, skillset)))
        return turns


class HFBackend(BaseBackend):
    def __init__(
        self,
        model_name_or_path: str,
        hmm_path: str | None = None,
        device: str = "cuda",
        dtype=None,
        gen_config: GenConfig | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = dtype or torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=dtype
        ).to(device)
        self.model.eval()
        self.device = device
        self.cfg = gen_config or GenConfig()

        self.hmm_model = None
        if hmm_path is not None:
            import ctrlg

            self.hmm_model = ctrlg.HMM.from_pretrained(hmm_path).to(device)
            self.vocab_size = self.hmm_model.vocab_size
        else:
            self.vocab_size = len(self.tokenizer)

    def _generate_until(
        self,
        prompt_text: str,
        stop_strings,
        max_new_tokens: int,
        temperature: float | None = None,
    ) -> GeneratedChunk:
        from transformers import StoppingCriteriaList

        prompt_ids = self.tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        stopper = StopOnStrings(
            stop_strings, self.tokenizer, prompt_ids.shape[1]
        )
        do_sample = temperature is not None and temperature > 0
        _synchronize_if_cuda(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_beams=1,
                num_return_sequences=1,
                stopping_criteria=StoppingCriteriaList([stopper]),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        _synchronize_if_cuda(self.device)
        latency = time.perf_counter() - started
        token_ids = output[0][prompt_ids.shape[1] :].tolist()
        text, token_ids, stop_found = _crop_chunk_at_stop(
            self.tokenizer, token_ids, stop_strings
        )
        return GeneratedChunk(
            text=text,
            token_ids=tuple(token_ids),
            stop_found=stop_found,
            truncated=not stop_found,
            latency_seconds=latency,
        )

    def _generate_evaluation_action(self, prompt_text):
        from transformers import StoppingCriteriaList
        _synchronize_if_cuda(self.device)
        started = time.perf_counter()
        ids = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        stopper = StopOnStrings([ACTION_CLOSE], self.tokenizer, ids.shape[1])
        with torch.no_grad():
            outputs = self.model.generate(input_ids=ids, min_new_tokens=self.cfg.min_action_tokens, max_new_tokens=self.cfg.max_action_tokens,
                do_sample=self.cfg.do_sample, temperature=self.cfg.temperature if self.cfg.do_sample else None,
                num_beams=1 if self.cfg.do_sample else self.cfg.beam_size, num_return_sequences=1,
                stopping_criteria=StoppingCriteriaList([stopper]), pad_token_id=self.tokenizer.eos_token_id)
        text, generated, found = _crop_chunk_at_stop(self.tokenizer, outputs[0][ids.shape[1]:].tolist(), [ACTION_CLOSE])
        _synchronize_if_cuda(self.device)
        return GeneratedChunk(text, tuple(generated), found, not found, time.perf_counter()-started)

    def _generate_state_hmm_span(self, prompt_text, graph, context, skillset, *, greedy):
        import ctrlg
        from ctrlg.utils import dfa_completion_log_probs, matmul_a_logb
        from transformers import LogitsProcessorList, StoppingCriteriaList
        _synchronize_if_cuda(self.device)
        started = time.perf_counter()
        if not hasattr(self, "_state_terminal_cache"):
            self._state_terminal_cache = {}
        key = (skillset.content_sha256, self.cfg.min_action_tokens, self.cfg.max_action_tokens)
        terminal = self._state_terminal_cache.get(key)
        if terminal is None:
            action_graph = lift_character_fsm(regex_fsm(action_span_pattern(skillset)), self.tokenizer, self.vocab_size)
            action_dfa = ctrlg.DFAModel(action_graph, self.vocab_size, dense_token_mask=False).to(self.device)
            terminal = dfa_completion_log_probs(self.hmm_model, action_dfa,
                self.cfg.min_action_tokens, self.cfg.max_action_tokens, [self.hmm_model.eos_token_id])
            for token in reversed(self.tokenizer.encode(ACTION_OPEN, add_special_tokens=False)):
                terminal = self.hmm_model.beta[:, token] + matmul_a_logb(
                    self.hmm_model.alpha_exp, terminal[:, None]).squeeze(-1)
            self._state_terminal_cache[key] = terminal
        ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        dfa = ctrlg.DFAModel(graph, self.vocab_size, dense_token_mask=False).to(self.device)
        prefix = list(context.token_ids) + self.tokenizer.encode(STATE_OPEN, add_special_tokens=False)
        processor = ctrlg.ConstraintLogitsProcessor(self.hmm_model, dfa, 1,
            self.cfg.max_state_tokens, ids, prefix_ids=prefix, suffix_ids=[],
            terminal_log_probs=terminal, temperature=1.0)
        hard = HardDFALogitsProcessor(graph, prompt_length=len(ids),
            max_new_tokens=self.cfg.max_state_tokens, eos_token_id=self.tokenizer.eos_token_id)
        stopper = StopOnStrings([STATE_CLOSE], self.tokenizer, len(ids))
        with torch.no_grad():
            outputs = self.model.generate(input_ids=torch.tensor([ids], device=self.device),
                max_new_tokens=self.cfg.max_state_tokens, do_sample=not greedy,
                temperature=self.cfg.rollout_temperature if not greedy else None,
                num_beams=1, num_return_sequences=1,
                logits_processor=LogitsProcessorList([processor, hard]),
                stopping_criteria=StoppingCriteriaList([stopper]), pad_token_id=self.tokenizer.eos_token_id)
        text, tokens, found = _crop_chunk_at_stop(self.tokenizer, outputs[0][len(ids):].tolist(), [STATE_CLOSE])
        if not found or not dfa_accepts(graph, tokens):
            raise RuntimeError("Ctrl-G did not complete a state within the state budget")
        _synchronize_if_cuda(self.device)
        return GeneratedChunk(text, tuple(tokens), found, not found, time.perf_counter()-started)

    def _generate_dfa_span(
        self,
        prompt_text: str,
        dfa_graph: dict,
        *,
        stop_string: str,
        max_new_tokens: int,
        sample: bool = False,
    ) -> GeneratedChunk:
        from transformers import LogitsProcessorList, StoppingCriteriaList

        _synchronize_if_cuda(self.device)
        started = time.perf_counter()
        prompt_ids = self.tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        processor = HardDFALogitsProcessor(
            dfa_graph,
            prompt_length=prompt_ids.shape[1],
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        stopper = StopOnStrings([stop_string], self.tokenizer, prompt_ids.shape[1])
        with torch.no_grad():
            output = self.model.generate(
                input_ids=prompt_ids,
                do_sample=sample,
                temperature=self.cfg.temperature if sample else None,
                num_beams=1 if sample else self.cfg.beam_size,
                num_return_sequences=1,
                max_new_tokens=max_new_tokens,
                logits_processor=LogitsProcessorList([processor]),
                stopping_criteria=StoppingCriteriaList([stopper]),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        token_ids = output[0][prompt_ids.shape[1] :].tolist()
        text, token_ids, stop_found = _crop_chunk_at_stop(
            self.tokenizer, token_ids, [stop_string]
        )
        _synchronize_if_cuda(self.device)
        latency = time.perf_counter() - started
        return GeneratedChunk(
            text=text,
            token_ids=tuple(token_ids),
            stop_found=stop_found,
            truncated=not stop_found,
            latency_seconds=latency,
        )

    def _generate_hmm_span(
        self,
        prompt_text: str,
        dfa_graph: dict,
        prefix_ids: list[int],
    ) -> GeneratedChunk:
        from transformers import LogitsProcessorList

        if self.hmm_model is None:
            raise ValueError("state_ctrlg requires an action HMM checkpoint")

        _synchronize_if_cuda(self.device)
        started = time.perf_counter()
        import ctrlg

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        dfa_model = ctrlg.DFAModel(
            dfa_graph, self.vocab_size, dense_token_mask=False
        ).to(self.device)
        suffix_ids = [self.hmm_model.eos_token_id]

        processor = ctrlg.ConstraintLogitsProcessor(
            self.hmm_model,
            dfa_model,
            self.cfg.min_action_tokens,
            self.cfg.max_action_tokens,
            prompt_ids,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            # Keep condition-specific reweighting at unit temperature. During
            # sampling, generate() applies cfg.temperature once afterwards,
            # matching the finite-action no-HMM path. Beam search is unscaled.
            temperature=1.0,
        )
        processor.hmm_batch_size = self.cfg.beam_size
        input_ids = torch.tensor([prompt_ids], device=self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature if self.cfg.do_sample else None,
                num_beams=1 if self.cfg.do_sample else self.cfg.beam_size,
                num_return_sequences=(
                    1
                ),
                min_new_tokens=self.cfg.min_action_tokens,
                max_new_tokens=self.cfg.max_action_tokens + 1,
                logits_processor=LogitsProcessorList([processor]),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        candidates = ctrlg.extract_generated_ids(
            outputs.tolist(),
            prompt_ids,
            suffix_ids,
            self.hmm_model.eos_token_id,
        )
        # The Ctrl-G processor keeps large DFA/HMM lookahead caches that are no
        # longer needed once the selected beam has been extracted.
        del processor, dfa_model, outputs
        # Beam search can also return incomplete or impossible beams. They must
        # never be selected, regardless of their decoding score.
        candidates = [
            candidate for candidate in candidates
            if dfa_accepts(dfa_graph, candidate)
        ]
        if not candidates:
            raise RuntimeError(
                "Ctrl-G produced no completed action accepted by the policy action DFA"
            )
        selected_ids = tuple(candidates[0])
        if not dfa_accepts(dfa_graph, selected_ids):
            decoded = self.tokenizer.decode(
                selected_ids, skip_special_tokens=False
            )
            raise RuntimeError(
                "Ctrl-G returned a token path outside the policy action DFA: "
                f"{decoded!r}"
            )
        _synchronize_if_cuda(self.device)
        latency = time.perf_counter() - started
        text = self.tokenizer.decode(selected_ids, skip_special_tokens=False)
        return GeneratedChunk(
            text=text,
            token_ids=selected_ids,
            stop_found=text.endswith(ACTION_CLOSE),
            truncated=not text.endswith(ACTION_CLOSE),
            latency_seconds=latency,
            generated_eos_tokens=1,
        )

    def _strict_head(
        self, prompt_text: str, skillset: SkillSet, task_key: str, *, greedy: bool, hmm_context=None
    ) -> HeadGeneration:
        temperature = None if greedy else self.cfg.rollout_temperature
        thought = self._generate_until(
            prompt_text, [THINK_CLOSE], self.cfg.max_thought_tokens, temperature
        )
        thought_text = thought.text + (THINK_CLOSE if not thought.stop_found else "")
        state_prompt = prompt_text + thought_text + STATE_OPEN
        compile_started = time.perf_counter()
        cache_key = (
            getattr(self.tokenizer, "name_or_path", type(self.tokenizer).__name__),
            skillset.content_sha256, task_key, self.vocab_size,
        )
        if not hasattr(self, "_state_dfa_cache"):
            self._state_dfa_cache = {}
        graph = self._state_dfa_cache.get(cache_key)
        if graph is None:
            pattern = state_span_pattern(skillset.state_schemas[task_key], skillset)
            graph = lift_character_fsm(regex_fsm(pattern), self.tokenizer, self.vocab_size)
            self._state_dfa_cache[cache_key] = graph
        compile_latency = time.perf_counter() - compile_started
        if hmm_context is not None:
            state = self._generate_state_hmm_span(state_prompt, graph, hmm_context, skillset, greedy=greedy)
        else:
            state = self._generate_dfa_span(state_prompt, graph, stop_string=STATE_CLOSE,
                max_new_tokens=self.cfg.max_state_tokens, sample=not greedy)
        state = replace(state, latency_seconds=state.latency_seconds + compile_latency)
        return self._assemble_head(thought, state, use_state=True)

    def generate_turn(
        self,
        prompt_text: str,
        skillset: SkillSet,
        task_key: str,
        *,
        constrained: bool,
        greedy_head: bool = True,
        hmm_context=None,
    ) -> TurnGeneration:
        """Generate one fair control or schema/policy-constrained turn."""

        if not constrained:
            return self.generate_turn_unconstrained(
                prompt_text, use_state=True, greedy=greedy_head, evaluation=True
            )
        if hmm_context is None:
            raise ValueError("state_ctrlg requires explicit observable HMM context")
        if self.hmm_model is None:
            raise ValueError("state_ctrlg requires a joint state/action HMM checkpoint")
        head = self._strict_head(prompt_text, skillset, task_key, greedy=greedy_head,
                                 **({"hmm_context": hmm_context} if hmm_context is not None else {}))
        action_prompt = prompt_text + head.chunk.text
        action_started = time.perf_counter()

        state_fields = parse_state(
            parse_turn(head.chunk.text, "x</action>", use_state=True).state,
            skillset.state_schemas[task_key],
            skillset,
        )
        policy = compile_policy_language(state_fields, skillset)
        if not hasattr(self, "_action_dfa_cache"):
            self._action_dfa_cache = OrderedDict()
        policy_key = (
            getattr(self.tokenizer, "name_or_path", type(self.tokenizer).__name__),
            skillset.content_sha256, policy.required_action, policy.forbidden_action,
            policy.take_type, policy.fallback_reason, self.vocab_size,
        )
        action_graph = self._action_dfa_cache.get(policy_key)
        if action_graph is None:
            action_graph = lift_character_fsm(policy.fsm, self.tokenizer, self.vocab_size)
            self._action_dfa_cache[policy_key] = action_graph
            if len(self._action_dfa_cache) > ACTION_DFA_CACHE_SIZE:
                self._action_dfa_cache.popitem(last=False)
        else:
            self._action_dfa_cache.move_to_end(policy_key)

        try:
            prefix_ids = hmm_prefix_token_ids(
                self.tokenizer, list(head.chunk.token_ids), head.chunk.text
            )
        except ValueError:
            prefix_ids = []

        if hmm_context is not None:
            prefix_ids = list(hmm_context.token_ids) + prefix_ids if prefix_ids else []

        # The HMM prefix begins after native thinking. A repaired thought close
        # is therefore outside the HMM sequence, but a repaired state close
        # makes the state/action prefix synthetic and cannot be used.
        hmm_skip_reason = None
        if not prefix_ids:
            hmm_skip_reason = "missing_exact_hmm_prefix"
        elif (
            self.cfg.max_hmm_prefix_tokens is not None
            and len(prefix_ids) > self.cfg.max_hmm_prefix_tokens
        ):
            hmm_skip_reason = "hmm_prefix_too_long"
        effective_hmm = hmm_skip_reason is None
        if effective_hmm:
            action_chunk = self._generate_hmm_span(action_prompt, action_graph, prefix_ids)
        else:
            action_chunk = self._generate_dfa_span(
                action_prompt,
                action_graph,
                stop_string=ACTION_CLOSE,
                max_new_tokens=self.cfg.max_action_tokens,
                sample=self.cfg.do_sample,
            )
        _synchronize_if_cuda(self.device)
        action_latency = time.perf_counter() - action_started
        raw_tail = action_chunk.text
        tail_ids = action_chunk.token_ids
        tail_stop_found = action_chunk.stop_found
        tail_truncated = action_chunk.truncated
        tail_span_exact = action_chunk.text.endswith(ACTION_CLOSE)
        try:
            action_ids, _ = exact_action_tail_token_ids(
                self.tokenizer, list(tail_ids), raw_tail
            )
        except ValueError:
            action_ids = []

        parsed = parse_turn(head.chunk.text, raw_tail, use_state=True)
        if not policy.fsm.accepts(raw_tail):
            raise RuntimeError(
                "token DFA output does not decode to the compiled policy language: "
                f"{raw_tail!r}"
            )
        return TurnGeneration(
            parsed=parsed,
            head_token_ids=head.chunk.token_ids,
            sampled_head_tokens=len(head.thought_chunk.token_ids) + len(head.state_chunk.token_ids),
            generated_eos_tokens=action_chunk.generated_eos_tokens,
            action_token_ids=tuple(action_ids),
            tail_token_ids=tuple(tail_ids),
            hmm_prefix_token_ids=tuple(prefix_ids),
            head_latency_seconds=head.chunk.latency_seconds,
            action_latency_seconds=action_latency,
            used_head_repair=head.used_repair,
            state_hmm_applied=hmm_context is not None,
            state_hmm_skip_reason=None if hmm_context is not None else "missing_hmm_context",
            hmm_context_token_ids=tuple(hmm_context.token_ids) if hmm_context is not None else (),
            hmm_context_truncated=hmm_context.truncated if hmm_context is not None else False,
            hmm_applied=effective_hmm,
            hmm_skip_reason=hmm_skip_reason,
            head_stop_found=head.chunk.stop_found,
            head_truncated=head.chunk.truncated,
            tail_stop_found=tail_stop_found,
            tail_truncated=tail_truncated,
            tail_span_exact=tail_span_exact,
            thought_token_ids=head.thought_chunk.token_ids,
            state_token_ids=(
                head.state_chunk.token_ids
                if head.state_chunk is not None
                else ()
            ),
            thought_latency_seconds=head.thought_chunk.latency_seconds,
            state_latency_seconds=(
                head.state_chunk.latency_seconds
                if head.state_chunk is not None
                else 0.0
            ),
            thought_stop_found=head.thought_chunk.stop_found,
            thought_truncated=head.thought_chunk.truncated,
            state_stop_found=(
                head.state_chunk.stop_found
                if head.state_chunk is not None
                else True
            ),
            state_truncated=(
                head.state_chunk.truncated
                if head.state_chunk is not None
                else False
            ),
            used_thought_repair=head.used_thought_repair,
            used_state_repair=head.used_state_repair,
            state_schema_valid=True,
            state_fields=state_fields,
            action_grammar_valid=action_grammar_valid(parsed.action, skillset),
            activated_policies=policy.activated,
            shadowed_policies=policy.shadowed,
            policy_evaluable=policy.evaluable,
            policy_satisfied=policy_satisfied(parsed.action, policy),
            policy_fallback_reason=policy.fallback_reason,
        )

    def generate_turns_training(self, prompt_text, skillset, task_key, *, count,
                                seed_context=()):
        return super().generate_turns_training(
            prompt_text, skillset, task_key, count=count, seed_context=seed_context)


class VLLMBackend(BaseBackend):
    """Unconstrained sampling against an OpenAI-compatible vLLM server."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        tokenizer_path: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 5,
        gen_config: GenConfig | None = None,
    ):
        from openai import OpenAI
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.cfg = gen_config or GenConfig()

    @staticmethod
    def _stable_seed(
        base_seed: int,
        seed_context: tuple[int, ...],
        candidate_index: int,
        phase: str,
    ) -> int:
        """Derive a request seed without depending on request scheduling."""

        coordinates = ":".join(str(value) for value in seed_context)
        payload = (
            f"ctrlg-alfworld-v1:{base_seed}:{coordinates}:"
            f"{candidate_index}:{phase}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
            (1 << 63) - 1
        )

    def _generate_until(
        self,
        prompt_text: str,
        stop_strings,
        max_new_tokens: int,
        temperature: float | None = None,
        *,
        seed: int | None = None,
        guided_regex: str | None = None,
    ) -> GeneratedChunk:
        started = time.perf_counter()
        request_seed = self.cfg.seed if seed is None else seed
        extra_body = {
            "include_stop_str_in_output": True,
            "return_token_ids": True,
            "seed": request_seed,
            "skip_special_tokens": False,
        }
        if guided_regex is not None:
            extra_body["structured_outputs"] = {"regex": guided_regex}
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt_text,
            max_tokens=max_new_tokens,
            temperature=(
                temperature if temperature is not None and temperature > 0 else 0.0
            ),
            stop=list(stop_strings),
            extra_body=extra_body,
        )
        latency = time.perf_counter() - started
        choice = response.choices[0]
        server_text = choice.text
        token_ids = getattr(choice, "token_ids", None)
        if token_ids is None and getattr(choice, "model_extra", None):
            token_ids = choice.model_extra.get("token_ids")
        if token_ids is None:
            raise RuntimeError(
                "vLLM did not return generated token IDs; use a vLLM version "
                "supporting return_token_ids or collect with --backend hf"
            )
        decoded_text = self.tokenizer.decode(
            token_ids, skip_special_tokens=False
        )
        # vLLM may truncate its displayed text at the stop string while retaining
        # the complete generated token that contained that string.
        if (
            decoded_text != server_text
            and not decoded_text.startswith(server_text)
        ):
            raise ValueError(
                "Unexpected vLLM tokenizer mismatch: "
                f"server_text={server_text!r}, "
                f"decoded_text={decoded_text!r}, "
                f"token_ids={token_ids!r}"
            )
        text, token_ids, stop_found = _crop_chunk_at_stop(
            self.tokenizer, list(token_ids), stop_strings
        )
        chunk = GeneratedChunk(
            text=text,
            token_ids=tuple(token_ids),
            stop_found=stop_found,
            truncated=not stop_found,
            latency_seconds=latency,
        )
        if guided_regex is not None and re.fullmatch(guided_regex, text) is None:
            raise StructuredOutputMismatchError(chunk)
        return chunk

    def _generate_batch_until(
        self,
        prompt_texts: list[str],
        stop_strings,
        max_new_tokens: int,
        temperature: float | None,
        seeds: list[int],
        guided_regex: str | None = None,
    ) -> list[GeneratedChunk]:
        """Submit one concurrent phase and return results in input order.

        The OpenAI completions protocol has one seed per request, not one seed
        per prompt. Separate concurrent requests therefore preserve independent
        candidate seeds while vLLM continuously batches them on one server.
        """

        if not prompt_texts:
            return []
        if len(prompt_texts) != len(seeds):
            raise ValueError("prompt_texts and seeds must have the same length")

        def generate_one(item) -> GeneratedChunk:
            prompt_text, request_seed = item
            return self._generate_until(
                prompt_text,
                stop_strings,
                max_new_tokens,
                temperature,
                seed=request_seed,
                guided_regex=guided_regex,
            )

        with ThreadPoolExecutor(max_workers=len(prompt_texts)) as executor:
            return list(executor.map(generate_one, zip(prompt_texts, seeds)))

    def generate_turns_unconstrained(
        self,
        prompt_text: str,
        *,
        count: int,
        use_state: bool,
        greedy: bool = False,
        seed_context: tuple[int, ...] = (),
    ) -> list[TurnGeneration]:
        """Batch thought, state, and action phases while preserving order."""

        if count < 1:
            raise ValueError("count must be at least one")
        temperature = None if greedy else self.cfg.rollout_temperature
        thought_seeds = [
            self._stable_seed(
                self.cfg.seed, seed_context, candidate_index, "thought"
            )
            for candidate_index in range(count)
        ]
        state_seeds = [
            self._stable_seed(
                self.cfg.seed, seed_context, candidate_index, "state"
            )
            for candidate_index in range(count)
        ]
        action_seeds = [
            self._stable_seed(
                self.cfg.seed, seed_context, candidate_index, "action"
            )
            for candidate_index in range(count)
        ]

        thoughts = self._generate_batch_until(
            [prompt_text] * count,
            [THINK_CLOSE],
            self.cfg.max_thought_tokens,
            temperature,
            thought_seeds,
        )
        if use_state:
            state_prompts = [
                prompt_text
                + thought.text
                + (THINK_CLOSE if not thought.stop_found else "")
                + STATE_OPEN
                for thought in thoughts
            ]
            states: list[GeneratedChunk | None] = list(
                self._generate_batch_until(
                    state_prompts,
                    [STATE_CLOSE],
                    self.cfg.max_state_tokens,
                    temperature,
                    state_seeds,
                )
            )
        else:
            states = [None] * count
            state_seeds = [None] * count

        heads = [
            self._assemble_head(
                thought, state, use_state=use_state
            )
            for thought, state in zip(thoughts, states)
        ]
        action_prompts = [
            prompt_text + head.chunk.text for head in heads
        ]
        tails = self._generate_batch_until(
            action_prompts,
            [ACTION_CLOSE],
            self.cfg.max_action_tokens,
            temperature,
            action_seeds,
        )
        return [
            self._assemble_unconstrained_turn(
                head,
                tail,
                use_state=use_state,
                head_seed=thought_seed,
                state_seed=state_seed,
                tail_seed=action_seed,
            )
            for head, tail, thought_seed, state_seed, action_seed in zip(
                heads,
                tails,
                thought_seeds,
                state_seeds,
                action_seeds,
            )
        ]

    def generate_turns_training(self, prompt_text, skillset, task_key, *, count,
                                seed_context=()):
        return super().generate_turns_training(
            prompt_text, skillset, task_key, count=count, seed_context=seed_context)
