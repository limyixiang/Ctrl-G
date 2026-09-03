from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .constraints import (
    FiniteActionLogitsProcessor,
    build_action_dfa,
    tokenize_fixed_suffix,
    tokenize_continuation,
)
from .generation import (
    GeneratedChunk,
    TurnGeneration,
    exact_action_tail_token_ids,
    hmm_prefix_token_ids,
    parse_turn,
    token_boundary_for_char_offset,
)
from .prompts import ACTION_CLOSE, ACTION_OPEN

torch.backends.cuda.enable_cudnn_sdp(False)


def _synchronize_if_cuda(device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class StopOnStrings(StoppingCriteria):
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
    max_head_tokens: int = 512
    max_action_tokens: int = 24
    min_action_tokens: int = 1
    beam_size: int = 8
    do_sample: bool = False
    temperature: float = 1.0
    rollout_temperature: float = 0.7
    seed: int = 42
    max_hmm_prefix_tokens: int | None = None


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

    def _generate_head(self, prompt_text: str, *, greedy: bool) -> GeneratedChunk:
        temperature = None if greedy else self.cfg.rollout_temperature
        return self._generate_until(
            prompt_text,
            [ACTION_OPEN],
            self.cfg.max_head_tokens,
            temperature,
        )

    def generate_turn_unconstrained(
        self, prompt_text: str, *, use_decision: bool, greedy: bool = False
    ) -> TurnGeneration:
        """Sample raw model data for evaluation controls or HMM distillation."""

        head = self._generate_head(prompt_text, greedy=greedy)
        used_head_repair = ACTION_OPEN not in head.text
        action_prompt = prompt_text + head.text
        if used_head_repair:
            action_prompt += ACTION_OPEN

        temperature = None if greedy else self.cfg.rollout_temperature
        tail = self._generate_until(
            action_prompt,
            [ACTION_CLOSE],
            self.cfg.max_action_tokens,
            temperature,
        )
        return self._assemble_unconstrained_turn(
            head,
            tail,
            use_decision=use_decision,
            used_head_repair=used_head_repair,
        )

    def generate_turns_unconstrained(
        self,
        prompt_text: str,
        *,
        count: int,
        use_decision: bool,
        greedy: bool = False,
        seed_context: tuple[int, ...] = (),
    ) -> list[TurnGeneration]:
        """Generate multiple candidates, sequentially unless overridden."""

        if count < 1:
            raise ValueError("count must be at least one")
        return [
            self.generate_turn_unconstrained(
                prompt_text, use_decision=use_decision, greedy=greedy
            )
            for _ in range(count)
        ]

    def _assemble_unconstrained_turn(
        self,
        head: GeneratedChunk,
        tail: GeneratedChunk,
        *,
        use_decision: bool,
        used_head_repair: bool,
        head_seed: int | None = None,
        tail_seed: int | None = None,
    ) -> TurnGeneration:
        """Parse and retain the exact token spans from a head/tail pair."""

        parsed = parse_turn(head.text, tail.text, use_decision=use_decision)

        try:
            prefix_ids = tuple(
                hmm_prefix_token_ids(self.tokenizer, list(head.token_ids), head.text)
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
            head_token_ids=head.token_ids,
            action_token_ids=action_ids,
            tail_token_ids=tail.token_ids,
            hmm_prefix_token_ids=prefix_ids,
            head_latency_seconds=head.latency_seconds,
            action_latency_seconds=tail.latency_seconds,
            used_head_repair=used_head_repair,
            hmm_applied=False,
            head_stop_found=head.stop_found,
            head_truncated=head.truncated,
            tail_stop_found=tail.stop_found,
            tail_truncated=tail.truncated,
            tail_span_exact=tail_span_exact,
            head_seed=head_seed,
            tail_seed=tail_seed,
        )


class HFBackend(BaseBackend):
    def __init__(
        self,
        model_name_or_path: str,
        hmm_path: str | None = None,
        device: str = "cuda",
        dtype=torch.bfloat16,
        gen_config: GenConfig | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype
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

    def _generate_dfa_action(
        self, prompt_text: str, allowed_actions: list[str]
    ) -> GeneratedChunk:
        _synchronize_if_cuda(self.device)
        started = time.perf_counter()
        prompt_ids = self.tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        processor = FiniteActionLogitsProcessor(
            self.tokenizer,
            prompt_text,
            allowed_actions,
            prompt_length=prompt_ids.shape[1],
            eos_token_id=self.tokenizer.eos_token_id,
        )
        stopper = StopOnStrings(
            [ACTION_CLOSE], self.tokenizer, prompt_ids.shape[1]
        )
        max_path_length = max(len(path) for path in processor.paths)
        with torch.no_grad():
            output = self.model.generate(
                input_ids=prompt_ids,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature if self.cfg.do_sample else None,
                num_beams=1 if self.cfg.do_sample else self.cfg.beam_size,
                num_return_sequences=1,
                min_new_tokens=self.cfg.min_action_tokens,
                max_new_tokens=max(max_path_length, self.cfg.max_action_tokens),
                logits_processor=LogitsProcessorList([processor]),
                stopping_criteria=StoppingCriteriaList([stopper]),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        token_ids = output[0][prompt_ids.shape[1] :].tolist()
        text, token_ids, stop_found = _crop_chunk_at_stop(
            self.tokenizer, token_ids, [ACTION_CLOSE]
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

    def _generate_hmm_action(
        self,
        prompt_text: str,
        allowed_actions: list[str],
        prefix_ids: list[int],
    ) -> tuple[str, tuple[int, ...], tuple[int, ...], float]:
        if self.hmm_model is None:
            raise ValueError("DFA+HMM condition requires an HMM checkpoint")

        _synchronize_if_cuda(self.device)
        started = time.perf_counter()
        import ctrlg

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        dfa_graph, action_sequences = build_action_dfa(
            allowed_actions, self.tokenizer, self.vocab_size, prompt_text
        )
        dfa_model = ctrlg.DFAModel(dfa_graph, self.vocab_size).to(self.device)

        suffix_variants = set()
        for action, body_ids in action_sequences.items():
            suffix_for_action = tokenize_fixed_suffix(
                self.tokenizer,
                prompt_ids + body_ids,
                ACTION_CLOSE,
            )
            suffix_variants.add(tuple(suffix_for_action))
        if len(suffix_variants) != 1:
            raise ValueError(
                "</action> does not have one stable tokenization across admissible actions"
            )
        suffix_ids = list(next(iter(suffix_variants))) + [
            self.hmm_model.eos_token_id
        ]
        max_body_tokens = max(len(ids) for ids in action_sequences.values())

        processor = ctrlg.ConstraintLogitsProcessor(
            self.hmm_model,
            dfa_model,
            self.cfg.min_action_tokens,
            max_body_tokens,
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
                    self.cfg.beam_size if not self.cfg.do_sample else 1
                ),
                min_new_tokens=self.cfg.min_action_tokens,
                max_new_tokens=max_body_tokens,
                logits_processor=LogitsProcessorList([processor]),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        candidates = ctrlg.extract_generated_ids(
            outputs.tolist(),
            prompt_ids,
            suffix_ids,
            self.hmm_model.eos_token_id,
        )
        candidates = ctrlg.rank_generated_ids(
            self.model, candidates, prompt_ids, suffix_ids
        )
        selected_ids = tuple(candidates[0])
        action_by_ids = {
            tuple(ids): action for action, ids in action_sequences.items()
        }
        if selected_ids not in action_by_ids:
            decoded = self.tokenizer.decode(
                selected_ids, skip_special_tokens=False
            )
            raise RuntimeError(
                "Ctrl-G returned a token path outside the admissible action DFA: "
                f"{decoded!r}"
            )
        emitted_tail_ids = selected_ids + tuple(suffix_ids[:-1])
        _synchronize_if_cuda(self.device)
        latency = time.perf_counter() - started
        return action_by_ids[selected_ids], selected_ids, emitted_tail_ids, latency

    def generate_turn(
        self,
        prompt_text: str,
        allowed_actions: list[str],
        *,
        use_decision: bool,
        use_hmm: bool,
        greedy_head: bool = True,
    ) -> TurnGeneration:
        """Generate a native-think head, then a hard-constrained action."""

        head = self._generate_head(prompt_text, greedy=greedy_head)
        used_head_repair = ACTION_OPEN not in head.text
        action_prompt = prompt_text + head.text
        if used_head_repair:
            action_prompt += ACTION_OPEN

        try:
            prefix_ids = hmm_prefix_token_ids(
                self.tokenizer, list(head.token_ids), head.text
            )
        except ValueError:
            prefix_ids = []

        # A malformed/non-aligned head cannot supply the agreed generated HMM
        # prefix. Keep the episode moving with the same hard DFA and record the
        # parse failure instead of silently feeding a synthetic HMM prefix.
        hmm_skip_reason = None
        if use_hmm:
            if used_head_repair:
                hmm_skip_reason = "synthetic_action_open"
            elif not prefix_ids:
                hmm_skip_reason = "missing_exact_hmm_prefix"
            elif (
                self.cfg.max_hmm_prefix_tokens is not None
                and len(prefix_ids) > self.cfg.max_hmm_prefix_tokens
            ):
                hmm_skip_reason = "hmm_prefix_too_long"
        effective_hmm = use_hmm and hmm_skip_reason is None
        if effective_hmm:
            action, action_ids, tail_ids, action_latency = self._generate_hmm_action(
                action_prompt, allowed_actions, prefix_ids
            )
            raw_tail = action + ACTION_CLOSE
            tail_stop_found = True
            tail_truncated = False
            tail_span_exact = True
        else:
            action_chunk = self._generate_dfa_action(
                action_prompt, allowed_actions
            )
            action_latency = action_chunk.latency_seconds
            raw_tail = action_chunk.text
            tail_ids = action_chunk.token_ids
            tail_stop_found = action_chunk.stop_found
            tail_truncated = action_chunk.truncated
            tail_span_exact = action_chunk.text.endswith(ACTION_CLOSE)
            parsed_action = parse_turn(
                head.text, raw_tail, use_decision=use_decision
            ).action
            action_ids = tuple(
                tokenize_continuation(
                    self.tokenizer, action_prompt, parsed_action
                )
            )

        parsed = parse_turn(head.text, raw_tail, use_decision=use_decision)
        if parsed.action not in allowed_actions:
            raise RuntimeError(
                f"hard DFA emitted non-admissible action {parsed.action!r}"
            )
        return TurnGeneration(
            parsed=parsed,
            head_token_ids=head.token_ids,
            action_token_ids=tuple(action_ids),
            tail_token_ids=tuple(tail_ids),
            hmm_prefix_token_ids=tuple(prefix_ids),
            head_latency_seconds=head.latency_seconds,
            action_latency_seconds=action_latency,
            used_head_repair=used_head_repair,
            hmm_applied=effective_hmm,
            hmm_skip_reason=hmm_skip_reason,
            head_stop_found=head.stop_found,
            head_truncated=head.truncated,
            tail_stop_found=tail_stop_found,
            tail_truncated=tail_truncated,
            tail_span_exact=tail_span_exact,
        )


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
    ) -> GeneratedChunk:
        started = time.perf_counter()
        request_seed = self.cfg.seed if seed is None else seed
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt_text,
            max_tokens=max_new_tokens,
            temperature=(
                temperature if temperature is not None and temperature > 0 else 0.0
            ),
            stop=list(stop_strings),
            extra_body={
                "include_stop_str_in_output": True,
                "return_token_ids": True,
                "seed": request_seed,
                "skip_special_tokens": False,
            },
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
        return GeneratedChunk(
            text=text,
            token_ids=tuple(token_ids),
            stop_found=stop_found,
            truncated=not stop_found,
            latency_seconds=latency,
        )

    def _generate_batch_until(
        self,
        prompt_texts: list[str],
        stop_strings,
        max_new_tokens: int,
        temperature: float | None,
        seeds: list[int],
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
            )

        with ThreadPoolExecutor(max_workers=len(prompt_texts)) as executor:
            return list(executor.map(generate_one, zip(prompt_texts, seeds)))

    def generate_turns_unconstrained(
        self,
        prompt_text: str,
        *,
        count: int,
        use_decision: bool,
        greedy: bool = False,
        seed_context: tuple[int, ...] = (),
    ) -> list[TurnGeneration]:
        """Generate candidates in one head batch followed by one tail batch."""

        if count < 1:
            raise ValueError("count must be at least one")
        temperature = None if greedy else self.cfg.rollout_temperature
        head_seeds = [
            self._stable_seed(self.cfg.seed, seed_context, candidate_index, "head")
            for candidate_index in range(count)
        ]
        tail_seeds = [
            self._stable_seed(self.cfg.seed, seed_context, candidate_index, "tail")
            for candidate_index in range(count)
        ]

        heads = self._generate_batch_until(
            [prompt_text] * count,
            [ACTION_OPEN],
            self.cfg.max_head_tokens,
            temperature,
            head_seeds,
        )
        used_head_repairs = [ACTION_OPEN not in head.text for head in heads]
        action_prompts = [
            prompt_text + head.text + (ACTION_OPEN if used_head_repair else "")
            for head, used_head_repair in zip(heads, used_head_repairs)
        ]
        tails = self._generate_batch_until(
            action_prompts,
            [ACTION_CLOSE],
            self.cfg.max_action_tokens,
            temperature,
            tail_seeds,
        )
        return [
            self._assemble_unconstrained_turn(
                head,
                tail,
                use_decision=use_decision,
                used_head_repair=used_head_repair,
                head_seed=head_seed,
                tail_seed=tail_seed,
            )
            for head, tail, used_head_repair, head_seed, tail_seed in zip(
                heads,
                tails,
                used_head_repairs,
                head_seeds,
                tail_seeds,
            )
        ]
