"""HuggingFace generation backend: unconstrained (baseline / data collection)
and Ctrl-G-constrained (<tool> span) generation.

Two-phase constrained decoding per step:
  phase 1 - free generation of the thought, stopping at '<tool>';
  phase 2 - constrained generation of the action span with Ctrl-G's
            ConstraintLogitsProcessor (HMM x token-trie DFA), suffix
            '</tool>' + EOS.

The HMM must be distilled from THIS base model (same vocabulary), on samples
shaped like phase 2: prompt = context + thought + '<tool>', response =
action + '</tool>' + EOS. See scripts/run_rollouts.py for the prompt dump.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .constraints import build_step_dfa, tokenize_continuation
from .prompts import TOOL_CLOSE, TOOL_OPEN

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)


class StopOnStrings(StoppingCriteria):
    """Stop generation when any stop string appears in the generated suffix."""

    def __init__(self, stop_strings, tokenizer, prompt_len: int):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0][self.prompt_len:])
        return any(s in text for s in self.stop_strings)


@dataclass
class GenConfig:
    max_thought_tokens: int = 512
    max_action_tokens: int = 24
    min_action_tokens: int = 1
    beam_size: int = 8          # phase-2 beams (or samples if do_sample)
    do_sample: bool = False
    temperature: float = 1.0
    rollout_temperature: float = 0.7  # unconstrained rollouts / baseline


class HFBackend:
    def __init__(self, model_name_or_path: str, hmm_path: str | None = None,
                 device: str = "cuda", dtype=torch.bfloat16,
                 gen_config: GenConfig | None = None):
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

    # ------------------------------------------------------------ primitives
    def _generate_until(self, prompt_text: str, stop_strings, max_new_tokens: int,
                        temperature: float | None = None) -> str:
        """Greedy/sampled free generation; returns generated text truncated at
        (and including) the first stop string."""
        ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        stopper = StopOnStrings(stop_strings, self.tokenizer, ids.shape[1])
        do_sample = temperature is not None and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                stopping_criteria=StoppingCriteriaList([stopper]),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        for s in stop_strings:
            idx = text.find(s)
            if idx != -1:
                text = text[: idx + len(s)]
        return text

    # -------------------------------------------------------- unconstrained
    def generate_turn_unconstrained(self, prompt_text: str, greedy: bool = False):
        """One full turn: thought + tool call. Returns (thought, action, prefix_to_tool).

        prefix_to_tool = prompt + generated text up to and including '<tool>'
        (this exact string is the distillation prompt for this step)."""
        temp = None if greedy else self.cfg.rollout_temperature
        text = self._generate_until(
            prompt_text, [TOOL_CLOSE],
            self.cfg.max_thought_tokens + self.cfg.max_action_tokens, temp,
        )
        thought = m.group(1).strip() if (m := THINK_RE.search(text)) else ""
        action = m.group(1).strip() if (m := TOOL_RE.search(text)) else None
        tool_open_idx = text.find(TOOL_OPEN)
        prefix_to_tool = (
            prompt_text + text[: tool_open_idx + len(TOOL_OPEN)]
            if tool_open_idx != -1 else None
        )
        return thought, action, prefix_to_tool

    # ---------------------------------------------------------- constrained
    def generate_turn_constrained(self, prompt_text: str, allowed_actions: list[str]):
        """Two-phase: free thought, then Ctrl-G-constrained action span.

        Returns (thought, action, prefix_to_tool)."""
        assert self.hmm_model is not None, "constrained generation needs an HMM"
        import ctrlg

        # phase 1: think until '<tool>'
        text = self._generate_until(
            prompt_text, [TOOL_OPEN], self.cfg.max_thought_tokens, None
        )
        thought = m.group(1).strip() if (m := THINK_RE.search(text)) else ""
        if TOOL_OPEN not in text:
            text = text.rstrip() + "\n" + TOOL_OPEN  # force the tool call open
        prefix_to_tool = prompt_text + text[: text.find(TOOL_OPEN) + len(TOOL_OPEN)]

        # phase 2: constrained action span
        seam = prefix_to_tool[-32:]  # boundary context for continuation tokenization
        dfa_graph, seqs = build_step_dfa(
            allowed_actions, self.tokenizer, self.vocab_size, prompt_tail=seam
        )
        dfa_model = ctrlg.DFAModel(dfa_graph, self.vocab_size).to(self.device)

        prompt_ids = self.tokenizer.encode(prefix_to_tool)
        # '</tool>' follows the action text, so tokenize it with an action tail
        # as boundary context; append the HMM's EOS (a suffix must end with it).
        suffix_ids = tokenize_continuation(
            self.tokenizer, seam + allowed_actions[0], TOOL_CLOSE
        ) + [self.hmm_model.eos_token_id]
        max_action_tokens = max(
            self.cfg.max_action_tokens, max(len(s) for s in seqs.values())
        )

        processor = ctrlg.ConstraintLogitsProcessor(
            self.hmm_model, dfa_model,
            self.cfg.min_action_tokens, max_action_tokens,
            prompt_ids, prefix_ids=[], suffix_ids=suffix_ids,
        )
        processor.hmm_batch_size = self.cfg.beam_size

        input_ids = torch.tensor([prompt_ids], device=self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                do_sample=self.cfg.do_sample,
                num_beams=1 if self.cfg.do_sample else self.cfg.beam_size,
                num_return_sequences=self.cfg.beam_size,
                min_new_tokens=self.cfg.min_action_tokens,
                max_new_tokens=max_action_tokens,
                logits_processor=LogitsProcessorList([processor]),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = ctrlg.extract_generated_ids(
            outputs.tolist(), prompt_ids, suffix_ids, self.hmm_model.eos_token_id
        )
        generated_ids = ctrlg.rank_generated_ids(
            self.model, generated_ids, prompt_ids, suffix_ids
        )
        action = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        if action not in allowed_actions:
            # tokenization drift safety net: snap to the closest allowed action
            action = min(allowed_actions, key=lambda a: abs(len(a) - len(action)))
        return thought, action, prefix_to_tool
