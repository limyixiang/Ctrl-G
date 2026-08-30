import re
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList
)

from .prompts import THINK_CLOSE, THINK_OPEN, ACTION_CLOSE, ACTION_OPEN

THINK_RE = re.compile(rf"{THINK_OPEN}(.*?){THINK_CLOSE}", re.DOTALL)
ACTION_RE = re.compile(rf"{ACTION_OPEN}(.*?){ACTION_CLOSE}", re.DOTALL)

class StopOnStrings(StoppingCriteria):
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
    beam_size: int = 8
    do_sample: bool = False
    temperature: float = 1.0
    rollout_temperature: float = 0.7

class HFBackend:
    def __init__(self, model_name_or_path: str, hmm_path: str | None = None, device: str = "cuda", dtype=torch.bfloat16, gen_config: GenConfig | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype).to(device)
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

    def _generate_until(self, prompt_text: str, stop_strings, max_new_tokens: int, temperature: float | None = None) -> str:
        """Returns generated text truncated at (and including) the first stop string"""
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
                pad_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        for s in stop_strings:
            idx = text.find(s)
            if idx != -1:
                text = text[: idx + len(s)]
        return text

    def generate_turn_unconstrained(self, prompt_text: str, greedy: bool = True):
        """Returns (thought, action, prefix_to_action)
        
        prefix_to_action = prompt + generated text up to and including <action> (this is the distillation prompt)"""
        temp = None if greedy else self.cfg.rollout_temperature
        text = self._generate_until(prompt_text, [ACTION_CLOSE], self.cfg.max_thought_tokens + self.cfg.max_action_tokens, temp)
        thought = m.group(1).strip() if (m := THINK_RE.search(text)) else ""
        action = m.group(1).strip() if (m := ACTION_RE.search(text)) else ""
        action_open_idx = text.find(ACTION_OPEN)
        prefix_to_action = (
            prompt_text + text[:action_open_idx + len(ACTION_OPEN)] if action_open_idx != -1 else None
        )
        return thought, action, prefix_to_action