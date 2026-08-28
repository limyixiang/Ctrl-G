"""Prompt assembly: system prompt (instructions + SKILLS.md + few-shot),
transcript rendering, and ReAct-prompt conversion to <tool> format.

Design: each step re-renders the episode as [system, user(transcript)] and the
model continues with `<think>...</think>\\n<tool>action</tool>`. Past thoughts
are kept in the transcript as plain text (ReAct-style). Rendering goes through
`tokenizer.apply_chat_template(..., tokenize=False)` so the exact prompt
string is available both for generation and for the distillation prompt dump
(the vLLM sampler consumes raw prompt strings).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TOOL_OPEN, TOOL_CLOSE = "<tool>", "</tool>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

# ReAct gamefile-prefix -> few-shot key mapping (ysymyth/ReAct alfworld.ipynb)
TASK_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

SYSTEM_TEMPLATE = """You are an agent interacting with a household environment to solve a task.

At every turn, first reason about what to do next, then emit EXACTLY ONE action
wrapped in {tool_open}{tool_close} tags, e.g. {tool_open}go to countertop 1{tool_close}.
The environment observation follows each action on a line starting with "OBS:".

The available actions ("skills"), including the preconditions under which each
action is admissible, are documented below.

{skills_md}

Here are {n_examples} example episodes:

{examples}
"""


def convert_react_example(text: str) -> str:
    """Convert a ReAct-format episode ('> think: ...' / '> action') into the
    <think>/<tool>/OBS transcript format used by this harness."""
    lines = text.strip().split("\n")
    out, pending_thoughts = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("> think:"):
            pending_thoughts.append(line[len("> think:"):].strip())
            # skip the 'OK.' observation line if present
            if i + 1 < len(lines) and lines[i + 1].strip() == "OK.":
                i += 1
        elif line.startswith(">"):
            action = line[1:].strip()
            if pending_thoughts:
                out.append(f"{THINK_OPEN}{' '.join(pending_thoughts)}{THINK_CLOSE}")
                pending_thoughts = []
            out.append(f"{TOOL_OPEN}{action}{TOOL_CLOSE}")
            # following non-'>' lines are the observation
            obs_lines = []
            while i + 1 < len(lines) and not lines[i + 1].startswith(">"):
                obs_lines.append(lines[i + 1])
                i += 1
            if obs_lines:
                out.append("OBS: " + " ".join(l.strip() for l in obs_lines))
            out.append("")
        else:
            out.append(line)  # header: room description / task line
        i += 1
    return "\n".join(out).strip()


def load_few_shot(json_path: str | Path, task_key: str, n: int = 2) -> list[str]:
    """Load and convert n ReAct examples for a task type key (e.g. 'put')."""
    d = json.loads(Path(json_path).read_text())
    return [convert_react_example(d[f"react_{task_key}_{i}"]) for i in range(n)]


def task_key_from_gamefile(gamefile: str) -> str:
    name = "/".join(gamefile.split("/")[-3:-1])
    for prefix, key in TASK_PREFIXES.items():
        if name.startswith(prefix):
            return key
    raise ValueError(f"Unknown task type for gamefile: {gamefile}")


def build_system_prompt(skills_md: str, examples: list[str]) -> str:
    joined = "\n\n---\n\n".join(examples)
    return SYSTEM_TEMPLATE.format(
        tool_open=TOOL_OPEN,
        tool_close=TOOL_CLOSE,
        skills_md=skills_md.strip(),
        n_examples=len(examples),
        examples=joined,
    )


@dataclass
class Step:
    thought: str
    action: str
    observation: str


def build_transcript(initial_obs: str, steps: list[Step]) -> str:
    parts = [initial_obs.strip(), ""]
    for s in steps:
        if s.thought:
            parts.append(f"{THINK_OPEN}{s.thought}{THINK_CLOSE}")
        parts.append(f"{TOOL_OPEN}{s.action}{TOOL_CLOSE}")
        parts.append(f"OBS: {s.observation.strip()}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def render_prompt(tokenizer, system_prompt: str, transcript: str,
                  enable_thinking: bool = False) -> str:
    """Exact prompt string the model is conditioned on at this step.

    enable_thinking=False by default: the harness manages reasoning via its own
    plain-text <think> convention inside the transcript, which keeps prompt
    strings identical between HF generation and vLLM distillation sampling.
    Set True to use Qwen's native thinking mode instead (then treat the
    template's own <think> block as the thought channel).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:  # tokenizer without enable_thinking kwarg
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
