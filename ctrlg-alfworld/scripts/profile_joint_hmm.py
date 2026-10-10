"""Profile joint Ctrl-G kernels with a real tokenizer and an untrained small HMM.

This checks resource use and token alignment, not model quality or task success.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctrlg import DFAModel, HMM
from ctrlg.utils import ConstraintLogitsProcessor, dfa_completion_log_probs, matmul_a_logb
from ctrlg_alfworld.constraints import action_span_pattern, lift_character_fsm, regex_fsm, state_span_pattern
from ctrlg_alfworld.hmm_context import build_hmm_context
from ctrlg_alfworld.skills import SkillSet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--hidden-states", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    skills = SkillSet.from_file(Path(__file__).resolve().parents[1]/"templates/SKILLS.md")
    context = build_hmm_context(tokenizer, "put the cup in cabinet 1", "You see countertop 1 and cabinet 1.",
                                "You see cup 1 on countertop 1.", [])
    model = HMM(args.hidden_states, len(tokenizer), tokenizer.eos_token_id).to(args.device)
    results = {}
    graphs = {}
    for phase, pattern in (("state", state_span_pattern(skills.state_schemas["put"], skills)),
                           ("action", action_span_pattern(skills))):
        started = time.perf_counter()
        graph = lift_character_fsm(regex_fsm(pattern), tokenizer, len(tokenizer))
        graphs[phase] = DFAModel(graph, len(tokenizer), dense_token_mask=False).to(args.device)
        results[phase] = dict(dfa_states=graphs[phase].num_states, dfa_edges=len(graph["edges"]),
                              compile_seconds=time.perf_counter()-started)
    started = time.perf_counter()
    terminal = dfa_completion_log_probs(model, graphs["action"], 1, 32, [tokenizer.eos_token_id])
    for token in reversed(tokenizer.encode("<action>", add_special_tokens=False)):
        terminal = model.beta[:,token] + matmul_a_logb(model.alpha_exp, terminal[:,None]).squeeze(-1)
    results["action_future_seconds"] = time.perf_counter()-started
    for phase, budget in (("state",512),("action",32)):
        started = time.perf_counter()
        prefix = list(context.token_ids)+tokenizer.encode(f"<{phase}>", add_special_tokens=False)
        processor = ConstraintLogitsProcessor(model, graphs[phase], 1, budget, [], prefix_ids=prefix,
            suffix_ids=[] if phase == "state" else [tokenizer.eos_token_id],
            terminal_log_probs=terminal if phase == "state" else None)
        results[phase]["lookahead_seconds"] = time.perf_counter()-started
        results[phase]["cache_bytes"] = sum(t.numel()*t.element_size() for t in processor.C_cache.values())
        started = time.perf_counter()
        scores = processor(torch.empty(1,0,dtype=torch.long,device=args.device),
                           torch.zeros(1,len(tokenizer),device=args.device))
        results[phase]["first_score_seconds"] = time.perf_counter()-started
        results[phase]["finite_scores"] = bool(torch.isfinite(scores).all())
        del processor
    results.update(tokenizer=args.tokenizer, hidden_states=args.hidden_states, device=args.device,
                   context_tokens=len(context.token_ids), vocab_size=len(tokenizer),
                   trained=False, llm_generation=False, torch_compile_disabled=bool(__import__("os").environ.get("TORCHDYNAMO_DISABLE")))
    # C caches are linear in hidden-state count; transition work scales quadratically.
    results["estimated_4096_cache_bytes"] = sum(results[p]["cache_bytes"] for p in ("state","action"))*4096//args.hidden_states
    output = Path(args.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
