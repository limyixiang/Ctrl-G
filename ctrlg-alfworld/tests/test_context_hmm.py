"""Exact small-model checks for context conditioning and censored attempts."""

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ctrlg import DFAModel, HMM
from ctrlg.utils import ConstraintLogitsProcessor, dfa_completion_log_probs
from ctrlg_alfworld.backends import BaseBackend, GenConfig
from ctrlg_alfworld.collection import executable_attempt, training_span, observed_attempt_span
from ctrlg_alfworld.constraints import build_trie_dfa
from ctrlg_alfworld.distillation import (extract_lvd_embeddings, pad_sequences, split_records,
    validate_record, build_contextual_records, summarize_context_usage)
from ctrlg_alfworld.generation import GeneratedChunk
from ctrlg_alfworld.hmm_context import HMM_SEQUENCE_FORMAT, build_hmm_context
from ctrlg_alfworld.prompts import Step
from ctrlg_alfworld.skills import SkillSet


class Tokenizer:
    eos_token_id = 0
    all_special_ids = [0]
    name_or_path = "character"

    def __len__(self):
        return 128

    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages) + "\n<think>"


def small_hmm():
    model = HMM(2, 5, 0)
    model.update_params(torch.tensor([[.8, .2], [.3, .7]]),
        torch.tensor([[.05, .2, .3, .4, .05], [.3, .1, .1, .1, .4]]).log(),
        torch.tensor([.6, .4]).log())
    return model


def paths(model, tokens, initial=None):
    for states in itertools.product(range(model.hidden_states), repeat=len(tokens)):
        probability = float(model.gamma.exp()[states[0]]) if initial is None else float(states[0] == initial)
        for index, (state, token) in enumerate(zip(states, tokens)):
            probability *= float(model.beta.exp()[state, token]) if token >= 0 else 1.0
            if index:
                probability *= float(model.alpha_exp[states[index-1], state])
        yield states, probability


def likelihood(model, tokens, initial=None):
    return sum(p for _, p in paths(model, tokens, initial))


def test_padding_likelihood_and_em_counts_match_exhaustive_latent_paths():
    model = small_hmm()
    sequences = [[1, 2, 0], [2], [1, -1, 3]]
    tensor = torch.tensor([[1, 2, 0], [2, -2, -2], [1, -1, 3]])
    original = tensor.clone()
    af, bf, gf = torch.zeros(2,2), torch.zeros(6,2), torch.zeros(2)
    probabilities = model.forward(tensor)
    model.backward(tensor, probabilities, af, bf, gf)
    expected_a, expected_b, expected_g = torch.zeros_like(af), torch.zeros_like(bf), torch.zeros_like(gf)
    for row, tokens in enumerate(sequences):
        evidence = likelihood(model, tokens)
        assert float(probabilities[-1][row]) == pytest.approx(np.log(evidence), abs=1e-6)
        for states, probability in paths(model, tokens):
            posterior = probability / evidence
            expected_g[states[0]] += posterior
            for index, (state, token) in enumerate(zip(states, tokens)):
                expected_b[token if token >= 0 else 5, state] += posterior
                if index:
                    expected_a[states[index-1], state] += posterior
    torch.testing.assert_close(af * model.alpha_exp, expected_a)
    torch.testing.assert_close(bf, expected_b)
    torch.testing.assert_close(gf, expected_g)
    assert torch.equal(original, tensor)
    assert float(bf[:5].sum()) == pytest.approx(6)


def test_action_future_and_state_guidance_match_complete_sequence_enumeration():
    model = small_hmm()
    action = DFAModel(build_trie_dfa([[4]], 5), 5, dense_token_mask=False)
    terminal = dfa_completion_log_probs(model, action, 1, 1, [0])
    for state in range(2):
        assert float(terminal[state]) == pytest.approx(np.log(likelihood(model, [4,0], state)), abs=1e-6)
    state_dfa = DFAModel(build_trie_dfa([[2], [3]], 5), 5, dense_token_mask=False)
    processor = ConstraintLogitsProcessor(model, state_dfa, 1, 1, [],
        prefix_ids=[1], suffix_ids=[], terminal_log_probs=terminal)
    constrained, plain = processor.compute_logits([(1,)], [(1,1)], 1)
    for token in (2,3):
        assert float(constrained[0,token]) == pytest.approx(np.log(likelihood(model, [1,token,4,0])), abs=2e-6)
        assert float(plain[0,token]) == pytest.approx(np.log(likelihood(model, [1,token])), abs=2e-6)
    # State termination has an action future, rather than a fabricated state EOS.
    assert not torch.allclose(constrained[0,2:4], torch.tensor([
        np.log(likelihood(model, [1,t,0])) for t in (2,3)], dtype=torch.float32))


def test_impossible_terminal_weights_do_not_create_nan_or_completion_mass():
    model = small_hmm()
    dfa = DFAModel(build_trie_dfa([[2]],5),5,dense_token_mask=False)
    processor = ConstraintLogitsProcessor(model,dfa,1,1,[],suffix_ids=[],
                                          terminal_log_probs=torch.full((2,),-torch.inf))
    scores, _ = processor.compute_logits([()],[(1,1)],1)
    assert not torch.isnan(scores).any()
    assert (scores < -1e29).all()


def test_context_contains_only_observable_recent_executed_turns_and_logs_shortening():
    tokenizer = Tokenizer()
    history = [Step(thought="secret", state="hidden", action="look", observation="old"),
               Step(thought="thought", state="state", action="take cup 1 from table 1", observation="taken")]
    context = build_hmm_context(tokenizer, "put cup", "room", "taken", history, k=1)
    value = json.loads(context.text[len("<context>"):-len("</context>\n")])
    assert value["history"] == [{"action":"take cup 1 from table 1"}]
    assert value["current_observation"] == "taken"
    assert "secret" not in context.text and "hidden" not in context.text
    short = build_hmm_context(tokenizer, "put cup", "x"*1000, "taken", history, k=1, max_tokens=160)
    assert short.truncated and len(short.token_ids) <= 160
    assert "taken" in short.text and "put cup" in short.text


def make_attempt(state, tail="", *, state_closed=False, action_closed=False):
    backend = object.__new__(BaseBackend)
    backend.tokenizer = Tokenizer()
    chunk = lambda text, closed: GeneratedChunk(text, tuple(backend.tokenizer.encode(text)), closed, not closed, .01)
    head = backend._assemble_head(chunk("reason</think>", True), chunk(state, state_closed), use_state=True)
    return backend._assemble_unconstrained_turn(head, chunk(tail, action_closed), use_state=True)


def record(turn, context, *, game="one", sample=0):
    return {**training_span(turn, context, 0), "prompt_token_ids":[7,8],
        "head_token_ids":list(turn.head_token_ids), "tail_token_ids":list(turn.tail_token_ids),
        "hmm_prefix_token_ids":list(turn.hmm_prefix_token_ids), "episode":0, "step":0,
        "sample":sample, "gamefile":game}


def raw_record(turn, *, game="one", episode=0, step=0, sample=0):
    return {**observed_attempt_span(turn, 0), "prompt_token_ids":[7,8],
        "head_token_ids":list(turn.head_token_ids), "tail_token_ids":list(turn.tail_token_ids),
        "hmm_prefix_token_ids":list(turn.hmm_prefix_token_ids), "action":turn.parsed.action,
        "episode":episode, "step":step, "sample":sample, "gamefile":game}


def test_builder_reuses_raw_targets_and_only_past_executed_history(tmp_path):
    steps = [dict(step=i, action=f"executed {i}", observation=f"feedback {i}",
                  selected_sample=None, state="secret state", candidates=["rejected action"])
             for i in range(4)]
    history = tmp_path/"history.jsonl"
    history.write_text(json.dumps(dict(episode=0, gamefile="one", task_description="task",
        initial_observation="room", steps=steps))+"\n")
    rows = [raw_record(make_attempt("partial"), step=3, sample=i) for i in range(2)]
    original = json.dumps(rows, sort_keys=True)
    outputs = [build_contextual_records(Tokenizer(), rows, history, k=k) for k in (0,1,3)]
    assert json.dumps(rows, sort_keys=True) == original
    for output in outputs:
        assert [r["hmm_target_token_ids"] for r in output] == [r["hmm_target_token_ids"] for r in rows]
        assert output[0]["hmm_context_token_ids"] == output[1]["hmm_context_token_ids"]
        assert "feedback 2" in output[0]["hmm_context_text"]
        assert "feedback 3" not in output[0]["hmm_context_text"]
        assert "secret state" not in output[0]["hmm_context_text"]
        assert "rejected action" not in output[0]["hmm_context_text"]
    assert outputs[0][0]["hmm_context_history_turns"] == 0
    assert outputs[1][0]["hmm_context_history_turns"] == 1
    assert outputs[2][0]["hmm_context_history_turns"] == 3
    assert "executed 0" not in outputs[1][0]["hmm_context_text"]
    assert "executed 0" in outputs[2][0]["hmm_context_text"]
    assert summarize_context_usage(outputs[2], 1024)["steps"] == 1
    shortened = build_contextual_records(Tokenizer(), rows, history, k=3, max_tokens=150)
    assert all(len(r["hmm_context_token_ids"]) <= 150 for r in shortened)
    assert shortened[0]["hmm_context_history_turns"] < 3
    assert shortened[0]["hmm_target_token_ids"] == rows[0]["hmm_target_token_ids"]
    assert summarize_context_usage(shortened, 150)["truncation_rate"] == 1
    with pytest.raises(ValueError, match="budget cannot hold"):
        build_contextual_records(Tokenizer(), rows, history, k=3, max_tokens=10)
    mismatched = [{**rows[0], "gamefile":"other"}]
    with pytest.raises(ValueError, match="episode/game history"):
        build_contextual_records(Tokenizer(), mismatched, history)
    steps[0]["action"] = ""
    history.write_text(json.dumps(dict(episode=0, gamefile="one", task_description="task",
        initial_observation="room", steps=steps))+"\n")
    with pytest.raises(ValueError, match="unexecuted or incomplete"):
        build_contextual_records(Tokenizer(), rows, history, k=0)


def test_builder_rebuilds_older_unmasked_context_without_changing_targets(tmp_path):
    context = build_hmm_context(Tokenizer(), "task", "room", "room", [], k=0)
    row = record(make_attempt("bad</state>", "go to", state_closed=True), context)
    row.pop("rollout_sample_format")
    row.pop("hmm_action_start_in_target")
    row.update(step=1, action="go to")
    history = tmp_path/"history.jsonl"
    history.write_text(json.dumps(dict(episode=0, gamefile="one", task_description="task",
        initial_observation="room", steps=[dict(step=0, action="look", observation="seen"),
                                           dict(step=1, action="", observation="seen")]))+"\n")
    rebuilt = build_contextual_records(Tokenizer(), [row], history, k=1)[0]
    assert rebuilt["hmm_target_token_ids"] == row["hmm_target_token_ids"]
    assert rebuilt["hmm_action_start"] - len(rebuilt["hmm_context_token_ids"]) == (
        row["hmm_action_start"] - len(row["hmm_context_token_ids"]))
    assert "look" in rebuilt["hmm_context_text"]


def test_all_attempts_are_separate_observed_prefixes_without_synthetic_closers():
    context = build_hmm_context(Tokenizer(), "task", "room", "room", [], k=1)
    partial = make_attempt("{task:")
    malformed = make_attempt("garbage</state>", "nonsense</action>", state_closed=True, action_closed=True)
    action_partial = make_attempt("garbage</state>", "go to", state_closed=True)
    rows = [record(t, context, sample=i) for i,t in enumerate((partial,malformed,action_partial))]
    for row in rows:
        validate_record(row)
        assert row["distill_eligible"] and row["hmm_context_token_ids"] == list(context.token_ids)
    assert rows[0]["hmm_target_token_ids"] == Tokenizer().encode("<state>{task:")
    assert not rows[0]["hmm_completed"] and not rows[2]["hmm_completed"]
    assert rows[1]["hmm_completed"] and rows[1]["hmm_appended_eos"]
    padded = pad_sequences(rows, eos_token_id=0)
    assert (padded[0,len(rows[0]["hmm_sequence_token_ids"]):] == -2).all()
    assert not executable_attempt(malformed)
    # Actual model EOS remains observed, even without a protocol close.
    eos = record(make_attempt("{task:\0"), context)
    assert eos["hmm_completed"] and not eos["hmm_appended_eos"]
    assert eos["hmm_target_token_ids"][-1] == 0


def test_game_split_keeps_attempt_siblings_and_all_steps_together():
    rows = [dict(hmm_sequence_format=HMM_SEQUENCE_FORMAT, gamefile=game, episode=episode,
                 step=step, sample=sample) for game,episode in (("a",0),("a",1),("b",2),("c",3))
            for step in (0,1) for sample in (0,1)]
    train, dev = split_records(rows, dev_fraction=.25, seed=1)
    assert {r["gamefile"] for r in train}.isdisjoint({r["gamefile"] for r in dev})
    assert len(train)+len(dev) == len(rows)


def test_context_and_teacher_embeddings_align_to_their_causal_predecessors():
    class Model:
        device = "cpu"
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(hidden_states=(input_ids.float().unsqueeze(-1),))
    context = build_hmm_context(Tokenizer(), "task", "room", "room", [])
    attempt = make_attempt("partial")
    row = record(attempt, context)
    sequences, embeddings = extract_lvd_embeddings(Model(), [row], eos_token_id=0)
    assert sequences[0].tolist() == row["hmm_sequence_token_ids"]
    n = len(context.token_ids)
    assert embeddings[0,:n,0].tolist() == [8]+list(context.token_ids[:-1])
    start = row["hmm_target_start_in_head"]
    target_n = len(row["hmm_target_token_ids"])
    teacher = row["prompt_token_ids"] + row["head_token_ids"]
    assert embeddings[0,n:,0].tolist() == teacher[2+start-1:2+start+target_n-1]


def test_output_conditional_likelihood_excludes_context_and_padding():
    model = small_hmm()
    tensor = torch.tensor([[1,2,0],[1,3,-2]])
    actual = model.conditional_loglikelihood(tensor, [1,1], batch_size=1)
    expected = sum(np.log(likelihood(model,t)) - np.log(likelihood(model,[1])) for t in ([1,2,0],[1,3]))
    assert float(actual) == pytest.approx(expected, abs=1e-6)


def test_censored_em_update_improves_observed_likelihood():
    model = small_hmm()
    tensor = torch.tensor([[1,2,0],[2,-2,-2],[1,3,-2]])
    before = model.forward(tensor)[-1].sum()
    af, bf, gf = torch.zeros(2,2), torch.zeros(6,2), torch.zeros(2)
    model.backward(tensor, model.forward(tensor), af, bf, gf)
    af *= model.alpha_exp
    af += 1e-7
    emissions = bf[:5].T.contiguous() + 1e-7
    model.update_params(af/af.sum(-1,keepdim=True),
                        (emissions/emissions.sum(-1,keepdim=True)).log(),
                        (gf/gf.sum()).log())
    assert model.forward(tensor)[-1].sum() > before


def test_dropout_never_manufactures_eos_or_changes_padding():
    import importlib.util
    source = Path(__file__).resolve().parents[2]/"distillation/train_hmm.py"
    spec = importlib.util.spec_from_file_location("censored_em", source)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    tokens = torch.tensor([[1,2,-2],[1,0,-2]])
    result = trainer.apply_dropout(tokens,1.0,5,0)
    assert result.tolist() == [[-1,-1,-2],[-1,0,-2]]


def test_lvd_counts_actual_eos_but_no_transition_into_eos_for_censored_rows(tmp_path):
    import importlib.util
    source = Path(__file__).resolve().parents[2]/"distillation/lvd_hmm.py"
    spec = importlib.util.spec_from_file_location("censored_lvd", source)
    lvd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lvd)
    sequences = torch.tensor([[1,2,3,-2], [2,3,0,-2]])
    torch.save(sequences, tmp_path/"seqs")
    torch.save(torch.zeros(2,4,2), tmp_path/"embeddings")
    _, _, lengths, offsets = lvd.load_examples(tmp_path/"seqs", tmp_path/"embeddings", 0)
    assert lengths.tolist() == [3,2]
    alpha, beta, gamma = torch.zeros(3,3), torch.zeros(3,5), torch.zeros(3)
    lvd.update_flows(alpha,beta,gamma,np.array([0,1,0,1,1]),np.array([1,2,3,2,3]),
                    lengths,offsets,4,3,0,completed=np.array([False,True]))
    assert alpha[0,2] == 0 and alpha[1,2] == 1
    assert alpha[0,1] == 1 and alpha[1,0] == 1 and alpha[1,1] == 1
    assert gamma.tolist() == [1,1,0]


def test_dataset_cli_writes_all_attempts_and_explicit_lengths(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import transformers
    source = Path(__file__).resolve().parents[1]/"scripts/build_hmm_data.py"
    spec = importlib.util.spec_from_file_location("censored_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    rows = []
    histories = []
    for episode, game in enumerate(("a","b")):
        histories.append(dict(episode=episode, gamefile=game, task_description="task",
            initial_observation="room", steps=[dict(step=i, action=f"executed {i}",
                observation=f"feedback {i}", selected_sample=None) for i in range(3)]))
        for step in range(3):
            for sample, attempt in enumerate((make_attempt("partial"),
                make_attempt("bad</state>", "bad</action>", state_closed=True, action_closed=True))):
                row = raw_record(attempt, game=game, episode=episode, step=step, sample=sample)
                row.update(use_state=True,constrained_state_collection=False,
                           no_oracle_filtering=True,schema_version=1,policy_version=1,
                           skills_sha256="skills",model="character",tokenizer="character",
                           prompt_format="state_with_persistent_history_no_oracle_v1",
                           collection_max_attempts=5,parse_ok=False)
                rows.append(row)
    samples = tmp_path/"samples.jsonl"
    samples.write_text("".join(json.dumps(r)+"\n" for r in rows))
    history = tmp_path/"history.jsonl"
    history.write_text("".join(json.dumps(r)+"\n" for r in histories))
    monkeypatch.setattr(transformers.AutoTokenizer,"from_pretrained",lambda *a,**kw: Tokenizer())
    output = tmp_path/"data"
    monkeypatch.setattr(sys,"argv",[str(source),"--samples",str(samples),"--tokenizer","character",
                                   "--output_dir",str(output),"--dev_fraction",".5"])
    builder.main()
    metadata = json.loads((output/"alfworld_actions.metadata.json").read_text())
    assert metadata["selection_policy"] == "all_observed_attempts"
    assert metadata["train_records"] == metadata["dev_records"] == 6
    assert metadata["completed_records"] == metadata["truncated_records"] == 6
    assert metadata["hmm_context_steps"] == 1 and metadata["max_hmm_context_tokens"] == 1024
    assert metadata["context_construction_stage"] == "dataset_build"
    assert metadata["hmm_context_usage"]["steps"] == 6
    assert metadata["history_source_sha256"]
    dev = torch.load(output/"alfworld_actions.dev",weights_only=True)
    spans = json.loads((output/"alfworld_actions.dev.spans.json").read_text())
    assert len(spans) == len(dev) == 6
    partial = next(i for i,s in enumerate(spans) if not s["completed"])
    assert (dev[partial,spans[partial]["length"]:] == -2).all()
    second = tmp_path/"k2"
    monkeypatch.setattr(sys,"argv",[str(source),"--samples",str(samples),"--tokenizer","character",
        "--output_dir",str(second),"--dev_fraction",".5","--hmm_context_steps","2",
        "--max_hmm_context_tokens","768"])
    builder.main()
    second_meta = json.loads((second/"alfworld_actions.metadata.json").read_text())
    assert second_meta["hmm_context_steps"] == 2 and second_meta["max_hmm_context_tokens"] == 768
    assert second_meta["source_sha256"] == metadata["source_sha256"]
    second_dev = torch.load(second/"alfworld_actions.dev",weights_only=True)
    second_spans = json.loads((second/"alfworld_actions.dev.spans.json").read_text())
    for first_span, second_span, first_row, second_row in zip(spans, second_spans, dev, second_dev):
        assert (first_span["episode"],first_span["step"],first_span["sample"]) == (
            second_span["episode"],second_span["step"],second_span["sample"])
        assert first_row[first_span["context_length"]:first_span["length"]].tolist() == (
            second_row[second_span["context_length"]:second_span["length"]].tolist())
    assert any(a["context_length"] != b["context_length"] for a,b in zip(spans,second_spans))
