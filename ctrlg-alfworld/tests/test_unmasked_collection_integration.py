"""Exercise durable collection, selection and resume across whole-turn retries."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from ctrlg_alfworld.backends import BaseBackend
from ctrlg_alfworld.generation import GeneratedChunk
from ctrlg_alfworld.distillation import load_eligible_records, validate_prompt_regime, build_contextual_records

from test_context_hmm import Tokenizer
from test_backend_integration import STATE


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rollouts.py"
SPEC = importlib.util.spec_from_file_location("unmasked_collection", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


@pytest.mark.parametrize("exhausted, accepted_state", [
    (False, "valid"), (False, "malformed"), (False, "truncated"), (True, None),
])
def test_retries_are_saved_and_resume_preserves_nonempty_rejected_attempts(tmp_path, exhausted, accepted_state):
    class Environment:
        actions = []
        def seed(self, seed):
            pass
        def reset(self):
            return ["welcome\n\nroom\n\nYour task is to: put cup"], {
                "extra.gamefile":["/games/pick_and_place_simple/trial/game.tw-pddl"], "admissible_commands":[["look"]]}
        def step(self, actions):
            self.actions.extend(actions)
            done = len(self.actions) == 2
            return ["complete" if done else "Nothing happens."], [0], [done], {
                "won":[done], "admissible_commands":[["look"]]}
    environment = Environment()
    class Factory:
        game_files = ["/games/pick_and_place_simple/trial/game.tw-pddl"]
        def init_env(self, batch_size):
            return environment
    alfworld = ModuleType("alfworld")
    agents = ModuleType("alfworld.agents")
    env_module = ModuleType("alfworld.agents.environment")
    alfworld.agents = agents
    agents.environment = env_module
    env_module.get_environment = lambda kind: lambda *args, **kwargs: Factory()
    class Backend(BaseBackend):
        def __init__(self, model, *, gen_config, **kwargs):
            self.tokenizer = Tokenizer()
            self.cfg = gen_config
            self.attempt = -1
        def _generate_until(self, prompt, stops, budget, temperature):
            if stops == ["</think>"]:
                self.attempt += 1
                text, closed = "reason</think>", True
            elif stops == ["</state>"]:
                if self.attempt == 0:
                    text, closed = "{task:", False
                elif self.attempt == 1 or exhausted:
                    text, closed = "malformed</state>", True
                elif self.attempt == 2 and accepted_state == "malformed":
                    text, closed = "malformed</state>", True
                elif self.attempt == 2 and accepted_state == "truncated":
                    text, closed = "{task:", False
                else:
                    text, closed = STATE + "</state>", True
            else:
                if self.attempt == 0:
                    text, closed = "look", False
                else:
                    text, closed = ("bad action</action>" if self.attempt == 1 or exhausted
                                else "take cup 1 from table 1</action>" if self.attempt == 2
                                else "look</action>"), True
            return GeneratedChunk(text, tuple(self.tokenizer.encode(text)), closed, not closed, .01)
    arguments = [str(SCRIPT), "--backend", "hf", "--model", "character", "--num_episodes", "1",
                 "--max_steps", "2", "--collection_max_attempts", "3", "--out", str(tmp_path)]
    with patch.dict(sys.modules, {"alfworld":alfworld, "alfworld.agents":agents,
                                 "alfworld.agents.environment":env_module}), \
         patch.object(collector, "HFBackend", Backend), patch.object(sys, "argv", arguments):
        collector.main()
    samples = [json.loads(line) for line in (tmp_path/"samples.jsonl").read_text().splitlines()]
    assert len(samples) == (3 if exhausted else 4)
    assert [s["selected_for_execution"] for s in samples] == ([False]*3 if exhausted else [False,False,True,True])
    assert all("hmm_context_token_ids" not in s and "hmm_context_steps" not in s for s in samples)
    metadata = json.loads((tmp_path/"metadata.json").read_text())
    assert "hmm_context_steps" not in metadata and "max_hmm_context_tokens" not in metadata
    assert "hmm_context_steps" not in collector.RESUME_COMPATIBILITY_FIELDS
    assert "max_hmm_context_tokens" not in collector.RESUME_COMPATIBILITY_FIELDS
    assert samples[0]["hmm_target_token_ids"] == Tokenizer().encode("<state>{task:")
    assert not samples[0]["hmm_completed"]
    assert samples[0]["action"] == "look" and samples[0]["used_state_repair"]
    assert "missing_action_close" in samples[0]["execution_rejection_reasons"]
    assert "invalid_action_grammar" in samples[1]["execution_rejection_reasons"]
    assert all(s["state_schema_error"] for s in samples[:2])
    assert metadata["collection_execution_policy"] == collector.COLLECTION_EXECUTION_POLICY
    assert not any(s["constrained_state_collection"] for s in samples)
    rows = load_eligible_records(tmp_path/"samples.jsonl")
    assert len(rows) == len(samples)
    regime = validate_prompt_regime(rows)
    assert regime["collection_execution_policy"] == metadata["collection_execution_policy"]
    contextual = build_contextual_records(Tokenizer(), rows, tmp_path/"history.jsonl", k=1)
    assert len({tuple(s["hmm_context_token_ids"]) for s in contextual[:3]}) == 1
    if exhausted:
        assert environment.actions == []
    else:
        assert environment.actions == ["take cup 1 from table 1", "look"]
        assert not samples[2]["action_was_admissible"]
        selected = samples[2]
        assert selected["state_schema_valid"] is (accepted_state == "valid")
        assert selected["execution_rejection_reasons"] == []
        assert selected["used_state_repair"] is (accepted_state == "truncated")
        histories = [json.loads(line) for line in (tmp_path/"history.jsonl").read_text().splitlines()]
        chosen_audit = histories[0]["steps"][0]["candidates"][2]
        assert chosen_audit["state_schema_valid"] == selected["state_schema_valid"]
        assert chosen_audit["used_state_repair"] == selected["used_state_repair"]
        assert chosen_audit["selected_for_execution"]
        if accepted_state == "truncated":
            assert selected["hmm_target_token_ids"] == Tokenizer().encode("<state>{task:")
            assert selected["synthetic_state_close_token_ids"] == Tokenizer().encode("</state>")
            assert selected["tail_token_ids"] == Tokenizer().encode("take cup 1 from table 1</action>")
            assert not selected["hmm_completed"] and not selected["hmm_appended_eos"]
            assert selected["hmm_action_start_in_target"] is None
            assert selected["hmm_action_exclusion_reason"] == "generated_after_synthetic_state_close"
            assert contextual[2]["hmm_action_start"] is None
        context = contextual[3]["hmm_context_text"]
        assert "Nothing happens." in context and "take cup 1 from table 1" in context
        assert "bad action" not in context and "malformed" not in context
        without_history = build_contextual_records(Tokenizer(), rows, tmp_path/"history.jsonl", k=0)
        assert "take cup 1 from table 1" not in without_history[3]["hmm_context_text"]
        assert "Nothing happens." in without_history[3]["hmm_context_text"]
    recovered = collector.recover_resume_state(tmp_path/"samples.jsonl", tmp_path/"episodes.jsonl",
                                              history_path=tmp_path/"history.jsonl", num_episodes=1)
    assert recovered[0] == 1 and recovered[1] == len(samples)
    assert recovered[-1]["retried_calls"] == 1
    assert recovered[-1]["exhausted_calls"] == int(exhausted)
