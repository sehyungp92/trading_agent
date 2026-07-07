"""Tests for phase_state — JSON persistence, advance_phase, per-phase retry counters."""

import json

import pytest

from crypto_trader.optimize.phase_state import PhaseState


class TestPhaseState:
    def test_fresh_state(self):
        state = PhaseState()
        assert state.current_phase == 0
        assert state.completed_phases == []
        assert state.cumulative_mutations == {}

    def test_start_phase(self):
        state = PhaseState()
        state.start_phase(1)
        assert state.current_phase == 1
        assert state.scoring_retries.get(1, 0) == 0
        assert state.diagnostic_retries.get(1, 0) == 0

    def test_advance_phase_merges_mutations(self):
        state = PhaseState()
        state.start_phase(1)
        state.advance_phase(1, {"entry.entry_on_break": True}, {"total_trades": 50.0})
        assert state.cumulative_mutations == {"entry.entry_on_break": True}
        assert 1 in state.completed_phases
        assert state.current_phase == 2

    def test_advance_phase_accumulates(self):
        state = PhaseState()
        state.advance_phase(1, {"a": 1}, {"m": 1.0})
        state.advance_phase(2, {"b": 2}, {"m": 2.0})
        assert state.cumulative_mutations == {"a": 1, "b": 2}
        assert state.completed_phases == [1, 2]
        assert 1 in state.phase_metrics
        assert 2 in state.phase_metrics

    def test_advance_phase_with_result_dict(self):
        """advance_phase extracts final_metrics from result dict."""
        state = PhaseState()
        result = {
            "final_metrics": {"total_trades": 50.0},
            "final_score": 0.8,
            "accepted_count": 3,
        }
        state.advance_phase(1, {"a": 1}, result)
        assert state.phase_metrics[1] == {"total_trades": 50.0}
        assert state.phase_results[1] == result

    def test_per_phase_retry_counters(self):
        state = PhaseState()
        assert state.increment_scoring_retry(1) == 1
        assert state.increment_scoring_retry(1) == 2
        assert state.increment_diagnostic_retry(1) == 1
        # Phase 2 has independent counters
        assert state.increment_scoring_retry(2) == 1
        assert state.scoring_retries[1] == 2
        assert state.scoring_retries[2] == 1

    def test_general_retry_counter(self):
        state = PhaseState()
        assert state.increment_retry(1) == 1
        assert state.increment_retry(1) == 2
        assert state.increment_retry(2) == 1

    def test_record_gate(self):
        state = PhaseState()
        gate_dict = {"passed": False, "failure_reasons": ["too low"]}
        state.record_gate(1, gate_dict)
        assert state.phase_gate_results[1] == gate_dict

    def test_record_result(self):
        state = PhaseState()
        result = {"final_score": 0.8, "accepted_count": 3}
        state.record_result(1, result)
        assert state.phase_results[1] == result

    def test_get_phase_metrics(self):
        state = PhaseState()
        assert state.get_phase_metrics(1) is None
        state.advance_phase(1, {"a": 1}, {"trades": 50.0})
        assert state.get_phase_metrics(1) == {"trades": 50.0}

    def test_complete_phase_timestamps(self):
        state = PhaseState()
        state.start_phase(1)
        assert "started" in state.phase_timestamps[1]
        state.complete_phase(1)
        assert "completed" in state.phase_timestamps[1]

    def test_save_load_roundtrip(self, tmp_path):
        state = PhaseState()
        state.advance_phase(1, {"entry.entry_on_break": True}, {"trades": 50.0})
        state.advance_phase(2, {"stops.atr_buffer_mult": 0.4}, {"trades": 60.0})
        state.increment_scoring_retry(1)
        state.record_gate(1, {"passed": True})

        path = tmp_path / "state.json"
        state.save(path)

        loaded = PhaseState.load(path)
        assert loaded.completed_phases == [1, 2]
        assert loaded.cumulative_mutations == {
            "entry.entry_on_break": True,
            "stops.atr_buffer_mult": 0.4,
        }
        assert loaded.phase_metrics[1] == {"trades": 50.0}
        assert loaded.phase_metrics[2] == {"trades": 60.0}
        assert loaded.scoring_retries[1] == 1
        assert loaded.phase_gate_results[1] == {"passed": True}

    def test_contract_roundtrip_and_validation(self, tmp_path):
        state = PhaseState()
        contract = {"contract_hash": "abc", "profile_hash": "profile"}
        state.ensure_contract(contract)
        state.mark_phase_invalid(1, reason="final_validation_failed", error="boom")

        path = tmp_path / "state.json"
        state.save(path)

        loaded = PhaseState.load(path)
        assert loaded.contract_hash == "abc"
        assert loaded.contract == contract
        assert loaded.invalid_phases[1]["reason"] == "final_validation_failed"
        loaded.ensure_contract(contract)

    def test_contract_mismatch_raises_in_strict_mode(self):
        state = PhaseState(contract_hash="old", contract={"contract_hash": "old"})

        with pytest.raises(RuntimeError, match="contract mismatch"):
            state.ensure_contract({"contract_hash": "new"}, strict=True)

    def test_legacy_completed_state_is_stale_in_strict_mode(self):
        state = PhaseState()
        state.advance_phase(1, {"a": 1}, {"m": 1.0})

        with pytest.raises(RuntimeError, match="contract mismatch"):
            state.ensure_contract({"contract_hash": "new"}, strict=True)

    def test_save_load_int_keys(self, tmp_path):
        """Phase metrics keys should survive JSON round-trip as ints."""
        state = PhaseState()
        state.phase_metrics[3] = {"x": 1.0}
        state.scoring_retries[3] = 2
        state.phase_results[3] = {"score": 0.5}
        path = tmp_path / "state.json"
        state.save(path)

        loaded = PhaseState.load(path)
        assert 3 in loaded.phase_metrics
        assert 3 in loaded.scoring_retries
        assert loaded.scoring_retries[3] == 2
        assert 3 in loaded.phase_results

    def test_rollback_clears_stale_phases(self):
        state = PhaseState()
        state.advance_phase(1, {"a": 1}, {
            "final_metrics": {"m": 1.0}, "final_mutations": {"a": 1},
        })
        state.advance_phase(2, {"b": 2}, {
            "final_metrics": {"m": 2.0}, "final_mutations": {"a": 1, "b": 2},
        })
        state.advance_phase(3, {"c": 3}, {
            "final_metrics": {"m": 3.0}, "final_mutations": {"a": 1, "b": 2, "c": 3},
        })
        state.increment_scoring_retry(2)
        state.record_gate(2, {"passed": True})

        rolled = state.rollback_to_phase(2)
        assert rolled == [2, 3]
        assert state.completed_phases == [1]
        assert state.current_phase == 2
        # Cumulative mutations re-derived from phase 1 only
        assert state.cumulative_mutations == {"a": 1}
        assert 2 not in state.phase_metrics
        assert 3 not in state.phase_metrics
        assert 2 not in state.scoring_retries
        assert 2 not in state.phase_gate_results

    def test_rollback_no_stale(self):
        state = PhaseState()
        state.advance_phase(1, {"a": 1}, {"m": 1.0})
        rolled = state.rollback_to_phase(2)
        assert rolled == []
        assert state.completed_phases == [1]

    def test_rollback_all_phases(self):
        state = PhaseState()
        state.advance_phase(1, {"a": 1}, {
            "final_metrics": {"m": 1.0}, "final_mutations": {"a": 1},
        })
        rolled = state.rollback_to_phase(1)
        assert rolled == [1]
        assert state.completed_phases == []
        assert state.cumulative_mutations == {}

    def test_load_or_create_fresh(self, tmp_path):
        path = tmp_path / "missing.json"
        state = PhaseState.load_or_create(path)
        assert state.current_phase == 0

    def test_load_or_create_existing(self, tmp_path):
        path = tmp_path / "existing.json"
        original = PhaseState()
        original.advance_phase(1, {"a": 1}, {"m": 1.0})
        original.save(path)

        loaded = PhaseState.load_or_create(path)
        assert loaded.completed_phases == [1]
