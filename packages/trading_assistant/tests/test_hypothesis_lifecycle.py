# tests/test_hypothesis_lifecycle.py
"""Tests for the adaptive hypothesis library lifecycle."""
from __future__ import annotations



from trading_assistant.skills.hypothesis_library import (
    HypothesisLibrary,
    Hypothesis,
    get_all,
    get_by_category,
    get_relevant,
)


class TestLegacyAPI:
    """Verify backward compatibility of module-level functions."""

    def test_get_all(self):
        result = get_all()
        assert len(result) == 12
        assert all(isinstance(h, Hypothesis) for h in result)

    def test_get_by_category(self):
        result = get_by_category("signal_decay")
        assert len(result) == 2
        assert all(h.category == "signal_decay" for h in result)

    def test_get_relevant_with_signal_keyword(self):
        class FakeSuggestion:
            title = "Signal decay detected"
            description = ""
        result = get_relevant([FakeSuggestion()])
        assert any(h.category == "signal_decay" for h in result)

    def test_get_relevant_empty(self):
        assert get_relevant([]) == []


class TestHypothesisLibrary:
    def test_seed_from_static(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        records = lib.get_all_records()
        assert len(records) == 12
        assert records[0].status == "active"
        assert records[0].times_proposed == 0

    def test_seed_idempotent(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        lib.seed_if_needed()  # should not duplicate
        assert len(lib.get_all_records()) == 12

    def test_record_proposal(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        lib.record_proposal("h-exit-trailing")
        records = lib.get_all_records()
        trailing = [r for r in records if r.id == "h-exit-trailing"][0]
        assert trailing.times_proposed == 1
        assert trailing.last_proposed_at != ""

    def test_record_acceptance(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        lib.record_acceptance("h-filter-loosen")
        records = lib.get_all_records()
        h = [r for r in records if r.id == "h-filter-loosen"][0]
        assert h.times_accepted == 1

    def test_record_rejection(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        lib.record_rejection("h-regime-pause")
        records = lib.get_all_records()
        h = [r for r in records if r.id == "h-regime-pause"][0]
        assert h.times_rejected == 1
        assert h.status == "active"  # not yet retired (only 1 rejection)

    def test_auto_retirement_neutral_retires(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        # 3 rejections + 0 outcomes = effectiveness 0.0 (neutral) → retires
        # A hypothesis with zero evidence and 3+ rejections should not survive
        for _ in range(3):
            lib.record_rejection("h-crowding-diversify")
        records = lib.get_all_records()
        h = [r for r in records if r.id == "h-crowding-diversify"][0]
        assert h.status == "retired"
        assert h.times_rejected == 3

    def test_auto_retirement_negative_effectiveness(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        # Negative outcome first → effectiveness < 0, then 3 rejections → retired
        lib.record_outcome("h-crowding-diversify", positive=False)
        for _ in range(3):
            lib.record_rejection("h-crowding-diversify")
        records = lib.get_all_records()
        h = [r for r in records if r.id == "h-crowding-diversify"][0]
        assert h.status == "retired"
        assert h.times_rejected == 3

    def test_retirement_not_triggered_with_positive_outcomes(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        lib.record_proposal("h-regime-pause")
        lib.record_outcome("h-regime-pause", positive=True)
        lib.record_outcome("h-regime-pause", positive=True)
        for _ in range(3):
            lib.record_rejection("h-regime-pause")
        records = lib.get_all_records()
        h = [r for r in records if r.id == "h-regime-pause"][0]
        # effectiveness = (2 - 0) / 1 = 2.0 > 0 → still active
        assert h.status == "active"

    def test_record_outcome(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        lib.record_outcome("h-exit-trailing", positive=True)
        lib.record_outcome("h-exit-trailing", positive=False)
        records = lib.get_all_records()
        h = [r for r in records if r.id == "h-exit-trailing"][0]
        assert h.outcomes_positive == 1
        assert h.outcomes_negative == 1

    def test_get_active_excludes_retired(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        lib.record_outcome("h-crowding-diversify", positive=False)
        for _ in range(3):
            lib.record_rejection("h-crowding-diversify")
        active = lib.get_active()
        assert not any(r.id == "h-crowding-diversify" for r in active)
        assert len(active) == 11  # 12 - 1 retired

    def test_add_candidate(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        hyp_id = lib.add_candidate(
            title="New pattern detected",
            category="custom",
            description="Auto-generated from consolidation",
            evidence="50 occurrences",
        )
        assert hyp_id.startswith("hyp_")
        records = lib.get_all_records()
        assert len(records) == 13
        candidate = [r for r in records if r.id == hyp_id][0]
        assert candidate.status == "candidate"
        assert candidate.title == "New pattern detected"

    def test_add_candidate_dedup(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        id1 = lib.add_candidate("X", "cat", "desc")
        id2 = lib.add_candidate("X", "cat", "desc")
        assert id1 == id2
        assert len(lib.get_all_records()) == 13  # 12 + 1

    def test_get_track_record(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        lib.record_proposal("h-exit-trailing")
        lib.record_outcome("h-exit-trailing", positive=True)
        track = lib.get_track_record()
        assert "h-exit-trailing" in track
        assert track["h-exit-trailing"]["effectiveness"] == 1.0
        assert track["h-exit-trailing"]["times_proposed"] == 1

    def test_effectiveness_property(self, tmp_path):
        lib = HypothesisLibrary(tmp_path)
        lib.seed_if_needed()
        lib.record_proposal("h-filter-loosen")
        lib.record_proposal("h-filter-loosen")
        lib.record_outcome("h-filter-loosen", positive=True)
        lib.record_outcome("h-filter-loosen", positive=False)
        records = lib.get_all_records()
        h = [r for r in records if r.id == "h-filter-loosen"][0]
        # effectiveness = (1 - 1) / 2 = 0.0
        assert h.effectiveness == 0.0
