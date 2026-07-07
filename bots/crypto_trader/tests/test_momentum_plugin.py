"""Tests for momentum_plugin — phase specs, weight sums, candidate validation."""

import pytest

from crypto_trader.optimize.momentum_plugin import (
    MomentumPlugin,
    HARD_REJECTS,
    PHASE_GATE_CRITERIA,
    PHASE_SCORING_EMPHASIS,
    SCORING_WEIGHTS,
    _phase1_candidates,
    _phase2_candidates,
    _phase3_candidates,
    _phase4_candidates,
    _phase5_candidates,
    _phase6_candidates,
)
from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.optimize.types import (
    GateResult,
    GreedyResult,
    GreedyRound,
    ScoredCandidate,
    Experiment,
)
from crypto_trader.strategy.momentum.config import MomentumConfig


def _make_phase_state_for_retry(
    scoring_retries: dict[int, int] | None = None,
    diagnostic_retries: dict[int, int] | None = None,
):
    from crypto_trader.optimize.phase_state import PhaseState
    state = PhaseState()
    if scoring_retries:
        state.scoring_retries = scoring_retries
    if diagnostic_retries:
        state.diagnostic_retries = diagnostic_retries
    return state


def _make_simple_greedy(accepted_count: int = 0, final_score: float = 0.5):
    accepted = [
        ScoredCandidate(Experiment(f"e{i}", {}), final_score, {})
        for i in range(accepted_count)
    ]
    return GreedyResult(
        accepted_experiments=accepted,
        rejected_experiments=[],
        final_mutations={},
        final_score=final_score,
        base_score=0.45,
        accepted_count=accepted_count,
        rounds=[GreedyRound(1, 5, "test", final_score, 1.0, True)],
    )


class TestScoringWeights:
    def test_weights_sum_to_one(self):
        total = sum(SCORING_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01), (
            f"Scoring weights sum to {total}"
        )

    def test_phase_specific_weights(self):
        """Each phase uses its PHASE_SCORING_EMPHASIS entry."""
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        from crypto_trader.optimize.phase_state import PhaseState
        state = PhaseState()
        for phase in range(1, 7):
            spec = plugin.get_phase_spec(phase, state)
            expected = PHASE_SCORING_EMPHASIS.get(phase, SCORING_WEIGHTS)
            assert spec.scoring_weights == expected

    def test_all_emphasis_profiles_sum_to_one(self):
        """Every PHASE_SCORING_EMPHASIS profile sums to 1.0."""
        for phase, weights in PHASE_SCORING_EMPHASIS.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=0.01), (
                f"Phase {phase} emphasis sums to {total}"
            )

    def test_base_scoring_weights_immutable(self):
        """Base SCORING_WEIGHTS dict is not modified by get_phase_spec."""
        original = dict(SCORING_WEIGHTS)
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        from crypto_trader.optimize.phase_state import PhaseState
        state = PhaseState()
        for phase in range(1, 7):
            plugin.get_phase_spec(phase, state)
        assert SCORING_WEIGHTS == original

    def test_gate_failed_with_accepts_advances_under_immutable_scoring(self):
        """Gate failures do not trigger scoring redesign in the round-3 replay."""
        from crypto_trader.optimize.phase_state import PhaseState
        from crypto_trader.optimize.types import GateResult
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        state = _make_phase_state_for_retry(scoring_retries={1: 0})
        greedy = _make_simple_greedy(accepted_count=2)
        gate = GateResult(passed=False, criteria_results=[], failure_reasons=["test"])
        decision = plugin._decide_action_fn(
            1, {}, state, greedy, gate, SCORING_WEIGHTS, {}, 2, 1)
        assert decision is not None
        assert decision.action == "advance"
        assert decision.scoring_weight_overrides is None


class TestPhaseCandidates:
    def test_phase1_nonempty(self):
        assert len(_phase1_candidates()) > 0

    def test_phase2_nonempty(self):
        assert len(_phase2_candidates()) > 0

    def test_phase3_nonempty(self):
        assert len(_phase3_candidates()) > 0

    def test_phase4_nonempty(self):
        assert len(_phase4_candidates()) > 0

    def test_phase5_nonempty(self):
        assert len(_phase5_candidates()) > 0

    def test_phase6_dynamic(self):
        mutations = {"entry.max_bars_after_confirmation": 2, "stops.atr_buffer_mult": 0.3}
        candidates = _phase6_candidates(mutations)
        assert len(candidates) > 0
        # Should have 4 variants per numeric mutation (x0.8, x0.9, x1.1, x1.2)
        names = [c.name for c in candidates]
        assert any("FINETUNE" in n for n in names)

    def test_phase6_skips_booleans(self):
        mutations = {"entry.entry_on_break": True, "stops.atr_buffer_mult": 0.3}
        candidates = _phase6_candidates(mutations)
        names = [c.name for c in candidates]
        assert not any("entry_on_break" in n for n in names)

    def test_phase6_rounds_ints(self):
        mutations = {"entry.max_bars_after_confirmation": 5}
        candidates = _phase6_candidates(mutations)
        dynamic = [
            c for c in candidates
            if list(c.mutations) == ["entry.max_bars_after_confirmation"]
        ]
        assert dynamic
        for c in dynamic:
            for v in c.mutations.values():
                assert isinstance(v, int)

    def test_historical_round_candidates_present(self):
        phase1 = {c.name for c in _phase1_candidates()}
        phase2 = {c.name for c in _phase2_candidates()}
        phase4 = {c.name for c in _phase4_candidates()}
        phase5 = {c.name for c in _phase5_candidates()}

        assert "TRAIL_TIGHT_0_15" in phase1
        assert "TP2_FRAC_0_40" in phase2
        assert "ETH_BOTH" in phase4
        assert "RISK_B_0_014" in phase5

    def test_phase6_includes_legacy_round_replay(self):
        names = {c.name for c in _phase6_candidates({})}
        assert "FINETUNE_trail.trail_r_ceiling_x0.9" in names
        assert "FINETUNE_risk.risk_pct_b_x1.1" in names

    def test_unique_names(self):
        for phase_fn in [_phase1_candidates, _phase2_candidates, _phase3_candidates,
                         _phase4_candidates, _phase5_candidates]:
            candidates = phase_fn()
            names = [c.name for c in candidates]
            assert len(names) == len(set(names)), f"Duplicate names in {phase_fn.__name__}"


class TestHardRejects:
    def test_immutable_across_phases(self):
        """All phases use the same hard rejects."""
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        from crypto_trader.optimize.phase_state import PhaseState
        state = PhaseState()
        for phase in range(1, 7):
            spec = plugin.get_phase_spec(phase, state)
            assert spec.hard_rejects == HARD_REJECTS

    def test_drawdown_limit(self):
        _, threshold = HARD_REJECTS["max_drawdown_pct"]
        assert threshold == 50.0

    def test_min_trades(self):
        _, threshold = HARD_REJECTS["total_trades"]
        assert threshold == 12

    def test_min_profit_factor(self):
        _, threshold = HARD_REJECTS["profit_factor"]
        assert threshold == 0.8


class TestMomentumPluginProperties:
    def test_name(self):
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        assert plugin.name == "momentum_pullback"

    def test_num_phases(self):
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        assert plugin.num_phases == 6

    def test_ultimate_targets(self):
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        targets = plugin.ultimate_targets
        assert "total_trades" in targets
        assert "sharpe_ratio" in targets

    def test_get_phase_spec(self):
        from crypto_trader.optimize.phase_state import PhaseState
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        state = PhaseState()
        spec = plugin.get_phase_spec(1, state)
        assert spec.phase_num == 1
        assert spec.name == "Trail & Stop Calibration"
        assert len(spec.candidates) > 0
        assert sum(spec.scoring_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_candidate_mutations_valid(self):
        """All mutation keys in candidates should be valid config paths."""
        from crypto_trader.optimize.config_mutator import apply_mutations
        config = MomentumConfig()
        for phase_fn in [_phase1_candidates, _phase2_candidates, _phase3_candidates,
                         _phase4_candidates, _phase5_candidates]:
            for candidate in phase_fn():
                # Should not raise
                apply_mutations(config, candidate.mutations)


class TestPhaseGateCriteria:
    def test_phase1_trades_gate(self):
        """Phase 1 (trail) has trades >= 10."""
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        from crypto_trader.optimize.phase_state import PhaseState
        spec = plugin.get_phase_spec(1, PhaseState())
        trade_criteria = [c for c in spec.gate_criteria if c.metric == "total_trades"]
        assert len(trade_criteria) == 1
        assert trade_criteria[0].threshold == 10

    def test_phase3_tighter_trades_gate(self):
        """Phase 3 (signal) has tighter trades >= 12."""
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        from crypto_trader.optimize.phase_state import PhaseState
        spec = plugin.get_phase_spec(3, PhaseState())
        trade_criteria = [c for c in spec.gate_criteria if c.metric == "total_trades"]
        assert len(trade_criteria) == 1
        assert trade_criteria[0].threshold == 12

    def test_phase_without_override_uses_base(self):
        """Phases without PHASE_GATE_CRITERIA entry use HARD_REJECTS-derived criteria."""
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        from crypto_trader.optimize.phase_state import PhaseState
        spec = plugin.get_phase_spec(2, PhaseState())
        # Phase 2 not in PHASE_GATE_CRITERIA, should use base
        trade_criteria = [c for c in spec.gate_criteria if c.metric == "total_trades"]
        assert len(trade_criteria) == 1
        assert trade_criteria[0].threshold == 12  # from HARD_REJECTS

    def test_hard_rejects_still_immutable(self):
        """HARD_REJECTS remain unchanged by phase-specific gate criteria."""
        original = dict(HARD_REJECTS)
        plugin = MomentumPlugin(BacktestConfig(), MomentumConfig())
        from crypto_trader.optimize.phase_state import PhaseState
        state = PhaseState()
        for phase in range(1, 7):
            plugin.get_phase_spec(phase, state)
        assert HARD_REJECTS == original
