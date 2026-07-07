"""Tests for the 4 crypto perpetual detectors in StrategyEngine."""
from __future__ import annotations


from trading_assistant.analysis.strategy_engine import StrategyEngine
from trading_assistant.schemas.strategy_profile import StrategyArchetype, StrategyProfile, StrategyRegistry
from trading_assistant.schemas.strategy_suggestions import RefinementReport, SuggestionTier
from trading_assistant.schemas.weekly_metrics import BotWeeklySummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**kwargs) -> StrategyEngine:
    defaults = {"week_start": "2026-04-28", "week_end": "2026-05-04"}
    defaults.update(kwargs)
    return StrategyEngine(**defaults)


def _make_crypto_registry(
    bot_id: str = "crypto_trader",
    archetype: StrategyArchetype = StrategyArchetype.MOMENTUM_PULLBACK_CRYPTO,
) -> StrategyRegistry:
    return StrategyRegistry(strategies={
        "MomentumPullback_M15": StrategyProfile(
            display_name="Momentum Pullback M15",
            bot_id=bot_id,
            family="crypto",
            archetype=archetype,
            asset_class="crypto_perpetual",
        ),
    })


def _make_institutional_registry(
    bot_id: str = "institutional_bot",
) -> StrategyRegistry:
    return StrategyRegistry(strategies={
        "InstitutionalAnchor_H1": StrategyProfile(
            display_name="Institutional Anchor H1",
            bot_id=bot_id,
            family="crypto",
            archetype=StrategyArchetype.INSTITUTIONAL_ANCHOR,
            asset_class="crypto_perpetual",
        ),
    })


def _minimal_bot_summary(bot_id: str = "crypto_trader") -> BotWeeklySummary:
    return BotWeeklySummary(
        week_start="2026-04-28",
        week_end="2026-05-04",
        bot_id=bot_id,
        total_trades=30,
        win_count=15,
        loss_count=15,
        net_pnl=-500.0,
        avg_win=100.0,
        avg_loss=-100.0,
    )


# ---------------------------------------------------------------------------
# detect_funding_impact
# ---------------------------------------------------------------------------

def test_detect_funding_impact_high_cost():
    """funding_pct_of_gross > 0.15 triggers a parameter suggestion."""
    engine = _make_engine()
    funding = {"funding_pct_of_gross": 0.25, "total_funding_paid": -150}

    suggestions = engine.detect_funding_impact("crypto_trader", funding)

    assert len(suggestions) >= 1
    s = suggestions[0]
    assert s.bot_id == "crypto_trader"
    assert s.tier == SuggestionTier.PARAMETER
    assert s.detection_context is not None
    assert s.detection_context.detector_name == "funding_impact"
    assert "25%" in s.title


def test_detect_funding_impact_acceptable():
    """funding_pct_of_gross below threshold returns empty list."""
    engine = _make_engine()
    funding = {"funding_pct_of_gross": 0.10, "total_funding_paid": -50}

    suggestions = engine.detect_funding_impact("crypto_trader", funding)

    cost_suggestions = [s for s in suggestions if s.tier == SuggestionTier.PARAMETER]
    assert len(cost_suggestions) == 0


def test_detect_funding_impact_funding_losers():
    """Trades where funding > pnl in funding_losers list trigger a hypothesis."""
    engine = _make_engine()
    funding = {
        "funding_pct_of_gross": 0.05,
        "funding_losers": [
            {"trade_id": "t1", "pnl": 10, "funding_paid": -15},
            {"trade_id": "t2", "pnl": 5, "funding_paid": -12},
        ],
    }

    suggestions = engine.detect_funding_impact("crypto_trader", funding)

    assert len(suggestions) >= 1
    loser_suggestions = [s for s in suggestions if "funding cost" in s.description.lower()]
    assert len(loser_suggestions) == 1
    s = loser_suggestions[0]
    assert s.tier == SuggestionTier.HYPOTHESIS
    assert "2" in s.title  # 2 trades
    assert s.detection_context.detector_name == "funding_impact"


def test_detect_funding_impact_archetype_override():
    """institutional_anchor archetype uses 0.20 threshold instead of 0.15."""
    registry = _make_institutional_registry(bot_id="inst_bot")
    engine = _make_engine(strategy_registry=registry)
    # 0.18 is above default 0.15 but below institutional_anchor's 0.20
    funding = {"funding_pct_of_gross": 0.18}

    suggestions = engine.detect_funding_impact("inst_bot", funding)

    cost_suggestions = [s for s in suggestions if s.tier == SuggestionTier.PARAMETER]
    assert len(cost_suggestions) == 0, (
        "institutional_anchor threshold is 0.20; 0.18 should not trigger"
    )

    # 0.22 should trigger even with the looser threshold
    funding_high = {"funding_pct_of_gross": 0.22}
    suggestions_high = engine.detect_funding_impact("inst_bot", funding_high)
    cost_high = [s for s in suggestions_high if s.tier == SuggestionTier.PARAMETER]
    assert len(cost_high) == 1


# ---------------------------------------------------------------------------
# detect_grade_selectivity
# ---------------------------------------------------------------------------

def test_detect_grade_selectivity_b_negative():
    """B-grade avg_pnl < 0 with sufficient trades triggers suggestion."""
    engine = _make_engine()
    grade_data = {
        "per_grade": {
            "A": {"count": 12, "avg_pnl": 0.0050},
            "B": {"count": 10, "avg_pnl": -0.0020},
        },
        "grade_expectancy_gap": 0.007,
    }

    suggestions = engine.detect_grade_selectivity("crypto_trader", grade_data)

    b_neg = [s for s in suggestions if "B-grade" in s.title and "negative" in s.title.lower()]
    assert len(b_neg) == 1
    assert b_neg[0].tier == SuggestionTier.HYPOTHESIS
    assert b_neg[0].detection_context.detector_name == "grade_selectivity"


def test_detect_grade_selectivity_similar_performance():
    """A and B within 10% gap triggers miscalibration warning."""
    engine = _make_engine()
    grade_data = {
        "per_grade": {
            "A": {"count": 15, "avg_pnl": 0.0100},
            "B": {"count": 10, "avg_pnl": 0.0095},  # within 10% of A
        },
        "grade_expectancy_gap": 0.0005,
    }

    suggestions = engine.detect_grade_selectivity("crypto_trader", grade_data)

    gap_suggestions = [s for s in suggestions if "negligible" in s.title.lower()]
    assert len(gap_suggestions) == 1
    assert gap_suggestions[0].tier == SuggestionTier.PARAMETER
    assert "miscalibrated" in gap_suggestions[0].description


def test_detect_grade_selectivity_insufficient_trades():
    """< min_trades (20) returns empty — not enough data."""
    engine = _make_engine()
    grade_data = {
        "per_grade": {
            "A": {"count": 5, "avg_pnl": 0.0100},
            "B": {"count": 5, "avg_pnl": -0.0050},
        },
        "grade_expectancy_gap": 0.015,
    }

    suggestions = engine.detect_grade_selectivity("crypto_trader", grade_data)

    assert len(suggestions) == 0


def test_detect_grade_selectivity_grade_inversion():
    """Negative grade_expectancy_gap means B outperforms A — flag inversion."""
    engine = _make_engine()
    grade_data = {
        "per_grade": {
            "A": {"count": 12, "avg_pnl": 0.0020},
            "B": {"count": 10, "avg_pnl": 0.0060},
        },
        "grade_expectancy_gap": -0.0040,
    }

    suggestions = engine.detect_grade_selectivity("crypto_trader", grade_data)

    inversion = [s for s in suggestions if "inverted" in s.title.lower()]
    assert len(inversion) == 1
    assert inversion[0].tier == SuggestionTier.HYPOTHESIS
    assert inversion[0].detection_context.detector_name == "grade_selectivity"


# ---------------------------------------------------------------------------
# detect_confluence_quality
# ---------------------------------------------------------------------------

def test_detect_confluence_quality_lift():
    """Significant win rate lift at higher confluence count triggers suggestion."""
    engine = _make_engine()
    confluence_data = {
        "by_count": {
            "2": {"win_rate": 0.45, "count": 20},
            "3": {"win_rate": 0.60, "count": 10},  # +15% lift, above 10% threshold
        },
        "by_factor": {},
    }

    suggestions = engine.detect_confluence_quality("crypto_trader", confluence_data)

    lift_suggestions = [s for s in suggestions if "Win rate jumps" in s.title]
    assert len(lift_suggestions) == 1
    s = lift_suggestions[0]
    assert s.tier == SuggestionTier.PARAMETER
    assert "3" in s.title  # at 3 confluences
    assert s.detection_context.detector_name == "confluence_quality"


def test_detect_confluence_quality_negative_factor():
    """Factor with negative lift is flagged as a hypothesis."""
    engine = _make_engine()
    confluence_data = {
        "by_count": {},
        "by_factor": {
            "rsi_divergence": {"lift": -0.15, "count": 12},
            "volume_spike": {"lift": 0.08, "count": 15},
        },
    }

    suggestions = engine.detect_confluence_quality("crypto_trader", confluence_data)

    neg = [s for s in suggestions if "rsi_divergence" in s.title]
    assert len(neg) == 1
    assert neg[0].tier == SuggestionTier.HYPOTHESIS
    assert neg[0].detection_context.detector_name == "confluence_quality"
    # volume_spike has positive lift, should NOT be flagged
    vol = [s for s in suggestions if "volume_spike" in s.title]
    assert len(vol) == 0


# ---------------------------------------------------------------------------
# detect_leverage_utilization
# ---------------------------------------------------------------------------

def test_detect_leverage_utilization_high():
    """> 80% utilization triggers a parameter suggestion."""
    engine = _make_engine()
    leverage_data = {
        "leverage_utilization_pct": 0.92,
        "near_liquidation_count": 0,
        "per_grade": {},
    }

    suggestions = engine.detect_leverage_utilization("crypto_trader", leverage_data)

    util = [s for s in suggestions if "utilization" in s.title.lower()]
    assert len(util) == 1
    assert util[0].tier == SuggestionTier.PARAMETER
    assert util[0].detection_context.detector_name == "leverage_utilization"
    assert "92%" in util[0].title


def test_detect_leverage_utilization_near_liquidation():
    """near_liquidation_count > 0 produces a critical hypothesis."""
    engine = _make_engine()
    leverage_data = {
        "leverage_utilization_pct": 0.50,
        "near_liquidation_count": 3,
        "per_grade": {},
    }

    suggestions = engine.detect_leverage_utilization("crypto_trader", leverage_data)

    liq = [s for s in suggestions if "liquidation" in s.title.lower()]
    assert len(liq) == 1
    assert liq[0].confidence == 0.9
    assert liq[0].tier == SuggestionTier.HYPOTHESIS
    assert "3" in liq[0].title


def test_detect_leverage_utilization_grade_mismatch():
    """B leverage >= A leverage triggers a parameter suggestion."""
    engine = _make_engine()
    leverage_data = {
        "leverage_utilization_pct": 0.50,
        "near_liquidation_count": 0,
        "per_grade": {"A": 5.0, "B": 7.0},
    }

    suggestions = engine.detect_leverage_utilization("crypto_trader", leverage_data)

    mismatch = [s for s in suggestions if "B-grade" in s.title]
    assert len(mismatch) == 1
    assert mismatch[0].tier == SuggestionTier.PARAMETER
    assert "leverage" in mismatch[0].description.lower()


# ---------------------------------------------------------------------------
# build_report integration
# ---------------------------------------------------------------------------

def test_build_report_with_crypto_params():
    """Verify build_report passes crypto data through to the 4 detectors."""
    engine = _make_engine()
    summaries = {"crypto_trader": _minimal_bot_summary()}

    funding = {"crypto_trader": {"funding_pct_of_gross": 0.25}}
    grade = {"crypto_trader": {
        "per_grade": {
            "A": {"count": 15, "avg_pnl": 0.005},
            "B": {"count": 10, "avg_pnl": -0.002},
        },
        "grade_expectancy_gap": 0.007,
    }}
    confluence = {"crypto_trader": {
        "by_count": {
            "2": {"win_rate": 0.40, "count": 20},
            "3": {"win_rate": 0.60, "count": 8},
        },
        "by_factor": {"bad_factor": {"lift": -0.20, "count": 10}},
    }}
    leverage = {"crypto_trader": {
        "leverage_utilization_pct": 0.90,
        "near_liquidation_count": 1,
        "per_grade": {"A": 3.0, "B": 5.0},
    }}

    report = engine.build_report(
        summaries,
        funding_data=funding,
        grade_data=grade,
        confluence_data=confluence,
        leverage_data=leverage,
    )

    assert isinstance(report, RefinementReport)
    # Each detector should produce at least one suggestion
    detectors_seen = {
        s.detection_context.detector_name
        for s in report.suggestions
        if s.detection_context is not None
    }
    assert "funding_impact" in detectors_seen
    assert "grade_selectivity" in detectors_seen
    assert "confluence_quality" in detectors_seen
    assert "leverage_utilization" in detectors_seen


def test_build_report_without_crypto_params():
    """None crypto params should not produce crypto detector suggestions."""
    engine = _make_engine()
    summaries = {"crypto_trader": _minimal_bot_summary()}

    report = engine.build_report(summaries)

    crypto_detector_names = {"funding_impact", "grade_selectivity",
                             "confluence_quality", "leverage_utilization"}
    crypto_suggestions = [
        s for s in report.suggestions
        if s.detection_context is not None
        and s.detection_context.detector_name in crypto_detector_names
    ]
    assert len(crypto_suggestions) == 0


# ---------------------------------------------------------------------------
# Archetype defaults and detector-to-category mappings
# ---------------------------------------------------------------------------

def test_archetype_defaults_crypto():
    """Verify _ARCHETYPE_DEFAULTS has funding_impact entries for crypto archetypes."""
    defaults = StrategyEngine._ARCHETYPE_DEFAULTS

    assert "funding_impact" in defaults
    fi = defaults["funding_impact"]
    assert "momentum_pullback_crypto" in fi
    assert fi["momentum_pullback_crypto"]["cost_threshold"] == 0.15
    assert "institutional_anchor" in fi
    assert fi["institutional_anchor"]["cost_threshold"] == 0.20
    assert "volume_profile_breakout" in fi
    assert fi["volume_profile_breakout"]["cost_threshold"] == 0.10
    assert defaults["funding_trend"]["volume_profile_breakout"]["cost_threshold"] == 0.10
    assert defaults["liquidation_proximity"]["volume_profile_breakout"]["proximity_threshold"] == 0.65


def test_detector_to_category_crypto():
    """Verify _DETECTOR_TO_CATEGORY maps crypto detectors to the correct categories."""
    mapping = StrategyEngine._DETECTOR_TO_CATEGORY

    assert mapping["funding_impact"] == "filter_threshold"
    assert mapping["grade_selectivity"] == "signal"
    assert mapping["confluence_quality"] == "filter_threshold"
    assert mapping["leverage_utilization"] == "position_sizing"
    assert mapping["liquidation_proximity"] == "leverage_cap"
    assert mapping["funding_trend"] == "funding_threshold"


def test_empty_crypto_data_graceful():
    """Empty dicts for crypto data should not crash and produce no suggestions."""
    engine = _make_engine()

    assert engine.detect_funding_impact("bot1", {}) == []
    assert engine.detect_grade_selectivity("bot1", {}) == []
    assert engine.detect_confluence_quality("bot1", {}) == []
    assert engine.detect_leverage_utilization("bot1", {}) == []
    assert engine.detect_mtf_alignment_drift("bot1", []) == []
    assert engine.detect_liquidation_proximity("bot1", []) == []
    assert engine.detect_symbol_concentration("bot1", []) == []
    assert engine.detect_session_patterns_24_7("bot1", []) == []
    assert engine.detect_funding_trend("bot1", []) == []


# ---------------------------------------------------------------------------
# Additional crypto detector coverage
# ---------------------------------------------------------------------------

def _trade(**overrides) -> dict:
    base = {
        "trade_id": "t",
        "bot_id": "crypto_trader",
        "strategy_id": "MomentumPullback_M15",
        "pair": "BTCUSDT",
        "side": "LONG",
        "bias_direction": "bullish",
        "entry_time": "2026-04-01T01:00:00+00:00",
        "exit_time": "2026-04-01T02:00:00+00:00",
        "pnl": 10.0,
        "mae_r": 0.1,
        "funding_paid": 0.0,
        "sizing_inputs": {"leverage": 3.0},
    }
    base.update(overrides)
    return base


def test_detect_mtf_alignment_drift_positive():
    engine = _make_engine()
    trades = [
        _trade(trade_id=f"a{i}", side="LONG", bias_direction="bullish", pnl=20)
        for i in range(10)
    ] + [
        _trade(trade_id=f"m{i}", side="SHORT", bias_direction="bullish", pnl=-10)
        for i in range(6)
    ]

    suggestions = engine.detect_mtf_alignment_drift("crypto_trader", trades)

    assert len(suggestions) == 1
    assert suggestions[0].detection_context.detector_name == "mtf_alignment_drift"


def test_detect_mtf_alignment_drift_threshold_edge_empty():
    engine = _make_engine()
    trades = [
        _trade(trade_id=f"a{i}", side="LONG", bias_direction="bullish", pnl=10)
        for i in range(10)
    ] + [
        _trade(trade_id=f"m{i}", side="SHORT", bias_direction="bullish", pnl=-1)
        for i in range(4)
    ]

    assert engine.detect_mtf_alignment_drift("crypto_trader", trades) == []


def test_detect_mtf_alignment_drift_requires_win_rate_gap():
    engine = _make_engine()
    trades = [
        _trade(trade_id=f"a{i}", side="LONG", bias_direction="bullish", pnl=10 if i < 5 else -10)
        for i in range(10)
    ] + [
        _trade(trade_id=f"m{i}", side="SHORT", bias_direction="bullish", pnl=8 if i < 3 else -10)
        for i in range(6)
    ]

    assert engine.detect_mtf_alignment_drift("crypto_trader", trades) == []


def test_detect_liquidation_proximity_positive():
    engine = _make_engine()
    trades = [_trade(trade_id="liq", mae_r=0.25, sizing_inputs={"leverage": 4.0})]

    suggestions = engine.detect_liquidation_proximity("crypto_trader", trades)

    assert len(suggestions) == 1
    assert suggestions[0].detection_context.detector_name == "liquidation_proximity"
    assert suggestions[0].detection_context.observed_value == 1.0


def test_detect_liquidation_proximity_edge_empty():
    engine = _make_engine()
    trades = [_trade(trade_id="safe", mae_r=0.2, sizing_inputs={"leverage": 3.5})]

    assert engine.detect_liquidation_proximity("crypto_trader", trades) == []


def test_detect_symbol_concentration_positive():
    engine = _make_engine()
    trades = [
        _trade(trade_id=f"btc{i}", pair="BTCUSDT", pnl=-10)
        for i in range(10)
    ] + [
        _trade(trade_id=f"eth{i}", pair="ETHUSDT", pnl=-1)
        for i in range(5)
    ]

    suggestions = engine.detect_symbol_concentration("crypto_trader", trades)

    assert len(suggestions) == 1
    assert "BTC" in suggestions[0].title
    assert suggestions[0].detection_context.detector_name == "symbol_concentration"


def test_detect_session_patterns_24_7_positive():
    engine = _make_engine()
    trades = [
        _trade(trade_id=f"asia{i}", exit_time=f"2026-04-01T0{i % 5}:00:00+00:00", pnl=-5)
        for i in range(10)
    ] + [
        _trade(trade_id=f"us{i}", exit_time=f"2026-04-01T1{3 + (i % 5)}:00:00+00:00", pnl=5)
        for i in range(10)
    ]

    suggestions = engine.detect_session_patterns_24_7("crypto_trader", trades)

    assert len(suggestions) == 1
    assert "Asia" in suggestions[0].title
    assert suggestions[0].detection_context.detector_name == "session_patterns_24_7"


def test_detect_funding_trend_positive():
    engine = _make_engine()
    trades = []
    for idx, (date, funding) in enumerate([
        ("2026-03-01T12:00:00+00:00", 5),
        ("2026-03-08T12:00:00+00:00", 10),
        ("2026-03-15T12:00:00+00:00", 20),
    ]):
        trades.append(_trade(trade_id=f"fund{idx}", exit_time=date, pnl=100, funding_paid=-funding))

    suggestions = engine.detect_funding_trend("crypto_trader", trades)

    assert len(suggestions) == 1
    assert suggestions[0].detection_context.detector_name == "funding_trend"
    assert "BTC" in suggestions[0].title


def test_detect_funding_trend_is_per_symbol():
    engine = _make_engine()
    trades = []
    for idx, (date, funding) in enumerate([
        ("2026-03-01T12:00:00+00:00", 2),
        ("2026-03-08T12:00:00+00:00", 4),
        ("2026-03-15T12:00:00+00:00", 20),
    ]):
        trades.append(_trade(trade_id=f"sol{idx}", pair="SOLUSDT", exit_time=date, pnl=100, funding_paid=-funding))
        trades.append(_trade(trade_id=f"btc{idx}", pair="BTCUSDT", exit_time=date, pnl=100, funding_paid=-1))

    suggestions = engine.detect_funding_trend("crypto_trader", trades)

    assert len(suggestions) == 1
    assert "SOL" in suggestions[0].title


def test_build_report_with_crypto_trade_data_runs_new_detectors():
    engine = _make_engine()
    summaries = {"crypto_trader": _minimal_bot_summary()}
    trades = [
        _trade(trade_id=f"a{i}", side="LONG", bias_direction="bullish", pnl=20)
        for i in range(10)
    ] + [
        _trade(trade_id=f"m{i}", side="SHORT", bias_direction="bullish", pnl=-10)
        for i in range(6)
    ]

    report = engine.build_report(summaries, crypto_trade_data={"crypto_trader": trades})
    detectors_seen = {
        s.detection_context.detector_name
        for s in report.suggestions
        if s.detection_context is not None
    }

    assert "mtf_alignment_drift" in detectors_seen
