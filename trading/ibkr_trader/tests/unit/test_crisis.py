"""Crisis detection: config, live service, advisory/action overlay, economic policy."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtests.regime.auto.crisis.economic import (
    CrisisEconomicPolicy,
    CrisisSleeveEconomicPolicy,
    build_exposure_series,
    build_sleeve_exposure_map,
    evaluate_policy,
)
from backtests.regime.auto.crisis.scoring import extract_crisis_metrics
from backtests.regime.crisis_validation import build_event_channel_chronology
from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
from regime.crisis import config as C
from regime.crisis.actions import resolve_crisis_action
from regime.crisis.context import CrisisContext, _dd_mult_for_level, _risk_mult_for_level
from regime.crisis.detector import (
    compute_advisory_level,
    compute_alert_level,
    is_hard_credit_impulse_warning_candidate,
)
from regime.crisis.hysteresis import HysteresisTracker
from regime.crisis.indicators import ChannelReading, CrisisIndicators, compute_indicators
from regime.crisis.integration import apply_crisis_overlay
from regime.crisis.service import CrisisService, _next_crisis_compute_after_close


ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _indicators(*, vix: int = 0, spread: int = 0, slope: int = 0,
                corr: int = 0, spy_dd: int = 0) -> CrisisIndicators:
    return CrisisIndicators(
        vix=ChannelReading("VIX", 30.0, vix),
        credit_spread=ChannelReading("CREDIT_SPREAD", 260.0, spread),
        yield_curve=ChannelReading("YIELD_CURVE", -0.6, slope),
        spy_tlt_corr=ChannelReading("SPY_TLT_CORR", 0.45, corr),
        spy_drawdown=ChannelReading("SPY_DRAWDOWN", -0.05, spy_dd),
    )


def _crisis_ctx(level: int) -> CrisisContext:
    return CrisisContext(
        alert_level=C.ALERT_LEVELS[level],
        alert_level_int=level,
        portfolio_action_level=C.ALERT_LEVELS[level] if level >= 2 else C.ALERT_NORMAL,
        portfolio_action_level_int=level if level >= 2 else 0,
        risk_multiplier=_risk_mult_for_level(level),
        dd_tier_multiplier=_dd_mult_for_level(level),
    )


# ---------------------------------------------------------------------------
# Detector config
# ---------------------------------------------------------------------------
def test_live_crisis_defaults_match_latest_optimized_config() -> None:
    optimized_path = Path("backtests/output/regime/crisis/round_9/optimized_config.json")
    optimized = json.loads(optimized_path.read_text(encoding="utf-8"))

    for key, expected in optimized.items():
        actual = getattr(C, key)
        if isinstance(expected, float):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected


def test_watch_is_internal_buffer_only() -> None:
    assert C.RISK_MULT_WATCH == 1.0
    assert C.DD_TIER_MULT_WATCH == 1.0
    assert C.ADVISORY_WATCH_MIN_PRIMARY > C.WATCH_MIN_PRIMARY
    assert C.ADVISORY_WATCH_MIN_WARNING >= C.WARNING_MIN_PRIMARY


# ---------------------------------------------------------------------------
# Live service: scheduling, fetching, publishing
# ---------------------------------------------------------------------------
def test_next_crisis_compute_runs_after_completed_daily_bars() -> None:
    before_cutoff = datetime(2026, 5, 4, 16, 59, tzinfo=ET)
    after_cutoff = datetime(2026, 5, 4, 17, 10, tzinfo=ET)
    friday_after_cutoff = datetime(2026, 5, 8, 17, 10, tzinfo=ET)

    assert _next_crisis_compute_after_close(before_cutoff).isoformat() == (
        "2026-05-04T17:05:00-04:00"
    )
    assert _next_crisis_compute_after_close(after_cutoff).isoformat() == (
        "2026-05-05T17:05:00-04:00"
    )
    assert _next_crisis_compute_after_close(friday_after_cutoff).isoformat() == (
        "2026-05-11T17:05:00-04:00"
    )


@pytest.mark.asyncio
async def test_crisis_fetches_completed_daily_spy_tlt_bars(tmp_path) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def req_historical_data(self, contract: object, **kwargs):
            self.calls.append(kwargs)
            start = datetime(2026, 4, 1, tzinfo=timezone.utc)
            return [
                SimpleNamespace(
                    date=start + timedelta(days=i),
                    close=100.0 + i,
                )
                for i in range(30)
            ]

    session = FakeSession()
    service = CrisisService(
        session,
        data_dir=tmp_path,
        compute_on_start=False,
        auto_schedule=False,
        now_provider=lambda: datetime(2026, 5, 6, 21, 10, tzinfo=timezone.utc),
    )
    service._contracts = {"SPY": object(), "TLT": object()}

    returns = await service._fetch_etf_returns()

    assert not returns.empty
    assert {call["barSizeSetting"] for call in session.calls} == {"1 day"}
    assert all(call["useRTH"] is True for call in session.calls)
    assert all(call["completed_only"] is True for call in session.calls)


@pytest.mark.asyncio
async def test_crisis_compute_publishes_fresh_context_to_listeners(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2026-01-21", periods=100, freq="D")
    market_df = pd.DataFrame(
        {"VIX": [20.0] * 100, "SPREAD": [2.0] * 100, "SLOPE_10Y2Y": [1.0] * 100},
        index=dates,
    )
    strat_ret_df = pd.DataFrame({"SPY": [0.0] * 100, "TLT": [0.0] * 100}, index=dates)

    service = CrisisService(
        object(),
        data_dir=tmp_path,
        compute_on_start=False,
        auto_schedule=False,
        now_provider=lambda: datetime(2026, 4, 30, 21, 10, tzinfo=timezone.utc),
    )

    async def fake_fetch_data():
        return market_df, strat_ret_df, None, "unit=fresh"

    monkeypatch.setattr(service, "_fetch_data", fake_fetch_data)
    delivered = []
    service.add_listener(lambda ctx: delivered.append(ctx))

    ctx = await service.compute_now()

    assert ctx.alert_level == C.ALERT_NORMAL
    assert ctx.data_as_of == "2026-04-30"
    assert ctx.data_status == "unit=fresh;backlog=ok"
    assert delivered == [ctx]


@pytest.mark.asyncio
async def test_crisis_compute_uses_latest_completed_etf_date_not_fred_only_date(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2026-01-21", periods=100, freq="D")
    market_df = pd.DataFrame(
        {"VIX": [20.0] * 100, "SPREAD": [2.0] * 100, "SLOPE_10Y2Y": [1.0] * 100},
        index=dates,
    )
    strat_ret_df = pd.DataFrame(
        {"SPY": [0.0] * 97 + [-0.01, -0.01, -0.01], "TLT": [0.0] * 100},
        index=dates,
    )
    market_df.to_parquet(tmp_path / "market_df.parquet")
    strat_ret_df.to_parquet(tmp_path / "strat_ret_df.parquet")

    fred_only_next_day = pd.DataFrame(
        {"VIX": [21.0], "SPREAD": [2.1], "SLOPE_10Y2Y": [1.0]},
        index=[dates[-1] + pd.Timedelta(days=1)],
    )
    service = CrisisService(
        object(),
        data_dir=tmp_path,
        compute_on_start=False,
        auto_schedule=False,
        now_provider=lambda: datetime(2026, 4, 30, 21, 10, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(service, "_fetch_fred", lambda: fred_only_next_day)

    async def no_fresh_returns():
        return pd.DataFrame()

    monkeypatch.setattr(service, "_fetch_etf_returns", no_fresh_returns)

    ctx = await service.compute_now(notify=False)

    assert ctx.data_as_of == dates[-1].date().isoformat()
    assert ctx.spy_3d_return == pytest.approx((0.99 ** 3) - 1.0)


@pytest.mark.asyncio
async def test_crisis_compute_keeps_prior_context_when_spy_tlt_backlog_is_incomplete(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2026-04-01", periods=10, freq="D")
    market_df = pd.DataFrame(
        {"VIX": [20.0] * 10, "SPREAD": [2.0] * 10, "SLOPE_10Y2Y": [1.0] * 10},
        index=dates,
    )
    strat_ret_df = pd.DataFrame({"SPY": [0.0] * 10, "TLT": [0.0] * 10}, index=dates)
    service = CrisisService(
        object(),
        data_dir=tmp_path,
        compute_on_start=False,
        auto_schedule=False,
        now_provider=lambda: datetime(2026, 4, 10, 21, 10, tzinfo=timezone.utc),
    )

    async def fake_fetch_data():
        return market_df, strat_ret_df, None, "unit=short"

    monkeypatch.setattr(service, "_fetch_data", fake_fetch_data)

    ctx = await service.compute_now(notify=False)

    assert ctx.alert_level == C.ALERT_NORMAL
    assert ctx.computed_at == ""


@pytest.mark.asyncio
async def test_crisis_startup_fails_loudly_when_live_backlog_is_incomplete(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2026-04-01", periods=10, freq="D")
    market_df = pd.DataFrame(
        {"VIX": [20.0] * 10, "SPREAD": [2.0] * 10, "SLOPE_10Y2Y": [1.0] * 10},
        index=dates,
    )
    strat_ret_df = pd.DataFrame({"SPY": [0.0] * 10, "TLT": [0.0] * 10}, index=dates)
    service = CrisisService(
        object(),
        data_dir=tmp_path,
        compute_on_start=True,
        auto_schedule=False,
        now_provider=lambda: datetime(2026, 4, 10, 21, 10, tzinfo=timezone.utc),
    )

    async def fake_fetch_data():
        return market_df, strat_ret_df, None, "unit=short"

    async def noop_qualify():
        return None

    monkeypatch.setattr(service, "_fetch_data", fake_fetch_data)
    monkeypatch.setattr(service, "_qualify_contracts", noop_qualify)

    with pytest.raises(RuntimeError, match="insufficient live backlog"):
        await service.start()


# ---------------------------------------------------------------------------
# Advisory / action overlay
# ---------------------------------------------------------------------------
def test_internal_watch_does_not_become_external_advisory() -> None:
    indicators = _indicators(vix=1)

    alert_level, alert_level_int = compute_alert_level(indicators)
    advisory_level, advisory_level_int, reason = compute_advisory_level(
        indicators,
        alert_level_int,
    )

    assert (alert_level, alert_level_int) == (C.ALERT_WATCH, 1)
    assert (advisory_level, advisory_level_int) == (C.ALERT_NORMAL, 0)
    assert reason == "internal watch only"


def test_external_watch_requires_broader_or_deeper_confirmation() -> None:
    broad_watch = _indicators(vix=1, spread=1, slope=1, corr=1)
    one_crisis = _indicators(vix=3)

    assert compute_advisory_level(broad_watch, 1)[:2] == (C.ALERT_WATCH, 1)
    assert compute_advisory_level(one_crisis, 1)[:2] == (C.ALERT_WATCH, 1)


def test_external_advisory_mirrors_actionable_warning() -> None:
    indicators = _indicators(vix=2, spread=2)
    alert_level, alert_level_int = compute_alert_level(indicators)

    assert (alert_level, alert_level_int) == (C.ALERT_WARNING, 2)
    assert compute_advisory_level(indicators, alert_level_int)[:2] == (
        C.ALERT_WARNING,
        2,
    )


def test_stock_crisis_policy_blocks_new_stock_entries_and_tightens_caps() -> None:
    base = PortfolioRulesConfig(
        directional_cap_R=10.0,
        directional_cap_long_R=8.0,
        directional_cap_short_R=6.0,
        priority_headroom_R=2.0,
        regime_unit_risk_mult=0.8,
        disabled_strategies=frozenset({"AlreadyDisabled"}),
    )

    result = apply_crisis_overlay(base, _crisis_ctx(3), family_id="stock")

    assert result.regime_unit_risk_mult == pytest.approx(0.24)
    assert result.directional_cap_R == pytest.approx(6.0)
    assert result.directional_cap_long_R == pytest.approx(3.2)
    assert result.directional_cap_short_R == pytest.approx(3.6)
    assert result.priority_headroom_R == pytest.approx(1.0)
    assert result.disabled_strategies == frozenset({
        "ALCB_v1",
        "AlreadyDisabled",
        "IARIC_v1",
    })


def test_momentum_crisis_policy_preserves_short_capacity_and_cuts_contracts() -> None:
    base = PortfolioRulesConfig(
        directional_cap_R=4.0,
        directional_cap_long_R=10.0,
        directional_cap_short_R=8.0,
        max_family_contracts_mnq_eq=40,
        regime_unit_risk_mult=1.0,
    )

    result = apply_crisis_overlay(base, _crisis_ctx(3), family_id="momentum")

    assert result.directional_cap_R == pytest.approx(2.2)
    assert result.directional_cap_long_R == pytest.approx(3.5)
    assert result.directional_cap_short_R == pytest.approx(8.0)
    assert result.max_family_contracts_mnq_eq == 20
    assert result.regime_unit_risk_mult == pytest.approx(1.0)
    assert result.regime_unit_risk_long_mult == pytest.approx(0.3)
    assert result.regime_unit_risk_short_mult == pytest.approx(1.0)


def test_shock_formation_watch_can_lightly_tighten_without_warning() -> None:
    base = PortfolioRulesConfig(directional_cap_R=10.0, regime_unit_risk_mult=1.0)
    ctx = CrisisContext(
        alert_level=C.ALERT_WATCH,
        alert_level_int=1,
        advisory_level=C.ALERT_WATCH,
        advisory_level_int=1,
        portfolio_action_level=C.ALERT_WATCH,
        portfolio_action_level_int=1,
        risk_multiplier=C.STRESS_FORMATION_RISK_MULT_SHOCK,
        stress_formation_score=C.STRESS_FORMATION_MIN_SCORE,
        stress_formation_mode="shock",
    )

    result = apply_crisis_overlay(base, ctx, family_id="stock")

    assert result.regime_unit_risk_mult == pytest.approx(0.75)
    assert result.directional_cap_R == pytest.approx(10.0)


def test_credit_impulse_bridge_warning_keeps_early_label_but_uses_lighter_action() -> None:
    base = PortfolioRulesConfig(
        directional_cap_R=10.0,
        directional_cap_long_R=8.0,
        priority_headroom_R=2.0,
        regime_unit_risk_mult=1.0,
    )
    ctx = CrisisContext(
        alert_level=C.ALERT_WARNING,
        alert_level_int=2,
        portfolio_action_level=C.ALERT_WARNING,
        portfolio_action_level_int=2,
        risk_multiplier=C.RISK_MULT_WARNING,
        dd_tier_multiplier=C.DD_TIER_MULT_WARNING,
        stress_formation_score=C.STRESS_FORMATION_MIN_SCORE,
        stress_formation_mode="credit_impulse",
        primary_warning_count=1,
        primary_crisis_count=0,
    )

    action = resolve_crisis_action(ctx, family_id="stock", regime="G")
    result = apply_crisis_overlay(base, ctx, family_id="stock", regime="G")

    assert action.alert_level == C.ALERT_WARNING
    assert action.action_provenance == "credit_impulse_bridge"
    assert result.regime_unit_risk_mult == pytest.approx(0.75)
    assert result.directional_cap_R == pytest.approx(9.0)
    assert result.directional_cap_long_R == pytest.approx(6.4)
    assert result.priority_headroom_R == pytest.approx(1.8)


def test_confirmed_warning_keeps_full_growth_warning_cut() -> None:
    base = PortfolioRulesConfig(directional_cap_R=10.0, regime_unit_risk_mult=1.0)
    ctx = CrisisContext(
        alert_level=C.ALERT_WARNING,
        alert_level_int=2,
        portfolio_action_level=C.ALERT_WARNING,
        portfolio_action_level_int=2,
        risk_multiplier=C.RISK_MULT_WARNING,
        dd_tier_multiplier=C.DD_TIER_MULT_WARNING,
        primary_warning_count=2,
    )

    result = apply_crisis_overlay(base, ctx, family_id="stock", regime="G")

    assert result.regime_unit_risk_mult == pytest.approx(0.65)
    assert result.directional_cap_R == pytest.approx(8.5)


def test_warning_in_stress_regime_adds_smaller_incremental_cut() -> None:
    base = PortfolioRulesConfig(directional_cap_R=10.0, regime_unit_risk_mult=0.75)
    ctx = CrisisContext(
        alert_level=C.ALERT_WARNING,
        alert_level_int=2,
        portfolio_action_level=C.ALERT_WARNING,
        portfolio_action_level_int=2,
        risk_multiplier=C.RISK_MULT_WARNING,
        dd_tier_multiplier=C.DD_TIER_MULT_WARNING,
        primary_warning_count=2,
    )

    action = resolve_crisis_action(ctx, family_id="stock", regime="S")
    result = apply_crisis_overlay(base, ctx, family_id="stock", regime="S")

    assert action.action_provenance == "stress_regime"
    assert result.regime_unit_risk_mult == pytest.approx(0.60)
    assert result.directional_cap_R == pytest.approx(9.0)


def test_credit_impulse_hard_bridge_requires_persistence(monkeypatch) -> None:
    monkeypatch.setattr(C, "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS", 3)
    monkeypatch.setattr(C, "HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY", 1)

    indicators = _indicators(spread=3)
    indicators.stress_formation_score = C.STRESS_FORMATION_MIN_SCORE
    indicators.stress_formation_mode = "credit_impulse"
    tracker = HysteresisTracker()

    raw_level_ints = []
    final_level_ints = []
    for _ in range(3):
        _, raw_level_int = compute_alert_level(indicators)
        raw_level_int = tracker.apply_hard_credit_impulse_bridge(
            raw_level_int,
            is_hard_credit_impulse_warning_candidate(indicators),
        )
        raw_level_ints.append(raw_level_int)
        final_level_ints.append(tracker.update(raw_level_int))

    assert raw_level_ints == [1, 1, 2]
    assert final_level_ints == [1, 1, 2]


def test_event_channel_chronology_identifies_confirmation_bottleneck() -> None:
    dates = pd.date_range("2020-02-15", periods=5, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 1, 2, 2, 3],
            "raw_level_int": [0, 1, 2, 2, 3],
            "advisory_level_int": [0, 1, 2, 2, 3],
            "vix_level_int": [0, 2, 2, 2, 3],
            "credit_spread_level_int": [0, 0, 2, 2, 2],
            "yield_curve_level_int": [0, 0, 0, 1, 1],
            "spy_tlt_corr_level_int": [0, 0, 0, 0, 0],
            "spy_drawdown_level_int": [0, 1, 1, 1, 3],
        },
        index=dates,
    )

    chronology = build_event_channel_chronology(
        alerts,
        {"Demo": ("2020-02-15", "2020-02-19", "D")},
    )

    demo = chronology["Demo"]
    assert demo["detected_at"] == "2020-02-17"
    assert demo["latency_days"] == 2
    assert demo["bottleneck_channel"] == "CREDIT_SPREAD"


# ---------------------------------------------------------------------------
# Economic overlay / sleeve policies
# ---------------------------------------------------------------------------
def test_stress_formation_can_surface_external_advisory_without_action() -> None:
    dates = pd.date_range("2020-01-01", periods=25, freq="D")
    market = pd.DataFrame(
        {
            "VIX": [20.0] * 21 + [21.0, 24.0, 30.0, 31.0],
            "SPREAD": [2.0] * 25,
            "SLOPE_10Y2Y": [1.0] * 25,
        },
        index=dates,
    )
    ret = pd.DataFrame(
        {
            "SPY": [0.0] * 20 + [-0.01, -0.02, -0.02, -0.015, 0.0],
            "TLT": [0.0] * 25,
        },
        index=dates,
    )

    indicators = compute_indicators(market, ret, date=dates[-1])
    advisory_level, advisory_level_int, reason = compute_advisory_level(
        indicators,
        action_level_int=0,
    )

    assert indicators.stress_formation_score >= C.STRESS_FORMATION_MIN_SCORE
    assert indicators.stress_formation_mode == "shock"
    assert (advisory_level, advisory_level_int) == (C.ALERT_WATCH, 1)
    assert "stress formation shock" in reason


def test_build_exposure_series_applies_pre_action_then_action_overrides() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 1, 1, 2, 3],
            "portfolio_action_level_int": [0, 0, 0, 2, 3],
            "advisory_level_int": [0, 1, 0, 2, 3],
            "stress_formation_mode": ["", "", "shock", "", ""],
        },
        index=dates,
    )
    policy = CrisisEconomicPolicy(
        advisory_mult=0.95,
        shock_mult=0.90,
        warning_mult=0.70,
        crisis_mult=0.40,
    )

    exposure = build_exposure_series(alerts, policy)

    assert exposure.tolist() == [1.0, 0.95, 0.90, 0.70, 0.40]


def test_credit_impulse_pre_action_has_separate_multiplier() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 0, 2],
            "portfolio_action_level_int": [1, 1, 2],
            "advisory_level_int": [1, 1, 2],
            "stress_formation_mode": ["credit_impulse", "shock+credit_impulse", ""],
        },
        index=dates,
    )
    policy = CrisisEconomicPolicy(
        advisory_mult=1.0,
        shock_mult=0.85,
        credit_impulse_mult=0.75,
        warning_mult=0.65,
    )

    exposure = build_exposure_series(alerts, policy)

    assert exposure.tolist() == [0.75, 0.75, 0.65]


def test_credit_bridge_warning_can_be_scored_separately_from_confirmed_warning() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [2, 2, 3],
            "portfolio_action_level_int": [2, 2, 3],
            "advisory_level_int": [2, 2, 3],
            "stress_formation_mode": ["credit_impulse", "", ""],
            "primary_warning_count": [1, 2, 2],
            "primary_crisis_count": [0, 0, 2],
        },
        index=dates,
    )
    policy = CrisisEconomicPolicy(
        warning_mult=0.65,
        crisis_mult=0.30,
        credit_bridge_warning_mult=0.75,
    )

    exposure = build_exposure_series(alerts, policy)

    assert exposure.tolist() == [0.75, 0.65, 0.30]


def test_regime_conditioned_warning_policy_reduces_overcut_in_stress_states() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [2, 2, 2, 3],
            "portfolio_action_level_int": [2, 2, 2, 3],
            "advisory_level_int": [2, 2, 2, 3],
            "hmm_regime": ["G", "S", "D", "D"],
        },
        index=dates,
    )
    policy = CrisisEconomicPolicy(
        warning_mult=0.65,
        crisis_mult=0.30,
        stress_regime_warning_mult=0.80,
        defensive_regime_warning_mult=0.85,
    )

    exposure = build_exposure_series(alerts, policy)

    assert exposure.tolist() == [0.65, 0.80, 0.85, 0.30]


def test_sleeve_policy_can_preserve_short_exposure_separately() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 2, 3, 0],
            "portfolio_action_level_int": [0, 2, 3, 0],
            "advisory_level_int": [0, 2, 3, 0],
            "stress_formation_mode": [""] * 4,
        },
        index=dates,
    )
    policy = CrisisSleeveEconomicPolicy(
        equity_warning_mult=0.65,
        equity_crisis_mult=0.30,
        gld_warning_mult=0.80,
        gld_crisis_mult=0.60,
        short_warning_mult=1.00,
        short_crisis_mult=1.00,
    )

    exposures = build_sleeve_exposure_map(alerts, policy)

    assert exposures["equity_beta"].tolist() == [1.0, 0.65, 0.30, 1.0]
    assert exposures["gld"].tolist() == [1.0, 0.80, 0.60, 1.0]
    assert exposures["short_spy"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_credit_impulse_can_surface_targeted_pre_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C, "CREDIT_IMPULSE_SPREAD_BPS", 336.0)
    monkeypatch.setattr(C, "CREDIT_IMPULSE_SPY_3D_RETURN", -0.015)
    monkeypatch.setattr(C, "CREDIT_IMPULSE_MIN_VIX", 18.0)

    dates = pd.date_range("2020-01-01", periods=25, freq="D")
    market = pd.DataFrame(
        {
            "VIX": [16.0] * 22 + [18.5, 19.0, 18.2],
            "SPREAD": [3.6] * 25,
            "SLOPE_10Y2Y": [1.0] * 25,
        },
        index=dates,
    )
    ret = pd.DataFrame(
        {
            "SPY": [0.0] * 22 + [-0.006, -0.006, -0.006],
            "TLT": [0.0] * 25,
        },
        index=dates,
    )

    indicators = compute_indicators(market, ret, date=dates[-1])

    assert indicators.stress_formation_score >= C.STRESS_FORMATION_MIN_SCORE
    assert indicators.stress_formation_mode == "credit_impulse"


def test_economic_policy_can_improve_drawdown_on_synthetic_crash() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 0, 2, 2, 0, 0],
            "portfolio_action_level_int": [0, 0, 2, 2, 0, 0],
            "advisory_level_int": [0, 0, 2, 2, 0, 0],
            "stress_formation_mode": [""] * 6,
        },
        index=dates,
    )
    base = pd.Series([0.01, 0.01, -0.10, -0.08, 0.03, 0.03], index=dates)
    cash = pd.Series(0.0, index=dates)

    result = evaluate_policy(
        alerts_df=alerts,
        base_portfolios={"regime_proxy": base},
        cash_returns=cash,
        policy=CrisisEconomicPolicy(warning_mult=0.50, crisis_mult=0.50),
        portfolio_score_weights={"regime_proxy": 1.0},
    )

    deltas = result["portfolio_results"]["regime_proxy"]["deltas"]
    assert deltas["max_drawdown_pct"] < 0
    assert result["score"] > 0


def test_crisis_metrics_track_advisory_and_action_latency_separately() -> None:
    dates = pd.date_range("2020-02-15", periods=6, freq="D")
    alerts = pd.DataFrame(
        {
            "alert_level_int": [0, 0, 0, 2, 2, 3],
            "advisory_level_int": [1, 1, 1, 2, 2, 3],
            "portfolio_action_level_int": [0, 1, 1, 2, 2, 3],
        },
        index=dates,
    )

    metrics = extract_crisis_metrics(alerts)

    assert metrics.avg_latency == 3.0
    assert metrics.avg_advisory_latency == 0.0
    assert metrics.avg_action_latency == 1.0
    assert metrics.avg_action_latency < metrics.avg_latency
