from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, time, timezone
from types import SimpleNamespace

import pytest

from strategies.stock.alcb.artifact_store import (
    load_candidate_artifact,
    persist_candidate_artifact,
)
from strategies.stock.alcb.models import (
    CandidateArtifact,
    CandidateItem,
    Campaign,
    CampaignState,
    RegimeSnapshot as ALCBRegimeSnapshot,
)
from strategies.stock.alcb.plugin import ALCBPlugin
from strategies.stock.iaric.artifact_store import (
    coerce_intraday_state_snapshot,
    load_intraday_state,
    load_watchlist_artifact,
    persist_intraday_state,
    persist_watchlist_artifact,
)
from strategies.stock.iaric.models import (
    HeldPositionDirective,
    IntradayStateSnapshot,
    PBSymbolState,
    PendingOrderState,
    PositionState,
    RegimeSnapshot as IARICRegimeSnapshot,
    WatchlistArtifact,
    WatchlistItem,
)
from strategies.stock.iaric.plugin import IARICPlugin

UTC = timezone.utc


def _runtime_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(connection_group="default"),
        registry=SimpleNamespace(
            connection_groups={"default": SimpleNamespace(account_id="DU123")}
        ),
        portfolio=SimpleNamespace(
            capital=SimpleNamespace(
                strategy_navs={
                    "ALCB_v1": 25_000.0,
                    "IARIC_v1": 20_000.0,
                },
                strategy_allocations={},
                paper_initial_equity=30_000.0,
            )
        ),
        oms=object(),
        instrumentation=SimpleNamespace(trade_recorder=None),
    )


def _alcb_artifact(trade_date: date) -> CandidateArtifact:
    item = CandidateItem(
        symbol="AAA",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector="Technology",
        adv20_usd=25_000_000.0,
        median_spread_pct=0.001,
        selection_score=87,
        selection_detail={"compression": 40, "momentum": 47},
        stock_regime="BULL",
        market_regime="BULL",
        sector_regime="BULL",
        daily_trend_sign=1,
        relative_strength_percentile=92.0,
        accumulation_score=1.4,
        ttm_squeeze_bonus=2,
        average_30m_volume=120_000.0,
        median_30m_volume=110_000.0,
        tradable_flag=True,
        direction_bias="LONG",
        price=12.45,
        earnings_risk_flag=False,
        campaign=Campaign(
            symbol="AAA",
            state=CampaignState.COMPRESSION,
            campaign_id=7,
            continuation_enabled=True,
        ),
    )
    return CandidateArtifact(
        trade_date=trade_date,
        generated_at=datetime.combine(trade_date, time(0, 0), tzinfo=UTC),
        regime=ALCBRegimeSnapshot(
            score=0.9,
            tier="A",
            risk_multiplier=1.0,
            price_ok=True,
            breadth_ok=True,
            vol_ok=True,
            credit_ok=True,
            market_regime="BULL",
        ),
        items=[item],
        tradable=[item],
        overflow=[],
        long_candidates=[item],
        short_candidates=[],
        market_wide_institutional_selling=False,
    )


def _iaric_watchlist_artifact(trade_date: date) -> WatchlistArtifact:
    item = WatchlistItem(
        symbol="MSFT",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector="Technology",
        regime_score=0.8,
        regime_tier="B",
        regime_risk_multiplier=1.0,
        sector_score=0.7,
        sector_rank_weight=0.8,
        sponsorship_score=0.9,
        sponsorship_state="POSITIVE",
        persistence=0.6,
        intensity_z=1.2,
        accel_z=0.8,
        rs_percentile=95.0,
        leader_pass=True,
        trend_pass=True,
        trend_strength=0.9,
        earnings_risk_flag=False,
        blacklist_flag=False,
        anchor_date=trade_date,
        anchor_type="AVWAP",
        acceptance_pass=True,
        avwap_ref=410.0,
        avwap_band_lower=407.5,
        avwap_band_upper=412.5,
        daily_atr_estimate=6.5,
        intraday_atr_seed=1.2,
        daily_rank=1.0,
        tradable_flag=True,
        conviction_bucket="HIGH",
        conviction_multiplier=1.2,
        recommended_risk_r=1.0,
        average_30m_volume=200_000.0,
        expected_5m_volume=25_000.0,
        entry_gap_pct=0.4,
        flow_proxy_gate_pass=True,
        daily_signal_score=88.0,
        trigger_types=["OPENING_RECLAIM", "VWAP_BOUNCE"],
        trigger_tier="PREMIUM",
        trend_tier="STRONG",
        rescue_flow_candidate=True,
        sizing_mult=1.15,
        cdd_value=2,
        ema10_daily=405.5,
        rsi14_daily=63.0,
    )
    held = HeldPositionDirective(
        symbol="MSFT",
        entry_time=datetime(2026, 4, 24, 14, 30, tzinfo=UTC),
        entry_price=408.5,
        size=50,
        stop=402.0,
        initial_r=6.5,
        setup_tag="PB_CARRY",
        time_stop_deadline=datetime(2026, 4, 25, 18, 0, tzinfo=UTC),
        carry_eligible_flag=True,
        flow_reversal_flag=False,
    )
    return WatchlistArtifact(
        trade_date=trade_date,
        generated_at=datetime.combine(trade_date, time(0, 0), tzinfo=UTC),
        regime=IARICRegimeSnapshot(
            score=0.75,
            tier="B",
            risk_multiplier=1.0,
            price_ok=True,
            breadth_ok=True,
            vol_ok=True,
            credit_ok=True,
        ),
        items=[item],
        tradable=[item],
        overflow=[],
        market_wide_institutional_selling=False,
        held_positions=[held],
    )


def _iaric_intraday_snapshot(trade_date: date) -> IntradayStateSnapshot:
    return IntradayStateSnapshot(
        trade_date=trade_date,
        saved_at=datetime(2026, 4, 25, 15, 30, tzinfo=UTC),
        symbols=[
            PBSymbolState(
                symbol="MSFT",
                stage="IN_POSITION",
                route_family="OPENING_RECLAIM",
                in_position=True,
                position=PositionState(
                    entry_price=408.5,
                    qty_entry=50,
                    qty_open=25,
                    final_stop=402.0,
                    current_stop=404.0,
                    entry_time=datetime(2026, 4, 25, 14, 35, tzinfo=UTC),
                    initial_risk_per_share=6.5,
                    max_favorable_price=414.0,
                    max_adverse_price=407.0,
                    partial_taken=True,
                    stop_order_id="STOP-1",
                    trade_id="TRADE-1",
                    realized_pnl_usd=180.0,
                    setup_tag="PB_V2",
                    time_stop_deadline=datetime(2026, 4, 25, 18, 0, tzinfo=UTC),
                ),
                entry_order=PendingOrderState(
                    oms_order_id="ENTRY-1",
                    submitted_at=datetime(2026, 4, 25, 14, 31, tzinfo=UTC),
                    role="ENTRY",
                    requested_qty=50,
                    limit_price=409.0,
                    stop_price=409.2,
                ),
                bars_seen_today=12,
                mfe_stage=2,
                breakeven_activated=True,
                trail_active=True,
                hold_bars=9,
                risk_per_share=6.5,
                v2_partial_taken=True,
                carry_decision_path="STRONG_TREND_CARRY",
                consecutive_bars_below_vwap=1,
            )
        ],
        last_decision_code="PARTIAL_EXIT_FILLED",
        meta={"active_symbols": ["MSFT"]},
    )


@pytest.mark.asyncio
async def test_alcb_plugin_replays_pending_snapshot_and_delegates_health(monkeypatch) -> None:
    plugin = ALCBPlugin(_runtime_ctx())
    plugin._artifact = object()

    class _FakeEngine:
        def __init__(self) -> None:
            self.hydrated: list[dict] = []
            self.started = False

        def hydrate_state(self, snapshot: dict) -> None:
            self.hydrated.append(snapshot)

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.started = False

        def health_status(self) -> dict:
            return {
                "strategy_id": "ALCB_v1",
                "running": self.started,
                "last_decision_code": "ENTRY_SUBMITTED",
            }

        def snapshot_state(self) -> dict:
            return {"positions": {"AAA": {"trade_class": "MOMENTUM"}}}

    fake_engine = _FakeEngine()
    monkeypatch.setattr(plugin, "_build_engine", lambda: fake_engine)

    pending_snapshot = {"positions": {"AAA": {"trade_class": "MOMENTUM"}}}
    await plugin.hydrate(pending_snapshot)
    await plugin.start()

    assert fake_engine.hydrated == [pending_snapshot]
    assert plugin.health_status()["last_decision_code"] == "ENTRY_SUBMITTED"
    assert plugin.snapshot_state() == {"positions": {"AAA": {"trade_class": "MOMENTUM"}}}


@pytest.mark.asyncio
async def test_iaric_plugin_coerces_snapshot_payload_for_engine_hydrate(monkeypatch) -> None:
    plugin = IARICPlugin(_runtime_ctx())
    plugin._artifact = object()
    payload = asdict(_iaric_intraday_snapshot(date(2026, 4, 25)))

    class _FakeEngine:
        def __init__(self) -> None:
            self.snapshot = None

        def hydrate_state(self, snapshot: IntradayStateSnapshot) -> None:
            self.snapshot = snapshot

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        def health_status(self) -> dict:
            return {"strategy_id": "IARIC_v1", "running": True}

        def snapshot_state(self) -> IntradayStateSnapshot:
            return self.snapshot

    fake_engine = _FakeEngine()
    monkeypatch.setattr(plugin, "_build_engine", lambda: fake_engine)

    await plugin.hydrate(payload)
    pending = plugin._pending_snapshot
    assert isinstance(pending, IntradayStateSnapshot)
    assert isinstance(pending.symbols[0], PBSymbolState)
    assert pending.symbols[0].route_family == "OPENING_RECLAIM"
    assert pending.symbols[0].v2_partial_taken is True

    await plugin.start()

    assert isinstance(fake_engine.snapshot, IntradayStateSnapshot)
    assert fake_engine.snapshot.symbols[0].carry_decision_path == "STRONG_TREND_CARRY"
    assert plugin.snapshot_state()["symbols"][0]["route_family"] == "OPENING_RECLAIM"


def test_alcb_candidate_artifact_roundtrip_preserves_campaign_fields(tmp_path) -> None:
    trade_date = date(2026, 4, 25)
    artifact = _alcb_artifact(trade_date)

    persist_candidate_artifact(artifact, root=tmp_path)
    restored = load_candidate_artifact(trade_date, root=tmp_path)

    assert restored.regime.market_regime == "BULL"
    assert restored.items[0].campaign.state == CampaignState.COMPRESSION
    assert restored.items[0].campaign.continuation_enabled is True
    assert restored.long_candidates[0].symbol == "AAA"


def test_iaric_watchlist_artifact_roundtrip_preserves_directive_fields(tmp_path) -> None:
    trade_date = date(2026, 4, 25)
    artifact = _iaric_watchlist_artifact(trade_date)

    persist_watchlist_artifact(artifact, root=tmp_path)
    restored = load_watchlist_artifact(trade_date, root=tmp_path)

    assert restored.items[0].trigger_types == ["OPENING_RECLAIM", "VWAP_BOUNCE"]
    assert restored.items[0].sizing_mult == pytest.approx(1.15)
    assert restored.items[0].entry_gap_pct == pytest.approx(0.4)
    assert restored.held_positions[0].setup_tag == "PB_CARRY"
    assert restored.held_positions[0].time_stop_deadline == datetime(2026, 4, 25, 18, 0, tzinfo=UTC)


def test_iaric_intraday_state_roundtrip_preserves_route_and_partial_fields(tmp_path) -> None:
    snapshot = _iaric_intraday_snapshot(date(2026, 4, 25))

    persist_intraday_state(snapshot, root=tmp_path)
    restored = load_intraday_state(snapshot.trade_date, root=tmp_path)
    from_payload = coerce_intraday_state_snapshot(asdict(snapshot))

    for item in (restored, from_payload):
        assert item.last_decision_code == "PARTIAL_EXIT_FILLED"
        assert item.meta["active_symbols"] == ["MSFT"]
        assert isinstance(item.symbols[0], PBSymbolState)
        assert item.symbols[0].route_family == "OPENING_RECLAIM"
        assert item.symbols[0].v2_partial_taken is True
        assert item.symbols[0].carry_decision_path == "STRONG_TREND_CARRY"
