from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from libs.oms.models.instrument import Instrument
from libs.oms.models.intent import IntentResult
from libs.oms.models.order import OrderType
from strategies.core.actions import SubmitAddOnEntry, SubmitEntry
from strategies.momentum.downturn import engine as downturn_engine_module
from strategies.momentum.downturn.engine import DownturnEngine
from strategies.momentum.downturn.models import EngineTag
from strategies.momentum.nqdtc import engine as nqdtc_engine_module
from strategies.momentum.nqdtc.engine import NQDTCEngine
from strategies.momentum.nqdtc.models import Direction as NQDTCDirection
from strategies.momentum.nqdtc.models import EntrySubtype
from strategies.momentum.nq_regime.engine import NQRegimeEngine
from strategies.momentum.vdub.engine import VdubNQv4Engine
from strategies.momentum.vdub.models import Direction as VdubDirection
from strategies.momentum.vdub.models import EntryType as VdubEntryType
from strategies.momentum.vdub.models import WorkingEntry as VdubWorkingEntry
from strategies.swing.akc_helix.engine import HelixEngine
from strategies.swing.akc_helix.models import Direction as HelixDirection
from strategies.swing.akc_helix.models import SetupClass, SetupInstance
from strategies.swing.atrss.config import SymbolConfig as ATRSSSymbolConfig
from strategies.swing.atrss.engine import ATRSSEngine
from strategies.swing.atrss.models import Candidate, CandidateType
from strategies.swing.atrss.models import Direction as ATRSSDirection
from strategies.swing.atrss.models import HourlyState, PositionBook
from strategies.stock.alcb.execution import build_entry_order as build_alcb_entry_order
from strategies.stock.alcb.models import (
    Campaign,
    CandidateItem,
    CampaignState,
    Direction,
    EntryType,
    PositionPlan,
)
from strategies.stock.iaric.execution import build_entry_order as build_iaric_entry_order
from strategies.stock.iaric.models import WatchlistItem
from strategies.swing.tpc.engine import TPCEngine


UTC = timezone.utc


def _instrument(symbol: str = "MNQ") -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="CME" if symbol.endswith("NQ") or symbol in {"MNQ", "NQ"} else "SMART",
        tick_size=0.25 if symbol in {"MNQ", "NQ"} else 0.01,
        tick_value=0.50 if symbol in {"MNQ", "NQ"} else 0.01,
        multiplier=2.0 if symbol in {"MNQ", "NQ"} else 1.0,
        point_value=2.0 if symbol in {"MNQ", "NQ"} else 1.0,
    )


class _CaptureOMS:
    def __init__(self) -> None:
        self.intents = []

    async def submit_intent(self, intent):
        self.intents.append(intent)
        return SimpleNamespace(
            result=IntentResult.ACCEPTED,
            oms_order_id=f"OMS-{len(self.intents)}",
        )

    def stream_events(self, _strategy_id):
        return SimpleNamespace()


def _iaric_item() -> WatchlistItem:
    return WatchlistItem(
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
        anchor_date=date(2026, 6, 4),
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
    )


def _alcb_item() -> CandidateItem:
    return CandidateItem(
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
        campaign=Campaign(symbol="AAA", state=CampaignState.COMPRESSION, campaign_id=7),
    )


def _alcb_plan() -> PositionPlan:
    return PositionPlan(
        symbol="AAA",
        direction=Direction.LONG,
        entry_type=EntryType.OR_BREAKOUT,
        entry_price=25.0,
        stop_price=23.5,
        tp1_price=26.5,
        tp2_price=28.0,
        quantity=100,
        risk_per_share=1.5,
        risk_dollars=150.0,
        quality_mult=1.0,
        regime_mult=1.0,
        corr_mult=1.0,
    )


def test_iaric_entry_builder_persists_source_signal_and_bar_context() -> None:
    ts = datetime(2026, 6, 4, 14, 30, tzinfo=UTC)
    order = build_iaric_entry_order(
        _iaric_item(),
        "paper",
        10,
        411.0,
        405.0,
        signal_id="MSFT:pb_open:20260604T143000Z",
        bar_id="MSFT:2026-06-04T14:30:00+00:00",
        exchange_timestamp=ts,
    )

    assert order.risk_context.signal_id == "MSFT:pb_open:20260604T143000Z"
    assert order.risk_context.bar_id == "MSFT:2026-06-04T14:30:00+00:00"
    assert order.risk_context.exchange_timestamp == ts


def test_alcb_entry_builder_persists_source_signal_and_bar_context() -> None:
    ts = datetime(2026, 6, 4, 15, 0, tzinfo=UTC)
    order = build_alcb_entry_order(
        _alcb_item(),
        "paper",
        _alcb_plan(),
        signal_id="AAA:OR_BREAKOUT:2026-06-04T15:00:00+00:00",
        bar_id="AAA:2026-06-04T15:00:00+00:00",
        exchange_timestamp=ts,
    )

    assert order.risk_context.signal_id == "AAA:OR_BREAKOUT:2026-06-04T15:00:00+00:00"
    assert order.risk_context.bar_id == "AAA:2026-06-04T15:00:00+00:00"
    assert order.risk_context.exchange_timestamp == ts


def test_nq_regime_order_builder_carries_candidate_signal_and_bar_context(tmp_path) -> None:
    engine = NQRegimeEngine(
        oms_service=SimpleNamespace(),
        state_dir=tmp_path,
    )
    action = SubmitEntry(
        client_order_id="NQ_REGIME-entry-1",
        symbol="MNQ",
        side="BUY",
        qty=1,
        order_type="LIMIT",
        limit_price=20_000.0,
        stop_price=19_980.0,
        risk_context={"stop_for_risk": 19_980.0, "planned_entry_price": 20_000.0},
        metadata={
            "module": "trend",
            "candidate_id": "nqreg-exp-202606041430-BULL",
            "signal_ts": "2026-06-04T14:30:00+00:00",
        },
    )

    order = engine._order_from_entry(action)

    assert order.risk_context.signal_id == "nqreg-exp-202606041430-BULL"
    assert order.risk_context.bar_id == "MNQ:2026-06-04T14:30:00+00:00"
    assert order.risk_context.exchange_timestamp == datetime(2026, 6, 4, 14, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_nqdtc_runtime_submit_carries_source_signal_and_bar_context(monkeypatch, tmp_path) -> None:
    ts = datetime(2026, 6, 4, 14, 30, tzinfo=UTC)
    oms = _CaptureOMS()
    engine = NQDTCEngine(
        ib_session=object(),
        oms_service=oms,
        instruments={"NQ": _instrument("NQ")},
        state_dir=tmp_path,
        disable_background_tasks=True,
    )
    engine._last_bar_ts = ts
    engine._bar_count_5m = 42

    def fake_on_bar(state, **kwargs):
        request = kwargs["entry_request"]
        return state, [
            SubmitEntry(
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                side="BUY",
                qty=request.qty,
                order_type="LIMIT",
                limit_price=20_000.0,
            )
        ], []

    monkeypatch.setattr(nqdtc_engine_module.nqdtc_core_logic, "on_bar", fake_on_bar)
    monkeypatch.setattr(
        nqdtc_engine_module.nqdtc_core_logic,
        "on_order_update",
        lambda state, _update: (state, [], []),
    )

    await engine._submit_order(
        subtype=EntrySubtype.A_RETEST,
        direction=NQDTCDirection.LONG,
        order_type=OrderType.LIMIT,
        price=20_000.0,
        stop_price=None,
        qty=1,
        stop_for_risk=19_980.0,
    )

    risk_context = oms.intents[-1].order.risk_context
    assert risk_context.signal_id == "NQ:A_retest:LONG:2026-06-04T14:30:00+00:00"
    assert risk_context.bar_id == "NQ:5m:2026-06-04T14:30:00+00:00"
    assert risk_context.exchange_timestamp == ts


@pytest.mark.asyncio
async def test_downturn_runtime_submit_carries_source_signal_and_bar_context(monkeypatch, tmp_path) -> None:
    ts = datetime(2026, 6, 4, 14, 35, tzinfo=UTC)
    oms = _CaptureOMS()
    engine = DownturnEngine(
        ib_session=object(),
        oms_service=oms,
        instruments={"MNQ": _instrument("MNQ")},
        state_dir=tmp_path,
        disable_background_tasks=True,
    )
    engine._last_bar_ts = ts
    engine._bar_count_5m = 43
    engine._bars_5m = {"close": [20_000.0]}

    monkeypatch.setattr(
        downturn_engine_module,
        "compute_entry_subtype_stop",
        lambda *_args, **_kwargs: (20_000.0, 20_020.0, "stop_market"),
    )
    monkeypatch.setattr(
        downturn_engine_module,
        "compute_tiered_tp_schedule",
        lambda *_args, **_kwargs: [],
    )

    def fake_on_bar(state, **kwargs):
        request = kwargs["entry_request"]
        return state, [
            SubmitEntry(
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                side="SELL",
                qty=request.qty,
                order_type="STOP",
                stop_price=request.stop_price,
            )
        ], []

    monkeypatch.setattr(downturn_engine_module.downturn_core_logic, "on_bar", fake_on_bar)
    monkeypatch.setattr(
        downturn_engine_module.downturn_core_logic,
        "on_order_update",
        lambda state, _update: (state, [], []),
    )

    await engine._submit_entry(
        SimpleNamespace(timestamp=ts),
        EngineTag.FADE,
    )

    risk_context = oms.intents[-1].order.risk_context
    assert risk_context.signal_id == "MNQ:fade:vwap_rejection:2026-06-04T14:35:00+00:00"
    assert risk_context.bar_id == "MNQ:5m:2026-06-04T14:35:00+00:00"
    assert risk_context.exchange_timestamp == ts


@pytest.mark.asyncio
async def test_vdub_runtime_submit_and_fallback_carry_source_signal_and_bar_context() -> None:
    ts = datetime(2026, 6, 4, 14, 45, tzinfo=UTC)
    oms = _CaptureOMS()
    engine = VdubNQv4Engine(
        ib_session=object(),
        oms_service=oms,
        instruments=[_instrument("NQ")],
        disable_background_tasks=True,
    )
    engine._last_bar_ts = ts
    engine._bar_idx = 44

    await engine._submit_entry(
        VdubDirection.LONG,
        1,
        20_000.0,
        20_001.0,
        19_980.0,
        VdubEntryType.TYPE_A,
        False,
        False,
        1.0,
        20_000.0,
        signal_id="A_LONG_44",
    )

    primary_context = oms.intents[-1].order.risk_context
    assert primary_context.signal_id == "A_LONG_44"
    assert primary_context.bar_id == "NQ:15m:2026-06-04T14:45:00+00:00"
    assert primary_context.exchange_timestamp == ts

    await engine._submit_fallback_market(
        VdubWorkingEntry(
            entry_type=VdubEntryType.TYPE_A,
            direction=VdubDirection.LONG,
            stop_entry=20_000.0,
            limit_entry=20_001.0,
            qty=1,
            submitted_bar_idx=44,
            initial_stop=19_980.0,
            signal_id="A_LONG_44",
            bar_id="NQ:15m:2026-06-04T14:45:00+00:00",
            exchange_timestamp=ts,
        )
    )

    fallback_context = oms.intents[-1].order.risk_context
    assert fallback_context.signal_id == "A_LONG_44"
    assert fallback_context.bar_id == "NQ:15m:2026-06-04T14:45:00+00:00"
    assert fallback_context.exchange_timestamp == ts


@pytest.mark.asyncio
async def test_atrss_runtime_submit_carries_candidate_signal_and_bar_context(tmp_path) -> None:
    ts = datetime(2026, 6, 4, 15, 0, tzinfo=UTC)
    oms = _CaptureOMS()
    engine = ATRSSEngine(
        ib_session=object(),
        oms_service=oms,
        instruments={"QQQ": _instrument("QQQ")},
        config={"QQQ": ATRSSSymbolConfig(symbol="QQQ", tick_size=0.01, multiplier=1.0)},
        equity=100_000.0,
        disable_background_tasks=True,
    )
    engine._last_bar_ts = ts
    engine._apply_core_bar_transition = lambda **_kwargs: (
        [
            SubmitEntry(
                client_order_id="QQQ-entry-1",
                symbol="QQQ",
                side="BUY",
                qty=10,
                order_type="STOP_LIMIT",
                stop_price=100.0,
                limit_price=100.05,
            )
        ],
        [],
    )
    candidate = Candidate(
        symbol="QQQ",
        type=CandidateType.BREAKOUT,
        direction=ATRSSDirection.LONG,
        trigger_price=100.0,
        initial_stop=98.0,
        qty=10,
        signal_bar=HourlyState(time=ts),
    )

    await engine._submit_entry(candidate)

    risk_context = oms.intents[-1].order.risk_context
    assert risk_context.signal_id == "QQQ:BREAKOUT:LONG:2026-06-04T15:00:00+00:00"
    assert risk_context.bar_id == "QQQ:1h:2026-06-04T15:00:00+00:00"
    assert risk_context.exchange_timestamp == ts


def test_akc_helix_setup_risk_context_carries_setup_signal_and_bar_context() -> None:
    ts = datetime(2026, 6, 4, 15, 30, tzinfo=UTC)
    engine = object.__new__(HelixEngine)
    engine._last_bar_ts = None
    engine._symbol_last_bar_ts = {}
    setup = SetupInstance(
        setup_id="AKC-setup-1",
        symbol="QQQ",
        setup_class=SetupClass.CLASS_B,
        direction=HelixDirection.LONG,
        origin_tf="1H",
        created_ts=ts,
    )

    risk_context = engine._setup_entry_risk_context(
        setup,
        planned_entry_price=100.0,
        stop_for_risk=98.0,
        qty=10,
        point_value=1.0,
    )
    add_context = engine._setup_entry_risk_context(
        setup,
        planned_entry_price=101.0,
        stop_for_risk=99.0,
        qty=5,
        point_value=1.0,
        role="add",
        bar_ts=ts,
    )

    assert risk_context.signal_id == "AKC-setup-1"
    assert risk_context.bar_id == "QQQ:1H:2026-06-04T15:30:00+00:00"
    assert risk_context.exchange_timestamp == ts
    assert add_context.signal_id == "AKC-setup-1:add"
    assert add_context.bar_id == "QQQ:1H:2026-06-04T15:30:00+00:00"
    assert add_context.exchange_timestamp == ts


def test_tpc_order_builder_carries_setup_signal_and_bar_context() -> None:
    engine = TPCEngine(
        ib_session=object(),
        oms_service=SimpleNamespace(),
        instruments={
            "QQQ": Instrument(
                symbol="QQQ",
                root="QQQ",
                venue="SMART",
                tick_size=0.01,
                tick_value=0.01,
                multiplier=1.0,
                point_value=1.0,
            )
        },
        config={},
        kit=SimpleNamespace(active=False),
        equity=10_000.0,
    )
    action = SubmitEntry(
        client_order_id="TPC-QQQ-entry-1",
        symbol="QQQ",
        side="BUY",
        qty=10,
        order_type="LIMIT",
        limit_price=100.0,
        stop_price=99.0,
        risk_context={
            "stop_for_risk": 99.0,
            "planned_entry_price": 100.0,
            "signal_id": "TPC-QQQ-setup-1",
            "bar_id": "QQQ:2026-06-04T14:30:00+00:00",
            "exchange_timestamp": "2026-06-04T14:30:00+00:00",
        },
        metadata={"setup_id": "TPC-QQQ-setup-1"},
    )

    risk_context = engine._build_risk_context(action, engine._instruments["QQQ"])

    assert risk_context.signal_id == "TPC-QQQ-setup-1"
    assert risk_context.bar_id == "QQQ:2026-06-04T14:30:00+00:00"
    assert risk_context.exchange_timestamp == datetime(2026, 6, 4, 14, 30, tzinfo=UTC)
