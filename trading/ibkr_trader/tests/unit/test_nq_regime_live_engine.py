from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from strategies.core.actions import SubmitEntry
from strategies.momentum.nq_regime.engine import NQRegimeEngine


class _IB:
    @staticmethod
    def isConnected() -> bool:
        return True

    async def qualifyContractsAsync(self, contract):
        return [contract]


class _Session:
    def __init__(self) -> None:
        self.ib = _IB()
        self.requests: list[str] = []

    async def req_historical_data(self, contract, **kwargs):
        bar_size = kwargs["barSizeSetting"]
        self.requests.append(bar_size)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        if bar_size == "1 day":
            return [
                {
                    "timestamp": now - timedelta(days=3),
                    "open": 100,
                    "high": 110,
                    "low": 95,
                    "close": 105,
                },
                {
                    "timestamp": now - timedelta(days=2),
                    "open": 105,
                    "high": 120,
                    "low": 101,
                    "close": 118,
                },
            ]
        return [
            {
                "timestamp": now - timedelta(minutes=5),
                "open": 118,
                "high": 121,
                "low": 117,
                "close": 120,
            }
        ]


@pytest.mark.asyncio
async def test_nq_regime_scheduled_bar_uses_cached_daily_levels():
    session = _Session()
    engine = NQRegimeEngine(ib_session=session)
    engine._get_analysis_contract = lambda: object()
    engine.on_bar = AsyncMock()

    bar_ts = datetime.now(timezone.utc) - timedelta(minutes=5)
    await engine._refresh_daily_context(force=True, for_ts=bar_ts)
    daily_requests_after_refresh = session.requests.count("1 day")
    await engine._fetch_and_emit_bar(request_kind="test")

    engine.on_bar.assert_awaited_once()
    levels = engine.on_bar.await_args.kwargs["daily_context"]
    assert levels.pdh == 120
    assert levels.pdl == 101
    assert levels.pdm == 110.5
    assert levels.weekly_high == 120
    assert levels.weekly_low == 95
    assert session.requests.count("1 day") == daily_requests_after_refresh
    assert session.requests[-1] == "5 mins"


@pytest.mark.asyncio
async def test_nq_regime_scheduled_bar_refreshes_daily_levels_on_session_change():
    session = _Session()
    engine = NQRegimeEngine(ib_session=session)
    engine._get_analysis_contract = lambda: object()
    engine.on_bar = AsyncMock()

    await engine._fetch_and_emit_bar(request_kind="test")

    engine.on_bar.assert_awaited_once()
    assert engine.on_bar.await_args.kwargs["daily_context"] is not None
    assert session.requests[0] == "5 mins"
    assert session.requests.count("1 day") >= 1


@pytest.mark.asyncio
async def test_nq_regime_on_bar_persists_after_successful_mutation(monkeypatch, tmp_path):
    import strategies.momentum.nq_regime.engine as engine_mod
    from strategies.momentum.nq_regime.core.levels import KeyLevels

    engine = NQRegimeEngine(state_dir=tmp_path)
    persisted = {"count": 0}
    engine._persist_state = lambda: persisted.__setitem__("count", persisted["count"] + 1)
    monkeypatch.setattr(
        engine_mod.CompletedBarPolicy,
        "build_event",
        lambda self, **kwargs: object(),
    )
    monkeypatch.setattr(
        engine_mod,
        "core_on_bar",
        lambda state, event, scheduled_news, settings: (state, [], []),
    )

    await engine.on_bar(
        {
            "timestamp": datetime(2026, 5, 8, 14, 35, tzinfo=timezone.utc),
            "open": 118,
            "high": 121,
            "low": 117,
            "close": 120,
        },
        daily_context=KeyLevels(pdh=120, pdl=101, pdm=110.5, weekly_high=120, weekly_low=95),
    )

    assert persisted["count"] == 1


@pytest.mark.asyncio
async def test_nq_regime_on_bar_routes_core_entry_action_to_oms(monkeypatch, tmp_path):
    import strategies.momentum.nq_regime.engine as engine_mod
    from strategies.momentum.nq_regime.core.levels import KeyLevels

    oms = SimpleNamespace(
        submit_intent=AsyncMock(return_value=SimpleNamespace(oms_order_id="oms-nqr-entry"))
    )
    engine = NQRegimeEngine(oms_service=oms, state_dir=tmp_path)
    action = SubmitEntry(
        client_order_id="NQ_REGIME-entry-1",
        symbol="MNQ",
        side="BUY",
        qty=1,
        order_type="LIMIT",
        limit_price=20000.0,
        stop_price=19980.0,
        risk_context={"stop_for_risk": 19980.0, "planned_entry_price": 20000.0},
        metadata={"module": "trend"},
    )
    monkeypatch.setattr(
        engine_mod.CompletedBarPolicy,
        "build_event",
        lambda self, **kwargs: object(),
    )
    monkeypatch.setattr(
        engine_mod,
        "core_on_bar",
        lambda state, event, scheduled_news, settings: (state, [action], []),
    )

    await engine.on_bar(
        {
            "timestamp": datetime(2026, 5, 8, 14, 35, tzinfo=timezone.utc),
            "open": 20000,
            "high": 20010,
            "low": 19990,
            "close": 20005,
        },
        daily_context=KeyLevels(pdh=20100, pdl=19900, pdm=20000, weekly_high=20200, weekly_low=19800),
    )

    oms.submit_intent.assert_awaited_once()
    intent = oms.submit_intent.await_args.args[0]
    assert intent.order.strategy_id == "NQ_REGIME"
    assert intent.order.client_order_id == "NQ_REGIME-entry-1"
    assert intent.order.risk_context.risk_dollars > 0


@pytest.mark.asyncio
async def test_nq_regime_forces_flatten_during_roll_blackout(monkeypatch, tmp_path):
    from libs.oms.models.events import OMSEventType
    import strategies.momentum.nq_regime.engine as engine_mod
    from strategies.momentum.nq_regime.config import TradeSide
    from strategies.momentum.nq_regime.core.levels import KeyLevels

    oms = SimpleNamespace(
        submit_intent=AsyncMock(return_value=SimpleNamespace(oms_order_id="roll-flat"))
    )
    engine = NQRegimeEngine(oms_service=oms, state_dir=tmp_path)
    engine._state.position_side = TradeSide.LONG
    engine._state.qty_open = 1
    engine._state.working_stop_order_id = "stop-1"
    engine._state.working_target_order_ids = ("target-1",)
    monkeypatch.setattr(
        engine_mod.CompletedBarPolicy,
        "build_event",
        lambda self, **kwargs: object(),
    )
    monkeypatch.setattr(
        engine_mod,
        "core_on_bar",
        lambda state, event, scheduled_news, settings: (state, [], []),
    )
    monkeypatch.setattr(
        engine_mod,
        "core_on_order_update",
        lambda state, update: (state, [], []),
    )
    monkeypatch.setattr(
        engine_mod,
        "roll_force_flatten_reason",
        lambda instrument, as_of=None: "roll safety",
    )

    await engine.on_bar(
        {
            "timestamp": datetime(2026, 3, 16, 14, 35, tzinfo=timezone.utc),
            "open": 20000,
            "high": 20010,
            "low": 19990,
            "close": 20005,
        },
        daily_context=KeyLevels(pdh=20100, pdl=19900, pdm=20000, weekly_high=20200, weekly_low=19800),
    )

    intents = [call.args[0] for call in oms.submit_intent.await_args_list]
    assert [intent.intent_type.value for intent in intents] == ["CANCEL_ORDER", "CANCEL_ORDER", "FLATTEN"]
    assert intents[-1].instrument_symbol == "MNQ"
    assert engine._roll_flatten_oms_id == "roll-flat"
    assert engine._last_decision_code == "ROLL_FORCE_FLATTEN"

    oms.submit_intent.reset_mock()
    await engine._handle_oms_event(
        SimpleNamespace(
            event_type=OMSEventType.ORDER_REJECTED,
            oms_order_id="roll-flat",
            payload={},
            timestamp=datetime(2026, 3, 16, 14, 36, tzinfo=timezone.utc),
        )
    )

    retry_intents = [call.args[0] for call in oms.submit_intent.await_args_list]
    assert [intent.intent_type.value for intent in retry_intents] == ["CANCEL_ORDER", "CANCEL_ORDER", "FLATTEN"]
    assert engine._roll_flatten_pending is True


def test_nq_regime_state_persist_uses_atomic_replace(tmp_path):
    engine = NQRegimeEngine(state_dir=tmp_path)

    engine._persist_state()

    snap_path = tmp_path / "NQ_REGIME.json"
    assert snap_path.exists()
    assert not (tmp_path / "NQ_REGIME.json.tmp").exists()
