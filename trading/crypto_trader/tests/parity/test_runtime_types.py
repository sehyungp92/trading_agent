"""Tests for additive parity runtime contracts."""

from datetime import datetime, timedelta, timezone

import pytest

from crypto_trader.core.models import Bar, Order, OrderStatus, OrderType, SetupGrade, Side, TimeFrame, Trade
from crypto_trader.core.runtime_types import (
    DecisionContext,
    DecisionEvent,
    ExecutionReport,
    ExecutionReportKind,
    MarketEvent,
    OrderIntent,
    PositionSnapshot,
    TimestampPolicy,
    TradeOutcome,
)


def _ts(hour: int = 12) -> datetime:
    return datetime(2026, 5, 24, hour, 0, tzinfo=timezone.utc)


def _bar() -> Bar:
    return Bar(
        timestamp=_ts(),
        symbol="BTC",
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        volume=1_000.0,
        timeframe=TimeFrame.M15,
    )


def test_market_event_from_bar_open_time_policy() -> None:
    event = MarketEvent.from_bar(
        _bar(),
        source="historical",
        timestamp_policy=TimestampPolicy.OPEN_TIME,
    )

    assert event.open_time == _ts()
    assert event.close_time == _ts() + timedelta(minutes=15)
    assert event.available_at == event.close_time
    assert event.to_dict()["timeframe"] == "15m"
    assert event.to_dict()["timestamp_policy"] == "open_time"


def test_market_event_to_bar_preserves_strategy_contract() -> None:
    event = MarketEvent.from_bar(
        _bar(),
        source="historical",
        timestamp_policy=TimestampPolicy.OPEN_TIME,
    )

    bar = event.to_bar()

    assert bar.timestamp == event.open_time
    assert bar.symbol == "BTC"
    assert bar.timeframe == TimeFrame.M15
    assert bar.close == pytest.approx(event.close)


def test_market_event_from_bar_close_time_policy() -> None:
    event = MarketEvent.from_bar(
        _bar(),
        source="live",
        timestamp_policy=TimestampPolicy.CLOSE_TIME,
    )

    assert event.open_time == _ts() - timedelta(minutes=15)
    assert event.close_time == _ts()
    assert event.available_at == _ts()
    assert event.raw_timestamp == _ts()


def test_decision_event_serializes_context() -> None:
    event = DecisionEvent(
        decision_id="d1",
        strategy_id="momentum",
        symbol="BTC",
        timeframe=TimeFrame.M15,
        decision_time=_ts(),
        decision_key="momentum|BTC|2026-05-24T12:00",
        action="enter",
        signal_context={"grade": "A"},
        metadata={"bar_id": "bar-1"},
    )

    assert event.to_dict()["signal_context"] == {"grade": "A"}
    assert event.to_dict()["timeframe"] == "15m"
    assert event.to_dict()["bar_id"] == "bar-1"


def test_order_intent_preserves_existing_order_enums() -> None:
    intent = OrderIntent(
        intent_id="i1",
        strategy_id="trend",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.STOP_LIMIT,
        qty=0.2,
        stop_price=101.0,
        limit_price=101.5,
        reduce_only=True,
        risk_metadata={"risk_R": 0.5},
    )

    payload = intent.to_dict()
    assert payload["side"] == "LONG"
    assert payload["order_type"] == "STOP_LIMIT"
    assert payload["risk_metadata"]["risk_R"] == pytest.approx(0.5)


def test_order_intent_from_order_attaches_decision_context() -> None:
    context = DecisionContext(
        decision_id="d1",
        strategy_id="momentum",
        symbol="BTC",
        timeframe=TimeFrame.M15,
        decision_time=_ts(),
        decision_key="k1",
    )
    order = Order(
        order_id="",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.MARKET,
        qty=0.1,
        tag="entry",
        metadata={"risk_R": 0.7},
    )

    intent = OrderIntent.from_order(order, context)

    assert intent.decision_id == "d1"
    assert intent.strategy_id == "momentum"
    assert intent.intent_id == "momentum:BTC:d1:intent:1"
    assert intent.metadata["tag"] == "entry"
    assert intent.risk_metadata["risk_R"] == pytest.approx(0.7)


def test_order_intent_from_order_prefers_context_identity_over_client_order_id() -> None:
    context = DecisionContext(
        decision_id="d1",
        strategy_id="trend",
        symbol="BTC",
        timeframe=TimeFrame.M15,
        decision_time=_ts(),
        decision_key="k1",
    )
    order = Order(
        order_id="trend_entry_BTC_random1234",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.MARKET,
        qty=0.1,
        tag="entry",
        metadata={"client_order_id": "trend_entry_BTC_random1234"},
    )

    intent = OrderIntent.from_order(order, context)

    assert intent.client_order_id == "trend_entry_BTC_random1234"
    assert intent.intent_id == "trend:BTC:d1:intent:1"
    assert intent.intent_id != intent.client_order_id


def test_execution_report_serializes_lifecycle_shapes() -> None:
    accepted = ExecutionReport(
        report_id="r1",
        kind=ExecutionReportKind.ACCEPTED,
        timestamp=_ts(),
        symbol="BTC",
        client_order_id="c1",
        exchange_order_id="100",
        order_status=OrderStatus.WORKING,
    )
    resting = ExecutionReport(
        report_id="r1b",
        kind=ExecutionReportKind.RESTING,
        timestamp=_ts(),
        symbol="BTC",
        client_order_id="c1",
        exchange_order_id="100",
        qty=0.2,
        order_status=OrderStatus.WORKING,
    )
    fill = ExecutionReport(
        report_id="r2",
        kind=ExecutionReportKind.FILL,
        timestamp=_ts(),
        symbol="BTC",
        side=Side.LONG,
        client_order_id="c1",
        exchange_order_id="100",
        fill_id="f1",
        filled_qty=0.1,
        fill_price=100.5,
        commission=0.01,
        liquidity="taker",
    )
    rejected = ExecutionReport(
        report_id="r3",
        kind=ExecutionReportKind.REJECTED,
        timestamp=_ts(),
        symbol="BTC",
        client_order_id="c2",
        reject_reason="insufficient_margin",
        order_status=OrderStatus.REJECTED,
    )

    assert accepted.to_dict()["kind"] == "accepted"
    assert accepted.to_dict()["order_status"] == "WORKING"
    assert resting.to_dict()["kind"] == "resting"
    assert resting.to_dict()["qty"] == pytest.approx(0.2)
    assert fill.to_dict()["side"] == "LONG"
    assert fill.to_dict()["fill_price"] == pytest.approx(100.5)
    assert rejected.to_dict()["reject_reason"] == "insufficient_margin"


def test_position_snapshot_serializes_owned_state() -> None:
    snapshot = PositionSnapshot(
        strategy_id="breakout",
        symbol="BTC",
        timestamp=_ts(),
        direction=Side.SHORT,
        qty=0.3,
        avg_entry=100.0,
        mfe_r=1.4,
        mae_r=-0.3,
        open_orders=["stop1"],
        risk_metadata={"stop": 105.0},
        source="live",
    )

    payload = snapshot.to_dict()
    assert payload["direction"] == "SHORT"
    assert payload["mfe_r"] == pytest.approx(1.4)
    assert payload["mae_r"] == pytest.approx(-0.3)
    assert payload["open_orders"] == ["stop1"]
    assert payload["risk_metadata"]["stop"] == pytest.approx(105.0)


def test_trade_outcome_from_trade_documents_current_economics() -> None:
    trade = Trade(
        trade_id="t1",
        symbol="BTC",
        direction=Side.LONG,
        entry_price=100.0,
        exit_price=110.0,
        qty=1.0,
        entry_time=_ts(10),
        exit_time=_ts(12),
        pnl=8.0,
        r_multiple=1.2,
        commission=0.5,
        bars_held=8,
        setup_grade=SetupGrade.A,
        exit_reason="trailing_stop",
        confluences_used=["ema"],
        confirmation_type="close",
        entry_method="market",
        funding_paid=2.0,
        mae_r=-0.2,
        mfe_r=1.5,
        realized_r_multiple=1.1,
    )

    outcome = TradeOutcome.from_trade(trade)
    payload = outcome.to_dict()

    assert outcome.price_pnl_gross == pytest.approx(10.0)
    assert outcome.price_pnl_after_funding == pytest.approx(8.0)
    assert outcome.total_fees == pytest.approx(0.5)
    assert outcome.funding_paid == pytest.approx(2.0)
    assert outcome.realized_pnl_net == pytest.approx(7.5)
    assert outcome.realized_pnl_net == pytest.approx(
        outcome.price_pnl_gross - outcome.total_fees - outcome.funding_paid
    )
    assert payload["geometric_r"] == pytest.approx(1.2)
    assert payload["price_pnl_after_funding"] == pytest.approx(8.0)
    assert payload["realized_r_net"] == pytest.approx(1.1)
