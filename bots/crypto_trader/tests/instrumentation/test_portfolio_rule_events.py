from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.core.models import Order, OrderStatus, OrderType, Side, TimeFrame
from crypto_trader.core.runtime_types import DecisionContext
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation
from crypto_trader.portfolio import backtest_runner
from crypto_trader.portfolio.backtest_runner import RuleEvent, run_portfolio_backtest
from crypto_trader.portfolio.coordinator import BrokerProxy
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import OpenRisk, PortfolioState


def _manager(max_total_positions: int = 9) -> tuple[PortfolioManager, PortfolioState]:
    state = PortfolioState(equity=10_000.0, peak_equity=10_000.0)
    config = PortfolioConfig(
        strategies=(StrategyAllocation(strategy_id="momentum"),),
        max_total_positions=max_total_positions,
    )
    return PortfolioManager(config, state), state


def test_portfolio_manager_returns_rule_trail_for_approval() -> None:
    manager, _ = _manager()

    result = manager.check_entry("momentum", "BTC", Side.LONG, 0.5)

    assert result.approved
    assert result.rule_event_id
    assert result.risk_decision_id
    assert [rule["rule"] for rule in result.rule_evaluations][:2] == [
        "strategy_enabled",
        "max_total_positions",
    ]
    assert result.state_after_preview["total_positions"] == 1


def test_broker_proxy_emits_portfolio_denial_and_order_reject_events() -> None:
    broker = MagicMock()
    manager, state = _manager(max_total_positions=1)
    state.add_risk(OpenRisk("momentum", "ETH", Side.LONG, 1.0))
    emitted: list[tuple[str, dict]] = []
    proxy = BrokerProxy(
        broker,
        manager,
        "momentum",
        event_callback=lambda event_type, payload: emitted.append((event_type, payload)),
    )

    order = Order(
        order_id="o1",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.MARKET,
        qty=0.1,
        tag="entry",
        metadata={
            "risk_R": 0.5,
            "decision_id": "d1",
            "bar_id": "bar1",
            "submitted_at": datetime(2026, 5, 31, tzinfo=timezone.utc).isoformat(),
        },
    )

    assert proxy.submit_order(order) == "o1"
    assert order.status == OrderStatus.REJECTED
    assert broker.submit_order.call_count == 0
    assert [event_type for event_type, _ in emitted] == [
        "portfolio_rule",
        "risk_decision",
        "order",
    ]
    assert emitted[0][1]["approved"] is False
    assert emitted[2][1]["rejection_stage"] == "portfolio_rule"


def test_broker_proxy_contextualizes_portfolio_rule_ids_and_intent_id() -> None:
    broker = MagicMock()
    broker.submit_order.side_effect = lambda order: order.order_id or "broker_1"
    manager, _ = _manager()
    emitted: list[tuple[str, dict]] = []
    proxy = BrokerProxy(
        broker,
        manager,
        "momentum",
        event_callback=lambda event_type, payload: emitted.append((event_type, payload)),
    )

    first_context = DecisionContext(
        decision_id="momentum|BTC|15m|2026-05-31T00:15:00+00:00",
        strategy_id="momentum",
        symbol="BTC",
        timeframe=TimeFrame.M15,
        decision_time=datetime(2026, 5, 31, 0, 15, tzinfo=timezone.utc),
        decision_key="d1",
        metadata={"bar_id": "bar-1"},
    )
    second_context = DecisionContext(
        decision_id="momentum|BTC|15m|2026-05-31T00:30:00+00:00",
        strategy_id="momentum",
        symbol="BTC",
        timeframe=TimeFrame.M15,
        decision_time=datetime(2026, 5, 31, 0, 30, tzinfo=timezone.utc),
        decision_key="d2",
        metadata={"bar_id": "bar-2"},
    )

    first = Order("", "BTC", Side.LONG, OrderType.MARKET, 0.1, tag="entry", metadata={"risk_R": 0.5})
    second = Order("", "BTC", Side.LONG, OrderType.MARKET, 0.1, tag="entry", metadata={"risk_R": 0.5})

    proxy.begin_decision_context(first_context)
    proxy.submit_order(first)
    proxy.end_decision_context(first_context)
    proxy.begin_decision_context(second_context)
    proxy.submit_order(second)
    proxy.end_decision_context(second_context)

    portfolio_events = [
        payload for event_type, payload in emitted
        if event_type == "portfolio_rule"
    ]

    assert len(portfolio_events) == 2
    assert portfolio_events[0]["event_type"] == "portfolio_rule"
    assert portfolio_events[0]["rule_event_id"] == portfolio_events[0]["portfolio_rule_event_id"]
    assert portfolio_events[0]["rule_evaluation_id"] == portfolio_events[1]["rule_evaluation_id"]
    assert portfolio_events[0]["portfolio_rule_event_id"] != portfolio_events[1]["portfolio_rule_event_id"]
    assert portfolio_events[0]["intent_id"].endswith(":intent:1")
    assert portfolio_events[0]["portfolio_config"]["max_total_positions"] == 9
    assert portfolio_events[0]["portfolio_config_version"]
    assert portfolio_events[0]["risk_config_version"]
    assert portfolio_events[0]["allocation_version"]
    assert first.metadata["intent_id"] == portfolio_events[0]["intent_id"]
    assert first.metadata["portfolio_rule_event_id"] == portfolio_events[0]["portfolio_rule_event_id"]


def test_broker_proxy_intent_id_prefers_decision_context_over_client_order_id() -> None:
    broker = MagicMock()
    broker.submit_order.side_effect = lambda order: order.order_id
    manager, _ = _manager()
    emitted: list[tuple[str, dict]] = []
    proxy = BrokerProxy(
        broker,
        manager,
        "momentum",
        event_callback=lambda event_type, payload: emitted.append((event_type, payload)),
    )
    context = DecisionContext(
        decision_id="momentum|BTC|15m|2026-05-31T00:15:00+00:00",
        strategy_id="momentum",
        symbol="BTC",
        timeframe=TimeFrame.M15,
        decision_time=datetime(2026, 5, 31, 0, 15, tzinfo=timezone.utc),
        decision_key="d1",
        metadata={"bar_id": "bar-1"},
    )
    order = Order(
        "trend_entry_BTC_random1234",
        "BTC",
        Side.LONG,
        OrderType.MARKET,
        0.1,
        tag="entry",
        metadata={"risk_R": 0.5},
    )

    proxy.begin_decision_context(context)
    proxy.submit_order(order)
    proxy.end_decision_context(context)

    portfolio_event = next(payload for event_type, payload in emitted if event_type == "portfolio_rule")
    assert portfolio_event["client_order_id"] == "trend_entry_BTC_random1234"
    assert portfolio_event["intent_id"] == (
        "momentum:BTC:momentum|BTC|15m|2026-05-31T00:15:00+00:00:intent:1"
    )
    assert portfolio_event["intent_id"] != portfolio_event["client_order_id"]


def test_backtest_rule_event_preserves_live_schema_fields() -> None:
    portfolio_config = _manager()[0].config.to_dict()
    payload = {
        "portfolio_rule_event_id": "pre_1",
        "risk_decision_id": "risk_1",
        "rule_evaluation_id": "eval_1",
        "strategy_id": "momentum",
        "symbol": "BTC",
        "direction": "LONG",
        "decision_id": "decision_1",
        "bar_id": "bar_1",
        "intent_id": "intent_1",
        "client_order_id": "client_1",
        "requested_risk_R": 0.5,
        "approved": True,
        "action": "allow",
        "size_multiplier": 1.0,
        "adjusted_risk_R": 0.5,
        "rule_evaluations": [{"rule": "strategy_enabled", "passed": True}],
        "state_before": {"total_positions": 0},
        "state_after_preview": {"total_positions": 1},
        "allocation": {"strategy_id": "momentum"},
        "request": {"requested_risk_R": 0.5},
    }

    event = RuleEvent.from_payload(
        payload,
        timestamp=datetime(2026, 5, 31, tzinfo=timezone.utc),
        portfolio_config=portfolio_config,
        portfolio_config_version="pcfg",
        risk_config_version="risk",
        allocation_version="alloc",
    )

    assert event.portfolio_rule_event_id == "pre_1"
    assert event.event_type == "portfolio_rule"
    assert event.rule_event_id == "pre_1"
    assert event.risk_decision_id == "risk_1"
    assert event.decision_id == "decision_1"
    assert event.intent_id == "intent_1"
    assert event.rule_evaluations == [{"rule": "strategy_enabled", "passed": True}]
    assert event.state_before == {"total_positions": 0}
    assert event.state_after_preview == {"total_positions": 1}
    assert event.portfolio_config == portfolio_config
    assert event.lineage["source"] == "portfolio_backtest"
    assert event.portfolio_config_version == "pcfg"
    assert event.payload["portfolio_config_version"] == "pcfg"
    assert event.payload["risk_config_version"] == "risk"
    assert event.payload["allocation_version"] == "alloc"
    assert event.to_dict()["event_type"] == "portfolio_rule"
    assert event.to_dict()["rule_event_id"] == "pre_1"
    assert event.to_dict()["state_before"] == {"total_positions": 0}


class _BacktestDeniedEntryStrategy:
    name = "momentum"
    symbols = ["BTC"]
    timeframes = [TimeFrame.M15]

    def __init__(self) -> None:
        self.submitted = False

    def on_init(self, _ctx) -> None:
        return None

    def on_bar(self, bar, ctx) -> None:
        if self.submitted:
            return
        ctx.broker.submit_order(Order(
            order_id="bt_entry_1",
            symbol=bar.symbol,
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={"risk_R": 0.5},
        ))
        self.submitted = True

    def on_fill(self, _fill, _ctx) -> None:
        return None

    def on_shutdown(self, _ctx) -> None:
        return None


class _OneBarStore:
    def load_candles(self, _symbol: str, _timeframe: str):
        ts = int(datetime(2026, 5, 31, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        return pd.DataFrame([{
            "ts": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }])

    def load_funding(self, _symbol: str):
        return None


def test_backtest_result_preserves_first_class_risk_decision_and_rejected_order_events(monkeypatch) -> None:
    monkeypatch.setattr(
        backtest_runner,
        "_create_strategy",
        lambda *_args, **_kwargs: (_BacktestDeniedEntryStrategy(), [TimeFrame.M15], TimeFrame.M15),
    )
    portfolio_config = PortfolioConfig(
        strategies=(StrategyAllocation(strategy_id="momentum"),),
        max_total_positions=0,
        initial_equity=10_000.0,
    )

    result = run_portfolio_backtest(
        portfolio_config,
        {"momentum": type("MomentumCfg", (), {"symbols": ["BTC"]})()},
        BacktestConfig(
            symbols=["BTC"],
            start_date=datetime(2026, 5, 31, tzinfo=timezone.utc).date(),
            end_date=datetime(2026, 5, 31, tzinfo=timezone.utc).date(),
            apply_funding=False,
        ),
        store=_OneBarStore(),
    )

    assert len(result.rule_events) == 1
    assert len(result.risk_decision_events) == 1
    assert len(result.order_events) == 1

    portfolio_rule = result.rule_events[0].to_dict()
    risk_decision = result.risk_decision_events[0].to_dict()
    rejected_order = result.order_events[0].to_dict()
    required_join_keys = {
        "portfolio_rule_event_id",
        "risk_decision_id",
        "decision_id",
        "bar_id",
        "intent_id",
        "client_order_id",
        "strategy_id",
        "symbol",
        "lineage",
        "portfolio_config_version",
        "risk_config_version",
        "allocation_version",
    }

    assert required_join_keys.issubset(portfolio_rule)
    assert required_join_keys.issubset(risk_decision)
    assert required_join_keys.issubset(rejected_order)
    assert risk_decision["event_type"] == "risk_decision"
    assert risk_decision["action"] == "block"
    assert rejected_order["event_type"] == "order"
    assert rejected_order["rejection_stage"] == "portfolio_rule"
    assert rejected_order["portfolio_rule_event_id"] == portfolio_rule["portfolio_rule_event_id"]
    assert rejected_order["risk_decision_id"] == risk_decision["risk_decision_id"]


def test_portfolio_denial_marks_decision_context_as_order_attempt() -> None:
    broker = MagicMock()
    manager, state = _manager(max_total_positions=1)
    state.add_risk(OpenRisk("momentum", "ETH", Side.LONG, 1.0))
    emitted: list[tuple[str, dict]] = []
    proxy = BrokerProxy(
        broker,
        manager,
        "momentum",
        event_callback=lambda event_type, payload: emitted.append((event_type, payload)),
    )
    context = DecisionContext(
        decision_id="momentum|BTC|15m|2026-05-31T00:15:00+00:00",
        strategy_id="momentum",
        symbol="BTC",
        timeframe=TimeFrame.M15,
        decision_time=datetime(2026, 5, 31, 0, 15, tzinfo=timezone.utc),
        decision_key="d1",
        metadata={"bar_id": "bar-1"},
    )
    order = Order("", "BTC", Side.LONG, OrderType.MARKET, 0.1, tag="entry", metadata={"risk_R": 0.5})

    proxy.begin_decision_context(context)
    proxy.submit_order(order)
    proxy.end_decision_context(context)

    assert order.status == OrderStatus.REJECTED
    assert context.action == "order"
    assert context.order_count == 1
    assert emitted[-1][0] == "order"
    assert emitted[-1][1]["intent_id"].endswith(":intent:1")
