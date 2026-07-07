"""Tests for portfolio coordinator (BrokerProxy + StrategyCoordinator)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.broker.sim_execution_adapter import SimExecutionAdapter
from crypto_trader.core.events import CanonicalRuntimeEvent, EventBus
from crypto_trader.core.execution_gateway import ExecutionGateway
from crypto_trader.core.models import Fill, Order, OrderStatus, OrderType, Position, SetupGrade, Side, Trade
from crypto_trader.live.lifecycle import PositionLifecycleLedger
from crypto_trader.live.oms_store import OmsStore
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation
from crypto_trader.portfolio.coordinator import BrokerProxy, StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager, PortfolioRuleResult
from crypto_trader.portfolio.state import OpenRisk, PortfolioState


def _make_components(max_total_positions=9):
    cfg = PortfolioConfig(
        strategies=(
            StrategyAllocation(strategy_id="momentum"),
            StrategyAllocation(strategy_id="trend"),
        ),
        max_total_positions=max_total_positions,
    )
    state = PortfolioState(equity=10000.0, peak_equity=10000.0)
    manager = PortfolioManager(cfg, state)
    broker = MagicMock()
    broker._orders = {}
    broker._closed_trades = []
    return broker, manager, state


class TestBrokerProxy:
    def test_entry_order_approved(self):
        broker, manager, state = _make_components()
        broker.submit_order.return_value = "order_1"
        proxy = BrokerProxy(broker, manager, "momentum")

        order = Order(
            order_id="o1", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1,
            tag="entry", metadata={"risk_R": 1.0},
        )
        result = proxy.submit_order(order)
        assert result == "o1"
        broker.submit_order.assert_called_once()
        # Strategy ID should be stamped in metadata
        assert order.metadata["strategy_id"] == "momentum"

    def test_entry_order_denied(self):
        broker, manager, state = _make_components(max_total_positions=3)
        proxy = BrokerProxy(broker, manager, "momentum")

        # Fill up positions to trigger denial (max_total_positions=3)
        from crypto_trader.portfolio.state import OpenRisk
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        state.add_risk(OpenRisk("trend", "ETH", Side.SHORT, 1.0))
        state.add_risk(OpenRisk("trend", "SOL", Side.LONG, 1.0))

        order = Order(
            order_id="o2", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1,
            tag="entry", metadata={"risk_R": 0.5},
        )
        result = proxy.submit_order(order)
        assert result == "o2"
        assert order.status == OrderStatus.REJECTED
        broker.submit_order.assert_not_called()

    def test_exit_order_passthrough(self):
        broker, manager, _ = _make_components()
        broker.submit_order.return_value = "order_3"
        proxy = BrokerProxy(broker, manager, "momentum")

        order = Order(
            order_id="o3", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.STOP, qty=0.1,
            stop_price=50000.0, tag="stop",
        )
        result = proxy.submit_order(order)
        assert result == "o3"
        broker.submit_order.assert_called_once()
        assert order.metadata["strategy_id"] == "momentum"
        assert order.metadata["reduce_only"] is True
        assert order.metadata["exit_only"] is True

    def test_exit_siblings_share_stable_oca_group_from_open_risk(self):
        broker, manager, state = _make_components()
        broker.submit_order.side_effect = lambda order: order.order_id
        state.add_risk(OpenRisk(
            "momentum",
            "BTC",
            Side.LONG,
            0.5,
            risk_id="entry_root_1",
            position_instance_id="pos_1",
            filled_qty=0.1,
        ))
        proxy = BrokerProxy(broker, manager, "momentum")
        stop = Order(
            order_id="stop_1", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.STOP, qty=0.1,
            stop_price=49_000.0, tag="protective_stop",
        )
        tp = Order(
            order_id="tp_1", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.LIMIT, qty=0.1,
            limit_price=52_000.0, tag="tp1",
        )

        proxy.submit_order(stop)
        proxy.submit_order(tp)

        assert stop.oca_group == "momentum:BTC:pos_1:exit_oca"
        assert tp.oca_group == stop.oca_group
        assert stop.metadata["oca_group"] == stop.oca_group
        assert tp.metadata["oca_policy"] == "broker_managed_cancel_siblings_on_terminal_close"
        assert stop.metadata["oca_role"] == "stop_loss"
        assert tp.metadata["oca_role"] == "take_profit"

    def test_cross_strategy_explicit_oca_group_is_rejected(self):
        broker, manager, _ = _make_components()
        proxy = BrokerProxy(broker, manager, "momentum")
        order = Order(
            order_id="bad_oca", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.STOP, qty=0.1,
            stop_price=49_000.0, tag="protective_stop",
            oca_group="trend:BTC:pos_1:exit_oca",
        )

        assert proxy.submit_order(order) == "bad_oca"

        assert order.status == OrderStatus.REJECTED
        assert order.metadata["oca_group_invalid_reason"] == "oca_group_not_strategy_symbol_scoped"
        broker.submit_order.assert_not_called()

    def test_fallback_exit_root_does_not_forge_position_instance_id(self):
        broker, manager, _ = _make_components()
        broker.submit_order.side_effect = lambda order: order.order_id
        proxy = BrokerProxy(broker, manager, "momentum")
        order = Order(
            order_id="stop_1", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.STOP, qty=0.1,
            stop_price=49_000.0, tag="protective_stop",
            metadata={"entry_intent_id": "entry_intent_1"},
        )

        proxy.submit_order(order)

        assert order.oca_group == "momentum:BTC:entry_intent_1:exit_oca"
        assert order.metadata["oca_root"] == "entry_intent_1"
        assert "position_instance_id" not in order.metadata

    def test_portfolio_block_records_canonical_rejected_order(self, tmp_path):
        cfg = PortfolioConfig(
            strategies=(StrategyAllocation(strategy_id="momentum"),),
            max_total_positions=0,
        )
        state = PortfolioState(equity=10000.0, peak_equity=10000.0)
        manager = PortfolioManager(cfg, state)
        sim_broker = SimBroker(initial_equity=10_000.0)
        events = EventBus()
        canonical = []
        events.subscribe(CanonicalRuntimeEvent, canonical.append)
        oms = OmsStore(tmp_path)
        gateway = ExecutionGateway(
            adapter=SimExecutionAdapter(sim_broker),
            broker=sim_broker,
            events=events,
            oms_store=oms,
        )
        proxy = BrokerProxy(gateway, manager, "momentum")

        order = Order(
            order_id="blocked_1",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={"risk_R": 1.0},
        )

        assert proxy.submit_order(order) == "blocked_1"
        row = oms.get_order("blocked_1")
        oms.close()

        assert order.status == OrderStatus.REJECTED
        assert [event.stream for event in canonical] == ["order_intent", "execution"]
        assert canonical[1].payload["order_status"] == OrderStatus.REJECTED.value
        assert canonical[1].payload["reject_reason"] == "max_total_positions reached"
        assert row is not None
        assert row["status"] == OrderStatus.REJECTED.value
        assert row["metadata"]["rejection_stage"] == "portfolio_rule"

    def test_size_multiplier_applied(self):
        broker, manager, state = _make_components()
        broker.submit_order.return_value = "order_4"
        proxy = BrokerProxy(broker, manager, "momentum")

        # Trigger drawdown tier
        state.peak_equity = 10000.0
        state.equity = 8700.0  # 13% DD → 0.50 multiplier

        order = Order(
            order_id="o4", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1,
            tag="entry", metadata={"risk_R": 0.5},
        )
        proxy.submit_order(order)
        assert order.qty == pytest.approx(0.05)
        assert order.metadata["risk_R"] == pytest.approx(0.25)

    def test_scaled_entry_fill_registers_full_scaled_risk(self):
        broker, manager, state = _make_components()
        state.peak_equity = 10000.0
        state.equity = 8700.0
        broker.submit_order.return_value = "entry_scaled"
        coord = StrategyCoordinator(broker, manager)
        proxy = coord.get_proxy("momentum")

        order = Order(
            order_id="entry_scaled",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=1.0,
            tag="entry",
            metadata={"risk_R": 1.0},
        )
        proxy.submit_order(order)
        coord.on_fill(Fill(
            order_id="entry_scaled",
            symbol="BTC",
            side=Side.LONG,
            qty=0.5,
            fill_price=50000.0,
            commission=1.0,
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            tag="entry",
            exchange_fill_id="fill_scaled",
        ))

        assert order.qty == pytest.approx(0.5)
        assert order.metadata["order_qty"] == pytest.approx(0.5)
        assert state.total_heat_R() == pytest.approx(0.5)
        assert state.open_risks[0].filled_qty == pytest.approx(0.5)

    def test_cancel_delegates(self):
        broker, manager, _ = _make_components()
        broker.cancel_order.return_value = True
        proxy = BrokerProxy(broker, manager, "momentum")
        assert proxy.cancel_order("x") is True
        broker.cancel_order.assert_called_once_with("x")

    def test_get_position_delegates(self):
        broker, manager, _ = _make_components()
        pos = Position(symbol="BTC", direction=Side.LONG, qty=0.1, avg_entry=50000.0)
        broker.get_position.return_value = pos
        proxy = BrokerProxy(broker, manager, "momentum")
        assert proxy.get_position("BTC") == pos

    def test_get_equity_delegates(self):
        broker, manager, _ = _make_components()
        broker.get_equity.return_value = 10500.0
        proxy = BrokerProxy(broker, manager, "momentum")
        assert proxy.get_equity() == 10500.0

    def test_getattr_fallback(self):
        broker, manager, _ = _make_components()
        broker.some_custom_method = MagicMock(return_value=42)
        proxy = BrokerProxy(broker, manager, "momentum")
        assert proxy.some_custom_method() == 42

    def test_get_open_orders_filters_to_strategy_owner(self):
        broker, manager, _ = _make_components()
        proxy = BrokerProxy(broker, manager, "momentum")
        own = Order(
            order_id="own", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP, qty=0.1,
            metadata={"strategy_id": "momentum"},
        )
        other = Order(
            order_id="other", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP, qty=0.1,
            metadata={"strategy_id": "trend"},
        )
        unknown = Order(
            order_id="unknown", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP, qty=0.1,
        )
        broker.get_open_orders.return_value = [own, other, unknown]

        assert proxy.get_open_orders("BTC") == [own]

    def test_client_order_ids_cancel_broker_assigned_ids(self):
        broker, manager, _ = _make_components()
        proxy = BrokerProxy(broker, manager, "trend")

        def assign_order_id(order):
            order.order_id = "broker_1"
            return "broker_1"

        broker.submit_order.side_effect = assign_order_id
        broker.cancel_order.return_value = True

        order = Order(
            order_id="trend_stop_BTC_abc",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.STOP,
            qty=0.1,
            stop_price=49000.0,
            tag="protective_stop",
        )

        assert proxy.submit_order(order) == "trend_stop_BTC_abc"
        assert proxy.cancel_order("trend_stop_BTC_abc") is True
        broker.cancel_order.assert_called_once_with("broker_1")

    def test_open_orders_expose_client_order_ids_when_broker_assigns_ids(self):
        broker, manager, _ = _make_components()
        proxy = BrokerProxy(broker, manager, "trend")

        def assign_order_id(order):
            order.order_id = "broker_1"
            return "broker_1"

        broker.submit_order.side_effect = assign_order_id
        order = Order(
            order_id="trend_stop_BTC_abc",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.STOP,
            qty=0.1,
            stop_price=49000.0,
            tag="protective_stop",
        )
        proxy.submit_order(order)
        broker.get_open_orders.return_value = [order]

        visible = proxy.get_open_orders("BTC")

        assert len(visible) == 1
        assert visible[0].order_id == "trend_stop_BTC_abc"
        assert visible[0].metadata["broker_order_id"] == "broker_1"
        assert order.order_id == "broker_1"

    def test_submit_registers_exchange_oid_when_broker_exposes_mapping(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)
        proxy = coord.get_proxy("momentum")

        def submit(order):
            order.order_id = "local_1"
            broker._local_to_oid = {"local_1": "999"}
            return "local_1"

        broker.submit_order.side_effect = submit

        order = Order(
            order_id="",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={"risk_R": 0.4},
        )

        assert proxy.submit_order(order) == "local_1"
        fill = Fill(
            order_id="999",
            symbol="BTC",
            side=Side.LONG,
            qty=0.1,
            fill_price=50000.0,
            commission=1.75,
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            tag="entry",
        )

        assert coord.on_fill(fill) == "momentum"
        assert state.total_heat_R() == pytest.approx(0.4)

    def test_submit_registers_gateway_exchange_id_for_fill_routing(self):
        _, manager, state = _make_components()
        sim_broker = SimBroker(initial_equity=10_000.0)
        gateway = ExecutionGateway(
            adapter=SimExecutionAdapter(sim_broker),
            broker=sim_broker,
        )
        coord = StrategyCoordinator(gateway, manager)
        proxy = coord.get_proxy("momentum")

        assert proxy.submit_order(Order(
            order_id="strategy_entry_1",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={"risk_R": 0.4},
        )) == "strategy_entry_1"

        fill = Fill(
            order_id="1",
            symbol="BTC",
            side=Side.LONG,
            qty=0.1,
            fill_price=50000.0,
            commission=1.75,
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            tag="entry",
        )

        assert coord.on_fill(fill) == "momentum"
        assert state.total_heat_R() == pytest.approx(0.4)

    def test_cancel_all_cancels_only_strategy_owned_orders(self):
        broker, manager, _ = _make_components()
        proxy = BrokerProxy(broker, manager, "momentum")
        own = Order(
            order_id="own", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP, qty=0.1,
            metadata={"strategy_id": "momentum"},
        )
        other = Order(
            order_id="other", symbol="BTC", side=Side.LONG,
            order_type=OrderType.STOP, qty=0.1,
            metadata={"strategy_id": "trend"},
        )
        broker.get_open_orders.return_value = [own, other]
        broker.cancel_order.return_value = True

        assert proxy.cancel_all("BTC") == 1
        broker.cancel_order.assert_called_once_with("own")

    def test_manager_equity_source_for_portfolio_backtests(self):
        broker, manager, state = _make_components()
        state.equity = 12345.0
        broker.get_equity.return_value = 999.0
        proxy = BrokerProxy(broker, manager, "momentum", use_manager_equity=True)

        assert proxy.get_equity() == pytest.approx(12345.0)


class TestStrategyCoordinator:
    def test_get_proxy_creates_once(self):
        broker, manager, _ = _make_components()
        coord = StrategyCoordinator(broker, manager)
        p1 = coord.get_proxy("momentum")
        p2 = coord.get_proxy("momentum")
        assert p1 is p2
        assert isinstance(p1, BrokerProxy)

    def test_get_proxy_different_strategies(self):
        broker, manager, _ = _make_components()
        coord = StrategyCoordinator(broker, manager)
        p1 = coord.get_proxy("momentum")
        p2 = coord.get_proxy("trend")
        assert p1 is not p2
        assert p1.strategy_id == "momentum"
        assert p2.strategy_id == "trend"

    def test_on_fill_entry(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)
        coord.get_proxy("momentum")  # register strategy

        # Set up order with strategy metadata
        order = Order(
            order_id="o1", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1, tag="entry",
            metadata={"strategy_id": "momentum", "risk_R": 1.0},
        )
        broker._orders = {"o1": order}
        broker.get_position.return_value = Position("BTC", Side.LONG, 0.1, 50000.0)

        fill = Fill(
            order_id="o1", symbol="BTC", side=Side.LONG,
            qty=0.1, fill_price=50000.0, commission=1.75,
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc), tag="entry",
        )

        strategy_id = coord.on_fill(fill)
        assert strategy_id == "momentum"

    def test_on_fill_entry_uses_registered_order_risk_metadata(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)
        order = Order(
            order_id="",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={"strategy_id": "momentum", "risk_R": 0.35},
        )
        coord.register_order("o_risk", "momentum", order)
        fill = Fill(
            order_id="o_risk", symbol="BTC", side=Side.LONG,
            qty=0.1, fill_price=50000.0, commission=1.75,
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc), tag="entry",
        )

        strategy_id = coord.on_fill(fill)

        assert strategy_id == "momentum"
        assert state.total_heat_R() == pytest.approx(0.35)

    def test_partial_entry_fills_update_one_prorated_open_risk(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)
        order = Order(
            order_id="entry_1",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=1.0,
            tag="entry",
            metadata={"strategy_id": "momentum", "risk_R": 1.0, "intent_id": "intent_1"},
        )
        coord.register_order("entry_1", "momentum", order)

        first = Fill(
            order_id="entry_1", symbol="BTC", side=Side.LONG,
            qty=0.4, fill_price=50000.0, commission=1.0,
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            tag="entry", exchange_fill_id="fill_1",
        )
        second = Fill(
            order_id="entry_1", symbol="BTC", side=Side.LONG,
            qty=0.6, fill_price=50100.0, commission=1.0,
            timestamp=datetime(2026, 4, 20, 0, 1, tzinfo=timezone.utc),
            tag="entry", exchange_fill_id="fill_2",
        )

        assert coord.on_fill(first) == "momentum"
        assert coord.on_fill(second) == "momentum"

        assert len(state.open_risks) == 1
        assert state.open_risks[0].risk_id == "intent_1"
        assert state.open_risks[0].filled_qty == pytest.approx(1.0)
        assert state.total_heat_R() == pytest.approx(1.0)

    def test_duplicate_entry_fill_does_not_double_count_portfolio_heat(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)
        order = Order(
            order_id="entry_1",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=1.0,
            tag="entry",
            metadata={"strategy_id": "momentum", "risk_R": 1.0, "intent_id": "intent_1"},
        )
        coord.register_order("entry_1", "momentum", order)
        fill = Fill(
            order_id="entry_1", symbol="BTC", side=Side.LONG,
            qty=0.4, fill_price=50000.0, commission=1.0,
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            tag="entry", exchange_fill_id="fill_1",
        )

        coord.on_fill(fill)
        coord.on_fill(fill)

        assert len(state.open_risks) == 1
        assert state.total_heat_R() == pytest.approx(0.4)

    def test_entry_fill_uses_same_position_instance_id_as_lifecycle_ledger(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)
        ledger = PositionLifecycleLedger()
        ts = datetime(2026, 4, 20, tzinfo=timezone.utc)
        order = Order(
            order_id="entry_1",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.4,
            tag="entry",
            metadata={"strategy_id": "momentum", "risk_R": 1.0, "intent_id": "intent_1"},
        )
        coord.register_order("entry_1", "momentum", order)
        fill = Fill(
            order_id="entry_1", symbol="BTC", side=Side.LONG,
            qty=0.4, fill_price=50000.0, commission=1.0,
            timestamp=ts, tag="entry", exchange_fill_id="fill_1",
        )

        ledger.apply_fill("momentum", fill)
        coord.on_fill(fill)

        assert len(state.open_risks) == 1
        assert len(ledger.snapshot()) == 1
        assert state.open_risks[0].position_instance_id == ledger.snapshot()[0].position_instance_id

    def test_trade_close_releases_matching_strategy_symbol_risks(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)
        manager.register_entry(
            "momentum", "BTC", Side.LONG, 0.4,
            risk_id="intent_1", order_id="entry_1", fill_qty=0.4,
        )
        manager.register_entry(
            "trend", "BTC", Side.LONG, 0.5,
            risk_id="trend_intent_1", order_id="trend_entry_1", fill_qty=0.5,
        )

        coord.on_trade_closed("momentum", "BTC", 0.25)

        assert state.total_heat_R() == pytest.approx(0.5)
        assert state.open_risks[0].strategy_id == "trend"

    def test_trade_close_with_trade_refs_releases_only_matching_stacked_risk(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)
        manager.register_entry(
            "momentum", "BTC", Side.LONG, 0.4,
            risk_id="intent_1", order_id="entry_1", fill_qty=0.4,
        )
        manager.register_entry(
            "momentum", "BTC", Side.LONG, 0.5,
            risk_id="intent_2", order_id="entry_2", fill_qty=0.5,
        )
        trade = Trade(
            trade_id="trade_1",
            symbol="BTC",
            direction=Side.LONG,
            entry_price=50_000.0,
            exit_price=51_000.0,
            qty=0.4,
            entry_time=datetime(2026, 4, 20, tzinfo=timezone.utc),
            exit_time=datetime(2026, 4, 20, 1, tzinfo=timezone.utc),
            pnl=400.0,
            r_multiple=0.25,
            commission=2.0,
            bars_held=4,
            setup_grade=SetupGrade.B,
            exit_reason="tp1",
            confluences_used=None,
            confirmation_type=None,
            entry_method=None,
            funding_paid=0.0,
            mae_r=None,
            mfe_r=None,
        )
        trade.instrumentation_context = {"entry_order_ids": ["entry_1"]}

        coord.on_trade_closed("momentum", "BTC", 0.25, trade=trade)

        assert state.total_heat_R() == pytest.approx(0.5)
        assert [risk.risk_id for risk in state.open_risks] == ["intent_2"]

    def test_on_trade_closed(self):
        broker, manager, state = _make_components()
        coord = StrategyCoordinator(broker, manager)

        # Register an entry first
        manager.register_entry("momentum", "BTC", Side.LONG, 1.0)
        assert state.total_heat_R() == 1.0

        coord.on_trade_closed("momentum", "BTC", 2.0)
        assert state.total_heat_R() == 0.0
        assert state.strategy_daily_pnl_R("momentum") == 2.0
