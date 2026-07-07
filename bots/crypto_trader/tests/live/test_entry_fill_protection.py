"""Live immediate fill sync regressions."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from crypto_trader.core.engine import MultiTimeFrameBars, StrategyContext
from crypto_trader.core.clock import SimClock
from crypto_trader.core.events import EventBus, PositionClosedEvent
from crypto_trader.core.execution_adapter import ExecutionCapabilities
from crypto_trader.core.execution_gateway import ExecutionGateway
from crypto_trader.core.models import Bar, Fill, Order, OrderStatus, OrderType, Side, TimeFrame, Trade
from crypto_trader.core.runtime_types import ExecutionReport, ExecutionReportKind
from crypto_trader.core.strategy_runtime import StrategySlotRuntime
from crypto_trader.live.engine import LiveEngine, _FillProcessingResult, _StrategySlot
from crypto_trader.live.oms_store import (
    FILL_STATUS_COORDINATOR_APPLIED,
    FILL_STATUS_LIFECYCLE_APPLIED,
    FILL_STATUS_PROCESSING_FAILED,
    FILL_STATUS_PROCESSED,
    FILL_STATUS_STRATEGY_DISPATCHED,
    OmsStore,
)
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation
from crypto_trader.portfolio.coordinator import StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import PortfolioState


def test_immediate_fill_sync_dispatches_entry_fill_once(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    engine._last_fill_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    engine._tracked_positions = {}
    engine._health = MagicMock()
    engine._detect_position_closures = MagicMock()
    engine._oms = OmsStore(tmp_path)
    engine._lifecycle = MagicMock()
    engine._lifecycle.apply_fill.return_value = None
    engine._lifecycle.snapshot.return_value = []
    engine._config = SimpleNamespace(state_dir=tmp_path)

    fill = Fill(
        order_id="entry1",
        symbol="BTC",
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
        tag="entry",
        exchange_order_id="123",
        exchange_fill_id="fill123",
    )
    broker = MagicMock()
    broker._local_to_oid = {"entry1": "123"}
    broker.get_fills_since.return_value = [fill]
    engine._broker = broker

    coordinator = MagicMock()
    coordinator.get_strategy_for_order.return_value = "momentum"
    coordinator.on_fill.return_value = "momentum"
    engine._coordinator = coordinator

    strategy = SimpleNamespace(symbols=["BTC"], on_fill=MagicMock())
    slot = _StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=MagicMock(),
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )
    engine._slots = [slot]

    engine._sync_fills_after_submit("entry1")
    engine._sync_fills_after_submit("entry1")

    strategy.on_fill.assert_called_once_with(fill, slot.ctx)
    assert engine._tracked_positions["BTC"]["strategy_id"] == "momentum"
    engine._oms.close()


class _AcceptedAdapter:
    capabilities = ExecutionCapabilities()

    def submit(self, intent):
        client_id = intent.client_order_id or intent.intent_id
        return [ExecutionReport(
            report_id=f"accepted_{client_id}",
            kind=ExecutionReportKind.ACCEPTED,
            timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
            symbol=intent.symbol,
            side=intent.side,
            client_order_id=client_id,
            order_status=OrderStatus.WORKING,
            qty=intent.qty,
            metadata=intent.metadata,
        )]

    def cancel(self, client_order_id):
        return []

    def sync_open_orders(self):
        return []

    def sync_positions(self):
        return []

    def sync_fills(self, watermark):
        return []


class _MetadataAfterSubmitStrategy:
    name = "race"
    symbols = ["BTC"]
    timeframes = [TimeFrame.M15]

    def __init__(self) -> None:
        self.meta_ready = False
        self.missed_fill = False
        self.stop_submitted = False

    def on_bar(self, bar, ctx) -> None:
        ctx.broker.submit_order(Order(
            order_id="",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={"strategy_id": "race"},
        ))
        self.meta_ready = True

    def on_fill(self, fill, ctx) -> None:
        if fill.tag != "entry":
            return
        if not self.meta_ready:
            self.missed_fill = True
            return
        ctx.broker.submit_order(Order(
            order_id="",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.STOP,
            qty=fill.qty,
            stop_price=95.0,
            tag="protective_stop",
            metadata={"strategy_id": "race"},
        ))
        self.stop_submitted = True


def test_immediate_fill_sync_drains_after_strategy_metadata_is_stored() -> None:
    strategy = _MetadataAfterSubmitStrategy()
    events = EventBus()
    bars = MultiTimeFrameBars()
    runtime = None

    def immediate_sync(order_id: str) -> None:
        assert runtime is not None
        runtime.dispatch_fill(Fill(
            order_id=order_id,
            symbol="BTC",
            side=Side.LONG,
            qty=0.1,
            fill_price=100.0,
            commission=0.01,
            timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
            tag="entry",
        ))

    gateway = ExecutionGateway(
        adapter=_AcceptedAdapter(),
        broker=SimpleNamespace(),
        events=events,
        immediate_fill_sync=immediate_sync,
    )
    ctx = StrategyContext(
        broker=gateway,
        clock=SimClock(),
        bars=bars,
        events=events,
    )
    runtime = StrategySlotRuntime(
        strategy=strategy,
        ctx=ctx,
        broker=SimpleNamespace(),
        bars=bars,
        events=events,
        primary_timeframe=TimeFrame.M15,
        strategy_id="race",
    )

    runtime.process_bar(Bar(
        timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
        symbol="BTC",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1.0,
        timeframe=TimeFrame.M15,
    ), process_broker=False)

    assert strategy.missed_fill is False
    assert strategy.stop_submitted is True


def test_fill_poll_uses_configured_overlap_and_exchange_timestamp_watermark(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    engine._last_fill_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    engine._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    engine._oms = OmsStore(tmp_path)
    fill = Fill(
        order_id="entry1",
        symbol="BTC",
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 1, tzinfo=timezone.utc),
        tag="entry",
    )
    broker = MagicMock()
    broker.get_fills_since.return_value = [fill]
    engine._broker = broker
    engine._process_fills = MagicMock(return_value=_FillProcessingResult(
        processed=[fill],
        duplicates=[],
        unresolved=[],
        safe_watermark_fills=[fill],
    ))

    engine._poll_and_process_fills()

    broker.get_fills_since.assert_called_once_with(
        datetime(2026, 5, 24, 11, 55, tzinfo=timezone.utc)
    )
    assert engine._last_fill_check == fill.timestamp
    assert engine._oms.get_watermark("fills_since") == fill.timestamp.isoformat()
    assert engine._oms.get_watermark("fills_last_poll_at") is not None
    engine._oms.close()


def test_fill_poll_does_not_watermark_unresolved_fill(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    last_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    engine._last_fill_check = last_check
    engine._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    engine._oms = OmsStore(tmp_path)
    engine._coordinator = None
    engine._slots = []
    engine._tracked_positions = {}
    engine._detect_position_closures = MagicMock()

    fill = Fill(
        order_id="unknown_exchange_order",
        symbol="BTC",
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 10, tzinfo=timezone.utc),
        tag="entry",
        exchange_fill_id="unknown_fill",
    )
    broker = MagicMock()
    broker.get_fills_since.return_value = [fill]
    broker.get_order_owner.return_value = None
    engine._broker = broker

    assert engine._poll_and_process_fills() == []
    assert engine._last_fill_check == last_check
    assert engine._oms.get_watermark("fills_since") == last_check.isoformat()
    discrepancy = engine._oms.list_unresolved_discrepancies()[0]
    assert discrepancy["kind"] == "unattributed_fill"
    assert discrepancy["metadata"]["fill_id"] == "unknown_fill"
    engine._oms.close()


def test_duplicate_fill_is_safe_for_watermark_without_redispatch(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    engine._last_fill_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    engine._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    engine._oms = OmsStore(tmp_path)
    engine._coordinator = MagicMock()
    engine._coordinator.get_strategy_for_order.return_value = "momentum"
    engine._slots = []
    engine._tracked_positions = {}
    engine._detect_position_closures = MagicMock()

    fill = Fill(
        order_id="entry1",
        symbol="BTC",
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 5, tzinfo=timezone.utc),
        tag="entry",
        exchange_fill_id="fill123",
    )
    engine._oms.record_fill(
        fill_id="fill123",
        client_order_id="entry1",
        exchange_order_id="",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.LONG.value,
        qty=0.1,
        price=100.0,
        commission=0.01,
        timestamp=fill.timestamp,
        exchange_fill_id="fill123",
    )
    broker = MagicMock()
    broker.get_fills_since.return_value = [fill]
    engine._broker = broker

    assert engine._poll_and_process_fills() == []
    assert engine._last_fill_check == fill.timestamp
    assert engine._oms.get_watermark("fills_since") == fill.timestamp.isoformat()
    engine._oms.close()


class _FailsOnceProtectiveStopStrategy:
    symbols = ["BTC"]

    def __init__(self) -> None:
        self.calls = 0

    def on_fill(self, fill, ctx) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary strategy failure")
        ctx.broker.submit_order(Order(
            order_id="stop1",
            symbol=fill.symbol,
            side=Side.SHORT,
            order_type=OrderType.STOP,
            qty=fill.qty,
            stop_price=95.0,
            tag="protective_stop",
        ))


class _ProtectiveStopStrategy:
    symbols = ["BTC"]

    def __init__(self) -> None:
        self.calls = 0

    def on_fill(self, fill, ctx) -> None:
        self.calls += 1
        ctx.broker.submit_order(Order(
            order_id=f"stop{self.calls}",
            symbol=fill.symbol,
            side=Side.SHORT,
            order_type=OrderType.STOP,
            qty=fill.qty,
            stop_price=95.0,
            tag="protective_stop",
        ))


def test_fill_retry_after_strategy_failure_is_not_consumed_or_watermarked(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    last_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    engine._last_fill_check = last_check
    engine._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    engine._tracked_positions = {}
    engine._health = MagicMock()
    engine._detect_position_closures = MagicMock()
    engine._oms = OmsStore(tmp_path)
    engine._lifecycle = MagicMock()
    engine._lifecycle.apply_fill.return_value = None
    engine._lifecycle.snapshot.return_value = []

    fill = Fill(
        order_id="entry1",
        symbol="BTC",
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 5, tzinfo=timezone.utc),
        tag="entry",
        exchange_fill_id="fill123",
    )
    broker = MagicMock()
    broker._local_to_oid = {"entry1": "123"}
    broker.get_fills_since.return_value = [fill]
    engine._broker = broker

    coordinator = MagicMock()
    coordinator.get_strategy_for_order.return_value = "momentum"
    coordinator.on_fill.return_value = "momentum"
    engine._coordinator = coordinator

    strategy_broker = MagicMock()
    strategy = _FailsOnceProtectiveStopStrategy()
    ctx = StrategyContext(
        broker=strategy_broker,
        clock=SimClock(),
        bars=MultiTimeFrameBars(),
        events=EventBus(),
    )
    slot = _StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=ctx,
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )
    engine._slots = [slot]

    assert engine._poll_and_process_fills() == []
    assert engine._last_fill_check == last_check
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_PROCESSING_FAILED
    assert engine._lifecycle.apply_fill.call_count == 0
    strategy_broker.submit_order.assert_not_called()

    processed = engine._poll_and_process_fills()

    assert processed == [fill]
    assert engine._last_fill_check == fill.timestamp
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_PROCESSED
    assert strategy.calls == 2
    strategy_broker.submit_order.assert_called_once()
    engine._lifecycle.apply_fill.assert_called_once_with("momentum", fill)
    engine._oms.close()


def test_fill_retry_after_lifecycle_failure_does_not_repeat_strategy_side_effects(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    last_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    engine._last_fill_check = last_check
    engine._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    engine._tracked_positions = {}
    engine._health = MagicMock()
    engine._detect_position_closures = MagicMock()
    engine._oms = OmsStore(tmp_path)
    engine._lifecycle = MagicMock()
    engine._lifecycle.apply_fill.side_effect = [RuntimeError("ledger down"), None]
    engine._lifecycle.snapshot.return_value = []

    fill = Fill(
        order_id="entry1",
        symbol="BTC",
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 5, tzinfo=timezone.utc),
        tag="entry",
        exchange_fill_id="fill123",
    )
    broker = MagicMock()
    broker._local_to_oid = {"entry1": "123"}
    broker.get_fills_since.return_value = [fill]
    engine._broker = broker

    coordinator = MagicMock()
    coordinator.get_strategy_for_order.return_value = "momentum"
    coordinator.on_fill.return_value = "momentum"
    engine._coordinator = coordinator

    strategy_broker = MagicMock()
    strategy = _ProtectiveStopStrategy()
    ctx = StrategyContext(
        broker=strategy_broker,
        clock=SimClock(),
        bars=MultiTimeFrameBars(),
        events=EventBus(),
    )
    engine._slots = [_StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=ctx,
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )]

    assert engine._poll_and_process_fills() == []
    assert engine._last_fill_check == last_check
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_COORDINATOR_APPLIED
    assert engine._oms.get_fill("fill123")["processing_error"] == "ledger down"
    assert strategy.calls == 1
    strategy_broker.submit_order.assert_called_once()

    processed = engine._poll_and_process_fills()

    assert processed == [fill]
    assert engine._last_fill_check == fill.timestamp
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_PROCESSED
    assert strategy.calls == 1
    strategy_broker.submit_order.assert_called_once()
    coordinator.on_fill.assert_called_once_with(fill)
    assert engine._lifecycle.apply_fill.call_count == 2
    engine._oms.close()


def test_lifecycle_failure_retry_does_not_double_count_real_portfolio_heat(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    last_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    engine._last_fill_check = last_check
    engine._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    engine._tracked_positions = {}
    engine._health = MagicMock()
    engine._detect_position_closures = MagicMock()
    engine._oms = OmsStore(tmp_path)
    engine._lifecycle = MagicMock()
    engine._lifecycle.apply_fill.side_effect = [RuntimeError("ledger down"), None]
    engine._lifecycle.snapshot.return_value = []

    fill = Fill(
        order_id="entry1",
        symbol="BTC",
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 5, tzinfo=timezone.utc),
        tag="entry",
        exchange_fill_id="fill123",
    )
    broker = MagicMock()
    broker._local_to_oid = {"entry1": "123"}
    broker.get_fills_since.return_value = [fill]
    broker.get_order_owner.return_value = None
    engine._broker = broker

    portfolio_state = PortfolioState(equity=10_000.0, peak_equity=10_000.0)
    manager = PortfolioManager(
        PortfolioConfig(strategies=(StrategyAllocation("momentum"),), symbol_collision="allow"),
        portfolio_state,
    )
    engine._coordinator = StrategyCoordinator(broker, manager)
    engine._coordinator.register_order(
        "entry1",
        "momentum",
        Order(
            order_id="entry1",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            qty=0.1,
            tag="entry",
            metadata={"risk_R": 0.75},
        ),
    )

    strategy_broker = MagicMock()
    strategy = _ProtectiveStopStrategy()
    ctx = StrategyContext(
        broker=strategy_broker,
        clock=SimClock(),
        bars=MultiTimeFrameBars(),
        events=EventBus(),
    )
    engine._slots = [_StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=ctx,
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )]

    assert engine._poll_and_process_fills() == []
    assert engine._last_fill_check == last_check
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_COORDINATOR_APPLIED
    assert strategy.calls == 1
    strategy_broker.submit_order.assert_called_once()
    assert len(portfolio_state.open_risks) == 1
    assert engine._lifecycle.apply_fill.call_count == 1

    processed = engine._poll_and_process_fills()

    assert processed == [fill]
    assert engine._last_fill_check == fill.timestamp
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_PROCESSED
    assert strategy.calls == 1
    strategy_broker.submit_order.assert_called_once()
    assert len(portfolio_state.open_risks) == 1
    assert portfolio_state.open_risks[0].risk_R == 0.75
    assert engine._lifecycle.apply_fill.call_count == 2
    engine._oms.close()


def test_finalization_retry_does_not_reaggregate_tracked_entry_fill(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    engine._last_fill_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    engine._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    engine._tracked_positions = {}
    engine._health = MagicMock()
    engine._detect_position_closures = MagicMock()
    engine._oms = OmsStore(tmp_path)
    engine._lifecycle = MagicMock()
    engine._lifecycle.apply_fill.return_value = None
    engine._lifecycle.snapshot.return_value = []

    fill = Fill(
        order_id="entry1",
        symbol="BTC",
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 5, tzinfo=timezone.utc),
        tag="entry",
        exchange_fill_id="fill123",
    )
    broker = MagicMock()
    broker._local_to_oid = {"entry1": "123"}
    broker.get_fills_since.return_value = [fill]
    engine._broker = broker

    coordinator = MagicMock()
    coordinator.get_strategy_for_order.return_value = "momentum"
    coordinator.on_fill.return_value = "momentum"
    engine._coordinator = coordinator

    strategy = _ProtectiveStopStrategy()
    ctx = StrategyContext(
        broker=MagicMock(),
        clock=SimClock(),
        bars=MultiTimeFrameBars(),
        events=EventBus(),
    )
    engine._slots = [_StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=ctx,
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )]

    original_mark_finalized = engine._mark_oms_fill_finalized
    finalize_attempts = 0

    def fail_finalized_once(fill_id: str, *, strategy_id: str) -> None:
        nonlocal finalize_attempts
        finalize_attempts += 1
        if finalize_attempts == 1:
            raise RuntimeError("finalize marker down")
        original_mark_finalized(fill_id, strategy_id=strategy_id)

    engine._mark_oms_fill_finalized = fail_finalized_once

    assert engine._poll_and_process_fills() == []
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_LIFECYCLE_APPLIED
    assert engine._tracked_positions["BTC"]["qty"] == 0.1

    processed = engine._poll_and_process_fills()

    assert processed == [fill]
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_PROCESSED
    assert engine._tracked_positions["BTC"]["qty"] == 0.1
    assert finalize_attempts == 2
    engine._lifecycle.apply_fill.assert_called_once_with("momentum", fill)
    engine._oms.close()


def test_lifecycle_persist_failure_retry_does_not_reapply_fill_same_process(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    engine._last_fill_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    engine._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    engine._tracked_positions = {}
    engine._health = MagicMock()
    engine._detect_position_closures = MagicMock()
    engine._oms = OmsStore(tmp_path)
    engine._lifecycle = MagicMock()
    engine._lifecycle.apply_fill.return_value = None

    persist_attempts = 0

    original_persist_phase = engine._oms.persist_lifecycle_phase

    def fail_persist_once(*args, **kwargs) -> None:
        nonlocal persist_attempts
        persist_attempts += 1
        if persist_attempts == 1:
            raise RuntimeError("persist down")
        original_persist_phase(*args, **kwargs)

    engine._oms.persist_lifecycle_phase = fail_persist_once

    fill = Fill(
        order_id="entry1",
        symbol="BTC",
        side=Side.LONG,
        qty=0.1,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 5, tzinfo=timezone.utc),
        tag="entry",
        exchange_fill_id="fill123",
    )
    broker = MagicMock()
    broker._local_to_oid = {"entry1": "123"}
    broker.get_fills_since.return_value = [fill]
    engine._broker = broker

    coordinator = MagicMock()
    coordinator.get_strategy_for_order.return_value = "momentum"
    coordinator.on_fill.return_value = "momentum"
    engine._coordinator = coordinator

    strategy = _ProtectiveStopStrategy()
    ctx = StrategyContext(
        broker=MagicMock(),
        clock=SimClock(),
        bars=MultiTimeFrameBars(),
        events=EventBus(),
    )
    engine._slots = [_StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=ctx,
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )]

    assert engine._poll_and_process_fills() == []
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_COORDINATOR_APPLIED
    engine._lifecycle.apply_fill.assert_called_once_with("momentum", fill)

    processed = engine._poll_and_process_fills()

    assert processed == [fill]
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_PROCESSED
    engine._lifecycle.apply_fill.assert_called_once_with("momentum", fill)
    assert persist_attempts == 2
    engine._oms.close()


def test_close_event_retry_after_position_oms_failure_does_not_reemit(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    engine._last_fill_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    engine._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    engine._tracked_positions = {"BTC": {"strategy_id": "momentum"}}
    engine._health = MagicMock()
    engine._detect_position_closures = MagicMock()
    engine._oms = OmsStore(tmp_path)
    engine._lifecycle = MagicMock()

    fill = Fill(
        order_id="exit1",
        symbol="BTC",
        side=Side.SHORT,
        qty=0.1,
        fill_price=105.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 5, tzinfo=timezone.utc),
        tag="protective_stop",
        exchange_fill_id="fill123",
    )
    trade = Trade(
        trade_id="trade1",
        symbol="BTC",
        direction=Side.LONG,
        entry_price=100.0,
        exit_price=105.0,
        qty=0.1,
        entry_time=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
        exit_time=fill.timestamp,
        pnl=0.5,
        r_multiple=0.5,
        commission=0.02,
        bars_held=0,
        setup_grade=None,
        exit_reason="protective_stop",
        confluences_used=None,
        confirmation_type=None,
        entry_method=None,
        funding_paid=0.0,
        mae_r=None,
        mfe_r=None,
    )
    engine._lifecycle.apply_fill.return_value = trade
    engine._lifecycle.snapshot.return_value = []

    original_upsert = engine._oms.upsert_position
    upsert_attempts = 0

    def fail_position_upsert_once(**kwargs) -> None:
        nonlocal upsert_attempts
        upsert_attempts += 1
        if upsert_attempts == 1:
            raise RuntimeError("position write down")
        original_upsert(**kwargs)

    engine._oms.upsert_position = fail_position_upsert_once

    broker = MagicMock()
    broker._local_to_oid = {"exit1": "456"}
    broker.get_fills_since.return_value = [fill]
    engine._broker = broker

    coordinator = MagicMock()
    coordinator.get_strategy_for_order.return_value = "momentum"
    coordinator.on_fill.return_value = "momentum"
    engine._coordinator = coordinator

    events = EventBus()
    closed_events = []
    events.subscribe(PositionClosedEvent, closed_events.append)
    strategy = _ProtectiveStopStrategy()
    ctx = StrategyContext(
        broker=MagicMock(),
        clock=SimClock(),
        bars=MultiTimeFrameBars(),
        events=events,
    )
    engine._slots = [_StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=ctx,
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )]

    assert engine._poll_and_process_fills() == []
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_LIFECYCLE_APPLIED
    assert len(closed_events) == 1
    coordinator.on_trade_closed.assert_called_once_with("momentum", "BTC", 0.5)

    processed = engine._poll_and_process_fills()

    assert processed == [fill]
    assert engine._oms.get_fill_status("fill123") == FILL_STATUS_PROCESSED
    assert len(closed_events) == 1
    coordinator.on_trade_closed.assert_called_once_with("momentum", "BTC", 0.5)
    assert upsert_attempts == 2
    assert engine._oms.list_open_positions() == []
    engine._oms.close()


def test_restart_at_lifecycle_applied_uses_durable_closed_trade_for_finalization(tmp_path) -> None:
    fill = Fill(
        order_id="exit1",
        symbol="BTC",
        side=Side.SHORT,
        qty=0.1,
        fill_price=105.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 5, tzinfo=timezone.utc),
        tag="protective_stop",
        exchange_fill_id="fill123",
    )
    trade = Trade(
        trade_id="trade1",
        symbol="BTC",
        direction=Side.LONG,
        entry_price=100.0,
        exit_price=105.0,
        qty=0.1,
        entry_time=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
        exit_time=fill.timestamp,
        pnl=0.5,
        r_multiple=0.5,
        commission=0.02,
        bars_held=0,
        setup_grade=None,
        exit_reason="protective_stop",
        confluences_used=None,
        confirmation_type=None,
        entry_method=None,
        funding_paid=0.0,
        mae_r=None,
        mfe_r=None,
    )

    first = object.__new__(LiveEngine)
    first._last_fill_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    first._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    first._tracked_positions = {"BTC": {"strategy_id": "momentum"}}
    first._health = MagicMock()
    first._detect_position_closures = MagicMock()
    first._oms = OmsStore(tmp_path)
    first._lifecycle = MagicMock()
    first._lifecycle.apply_fill.return_value = trade
    first._lifecycle.snapshot.return_value = []
    broker = MagicMock()
    broker._local_to_oid = {"exit1": "456"}
    broker.get_fills_since.return_value = [fill]
    first._broker = broker
    first._coordinator = MagicMock()
    first._coordinator.get_strategy_for_order.return_value = "momentum"
    first._coordinator.on_fill.return_value = "momentum"
    first._slots = [_StrategySlot(
        strategy_id="momentum",
        strategy=SimpleNamespace(symbols=["BTC"], on_fill=MagicMock()),
        ctx=StrategyContext(
            broker=MagicMock(),
            clock=SimClock(),
            bars=MultiTimeFrameBars(),
            events=EventBus(),
        ),
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )]
    first._emit_lifecycle_trade_once = MagicMock(side_effect=RuntimeError("finalization down"))

    assert first._poll_and_process_fills() == []
    assert first._oms.get_fill_status("fill123") == FILL_STATUS_LIFECYCLE_APPLIED
    assert first._oms.list_events("fill_lifecycle_closed_trade")[0]["payload"]["fill_id"] == "fill123"
    first._oms.close()

    second = object.__new__(LiveEngine)
    second._last_fill_check = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    second._config = SimpleNamespace(state_dir=tmp_path, fill_query_overlap_sec=300.0)
    second._tracked_positions = {"BTC": {"strategy_id": "momentum"}}
    second._health = MagicMock()
    second._detect_position_closures = MagicMock()
    second._oms = OmsStore(tmp_path)
    second._lifecycle = MagicMock()
    second._lifecycle.snapshot.return_value = []
    broker.get_fills_since.return_value = [fill]
    second._broker = broker
    second._coordinator = MagicMock()
    second._coordinator.get_strategy_for_order.return_value = "momentum"
    second._coordinator.on_fill.return_value = "momentum"
    events = EventBus()
    closed_events = []
    events.subscribe(PositionClosedEvent, closed_events.append)
    second._slots = [_StrategySlot(
        strategy_id="momentum",
        strategy=SimpleNamespace(symbols=["BTC"], on_fill=MagicMock()),
        ctx=StrategyContext(
            broker=MagicMock(),
            clock=SimClock(),
            bars=MultiTimeFrameBars(),
            events=events,
        ),
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )]

    assert second._poll_and_process_fills() == [fill]
    assert second._oms.get_fill_status("fill123") == FILL_STATUS_PROCESSED
    second._lifecycle.apply_fill.assert_not_called()
    second._coordinator.on_fill.assert_not_called()
    second._coordinator.on_trade_closed.assert_called_once_with("momentum", "BTC", 0.5)
    assert len(closed_events) == 1
    second._oms.close()
