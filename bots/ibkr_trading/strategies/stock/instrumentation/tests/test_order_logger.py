"""Tests for OrderEvent and OrderLogger."""
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from strategies.stock.instrumentation.src.order_logger import OrderEvent, OrderLogger
from strategies.stock.instrumentation.src.facade import InstrumentationKit
from strategies.stock.instrumentation.src.sidecar import _DIR_TO_EVENT_TYPE, _EVENT_PRIORITY


class TestOrderEvent:
    def test_all_fields_populated(self):
        event = OrderEvent(
            order_id="ORD_001",
            bot_id="test_bot",
            pair="NQ",
            side="LONG",
            order_type="LIMIT",
            status="FILLED",
            requested_qty=2.0,
            filled_qty=2.0,
            requested_price=21450.00,
            fill_price=21450.50,
            slippage_bps=0.23,
            timestamp="2026-03-06T14:30:00+00:00",
            latency_ms=12.5,
            related_trade_id="t_001",
            experiment_id="exp_001",
            experiment_variant="control",
            strategy_type="helix",
            session="RTH",
            contract_month="2026-06",
            order_book_depth={"bid_levels": [[21449.75, 15]], "ask_levels": [[21450.00, 12]]},
        )
        d = event.to_dict()
        assert d["order_id"] == "ORD_001"
        assert d["strategy_type"] == "helix"
        assert d["session"] == "RTH"
        assert d["contract_month"] == "2026-06"
        assert d["order_book_depth"]["bid_levels"] == [[21449.75, 15]]

    def test_to_dict_excludes_none(self):
        event = OrderEvent(
            order_id="ORD_002",
            bot_id="test_bot",
            pair="NQ",
        )
        d = event.to_dict()
        assert "requested_price" not in d
        assert "fill_price" not in d
        assert "slippage_bps" not in d
        assert "order_book_depth" not in d


class TestOrderLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "data_source_id": "ibkr_cme_nq",
            "experiment_id": "exp_001",
            "experiment_variant": "control",
        }

    def test_slippage_for_limit_order(self):
        """Requested_price vs fill_price → correct bps."""
        logger = OrderLogger(self.config, strategy_type="helix")
        event = logger.log_order(
            order_id="ORD_001",
            pair="NQ",
            side="LONG",
            order_type="LIMIT",
            status="FILLED",
            requested_qty=2,
            filled_qty=2,
            requested_price=21450.00,
            fill_price=21450.50,
        )
        # slippage = |21450.50 - 21450.00| / 21450.00 * 10000
        expected = round(abs(21450.50 - 21450.00) / 21450.00 * 10_000, 2)
        assert event.slippage_bps == expected

    def test_slippage_for_market_order(self):
        """Requested_price=None → slippage_bps=None."""
        logger = OrderLogger(self.config, strategy_type="helix")
        event = logger.log_order(
            order_id="ORD_002",
            pair="NQ",
            side="LONG",
            order_type="MARKET",
            status="FILLED",
            requested_qty=2,
            filled_qty=2,
            fill_price=21450.00,
        )
        assert event.slippage_bps is None

    def test_partial_fill(self):
        """PARTIAL_FILL: filled_qty < requested_qty."""
        logger = OrderLogger(self.config, strategy_type="helix")
        event = logger.log_order(
            order_id="ORD_003",
            pair="NQ",
            side="LONG",
            order_type="LIMIT",
            status="PARTIAL_FILL",
            requested_qty=5,
            filled_qty=2,
            requested_price=21450.00,
            fill_price=21450.00,
        )
        assert event.status == "PARTIAL_FILL"
        assert event.filled_qty == 2
        assert event.requested_qty == 5

    def test_rejected_order(self):
        """REJECTED: reject_reason populated, filled_qty=0."""
        logger = OrderLogger(self.config, strategy_type="helix")
        event = logger.log_order(
            order_id="ORD_004",
            pair="NQ",
            side="LONG",
            order_type="LIMIT",
            status="REJECTED",
            requested_qty=2,
            filled_qty=0,
            reject_reason="Insufficient margin",
        )
        assert event.status == "REJECTED"
        assert event.reject_reason == "Insufficient margin"
        assert event.filled_qty == 0

    def test_writes_to_jsonl(self):
        """Verify JSONL file is created with correct content."""
        logger = OrderLogger(self.config, strategy_type="helix")
        ts = datetime(2026, 3, 6, 14, 30, 0, tzinfo=timezone.utc)
        logger.log_order(
            order_id="ORD_005",
            pair="NQ",
            side="LONG",
            order_type="MARKET",
            status="FILLED",
            requested_qty=2,
            filled_qty=2,
            fill_price=21450.00,
            exchange_timestamp=ts,
        )
        filepath = Path(self.tmpdir) / "orders" / "orders_2026-03-06.jsonl"
        assert filepath.exists()
        data = json.loads(filepath.read_text().strip())
        assert data["order_id"] == "ORD_005"
        assert data["status"] == "FILLED"

    def test_experiment_tracking_propagated(self):
        """Experiment id/variant from config propagated to events."""
        logger = OrderLogger(self.config, strategy_type="helix")
        event = logger.log_order(
            order_id="ORD_006",
            pair="NQ",
            side="LONG",
            order_type="MARKET",
            status="SUBMITTED",
            requested_qty=2,
        )
        assert event.experiment_id == "exp_001"
        assert event.experiment_variant == "control"

    def test_event_metadata_attached(self):
        """Event metadata is attached to the order event."""
        logger = OrderLogger(self.config, strategy_type="helix")
        event = logger.log_order(
            order_id="ORD_007",
            pair="NQ",
            side="LONG",
            order_type="MARKET",
            status="SUBMITTED",
            requested_qty=2,
        )
        assert "event_id" in event.event_metadata
        assert event.event_metadata["bot_id"] == "test_bot"


class TestSidecarOrderForwarding:
    def test_orders_directory_in_type_mapping(self):
        """Verify orders/ directory included in sidecar type mapping."""
        assert "orders" in _DIR_TO_EVENT_TYPE
        assert _DIR_TO_EVENT_TYPE["orders"] == "order"

    def test_order_priority_set(self):
        """Verify order events have priority set."""
        assert "order" in _EVENT_PRIORITY
        assert _EVENT_PRIORITY["order"] == 3


class TestFacadeOrderEvent:
    def test_on_order_event_delegates(self):
        mgr = MagicMock()
        mgr.order_logger = MagicMock()
        kit = InstrumentationKit(mgr, strategy_type="helix")
        kit.on_order_event(
            order_id="ORD_001",
            pair="NQ",
            side="LONG",
            order_type="LIMIT",
            status="FILLED",
            requested_qty=2,
            filled_qty=2,
            fill_price=21450.00,
            strategy_type="helix",
            session="RTH",
            contract_month="2026-06",
        )
        assert mgr.order_logger.log_order.called
        kwargs = mgr.order_logger.log_order.call_args[1]
        assert kwargs["order_id"] == "ORD_001"
        assert kwargs["strategy_type"] == "helix"

    def test_on_order_event_uses_kit_strategy_type(self):
        mgr = MagicMock()
        mgr.order_logger = MagicMock()
        kit = InstrumentationKit(mgr, strategy_type="nqdtc")
        kit.on_order_event(
            order_id="ORD_002",
            pair="NQ",
            side="SHORT",
            order_type="MARKET",
            status="SUBMITTED",
            requested_qty=1,
        )
        kwargs = mgr.order_logger.log_order.call_args[1]
        assert kwargs["strategy_type"] == "nqdtc"

    def test_on_order_event_noop_without_manager(self):
        kit = InstrumentationKit(None, strategy_type="helix")
        # Should not raise
        kit.on_order_event(
            order_id="ORD_003",
            pair="NQ",
            side="LONG",
            order_type="MARKET",
            status="FILLED",
            requested_qty=2,
        )
