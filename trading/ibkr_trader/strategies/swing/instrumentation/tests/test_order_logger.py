"""Tests for OrderLogger and OrderEvent."""
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from strategies.swing.instrumentation.src.order_logger import OrderLogger, OrderEvent


class TestOrderLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "experiment_id": "exp1",
            "experiment_variant": "A",
        }
        self.logger = OrderLogger(self.config)
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def test_log_order_regular(self):
        """Test #6: Regular OrderEvent — all base + swing_trader fields populated."""
        event = self.logger.log_order(
            order_id="IBKR_001",
            pair="QQQ",
            side="LONG",
            order_type="MARKET",
            status="FILLED",
            requested_qty=50,
            filled_qty=50,
            fill_price=485.32,
            latency_ms=18.0,
            related_trade_id="t_qqq_001",
            strategy_id="ATRSS",
            overlay_state={"qqq_ema_bullish": True},
            drawdown_tier="NORMAL",
            market_session="RTH",
        )
        assert event.order_id == "IBKR_001"
        assert event.bot_id == "test_bot"
        assert event.pair == "QQQ"
        assert event.side == "LONG"
        assert event.order_type == "MARKET"
        assert event.status == "FILLED"
        assert event.requested_qty == 50
        assert event.filled_qty == 50
        assert event.fill_price == 485.32
        assert event.strategy_id == "ATRSS"
        assert event.order_action == "NEW"
        assert event.coordinator_triggered is False
        assert event.overlay_state == {"qqq_ema_bullish": True}
        assert event.drawdown_tier == "NORMAL"
        assert event.market_session == "RTH"
        assert event.experiment_id == "exp1"
        assert event.experiment_variant == "A"
        assert "event_id" in event.event_metadata

    def test_log_order_coordinator_triggered(self):
        """Test #7: Coordinator-triggered OrderEvent."""
        event = self.logger.log_order(
            order_id="IBKR_001_STOP_MOD",
            pair="QQQ",
            side="LONG",
            order_type="STOP",
            status="SUBMITTED",
            requested_qty=0,
            strategy_id="ATRSS",
            order_action="MODIFY",
            coordinator_triggered=True,
            coordinator_rule="tighten_stop_be",
            modification_details={
                "field": "stop_price",
                "old_value": 480.0,
                "new_value": 485.32,
                "reason": "Move stop to breakeven after 2ATR move",
            },
        )
        assert event.coordinator_triggered is True
        assert event.coordinator_rule == "tighten_stop_be"
        assert event.order_action == "MODIFY"

    def test_modification_details_structure(self):
        """Test #8: Modification details with old_value/new_value/reason."""
        details = {
            "field": "stop_price",
            "old_value": 480.0,
            "new_value": 485.32,
            "reason": "BE move",
        }
        event = self.logger.log_order(
            order_id="mod_001",
            pair="QQQ",
            side="LONG",
            order_type="STOP",
            status="SUBMITTED",
            requested_qty=0,
            order_action="MODIFY",
            coordinator_triggered=True,
            coordinator_rule="tighten_stop_be",
            modification_details=details,
        )
        assert event.modification_details["old_value"] == 480.0
        assert event.modification_details["new_value"] == 485.32
        assert event.modification_details["reason"] == "BE move"

    def test_rejected_status(self):
        """Test #13: REJECTED status with reject_reason."""
        event = self.logger.log_order(
            order_id="IBKR_REJ_001",
            pair="SPY",
            side="LONG",
            order_type="MARKET",
            status="REJECTED",
            requested_qty=100,
            reject_reason="Insufficient margin",
            related_trade_id="t_spy_001",
            strategy_id="ATRSS",
        )
        assert event.status == "REJECTED"
        assert event.reject_reason == "Insufficient margin"
        assert event.coordinator_triggered is False

    def test_slippage_calculation(self):
        """Slippage BPS calculated when both prices present."""
        event = self.logger.log_order(
            order_id="slip_001",
            pair="QQQ",
            side="LONG",
            order_type="LIMIT",
            status="FILLED",
            requested_qty=50,
            filled_qty=50,
            requested_price=485.00,
            fill_price=485.50,
        )
        expected_bps = round(abs(485.50 - 485.00) / 485.00 * 10_000, 2)
        assert event.slippage_bps == expected_bps

    def test_slippage_none_without_prices(self):
        """Slippage is None when prices not provided."""
        event = self.logger.log_order(
            order_id="noslip_001",
            pair="QQQ",
            side="LONG",
            order_type="MARKET",
            status="FILLED",
            requested_qty=50,
            filled_qty=50,
        )
        assert event.slippage_bps is None

    def test_writes_jsonl_file(self):
        """Verify JSONL file is written."""
        self.logger.log_order(
            order_id="file_001",
            pair="QQQ",
            side="LONG",
            order_type="MARKET",
            status="FILLED",
            requested_qty=50,
            filled_qty=50,
        )
        filepath = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        assert filepath.exists()
        data = json.loads(filepath.read_text().strip())
        assert data["order_id"] == "file_001"

    def test_to_dict_excludes_none(self):
        """OrderEvent.to_dict() excludes None values."""
        event = OrderEvent(
            order_id="dict_001",
            bot_id="test",
            pair="QQQ",
            requested_price=None,
            fill_price=None,
        )
        d = event.to_dict()
        assert "requested_price" not in d
        assert "fill_price" not in d
        assert "order_id" in d

    def test_event_metadata_present(self):
        """Verify event_metadata has event_id."""
        event = self.logger.log_order(
            order_id="meta_001",
            pair="QQQ",
            side="LONG",
            order_type="MARKET",
            status="SUBMITTED",
            requested_qty=50,
        )
        assert "event_id" in event.event_metadata
        assert len(event.event_metadata["event_id"]) == 16

    def test_multiple_orders_same_day(self):
        """Multiple orders append to same JSONL file."""
        for i in range(3):
            self.logger.log_order(
                order_id=f"multi_{i}",
                pair="QQQ",
                side="LONG",
                order_type="MARKET",
                status="SUBMITTED",
                requested_qty=50,
            )
        filepath = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        lines = filepath.read_text().strip().split("\n")
        assert len(lines) == 3
