"""Comprehensive tests for bot-side observation changes.

Covers experiment tracking (A), order events (B), and heartbeat enrichment (C).
"""
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from strategies.swing.instrumentation.src.daily_snapshot import DailySnapshotBuilder, DailySnapshot
from strategies.swing.instrumentation.src.coordination_logger import CoordinationLogger
from strategies.swing.instrumentation.src.order_logger import OrderLogger
from strategies.swing.instrumentation.src.sidecar import Sidecar
from strategies.swing.instrumentation.src.kit import InstrumentationKit
from strategies.swing.instrumentation.src.context import InstrumentationContext


# ============================================================
# A. Experiment Tracking Tests
# ============================================================

class TestExperimentTracking:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "strategy_type": "multi_strategy",
            "data_dir": self.tmpdir,
        }
        self.builder = DailySnapshotBuilder(self.config)
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _write_trades(self, trades):
        trades_dir = Path(self.tmpdir) / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        filepath = trades_dir / f"trades_{self.today}.jsonl"
        with open(filepath, "w") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")

    def test_strategy_id_in_breakdown_multi_strategy(self):
        """Test #1: Experiment trades from ATRSS + AKC_HELIX — strategy_id
        reflects dominant, strategy_ids lists both."""
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100,
             "experiment_id": "exp1", "experiment_variant": "A", "strategy_id": "ATRSS"},
            {"stage": "exit", "trade_id": "t2", "pnl": -50,
             "experiment_id": "exp1", "experiment_variant": "A", "strategy_id": "ATRSS"},
            {"stage": "exit", "trade_id": "t3", "pnl": 200,
             "experiment_id": "exp1", "experiment_variant": "A", "strategy_id": "AKC_HELIX"},
        ])
        snap = self.builder.build(self.today)
        assert snap.experiment_breakdown is not None
        entry = snap.experiment_breakdown["exp1:A"]
        assert entry["strategy_id"] == "ATRSS"  # dominant (2 out of 3)
        assert sorted(entry["strategy_ids"]) == ["AKC_HELIX", "ATRSS"]

    def test_strategy_id_in_breakdown_single_strategy(self):
        """Test #2: All trades from ATRSS — strategy_id="ATRSS",
        strategy_ids=["ATRSS"]."""
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100,
             "experiment_id": "exp1", "experiment_variant": "B", "strategy_id": "ATRSS"},
            {"stage": "exit", "trade_id": "t2", "pnl": -50,
             "experiment_id": "exp1", "experiment_variant": "B", "strategy_id": "ATRSS"},
        ])
        snap = self.builder.build(self.today)
        entry = snap.experiment_breakdown["exp1:B"]
        assert entry["strategy_id"] == "ATRSS"
        assert entry["strategy_ids"] == ["ATRSS"]

    def test_active_experiments_from_yaml(self):
        """Test #3: Load experiments.yaml — correct structure."""
        config_dir = Path(self.tmpdir).parent / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        experiments = {
            "exp_test": {
                "hypothesis": "Test hypothesis",
                "variants": ["control", "variant_a"],
                "primary_metric": "sharpe",
                "start_date": "2026-03-01",
            }
        }
        import yaml
        with open(config_dir / "experiments.yaml", "w") as f:
            yaml.safe_dump(experiments, f)

        snap = self.builder.build(self.today)
        assert "exp_test" in snap.active_experiments
        assert snap.active_experiments["exp_test"]["hypothesis"] == "Test hypothesis"
        assert snap.active_experiments["exp_test"]["variants"] == ["control", "variant_a"]
        assert snap.active_experiments["exp_test"]["primary_metric"] == "sharpe"
        assert snap.active_experiments["exp_test"]["start_date"] == "2026-03-01"

    def test_missing_experiments_yaml(self):
        """Test #4: File absent — active_experiments: {} (no error)."""
        # Use a deeply nested tmpdir so no sibling config/ exists
        nested = tempfile.mkdtemp()
        nested_data = Path(nested) / "sub" / "data"
        nested_data.mkdir(parents=True, exist_ok=True)
        builder = DailySnapshotBuilder({
            "bot_id": "test_bot",
            "strategy_type": "multi_strategy",
            "data_dir": str(nested_data),
        })
        snap = builder.build(self.today)
        assert snap.active_experiments == {}

    def test_concluded_experiments_excluded(self):
        """Test #5: Experiments with concluded: true excluded."""
        config_dir = Path(self.tmpdir).parent / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        experiments = {
            "exp_active": {
                "hypothesis": "Active",
                "variants": ["A", "B"],
                "primary_metric": "sharpe",
                "start_date": "2026-03-01",
            },
            "exp_done": {
                "hypothesis": "Done",
                "variants": ["A", "B"],
                "primary_metric": "win_rate",
                "start_date": "2026-01-01",
                "concluded": True,
            },
        }
        import yaml
        with open(config_dir / "experiments.yaml", "w") as f:
            yaml.safe_dump(experiments, f)

        snap = self.builder.build(self.today)
        assert "exp_active" in snap.active_experiments
        assert "exp_done" not in snap.active_experiments

    def test_experiment_breakdown_no_strategy_id_trades(self):
        """Trades without strategy_id get empty strategy fields."""
        self._write_trades([
            {"stage": "exit", "trade_id": "t1", "pnl": 100,
             "experiment_id": "exp1", "experiment_variant": "A"},
        ])
        snap = self.builder.build(self.today)
        entry = snap.experiment_breakdown["exp1:A"]
        assert entry["strategy_id"] == ""
        assert entry["strategy_ids"] == []


# ============================================================
# B. Order Events Tests
# ============================================================

class TestCoordinationLoggerOrderIntegration:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "data_source_id": "test",
        }
        self.coord_logger = CoordinationLogger(self.config)
        self.order_logger = OrderLogger(self.config)
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def test_coordination_emits_order_event(self):
        """Test #10: log_action() with order_logger emits both events."""
        event = self.coord_logger.log_action(
            action="tighten_stop_be",
            trigger_strategy="ATRSS",
            target_strategy="AKC_HELIX",
            symbol="QQQ",
            rule="rule_1",
            details={"old_stop": 480.0, "new_stop": 485.0},
            outcome="applied",
            order_logger=self.order_logger,
            related_order_id="IBKR_STOP_001",
            related_trade_id="t_qqq_001",
        )
        assert event is not None
        # Coordination event written
        coord_file = Path(self.tmpdir) / "coordination" / f"coordination_{self.today}.jsonl"
        assert coord_file.exists()
        # Order event also written
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        assert order_file.exists()
        order_data = json.loads(order_file.read_text().strip())
        assert order_data["coordinator_triggered"] is True
        assert order_data["coordinator_rule"] == "rule_1"
        assert order_data["order_action"] == "MODIFY"
        assert order_data["strategy_id"] == "AKC_HELIX"

    def test_non_order_action_no_order_event(self):
        """Test #11: overlay_signal_change — no OrderEvent emitted."""
        self.coord_logger.log_action(
            action="overlay_signal_change",
            trigger_strategy="OVERLAY",
            target_strategy="ATRSS",
            symbol="QQQ",
            rule="ema_crossover",
            outcome="applied",
            order_logger=self.order_logger,
        )
        # No order file should be created for non-order actions
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        assert not order_file.exists()

    def test_skipped_outcome_no_order_event(self):
        """Skipped actions don't emit OrderEvents."""
        self.coord_logger.log_action(
            action="tighten_stop_be",
            trigger_strategy="ATRSS",
            target_strategy="AKC_HELIX",
            symbol="QQQ",
            rule="rule_1",
            outcome="skipped_already_tighter",
            order_logger=self.order_logger,
        )
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        assert not order_file.exists()

    def test_no_order_logger_no_crash(self):
        """Without order_logger param, existing behavior unchanged."""
        event = self.coord_logger.log_action(
            action="tighten_stop_be",
            trigger_strategy="ATRSS",
            target_strategy="AKC_HELIX",
            symbol="QQQ",
            rule="rule_1",
            outcome="applied",
        )
        assert event is not None
        assert event.action == "tighten_stop_be"

    def test_force_exit_emits_new_order(self):
        """force_exit action should emit order_action=NEW."""
        self.coord_logger.log_action(
            action="force_exit",
            trigger_strategy="ATRSS",
            target_strategy="AKC_HELIX",
            symbol="SPY",
            rule="drawdown_halt",
            outcome="applied",
            order_logger=self.order_logger,
        )
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        assert order_file.exists()
        data = json.loads(order_file.read_text().strip())
        assert data["order_action"] == "NEW"
        assert data["order_type"] == "MARKET"

    def test_cancel_order_emits_cancel(self):
        """cancel_order action should emit order_action=CANCEL."""
        self.coord_logger.log_action(
            action="cancel_order",
            trigger_strategy="ATRSS",
            target_strategy="AKC_HELIX",
            symbol="SPY",
            rule="exposure_limit",
            outcome="applied",
            order_logger=self.order_logger,
        )
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        data = json.loads(order_file.read_text().strip())
        assert data["order_action"] == "CANCEL"


class TestKitOnOrderEvent:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.order_logger = OrderLogger({
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
        })
        self.mock_drawdown = MagicMock()
        self.mock_drawdown.get_entry_context.return_value = {
            "drawdown_tier_at_entry": "CAUTION",
        }
        self.mock_overlay = MagicMock(return_value={"qqq_ema_bullish": True})

        self.ctx = InstrumentationContext(
            order_logger=self.order_logger,
            drawdown_tracker=self.mock_drawdown,
            overlay_state_provider=self.mock_overlay,
            data_dir=self.tmpdir,
        )
        self.kit = InstrumentationKit(self.ctx, strategy_id="ATRSS")
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def test_auto_captured_context(self):
        """Test #9: on_order_event() populates overlay_state, drawdown_tier,
        market_session from trackers."""
        self.kit.on_order_event(
            order_id="auto_001",
            pair="QQQ",
            side="LONG",
            order_type="MARKET",
            status="SUBMITTED",
            requested_qty=50,
            strategy_id="ATRSS",
        )
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        assert order_file.exists()
        data = json.loads(order_file.read_text().strip())
        assert data["drawdown_tier"] == "CAUTION"
        assert data["market_session"] != ""
        assert data["overlay_state"] == {"qqq_ema_bullish": True}

    def test_on_order_event_never_raises(self):
        """on_order_event must never crash trading."""
        kit = InstrumentationKit(None, strategy_id="ATRSS")
        # Should not raise
        kit.on_order_event(
            order_id="safe_001",
            pair="QQQ",
            side="LONG",
            order_type="MARKET",
            status="SUBMITTED",
            requested_qty=50,
        )

    def test_on_order_event_no_order_logger(self):
        """Graceful when order_logger is None."""
        ctx = InstrumentationContext(data_dir=self.tmpdir)
        kit = InstrumentationKit(ctx, strategy_id="ATRSS")
        # Should not raise
        kit.on_order_event(
            order_id="nolog_001",
            pair="QQQ",
            side="LONG",
            order_type="MARKET",
            status="SUBMITTED",
            requested_qty=50,
        )


# ============================================================
# B. Sidecar Priority Tests
# ============================================================

class TestSidecarOrderPriority:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "sidecar": {
                "relay_url": "",
                "hmac_secret_env": "TEST_HMAC_SECRET",
                "batch_size": 10,
                "retry_max": 1,
                "buffer_dir": str(Path(self.tmpdir) / ".sidecar_buffer"),
            },
        }
        self.sidecar = Sidecar(self.config)

    def test_coordinator_triggered_order_priority_2(self):
        """Test #12: Coordinator-triggered orders get priority 2."""
        raw = {
            "order_id": "coord_001",
            "coordinator_triggered": True,
            "event_metadata": {"event_id": "e1"},
        }
        wrapped = self.sidecar._wrap_event(raw, "order")
        assert wrapped["priority"] == 2

    def test_regular_order_priority_3(self):
        """Test #12: Regular orders get priority 3."""
        raw = {
            "order_id": "reg_001",
            "coordinator_triggered": False,
            "event_metadata": {"event_id": "e2"},
        }
        wrapped = self.sidecar._wrap_event(raw, "order")
        assert wrapped["priority"] == 3

    def test_order_dir_mapped(self):
        """Verify orders directory is in event type mapping."""
        (Path(self.tmpdir) / "orders").mkdir(parents=True, exist_ok=True)
        (Path(self.tmpdir) / "orders" / "orders_2026-03-06.jsonl").write_text(
            json.dumps({"order_id": "t1", "event_metadata": {}}) + "\n"
        )
        files = self.sidecar._get_event_files()
        event_types = [et for _, et in files]
        assert "order" in event_types


# ============================================================
# C. Heartbeat Enrichment Tests
# ============================================================

class TestHeartbeatEnrichment:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = InstrumentationContext(data_dir=self.tmpdir)
        self.kit = InstrumentationKit(self.ctx, strategy_id="ATRSS")
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def test_basic_heartbeat_without_positions(self):
        """Test #20: Heartbeat without new kwargs still works."""
        self.kit.emit_heartbeat(
            active_positions=2,
            open_orders=1,
            uptime_s=3600.0,
            error_count_1h=0,
        )
        filepath = Path(self.tmpdir) / "heartbeat" / f"heartbeat_{self.today}.jsonl"
        assert filepath.exists()
        data = json.loads(filepath.read_text().strip())
        assert data["active_positions"] == 2
        assert data["open_orders"] == 1
        assert data["uptime_s"] == 3600.0
        assert data["error_count_1h"] == 0

    def test_heartbeat_with_positions(self):
        """Test #14/15: Overlay vs main positions, exposure split."""
        positions = [
            {
                "pair": "QQQ", "side": "LONG", "qty": 50,
                "entry_price": 485.32, "current_price": 487.15,
                "unrealized_pnl": 91.50, "unrealized_pnl_pct": 0.38,
                "duration_minutes": 75, "strategy_id": "ATRSS",
                "is_overlay": False,
                "drawdown_pct_current": 2.1, "position_size_multiplier": 1.0,
            },
            {
                "pair": "QQQ", "side": "LONG", "qty": 20,
                "entry_price": 484.00, "current_price": 487.15,
                "unrealized_pnl": 63.00, "unrealized_pnl_pct": 0.65,
                "duration_minutes": 200, "strategy_id": "OVERLAY",
                "is_overlay": True,
                "drawdown_pct_current": 2.1, "position_size_multiplier": 1.0,
            },
        ]
        exposure = {
            "total_positions": 2,
            "main_strategy_positions": 1,
            "overlay_positions": 1,
            "total_exposure_pct": 7.8,
            "main_exposure_pct": 5.5,
            "overlay_exposure_pct": 2.3,
            "total_unrealized_pnl": 154.50,
            "drawdown_tier": "NORMAL",
            "market_session": "RTH",
            "by_strategy": {},
        }

        self.kit.emit_heartbeat(
            active_positions=2,
            open_orders=0,
            uptime_s=7200.0,
            error_count_1h=0,
            positions=positions,
            portfolio_exposure=exposure,
        )
        filepath = Path(self.tmpdir) / "heartbeat" / f"heartbeat_{self.today}.jsonl"
        data = json.loads(filepath.read_text().strip())
        assert len(data["positions"]) == 2
        assert data["positions"][0]["is_overlay"] is False
        assert data["positions"][1]["is_overlay"] is True
        assert data["portfolio_exposure"]["main_strategy_positions"] == 1
        assert data["portfolio_exposure"]["overlay_positions"] == 1

    def test_heartbeat_with_drawdown_context(self):
        """Test #16: Each position carries drawdown_pct and multiplier."""
        positions = [
            {
                "pair": "QQQ", "side": "LONG", "qty": 50,
                "entry_price": 485.0, "current_price": 480.0,
                "unrealized_pnl": -250.0, "unrealized_pnl_pct": -1.03,
                "duration_minutes": 120, "strategy_id": "ATRSS",
                "is_overlay": False,
                "drawdown_pct_current": 5.2,
                "position_size_multiplier": 0.75,
            },
        ]
        self.kit.emit_heartbeat(
            active_positions=1,
            open_orders=0,
            uptime_s=3600.0,
            error_count_1h=0,
            positions=positions,
        )
        filepath = Path(self.tmpdir) / "heartbeat" / f"heartbeat_{self.today}.jsonl"
        data = json.loads(filepath.read_text().strip())
        pos = data["positions"][0]
        assert pos["drawdown_pct_current"] == 5.2
        assert pos["position_size_multiplier"] == 0.75

    def test_heartbeat_with_coordinator_rules(self):
        """Test #17: coordinator_active_rules in portfolio_exposure."""
        exposure = {
            "total_positions": 1,
            "coordinator_active_rules": ["tighten_stop_be", "size_boost"],
            "by_strategy": {},
        }
        self.kit.emit_heartbeat(
            active_positions=1,
            open_orders=0,
            uptime_s=3600.0,
            error_count_1h=0,
            positions=[],
            portfolio_exposure=exposure,
        )
        filepath = Path(self.tmpdir) / "heartbeat" / f"heartbeat_{self.today}.jsonl"
        data = json.loads(filepath.read_text().strip())
        assert data["portfolio_exposure"]["coordinator_active_rules"] == [
            "tighten_stop_be", "size_boost"
        ]

    def test_heartbeat_empty_positions(self):
        """Test #19: No open trades — positions: [], exposure zeros."""
        exposure = {
            "total_positions": 0,
            "main_strategy_positions": 0,
            "overlay_positions": 0,
            "total_exposure_pct": 0.0,
            "main_exposure_pct": 0.0,
            "overlay_exposure_pct": 0.0,
            "total_unrealized_pnl": 0.0,
            "by_strategy": {},
        }
        self.kit.emit_heartbeat(
            active_positions=0,
            open_orders=0,
            uptime_s=3600.0,
            error_count_1h=0,
            positions=[],
            portfolio_exposure=exposure,
        )
        filepath = Path(self.tmpdir) / "heartbeat" / f"heartbeat_{self.today}.jsonl"
        data = json.loads(filepath.read_text().strip())
        assert data["positions"] == []
        assert data["portfolio_exposure"]["total_positions"] == 0

    def test_heartbeat_never_raises(self):
        """Heartbeat must never crash trading."""
        kit = InstrumentationKit(None, strategy_id="ATRSS")
        # Should not raise
        kit.emit_heartbeat(
            active_positions=0,
            open_orders=0,
            uptime_s=0.0,
            error_count_1h=0,
        )

    def test_heartbeat_backward_compatible(self):
        """Test #20: Basic heartbeat without new kwargs works."""
        self.kit.emit_heartbeat(
            active_positions=3,
            open_orders=2,
            uptime_s=10000.0,
            error_count_1h=1,
        )
        filepath = Path(self.tmpdir) / "heartbeat" / f"heartbeat_{self.today}.jsonl"
        data = json.loads(filepath.read_text().strip())
        assert data["active_positions"] == 3
        assert data["open_orders"] == 2
        assert "positions" not in data
        assert "portfolio_exposure" not in data


# ============================================================
# D. Order Event Integration Tests (FILLED, REJECTED, terminal)
# ============================================================

class TestOrderEventFilled:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.order_logger = OrderLogger({
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
        })
        self.ctx = InstrumentationContext(
            order_logger=self.order_logger,
            data_dir=self.tmpdir,
        )
        self.kit = InstrumentationKit(self.ctx, strategy_id="ATRSS")
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def test_filled_order_has_fill_fields(self):
        """FILLED order event includes fill_price, filled_qty, and strategy_id."""
        self.kit.on_order_event(
            order_id="fill_001",
            pair="SPY",
            side="LONG",
            order_type="STOP_LIMIT",
            status="FILLED",
            requested_qty=100,
            filled_qty=100,
            requested_price=450.50,
            fill_price=450.55,
            related_trade_id="trade_001",
            strategy_id="ATRSS",
        )
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        assert order_file.exists()
        data = json.loads(order_file.read_text().strip())
        assert data["status"] == "FILLED"
        assert data["fill_price"] == 450.55
        assert data["filled_qty"] == 100
        assert data["strategy_id"] == "ATRSS"
        assert data["related_trade_id"] == "trade_001"

    def test_filled_order_slippage_computed(self):
        """FILLED order with price difference computes slippage_bps."""
        self.kit.on_order_event(
            order_id="slip_001",
            pair="QQQ",
            side="LONG",
            order_type="STOP_LIMIT",
            status="FILLED",
            requested_qty=50,
            filled_qty=50,
            requested_price=400.00,
            fill_price=400.20,
            strategy_id="ATRSS",
        )
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        data = json.loads(order_file.read_text().strip())
        # slippage_bps = (400.20 - 400.00) / 400.00 * 10000 = 5.0
        assert "slippage_bps" in data
        assert data["slippage_bps"] is not None


class TestOrderEventRejected:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.order_logger = OrderLogger({
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
        })
        self.ctx = InstrumentationContext(
            order_logger=self.order_logger,
            data_dir=self.tmpdir,
        )
        self.kit = InstrumentationKit(self.ctx, strategy_id="ATRSS")
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def test_rejected_order_has_status(self):
        """REJECTED order event writes correct status."""
        self.kit.on_order_event(
            order_id="rej_001",
            pair="QQQ",
            side="LONG",
            order_type="STOP_LIMIT",
            status="REJECTED",
            requested_qty=50,
            requested_price=400.00,
            reject_reason="insufficient margin",
            strategy_id="ATRSS",
        )
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        data = json.loads(order_file.read_text().strip())
        assert data["status"] == "REJECTED"
        assert data["reject_reason"] == "insufficient margin"

    def test_cancelled_status_string(self):
        """CANCELLED order event writes correct status string."""
        self.kit.on_order_event(
            order_id="can_001",
            pair="SPY",
            side="SHORT",
            order_type="STOP_LIMIT",
            status="CANCELLED",
            requested_qty=30,
            strategy_id="ATRSS",
        )
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        data = json.loads(order_file.read_text().strip())
        assert data["status"] == "CANCELLED"

    def test_expired_status_string(self):
        """EXPIRED order event writes correct status string."""
        self.kit.on_order_event(
            order_id="exp_001",
            pair="IWM",
            side="LONG",
            order_type="STOP_LIMIT",
            status="EXPIRED",
            requested_qty=40,
            strategy_id="AKC_HELIX",
        )
        order_file = Path(self.tmpdir) / "orders" / f"orders_{self.today}.jsonl"
        data = json.loads(order_file.read_text().strip())
        assert data["status"] == "EXPIRED"
        assert data["strategy_id"] == "AKC_HELIX"


# ============================================================
# E. log_missed Integration Tests
# ============================================================

class TestLogMissedIntegration:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = InstrumentationContext(data_dir=self.tmpdir)
        # Provide a mock missed_logger
        self.mock_missed = MagicMock()
        mock_event = MagicMock()
        mock_event.to_dict.return_value = {"signal_id": "test", "blocked_by": "test"}
        self.mock_missed.log_missed.return_value = mock_event
        self.ctx.missed_logger = self.mock_missed
        self.kit = InstrumentationKit(self.ctx, strategy_id="AKC_HELIX")

    def test_log_missed_hard_block_captures_direction(self):
        """log_missed called with direction before it's set to None."""
        result = self.kit.log_missed(
            pair="SPY",
            side="LONG",
            signal="breakout",
            signal_id="SPY_hard_block_2026-03-06",
            signal_strength=0.0,
            blocked_by="hard_block",
            block_reason="4H regime RANGE_CHOP blocks LONG",
        )
        self.mock_missed.log_missed.assert_called_once()
        call_kwargs = self.mock_missed.log_missed.call_args[1]
        assert call_kwargs["pair"] == "SPY"
        assert call_kwargs["side"] == "LONG"
        assert call_kwargs["blocked_by"] == "hard_block"

    def test_log_missed_volume_filter(self):
        """log_missed for a volume filter has correct signal_id and block_reason."""
        result = self.kit.log_missed(
            pair="QQQ",
            side="LONG",
            signal="breakout",
            signal_id="QQQ_vol_filter_2026-03-06",
            signal_strength=0.0,
            blocked_by="volume_filter",
            block_reason="volume 500000 < SMA 750000",
        )
        self.mock_missed.log_missed.assert_called_once()
        call_kwargs = self.mock_missed.log_missed.call_args[1]
        assert call_kwargs["signal"] == "breakout"
        assert call_kwargs["blocked_by"] == "volume_filter"
        assert "volume" in call_kwargs["block_reason"]
        assert "SMA" in call_kwargs["block_reason"]

    def test_log_missed_score_threshold_has_strength(self):
        """log_missed for score threshold includes signal_strength ratio."""
        result = self.kit.log_missed(
            pair="QQQ",
            side="SHORT",
            signal="breakout",
            signal_id="QQQ_score_threshold_2026-03-06",
            signal_strength=0.67,  # score_total / score_threshold
            blocked_by="score_threshold",
            block_reason="score 2 < threshold 3",
        )
        self.mock_missed.log_missed.assert_called_once()
        call_kwargs = self.mock_missed.log_missed.call_args[1]
        assert call_kwargs["signal_strength"] == 0.67

    def test_log_missed_never_raises(self):
        """log_missed with None context must not crash."""
        kit = InstrumentationKit(None, strategy_id="TEST")
        result = kit.log_missed(
            pair="SPY",
            side="LONG",
            signal="breakout",
            signal_id="test",
            signal_strength=0.0,
            blocked_by="test",
        )
        assert result == {}


# ============================================================
# F. classify_regime Integration Tests
# ============================================================

class TestClassifyRegimeIntegration:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = InstrumentationContext(data_dir=self.tmpdir)
        self.mock_classifier = MagicMock()
        self.mock_classifier.classify.return_value = "trending_up"
        self.ctx.regime_classifier = self.mock_classifier
        self.kit = InstrumentationKit(self.ctx, strategy_id="AKC_HELIX")

    def test_classify_regime_returns_result(self):
        """classify_regime returns classifier result."""
        result = self.kit.classify_regime("SPY")
        assert result == "trending_up"
        self.mock_classifier.classify.assert_called_once_with("SPY")

    def test_classify_regime_never_raises(self):
        """classify_regime with None context returns 'unknown'."""
        kit = InstrumentationKit(None, strategy_id="TEST")
        result = kit.classify_regime("SPY")
        assert result == "unknown"

    def test_classify_regime_no_classifier_returns_unknown(self):
        """classify_regime without classifier returns 'unknown'."""
        ctx = InstrumentationContext(data_dir=self.tmpdir)
        kit = InstrumentationKit(ctx, strategy_id="TEST")
        result = kit.classify_regime("QQQ")
        assert result == "unknown"


# ============================================================
# G. Heartbeat Emission Integration (main_multi pattern)
# ============================================================

class TestHeartbeatMainMultiPattern:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = InstrumentationContext(data_dir=self.tmpdir)
        self.kit = InstrumentationKit(self.ctx, strategy_id="ATRSS")
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def test_heartbeat_with_aggregated_counts(self):
        """Heartbeat with active_positions and open_orders from multiple engines."""
        self.kit.emit_heartbeat(
            active_positions=5,  # sum across all engines
            open_orders=3,
            uptime_s=120.0,
            error_count_1h=0,
        )
        filepath = Path(self.tmpdir) / "heartbeat" / f"heartbeat_{self.today}.jsonl"
        assert filepath.exists()
        data = json.loads(filepath.read_text().strip())
        assert data["active_positions"] == 5
        assert data["open_orders"] == 3
        assert data["uptime_s"] == 120.0
