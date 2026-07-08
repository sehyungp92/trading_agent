"""Tests for heartbeat enrichment with position state."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from strategies.stock.instrumentation.src.facade import InstrumentationKit


def _make_kit_with_data_dir():
    """Create a kit with a real tmpdir-backed manager mock."""
    tmpdir = tempfile.mkdtemp()
    mgr = MagicMock()
    mgr._config = {"data_dir": tmpdir}
    mgr._strategy_id = "test_bot"
    mgr.get_sidecar_diagnostics.return_value = {
        "sidecar_buffer_depth": 0,
        "relay_reachable": True,
    }
    kit = InstrumentationKit(mgr, strategy_type="helix")
    return kit, tmpdir


class TestHeartbeatEnrichment:
    def test_positions_with_r_multiples(self):
        """Verify position data with R-multiples passes through."""
        kit, tmpdir = _make_kit_with_data_dir()
        positions = [
            {
                "pair": "NQ",
                "contract_month": "2026-06",
                "side": "LONG",
                "qty": 2,
                "entry_price": 21450.50,
                "current_price": 21475.25,
                "unrealized_pnl": 990.0,
                "unrealized_pnl_pct": 0.12,
                "unrealized_pnl_r": 0.45,
                "duration_minutes": 35,
                "session": "RTH",
                "strategy_type": "helix",
                "mfe_since_entry_r": 0.8,
                "mae_since_entry_r": -0.2,
            },
        ]
        kit.emit_heartbeat(
            active_positions=1,
            open_orders=0,
            uptime_s=28800.0,
            error_count_1h=0,
            positions=positions,
        )
        # Read the written file
        hb_files = list((Path(tmpdir) / "heartbeats").glob("*.jsonl"))
        assert len(hb_files) == 1
        data = json.loads(hb_files[0].read_text().strip())
        assert len(data["positions"]) == 1
        assert data["positions"][0]["unrealized_pnl_r"] == 0.45
        assert data["positions"][0]["mfe_since_entry_r"] == 0.8

    def test_futures_specific_fields(self):
        """Contract_month, session, margin_used_pct present in portfolio_exposure."""
        kit, tmpdir = _make_kit_with_data_dir()
        portfolio_exposure = {
            "total_positions": 2,
            "total_contracts": 3,
            "margin_used_pct": 42.5,
            "total_unrealized_pnl": 1685.0,
            "daily_realized_pnl": 1200.0,
            "session": "RTH",
            "by_strategy": {
                "helix": {"positions": 1, "contracts": 2, "unrealized_pnl": 990.0},
                "nqdtc": {"positions": 1, "contracts": 1, "unrealized_pnl": 695.0},
            },
        }
        kit.emit_heartbeat(
            active_positions=2,
            open_orders=1,
            uptime_s=28800.0,
            error_count_1h=0,
            portfolio_exposure=portfolio_exposure,
        )
        hb_files = list((Path(tmpdir) / "heartbeats").glob("*.jsonl"))
        data = json.loads(hb_files[0].read_text().strip())
        assert data["portfolio_exposure"]["margin_used_pct"] == 42.5
        assert data["portfolio_exposure"]["session"] == "RTH"
        assert "helix" in data["portfolio_exposure"]["by_strategy"]

    def test_by_strategy_grouping(self):
        """Multiple strategies → correct breakdown in portfolio_exposure."""
        kit, tmpdir = _make_kit_with_data_dir()
        exposure = {
            "total_positions": 3,
            "by_strategy": {
                "helix": {"positions": 2, "contracts": 3, "unrealized_pnl": 1500.0},
                "nqdtc": {"positions": 1, "contracts": 2, "unrealized_pnl": 950.0},
            },
        }
        kit.emit_heartbeat(
            active_positions=3,
            open_orders=0,
            uptime_s=3600.0,
            error_count_1h=0,
            portfolio_exposure=exposure,
        )
        hb_files = list((Path(tmpdir) / "heartbeats").glob("*.jsonl"))
        data = json.loads(hb_files[0].read_text().strip())
        by_st = data["portfolio_exposure"]["by_strategy"]
        assert by_st["helix"]["positions"] == 2
        assert by_st["nqdtc"]["contracts"] == 2

    def test_empty_positions(self):
        """No open trades → positions: [], portfolio_exposure with defaults."""
        kit, tmpdir = _make_kit_with_data_dir()
        kit.emit_heartbeat(
            active_positions=0,
            open_orders=0,
            uptime_s=100.0,
            error_count_1h=0,
        )
        hb_files = list((Path(tmpdir) / "heartbeats").glob("*.jsonl"))
        data = json.loads(hb_files[0].read_text().strip())
        assert data["positions"] == []
        assert data["portfolio_exposure"] == {}
        assert data["active_positions"] == 0

    def test_backward_compatible(self):
        """Heartbeat without new kwargs still works."""
        kit, tmpdir = _make_kit_with_data_dir()
        kit.emit_heartbeat(
            active_positions=1,
            open_orders=2,
            uptime_s=500.0,
            error_count_1h=3,
        )
        hb_files = list((Path(tmpdir) / "heartbeats").glob("*.jsonl"))
        data = json.loads(hb_files[0].read_text().strip())
        assert data["active_positions"] == 1
        assert data["open_orders"] == 2
        assert data["uptime_s"] == 500.0
        assert data["error_count_1h"] == 3
        assert data["positions"] == []
        assert data["portfolio_exposure"] == {}

    def test_heartbeat_noop_without_manager(self):
        """emit_heartbeat with None manager does not raise."""
        kit = InstrumentationKit(None, strategy_type="helix")
        kit.emit_heartbeat(
            active_positions=0,
            open_orders=0,
            uptime_s=0,
            error_count_1h=0,
        )

    def test_sidecar_includes_heartbeats(self):
        """Verify heartbeats directory is in sidecar type mapping."""
        from strategies.stock.instrumentation.src.sidecar import _DIR_TO_EVENT_TYPE, _EVENT_PRIORITY
        assert "heartbeats" in _DIR_TO_EVENT_TYPE
        assert _DIR_TO_EVENT_TYPE["heartbeats"] == "heartbeat"
        assert "heartbeat" in _EVENT_PRIORITY
