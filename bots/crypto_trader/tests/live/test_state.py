"""Tests for persistent state — save/load roundtrip and crash recovery."""

import json
from pathlib import Path

import pytest

from crypto_trader.live.state import PersistentState


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "live_state"


class TestPersistentState:
    def test_portfolio_state_roundtrip(self, state_dir):
        ps = PersistentState(state_dir)
        data = {"equity": 10000, "peak_equity": 10500, "open_risks": []}
        ps.save_portfolio_state(data)

        loaded = ps.load_portfolio_state()
        assert loaded == data

    def test_portfolio_state_not_found(self, state_dir):
        ps = PersistentState(state_dir)
        assert ps.load_portfolio_state() is None

    def test_trades_append_and_load(self, state_dir):
        ps = PersistentState(state_dir)
        ps.append_trade({"trade_id": "t1", "pnl": 100})
        ps.append_trade({"trade_id": "t2", "pnl": -50})

        trades = ps.load_trades()
        assert len(trades) == 2
        assert trades[0]["trade_id"] == "t1"
        assert trades[1]["pnl"] == -50

    def test_equity_snapshots(self, state_dir):
        ps = PersistentState(state_dir)
        ps.append_equity_snapshot(10000.0)
        ps.append_equity_snapshot(10100.0)

        snapshots = ps.load_equity_snapshots()
        assert len(snapshots) == 2
        assert snapshots[0]["equity"] == 10000.0
        assert snapshots[1]["equity"] == 10100.0
        assert "timestamp" in snapshots[0]

    def test_rule_events(self, state_dir):
        ps = PersistentState(state_dir)
        ps.append_rule_event({"strategy": "momentum", "approved": True})
        ps.append_rule_event({"strategy": "trend", "approved": False, "reason": "heat_cap"})

        events = ps.load_equity_snapshots()  # wrong path, should be rule_events
        events = [json.loads(line) for line in open(ps.rule_events_path, encoding="utf-8")]
        assert len(events) == 2

    def test_atomic_write_survives_corrupt(self, state_dir):
        ps = PersistentState(state_dir)
        ps.save_portfolio_state({"equity": 10000})

        # Corrupt the file
        with open(ps.portfolio_state_path, "w") as f:
            f.write("{corrupt")

        # Load should handle gracefully
        result = ps.load_portfolio_state()
        # JSON parsing fails, returns None
        assert result is None

        # Write new valid state
        ps.save_portfolio_state({"equity": 9000})
        result = ps.load_portfolio_state()
        assert result == {"equity": 9000}

    def test_empty_jsonl(self, state_dir):
        ps = PersistentState(state_dir)
        assert ps.load_trades() == []
        assert ps.load_equity_snapshots() == []

    def test_creates_directory(self, tmp_path):
        deep_dir = tmp_path / "a" / "b" / "c"
        ps = PersistentState(deep_dir)
        assert deep_dir.exists()

    def test_atomic_write_no_partial(self, state_dir):
        """Atomic write should not leave .tmp files on success."""
        ps = PersistentState(state_dir)
        ps.save_portfolio_state({"test": True})

        tmp_path = ps.portfolio_state_path.with_suffix(".tmp")
        assert not tmp_path.exists()
        assert ps.portfolio_state_path.exists()
