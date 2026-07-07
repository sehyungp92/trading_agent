"""Unit tests for libs.oms.instrumentation.correlation_snapshot."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from libs.oms.instrumentation.correlation_snapshot import (
    capture_concurrent_positions,
    capture_concurrent_positions_from_coordinator,
    run_async_safely,
)


# ---------------------------------------------------------------------------
# capture_concurrent_positions (async, DB-backed)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pg_store():
    store = AsyncMock()
    return store


class TestCaptureConcurrentPositions:
    def test_returns_empty_when_no_store(self):
        result = asyncio.run(
            capture_concurrent_positions(None, "momentum", "helix", "NQ", ["helix", "nqdtc"])
        )
        assert result == []

    def test_returns_empty_when_no_siblings(self):
        store = AsyncMock()
        result = asyncio.run(
            capture_concurrent_positions(store, "momentum", "helix", "NQ", [])
        )
        assert result == []

    def test_excludes_current_strategy(self):
        store = AsyncMock()
        store.fetch = AsyncMock(return_value=[])
        result = asyncio.run(
            capture_concurrent_positions(store, "momentum", "helix", "NQ", ["helix"])
        )
        assert result == []
        store.fetch.assert_not_called()

    def test_returns_sibling_positions(self, mock_pg_store):
        mock_pg_store.fetch = AsyncMock(return_value=[
            {"strategy_id": "nqdtc", "instrument_symbol": "NQ", "direction": "LONG"},
            {"strategy_id": "vdubus", "instrument_symbol": "ES", "direction": "SHORT"},
        ])

        result = asyncio.run(
            capture_concurrent_positions(
                mock_pg_store, "momentum", "helix", "NQ",
                ["helix", "nqdtc", "vdubus"],
            )
        )

        assert len(result) == 2
        assert result[0] == {
            "sibling_strategy_id": "nqdtc",
            "sibling_symbol": "NQ",
            "sibling_direction": "LONG",
            "same_symbol": True,
        }
        assert result[1] == {
            "sibling_strategy_id": "vdubus",
            "sibling_symbol": "ES",
            "sibling_direction": "SHORT",
            "same_symbol": False,
        }

    def test_query_uses_correct_siblings(self, mock_pg_store):
        mock_pg_store.fetch = AsyncMock(return_value=[])
        asyncio.run(
            capture_concurrent_positions(
                mock_pg_store, "stock", "IARIC_v1", "AAPL",
                ["IARIC_v1", "ALCB_v1"],
            )
        )
        call_args = mock_pg_store.fetch.call_args
        # Second positional arg is the sibling list (excluding current)
        siblings_passed = call_args[0][1]
        assert "IARIC_v1" not in siblings_passed
        assert "ALCB_v1" in siblings_passed


# ---------------------------------------------------------------------------
# capture_concurrent_positions_from_coordinator (sync, in-process)
# ---------------------------------------------------------------------------

class TestCaptureFromCoordinator:
    def test_returns_empty_when_no_coordinator(self):
        result = capture_concurrent_positions_from_coordinator(None, "ATRSS", "QQQ")
        assert result == []

    def test_returns_sibling_positions(self):
        pos1 = MagicMock(symbol="QQQ", net_qty=100)
        pos2 = MagicMock(symbol="GLD", net_qty=-50)
        coordinator = MagicMock()
        coordinator.get_all_positions.return_value = {
            "ATRSS": [MagicMock(symbol="QQQ", net_qty=200)],
            "AKC_HELIX": [pos1],
            "OVERLAY": [pos2],
        }

        result = capture_concurrent_positions_from_coordinator(
            coordinator, "ATRSS", "QQQ"
        )

        assert len(result) == 2
        assert result[0]["sibling_strategy_id"] == "AKC_HELIX"
        assert result[0]["same_symbol"] is True
        assert result[0]["sibling_direction"] == "LONG"
        assert result[1]["sibling_strategy_id"] == "OVERLAY"
        assert result[1]["sibling_direction"] == "SHORT"
        assert result[1]["same_symbol"] is False

    def test_skips_flat_positions(self):
        coordinator = MagicMock()
        coordinator.get_all_positions.return_value = {
            "ATRSS": [],
            "AKC_HELIX": [MagicMock(symbol="QQQ", net_qty=0)],
        }

        result = capture_concurrent_positions_from_coordinator(
            coordinator, "ATRSS", "QQQ"
        )
        assert result == []

    def test_handles_coordinator_error_gracefully(self):
        coordinator = MagicMock()
        coordinator.get_all_positions.side_effect = RuntimeError("DB down")

        result = capture_concurrent_positions_from_coordinator(
            coordinator, "ATRSS", "QQQ"
        )
        assert result == []

    def test_handles_dict_positions(self):
        """Coordinator may return positions as dicts instead of objects."""
        coordinator = MagicMock()
        coordinator.get_all_positions.return_value = {
            "ATRSS": [],
            "AKC_HELIX": [{"symbol": "GLD", "net_qty": 75}],
        }

        result = capture_concurrent_positions_from_coordinator(
            coordinator, "ATRSS", "QQQ"
        )
        assert len(result) == 1
        assert result[0]["sibling_symbol"] == "GLD"
        assert result[0]["sibling_direction"] == "LONG"
        assert result[0]["same_symbol"] is False


# ---------------------------------------------------------------------------
# run_async_safely helper
# ---------------------------------------------------------------------------

class TestRunAsyncSafely:
    def test_runs_coroutine_from_sync_context(self):
        async def add(a, b):
            return a + b

        result = run_async_safely(add(1, 2))
        assert result == 3

    def test_runs_coroutine_when_loop_already_running(self):
        """Verify it works even when called from inside an async context."""
        async def inner():
            async def double(x):
                return x * 2
            return run_async_safely(double(5))

        result = asyncio.run(inner())
        assert result == 10
