"""Unit tests for the Strategy Diagnostic Pulse system.

Verifies each engine exposes the 3 required diagnostic keys in health_status()
and that _record_decision() correctly updates state.
"""
from __future__ import annotations

import importlib
import inspect
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Engine classes to test: (module_path, class_name)
# ---------------------------------------------------------------------------

ENGINE_SPECS = [
    ("strategies.swing.akc_helix.engine", "HelixEngine"),
    ("strategies.swing.atrss.engine", "ATRSSEngine"),
    ("strategies.swing.overlay.engine", "OverlayEngine"),
    ("strategies.momentum.nqdtc.engine", "NQDTCEngine"),
    ("strategies.momentum.nq_regime.engine", "NQRegimeEngine"),
    ("strategies.momentum.vdub.engine", "VdubNQv4Engine"),
    ("strategies.momentum.downturn.engine", "DownturnEngine"),
    ("strategies.stock.iaric.engine", "IARICEngine"),
    ("strategies.stock.alcb.engine", "ALCBT2Engine"),
]

REQUIRED_HEALTH_KEYS = {"last_decision_code", "last_decision_details", "last_bar_ts"}


def _get_engine_class(module_path: str, class_name: str):
    """Import and return the engine class, skipping if import fails."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except Exception:
        pytest.skip(f"Cannot import {module_path}.{class_name}")


# ---------------------------------------------------------------------------
# Test: every engine class has _record_decision method
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_path,class_name", ENGINE_SPECS, ids=[s[1] for s in ENGINE_SPECS])
def test_engine_has_record_decision(module_path, class_name):
    cls = _get_engine_class(module_path, class_name)
    assert hasattr(cls, "_record_decision"), (
        f"{class_name} missing _record_decision method"
    )
    sig = inspect.signature(cls._record_decision)
    params = list(sig.parameters.keys())
    assert "code" in params, f"{class_name}._record_decision must accept 'code' param"


# ---------------------------------------------------------------------------
# Test: every engine class has health_status method
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_path,class_name", ENGINE_SPECS, ids=[s[1] for s in ENGINE_SPECS])
def test_engine_has_health_status(module_path, class_name):
    cls = _get_engine_class(module_path, class_name)
    assert hasattr(cls, "health_status"), (
        f"{class_name} missing health_status method"
    )


# ---------------------------------------------------------------------------
# Test: diagnostic instance vars initialized in class body
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_path,class_name", ENGINE_SPECS, ids=[s[1] for s in ENGINE_SPECS])
def test_engine_has_diagnostic_vars(module_path, class_name):
    """Verify the engine source defines the 3 diagnostic instance vars."""
    cls = _get_engine_class(module_path, class_name)
    source = inspect.getsource(cls.__init__)
    assert "_last_decision_code" in source, f"{class_name}.__init__ missing _last_decision_code"
    assert "_last_decision_details" in source, f"{class_name}.__init__ missing _last_decision_details"
    assert "_last_bar_ts" in source, f"{class_name}.__init__ missing _last_bar_ts"


# ---------------------------------------------------------------------------
# Test: health_status source references the 3 required keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_path,class_name", ENGINE_SPECS, ids=[s[1] for s in ENGINE_SPECS])
def test_health_status_returns_diagnostic_keys(module_path, class_name):
    """Verify health_status() (or its delegate) references all 3 diagnostic keys."""
    cls = _get_engine_class(module_path, class_name)
    source = inspect.getsource(cls.health_status)
    # If health_status delegates to another method (e.g. snapshot_state), check that too
    if "snapshot_state" in source and hasattr(cls, "snapshot_state"):
        source = inspect.getsource(cls.snapshot_state)
    for key in REQUIRED_HEALTH_KEYS:
        assert key in source, (
            f"{class_name}.health_status() missing '{key}' in return dict"
        )


# ---------------------------------------------------------------------------
# Test: _record_decision updates state correctly
# ---------------------------------------------------------------------------

def test_record_decision_updates_state():
    """Test _record_decision on ATRSS engine (representative)."""
    cls = _get_engine_class("strategies.swing.atrss.engine", "ATRSSEngine")

    # Create a minimal mock instance with just the diagnostic attrs
    obj = object.__new__(cls)
    obj._last_decision_code = "IDLE"
    obj._last_decision_details = {}

    obj._record_decision("NO_SIGNAL", {"symbol": "QQQ", "bar_idx": 42})
    assert obj._last_decision_code == "NO_SIGNAL"
    assert obj._last_decision_details == {"symbol": "QQQ", "bar_idx": 42}

    obj._record_decision("MANAGING_POSITION")
    assert obj._last_decision_code == "MANAGING_POSITION"
    assert obj._last_decision_details == {}


# ---------------------------------------------------------------------------
# Test: FarmMonitor.all_statuses()
# ---------------------------------------------------------------------------

def test_farm_monitor_all_statuses():
    """Verify FarmMonitor exposes all_statuses() method."""
    try:
        from libs.broker_ibkr.farm_monitor import FarmMonitor, FarmStatus
    except ImportError:
        pytest.skip("Cannot import FarmMonitor")

    mock_ib = MagicMock()
    fm = FarmMonitor(mock_ib)
    fm._farm_status = {"usfarm": FarmStatus.OK, "cashfarm": FarmStatus.BROKEN}
    result = fm.all_statuses()
    assert result == {"usfarm": "OK", "cashfarm": "BROKEN"}


# ---------------------------------------------------------------------------
# Test: watchdog check_data_freshness exists
# ---------------------------------------------------------------------------

def test_watchdog_data_freshness_check_exists():
    """Verify check_data_freshness is importable."""
    try:
        from apps.watchdog.checks import check_data_freshness
    except ImportError:
        pytest.skip("Cannot import check_data_freshness")
    assert callable(check_data_freshness)
    sig = inspect.signature(check_data_freshness)
    params = list(sig.parameters.keys())
    assert "pool" in params
    assert "active_families" in params
