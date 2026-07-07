"""Unit tests for the 2026-04-10 audit fixes.

Covers:
  Fix 1 — Cancel status normalization (adapter emits "Cancelled", status map matches)
  Fix 2 — Family-scoped risk aggregation (strategy_ids filter in postgres queries)
  Fix 3 — CURRENT_DATE replaced with explicit ET trade date
  Fix 4 — Order event payload (symbol field + reject_reason key)
  Fix 6 — Relay ack data-loss (monotonic id ordering, no priority)
  Fix 8 — Order bus symbol field present
  Fix 9 — Relay HMAC auth enforcement in paper/live
  Fix 10 — Dashboard registry-backed strategy metadata
  Fix 13 — TA order/process_quality curated artifacts
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ASSISTANT_SRC = Path("packages/trading_assistant/src")

# ===========================================================================
# Fix 1 — Cancel status normalization
# ===========================================================================


class TestCancelStatusNormalization:
    """Adapter emits 'Cancelled' (PascalCase); both single and multi OMS
    status maps must map it to OrderStatus.CANCELLED."""

    def test_adapter_emits_pascal_case_cancelled(self):
        """execution_adapter._handle_order_status sends 'Cancelled' for IB cancels."""
        from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter

        # We inspect the source to confirm the string literal is "Cancelled"
        import inspect

        src = inspect.getsource(IBKRExecutionAdapter._handle_order_status)
        # The adapter should emit "Cancelled", NOT "CANCELLED"
        assert '"Cancelled"' in src or "'Cancelled'" in src
        assert '"CANCELLED"' not in src

    def test_single_oms_status_map_contains_cancelled(self):
        """The on_status callback in build_oms_service maps 'Cancelled'."""
        # Read factory source to verify status_map keys
        src_path = Path("libs/oms/services/factory.py")
        src = src_path.read_text(encoding="utf-8")
        # Both single and multi OMS status maps must have "Cancelled" key
        assert '"Cancelled": OrderStatus.CANCELLED' in src

    def test_cancel_status_roundtrip(self):
        """Drive 'Cancelled' through a mock status map and verify it resolves."""
        from libs.oms.models.order import OrderStatus

        status_map = {
            "Submitted": OrderStatus.WORKING,
            "PreSubmitted": OrderStatus.ROUTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "ApiCancelled": OrderStatus.CANCELLED,
            "PendingCancel": OrderStatus.CANCEL_REQUESTED,
        }
        assert status_map.get("Cancelled") == OrderStatus.CANCELLED
        # UPPERCASE must NOT match (this was the bug)
        assert status_map.get("CANCELLED") is None


class TestIBKRFillCallbackOrdering:
    """Live fill callbacks must not suppress executions before OMS mapping exists."""

    def test_unmapped_execution_id_is_not_marked_seen(self):
        from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter
        from libs.broker_ibkr.state.cache import IBCache

        adapter = object.__new__(IBKRExecutionAdapter)
        adapter._cache = IBCache()
        adapter.on_fill = MagicMock()

        trade = SimpleNamespace(order=SimpleNamespace(orderId=101))
        fill = SimpleNamespace(
            execution=SimpleNamespace(
                execId="exec-1",
                price=100.25,
                shares=1,
                time=datetime.now(timezone.utc),
            ),
            commissionReport=SimpleNamespace(commission=1.25),
        )

        adapter._handle_exec_details(trade, fill)

        assert not adapter._cache.is_fill_seen("exec-1")
        adapter.on_fill.assert_not_called()

        adapter._cache.register_order("oms-1", 101, 9001)
        adapter._handle_exec_details(trade, fill)

        assert adapter._cache.is_fill_seen("exec-1")
        adapter.on_fill.assert_called_once()


# ===========================================================================
# Fix 4 + 8 — Order event payload (symbol field + reject_reason key)
# ===========================================================================


class TestOrderEventPayload:
    """EventBus.emit_order_event must include 'symbol' and 'reject_reason'."""

    def test_order_event_contains_symbol_field(self):
        from libs.oms.events.bus import EventBus
        from libs.oms.models.instrument import Instrument
        from libs.oms.models.order import (
            OMSOrder,
            OrderRole,
            OrderSide,
            OrderStatus,
            OrderType,
        )

        bus = EventBus()
        q = bus.subscribe("test_strat")

        instr = Instrument(
            symbol="AAPL", root="AAPL", venue="SMART",
            tick_size=0.01, tick_value=0.01, multiplier=1.0,
        )
        order = OMSOrder(
            oms_order_id="ord-1",
            strategy_id="test_strat",
            instrument=instr,
            status=OrderStatus.FILLED,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            role=OrderRole.ENTRY,
        )
        bus.emit_order_event(order)

        event = q.get_nowait()
        assert event.payload["symbol"] == "AAPL"

    def test_order_event_contains_reject_reason_key(self):
        from libs.oms.events.bus import EventBus
        from libs.oms.models.order import (
            OMSOrder,
            OrderRole,
            OrderSide,
            OrderStatus,
            OrderType,
        )

        bus = EventBus()
        q = bus.subscribe("test_strat")

        order = OMSOrder(
            oms_order_id="ord-2",
            strategy_id="test_strat",
            status=OrderStatus.REJECTED,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            role=OrderRole.ENTRY,
            reject_reason="Insufficient margin",
        )
        bus.emit_order_event(order)

        event = q.get_nowait()
        # Must use "reject_reason" key (not "rejection_reason")
        assert "reject_reason" in event.payload
        assert event.payload["reject_reason"] == "Insufficient margin"
        assert "rejection_reason" not in event.payload

    def test_order_event_symbol_empty_when_no_instrument(self):
        from libs.oms.events.bus import EventBus
        from libs.oms.models.order import (
            OMSOrder,
            OrderRole,
            OrderSide,
            OrderStatus,
            OrderType,
        )

        bus = EventBus()
        q = bus.subscribe("test_strat")

        order = OMSOrder(
            oms_order_id="ord-3",
            strategy_id="test_strat",
            instrument=None,
            status=OrderStatus.CANCELLED,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            role=OrderRole.EXIT,
        )
        bus.emit_order_event(order)

        event = q.get_nowait()
        assert event.payload["symbol"] == ""


# ===========================================================================
# Fix 6 — Relay ack data-loss (monotonic id ordering)
# ===========================================================================


class TestRelayAckDataLoss:
    """get_events must return events by id ASC (not priority ASC) so that
    watermark-based ack cannot skip unseen events."""

    def _make_store(self, db_path: str):
        from apps.relay.db.store import EventStore
        return EventStore(db_path=db_path)

    def test_events_returned_in_id_order_not_priority(self):
        """Insert events with mixed priorities; verify fetch order is by id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(f"{tmpdir}/test.db")
            store.insert_events([
                {"event_id": "e1", "bot_id": "b1", "priority": 3},
                {"event_id": "e2", "bot_id": "b1", "priority": 3},
                {"event_id": "e3", "bot_id": "b1", "priority": 0},  # high priority, higher id
            ])

            events = store.get_events(limit=2)
            ids = [e["event_id"] for e in events]
            # Must be in insertion order (id ASC), NOT priority order
            assert ids == ["e1", "e2"]

    def test_ack_does_not_skip_unseen_events(self):
        """Acking the last event in a batch must not mark later-id events as acked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(f"{tmpdir}/test.db")
            store.insert_events([
                {"event_id": "e1", "bot_id": "b1", "priority": 3},
                {"event_id": "e2", "bot_id": "b1", "priority": 3},
                {"event_id": "e3", "bot_id": "b1", "priority": 0},
                {"event_id": "e4", "bot_id": "b1", "priority": 3},
            ])

            # Fetch first 2 events
            batch = store.get_events(limit=2)
            assert len(batch) == 2

            # Ack up to the last event in the batch
            last_event_id = batch[-1]["event_id"]
            store.ack_up_to(last_event_id)

            # Remaining events must all still be fetchable
            remaining = store.get_events(limit=10)
            remaining_ids = [e["event_id"] for e in remaining]
            assert "e3" in remaining_ids
            assert "e4" in remaining_ids

    def test_all_events_eventually_delivered(self):
        """Repeated fetch-ack cycles must deliver every event exactly once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(f"{tmpdir}/test.db")
            store.insert_events([
                {"event_id": f"e{i}", "bot_id": "b1", "priority": (0 if i % 3 == 0 else 3)}
                for i in range(1, 11)
            ])

            delivered = []
            for _ in range(20):  # safety limit
                batch = store.get_events(limit=3)
                if not batch:
                    break
                delivered.extend(e["event_id"] for e in batch)
                store.ack_up_to(batch[-1]["event_id"])

            assert sorted(delivered, key=lambda x: int(x[1:])) == [f"e{i}" for i in range(1, 11)]


# ===========================================================================
# Fix 2 — Family-scoped risk aggregation (strategy_ids filter)
# ===========================================================================


class TestFamilyScopedRiskAggregation:
    """Postgres queries must filter by strategy_ids to prevent cross-family bleed."""

    def test_postgres_get_risk_daily_strategies_accepts_strategy_ids(self):
        """Verify the method signature accepts strategy_ids parameter."""
        import inspect
        from libs.oms.persistence.postgres import PgStore

        sig = inspect.signature(PgStore.get_risk_daily_strategies_for_date)
        assert "strategy_ids" in sig.parameters

    def test_postgres_get_risk_daily_strategy_totals_accepts_strategy_ids(self):
        """Verify the method signature accepts strategy_ids parameter."""
        import inspect
        from libs.oms.persistence.postgres import PgStore

        sig = inspect.signature(PgStore.get_risk_daily_strategy_totals)
        assert "strategy_ids" in sig.parameters

    def test_factory_single_oms_defines_family_sids(self):
        """build_oms_service must accept family_strategy_ids for scoped queries."""
        src = Path("libs/oms/services/factory.py").read_text(encoding="utf-8")
        # Single-OMS builder should use family_strategy_ids when provided, else [strategy_id]
        assert "family_strategy_ids" in src
        assert "_family_sids = list(family_strategy_ids) if family_strategy_ids else [strategy_id]" in src

    def test_factory_multi_oms_defines_family_sids(self):
        """build_multi_strategy_oms must define _family_sids for scoped queries."""
        src = Path("libs/oms/services/factory.py").read_text(encoding="utf-8")
        # Multi-OMS builder should define _family_sids from strategies list
        assert '_family_sids = [s["id"] for s in strategies]' in src

    def test_factory_passes_strategy_ids_to_queries(self):
        """Both load and persist paths must pass strategy_ids filter."""
        src = Path("libs/oms/services/factory.py").read_text(encoding="utf-8")
        assert "strategy_ids=_family_sids" in src


# ===========================================================================
# Fix 3 — CURRENT_DATE replaced with explicit ET trade date
# ===========================================================================


class TestCurrentDateRemoved:
    """All SQL queries must use explicit ET trade date, never CURRENT_DATE."""

    _SQL_FILES = [
        "libs/risk/account_risk_gate.py",
        "apps/dashboard/src/app/api/live/route.ts",
        "apps/dashboard/src/app/api/portfolio/route.ts",
        "apps/dashboard/src/app/api/strategies/route.ts",
        "apps/dashboard/src/app/api/charts/route.ts",
    ]

    def test_no_current_date_in_sql_files(self):
        """CURRENT_DATE must not appear in any query file."""
        for rel_path in self._SQL_FILES:
            p = Path(rel_path)
            if not p.exists():
                continue
            src = p.read_text(encoding="utf-8")
            assert "CURRENT_DATE" not in src, f"CURRENT_DATE found in {rel_path}"

    def test_account_risk_gate_uses_et_timezone(self):
        """account_risk_gate.py must use explicit ET trade date expression."""
        src = Path("libs/risk/account_risk_gate.py").read_text(encoding="utf-8")
        assert "(now() AT TIME ZONE 'America/New_York')::date" in src

    def test_dashboard_live_route_uses_et_timezone(self):
        """Live batch route must use explicit ET trade date expression."""
        src = Path("apps/dashboard/src/app/api/live/route.ts").read_text(encoding="utf-8")
        assert "(now() AT TIME ZONE 'America/New_York')::date" in src


# ===========================================================================
# Fix 9 — HMAC auth enforcement in paper/live
# ===========================================================================


class TestHMACFailOpen:
    """Sidecar HMAC gaps must disable forwarding, not local instrumentation."""

    def test_swing_context_disables_forwarding_without_hmac(self):
        """Swing InstrumentationContext.start() warns and skips forwarding."""
        src = Path("strategies/swing/instrumentation/src/context.py").read_text(encoding="utf-8")
        assert 'env in ("paper", "live")' in src
        assert "Sidecar forwarding disabled" in src
        assert "Local startup instrumentation will continue" in src

    def test_momentum_bootstrap_disables_forwarding_without_hmac(self):
        """Momentum InstrumentationManager.start() warns and skips forwarding."""
        src = Path("strategies/momentum/instrumentation/src/bootstrap.py").read_text(encoding="utf-8")
        assert 'env in ("paper", "live")' in src
        assert "Sidecar forwarding disabled" in src
        assert "Local startup instrumentation will continue" in src

    def test_swing_hmac_gate_before_sidecar_start(self):
        """HMAC forwarding gate must happen before sidecar.start()."""
        src = Path("strategies/swing/instrumentation/src/context.py").read_text(encoding="utf-8")
        hmac_pos = src.index("Sidecar forwarding disabled")
        start_pos = src.index("self.sidecar.start()")
        assert hmac_pos < start_pos, "HMAC forwarding gate must precede sidecar.start()"

    def test_momentum_hmac_gate_before_running_flag(self):
        """HMAC forwarding gate must happen before self._running = True."""
        src = Path("strategies/momentum/instrumentation/src/bootstrap.py").read_text(encoding="utf-8")
        hmac_pos = src.index("Sidecar forwarding disabled")
        running_pos = src.index("self._running = True")
        assert hmac_pos < running_pos, "HMAC forwarding gate must precede _running = True"


# ===========================================================================
# Fix 10 — Dashboard registry-backed strategy metadata
# ===========================================================================


class TestDashboardRegistryMetadata:
    """Dashboard must derive system grouping from DB registry, not only hardcoded config."""

    def test_types_has_registry_cache(self):
        """types.ts must define a registry cache mechanism."""
        src = Path("apps/dashboard/src/lib/types.ts").read_text(encoding="utf-8")
        assert "_registryCache" in src
        assert "setRegistryCache" in src

    def test_types_get_system_checks_registry_first(self):
        """getSystem() must check registry cache before falling back to STRATEGY_CONFIG."""
        src = Path("apps/dashboard/src/lib/types.ts").read_text(encoding="utf-8")
        # Registry check must come before STRATEGY_CONFIG fallback
        registry_pos = src.index("_registryCache")
        config_pos = src.index("STRATEGY_CONFIG[strategyId]?.system")
        assert registry_pos < config_pos, "Registry cache must be checked before hardcoded config"

    def test_types_family_to_system_mapping(self):
        """FAMILY_TO_SYSTEM must map family_id strings to SystemId values."""
        src = Path("apps/dashboard/src/lib/types.ts").read_text(encoding="utf-8")
        assert "FAMILY_TO_SYSTEM" in src
        assert "'swing': 'swing_trader'" in src or "swing: 'swing_trader'" in src
        assert "'momentum': 'momentum_trader'" in src or "momentum: 'momentum_trader'" in src
        assert "'stock': 'stock_trader'" in src or "stock: 'stock_trader'" in src

    def test_live_route_populates_registry(self):
        """Live batch route must query registry and call setRegistryCache."""
        src = Path("apps/dashboard/src/app/api/live/route.ts").read_text(encoding="utf-8")
        assert "setRegistryCache" in src
        assert "family_id" in src

    def test_unknown_strategy_does_not_fallback_to_swing(self):
        """Unknown strategies must be surfaced as unknown, not mislabeled as swing."""
        src = Path("apps/dashboard/src/lib/types.ts").read_text(encoding="utf-8")
        assert "return STRATEGY_CONFIG[strategyId]?.system ?? 'unknown'" in src
        assert "FAMILY_TO_SYSTEM[familyId] ?? 'unknown'" in src


# ===========================================================================
# Fix 13 — TA order/process_quality curated artifacts
# ===========================================================================


class TestTACuratedArtifacts:
    """Trading Assistant must load order + process_quality events and write curated artifacts."""

    HANDLERS_PATH = ASSISTANT_SRC / "trading_assistant/orchestrator/handlers.py"
    BUILDER_PATH = ASSISTANT_SRC / "trading_assistant/skills/build_daily_metrics.py"

    def test_handlers_maps_order_events(self):
        """handlers.py event_type mapping must include 'order' -> 'order_events'."""
        src = self.HANDLERS_PATH.read_text(encoding="utf-8")
        assert '"order": "order_events"' in src or "'order': 'order_events'" in src

    def test_handlers_maps_process_quality_events(self):
        """handlers.py event_type mapping must include 'process_quality' -> 'process_quality_events'."""
        src = self.HANDLERS_PATH.read_text(encoding="utf-8")
        assert '"process_quality": "process_quality_events"' in src or \
               "'process_quality': 'process_quality_events'" in src

    def test_builder_has_order_lifecycle_method(self):
        """DailyMetricsBuilder must have build_order_lifecycle_summary method."""
        src = self.BUILDER_PATH.read_text(encoding="utf-8")
        assert "def build_order_lifecycle_summary" in src

    def test_builder_has_process_quality_method(self):
        """DailyMetricsBuilder must have build_process_quality_summary method."""
        src = self.BUILDER_PATH.read_text(encoding="utf-8")
        assert "def build_process_quality_summary" in src

    def test_write_curated_accepts_order_events(self):
        """write_curated() must accept order_events parameter."""
        src = self.BUILDER_PATH.read_text(encoding="utf-8")
        assert "order_events" in src

    def test_write_curated_accepts_process_quality_events(self):
        """write_curated() must accept process_quality_events parameter."""
        src = self.BUILDER_PATH.read_text(encoding="utf-8")
        assert "process_quality_events" in src


class TestJune2026AuditContracts:
    """Contracts added for the 2026-06-04 live/paper audit."""

    OMS_EVENT_TYPES = [
        "risk_decision",
        "risk_denial",
        "risk_halt",
        "reconciliation_alert",
        "allocation_drift",
        "admin_correction",
        "inferred_fill",
        "allocation_snapshot",
        "position_snapshot",
        "portfolio_snapshot",
        "family_daily_snapshot",
        "deployment",
        "config_snapshot",
    ]

    CURATED_FILES = [
        "risk_decision_summary.json",
        "risk_denial_summary.json",
        "account_gate_summary.json",
        "reconciliation_summary.json",
        "allocation_drift_summary.json",
        "portfolio_state_summary.json",
        "deployment_config_lineage.json",
    ]

    @staticmethod
    def _run_dashboard_ts_export(module_path: str, export_name: str, *args):
        """Execute a pure exported TS helper through the dashboard TypeScript toolchain."""
        ts_module = Path("apps/dashboard/node_modules/typescript")
        if not ts_module.exists():
            pytest.skip("dashboard TypeScript dependency is not installed")

        source = Path(module_path).read_text(encoding="utf-8")
        script = f"""
const path = require('path');
const ts = require(path.resolve('apps/dashboard/node_modules/typescript'));
const source = {json.dumps(source)};
const output = ts.transpileModule(source, {{
  compilerOptions: {{
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2020,
  }},
}}).outputText;
const moduleObj = {{ exports: {{}} }};
new Function('exports', 'require', 'module', 'process', output)(
  moduleObj.exports,
  require,
  moduleObj,
  {{ env: {{}} }},
);
const fn = moduleObj.exports[{json.dumps(export_name)}];
const args = JSON.parse({json.dumps(json.dumps(args))});
const result = fn(...args);
process.stdout.write(JSON.stringify(result));
"""
        script_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".cjs",
                encoding="utf-8",
                delete=False,
            ) as script_file:
                script_file.write(script)
                script_path = Path(script_file.name)
            result = subprocess.run(
                ["node", str(script_path)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=True,
            )
        finally:
            if script_path is not None:
                script_path.unlink(missing_ok=True)
        return json.loads(result.stdout)

    @staticmethod
    def _run_dashboard_env_route(env: dict[str, str]) -> dict:
        """Execute /api/env through transpiled route code with mocked NextResponse."""
        ts_module = Path("apps/dashboard/node_modules/typescript")
        if not ts_module.exists():
            pytest.skip("dashboard TypeScript dependency is not installed")

        active_source = Path("apps/dashboard/src/lib/active-config.ts").read_text(
            encoding="utf-8"
        )
        route_source = Path("apps/dashboard/src/app/api/env/route.ts").read_text(
            encoding="utf-8"
        )
        script = f"""
const path = require('path');
const ts = require(path.resolve('apps/dashboard/node_modules/typescript'));
const env = {json.dumps(env)};
function compile(source) {{
  return ts.transpileModule(source, {{
    compilerOptions: {{
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
    }},
  }}).outputText;
}}
function loadModule(source, processObj, customRequire) {{
  const moduleObj = {{ exports: {{}} }};
  new Function('exports', 'require', 'module', 'process', compile(source))(
    moduleObj.exports,
    customRequire,
    moduleObj,
    processObj,
  );
  return moduleObj.exports;
}}
const processObj = {{ env }};
const activeExports = loadModule({json.dumps(active_source)}, processObj, require);
const routeExports = loadModule(
  {json.dumps(route_source)},
  processObj,
  (specifier) => {{
    if (specifier === 'next/server') {{
      return {{ NextResponse: {{ json: (body, init) => ({{ body, init }}) }} }};
    }}
    if (specifier === '@/lib/active-config') {{
      return activeExports;
    }}
    return require(specifier);
  }},
);
Promise.resolve(routeExports.GET()).then((result) => {{
  process.stdout.write(JSON.stringify(result.body));
}}).catch((err) => {{
  console.error(err);
  process.exit(1);
}});
"""
        script_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".cjs",
                encoding="utf-8",
                delete=False,
            ) as script_file:
                script_file.write(script)
                script_path = Path(script_file.name)
            result = subprocess.run(
                ["node", str(script_path)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=True,
            )
        finally:
            if script_path is not None:
                script_path.unlink(missing_ok=True)
        return json.loads(result.stdout)

    @staticmethod
    def _run_dashboard_api_route(
        route_path: str,
        env: dict[str, str],
        active_rows: list[dict],
        *,
        url: str,
    ) -> dict:
        """Execute a dashboard API route with mocked DB rows."""
        ts_module = Path("apps/dashboard/node_modules/typescript")
        if not ts_module.exists():
            pytest.skip("dashboard TypeScript dependency is not installed")

        active_source = Path("apps/dashboard/src/lib/active-config.ts").read_text(
            encoding="utf-8"
        )
        types_source = Path("apps/dashboard/src/lib/types.ts").read_text(
            encoding="utf-8"
        )
        route_source = Path(route_path).read_text(encoding="utf-8")
        script = f"""
const path = require('path');
const ts = require(path.resolve('apps/dashboard/node_modules/typescript'));
const env = {json.dumps(env)};
const routeUrl = {json.dumps(url)};
const activeRows = {json.dumps(active_rows)};
const strategyRows = [{{
  strategy_id: 'NQ_REGIME',
  mode: 'paper',
  last_heartbeat_ts: '2026-06-05T12:00:00.000Z',
  heartbeat_age_sec: 3,
  health_status: 'OK',
  heat_r: 0.25,
  daily_pnl_r: 0.1,
  last_error: null,
  last_error_ts: null,
  daily_realized_r: 0.1,
  daily_realized_usd: 20,
  open_risk_r: 0.25,
  filled_entries: 1,
  halted: false,
  halt_reason: null,
}}];
function compile(source) {{
  return ts.transpileModule(source, {{
    compilerOptions: {{
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
    }},
  }}).outputText;
}}
function loadModule(source, processObj, customRequire) {{
  const moduleObj = {{ exports: {{}} }};
  new Function('exports', 'require', 'module', 'process', compile(source))(
    moduleObj.exports,
    customRequire,
    moduleObj,
    processObj,
  );
  return moduleObj.exports;
}}
function runtimeConfigRow(row) {{
  return {{
    config_version: '2026-06-04',
    deployment_id: null,
    source_hash: null,
    applied_at: row.applied_at || '2026-06-05T12:00:00.000Z',
    expires_at: row.expires_at === undefined ? null : row.expires_at,
    freshness_status: row.freshness_status || 'fresh',
    ...row,
  }};
}}
function activeConfigQuery(sql, params) {{
  const hasRuntimePredicate = /runtime_env\\s*=\\s*\\$1/.test(sql);
  const hasAccountPredicate = /account_id\\s*=\\s*\\$2/.test(sql);
  let rows = activeRows.map(runtimeConfigRow);
  if (hasRuntimePredicate) {{
    rows = rows.filter(row => row.runtime_env === params[0]);
  }}
  if (hasAccountPredicate) {{
    rows = rows.filter(row => row.account_id === params[1]);
  }}
  if (sql.includes(\"config_scope = 'account'\")) {{
    rows = rows.filter(row => row.config_scope === 'account');
    if (sql.includes('scope_id = $2')) {{
      rows = rows.filter(row => row.scope_id === params[1]);
    }}
  }}
  if (sql.includes(\"config_scope IN ('family', 'strategy')\")) {{
    rows = rows.filter(row => row.config_scope === 'family' || row.config_scope === 'strategy');
  }}
  if (sql.includes('ORDER BY applied_at DESC')) {{
    rows = rows.sort((a, b) => Date.parse(b.applied_at) - Date.parse(a.applied_at));
  }}
  return rows;
}}
async function query(sql, params = []) {{
  if (sql.includes('FROM active_runtime_config')) {{
    return activeConfigQuery(sql, params);
  }}
  if (sql.includes('FROM v_portfolio_daily_summary')) {{
    return [{{
      daily_realized_r: 0.1,
      daily_realized_usd: 20,
      portfolio_open_risk_r: 0.25,
      halted: false,
      halt_reason: null,
    }}];
  }}
  if (sql.includes('COALESCE(SUM(unrealized_pnl)')) {{
    return [{{ unrealized_pnl: 10, heat_r: 0.25 }}];
  }}
  if (sql.includes('FROM v_strategy_health')) {{
    return strategyRows;
  }}
  if (sql.includes('FROM v_adapter_health')) {{
    return [];
  }}
  if (sql.includes('FROM v_active_halts')) {{
    return [];
  }}
  if (sql.includes('FROM positions')) {{
    return [];
  }}
  if (sql.includes('FROM v_today_trades')) {{
    return [];
  }}
  if (sql.includes('FROM v_working_orders')) {{
    return [];
  }}
  if (sql.includes('FROM risk_daily_strategy')) {{
    return [{{ strategy_id: 'NQ_REGIME', family_id: 'momentum' }}];
  }}
  if (sql.includes(\"FROM orders\")) {{
    return [{{ queued_count: 0, oldest_queued_at: null, oldest_queued_age_seconds: null }}];
  }}
  throw new Error(`Unexpected SQL: ${{sql}}`);
}}
const processObj = {{ env }};
const activeExports = loadModule({json.dumps(active_source)}, processObj, require);
const typesExports = loadModule({json.dumps(types_source)}, processObj, require);
const routeExports = loadModule(
  {json.dumps(route_source)},
  processObj,
  (specifier) => {{
    if (specifier === 'next/server') {{
      return {{ NextResponse: {{ json: (body, init) => ({{ body, init }}) }} }};
    }}
    if (specifier === '@/lib/db') {{
      return {{ query }};
    }}
    if (specifier === '@/lib/active-config') {{
      return activeExports;
    }}
    if (specifier === '@/lib/evidence-health') {{
      return {{ getEvidencePipelineHealth: async () => null }};
    }}
    if (specifier === '@/lib/types') {{
      return typesExports;
    }}
    return require(specifier);
  }},
);
Promise.resolve(routeExports.GET({{ url: routeUrl }})).then((result) => {{
  process.stdout.write(JSON.stringify(result.body));
}}).catch((err) => {{
  console.error(err);
  process.exit(1);
}});
"""
        script_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".cjs",
                encoding="utf-8",
                delete=False,
            ) as script_file:
                script_file.write(script)
                script_path = Path(script_file.name)
            result = subprocess.run(
                ["node", str(script_path)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=True,
            )
        finally:
            if script_path is not None:
                script_path.unlink(missing_ok=True)
        return json.loads(result.stdout)

    @staticmethod
    def _add_trading_assistant_src_to_path():
        src = ASSISTANT_SRC.resolve()
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

    def test_runtime_postgres_schema_contains_portfolio_summary_view(self):
        src = Path("libs/oms/persistence/postgres.py").read_text(encoding="utf-8")
        assert "CREATE OR REPLACE VIEW v_portfolio_daily_summary AS" in src
        assert "CREATE TABLE IF NOT EXISTS active_runtime_config" in src
        assert "account_id TEXT NOT NULL DEFAULT ''" in src
        assert "PRIMARY KEY (account_id, config_scope, scope_id, runtime_env)" in src
        assert "idx_active_runtime_config_scope" in src
        assert "CREATE OR REPLACE VIEW v_active_runtime_config AS" in src
        assert '"v_active_runtime_config"' in src
        assert '"v_portfolio_daily_summary"' in src
        assert '"v_strategy_diagnostics"' in src
        assert '"v_daily_strategy_activity"' in src
        assert "_verify_required_views" in src

    def test_active_runtime_config_payload_builders_match_contract(self):
        from libs.runtime.active_config import (
            ActiveRuntimeConfigRecord,
            build_account_runtime_config,
            build_active_runtime_config_artifact,
            build_family_runtime_config,
            build_strategy_runtime_config,
            hash_config_payload,
        )

        account = build_account_runtime_config(
            account_id="DU123",
            heat_cap_R=2.5,
            portfolio_daily_stop_R=3.0,
            portfolio_weekly_stop_R=5.0,
            global_standdown=False,
            account_urd=200.0,
        )
        family = build_family_runtime_config(
            account_id="DU123",
            family_id="momentum",
            family_allocation_pct=0.35,
            family_nav=35_000.0,
            family_heat_cap_R=10.0,
            family_daily_stop_R=3.0,
            family_weekly_stop_R=5.0,
            active_strategy_ids=["NQDTC_v2.1", "NQ_REGIME"],
        )
        strategy = build_strategy_runtime_config(
            account_id="DU123",
            strategy_id="NQ_REGIME",
            family_id="momentum",
            enabled=True,
            live=True,
            allocated_nav=10_000.0,
            unit_risk_dollars=200.0,
            max_heat_R=1.5,
            max_daily_loss_R=2.0,
            max_weekly_loss_R=5.0,
            risk_per_trade=0.01,
            regime_overlays={},
        )

        assert set(account) >= {
            "account_id",
            "heat_cap_R",
            "portfolio_daily_stop_R",
            "portfolio_weekly_stop_R",
            "global_standdown",
            "account_urd",
            "source",
        }
        assert set(family) >= {
            "account_id",
            "family_id",
            "family_allocation_pct",
            "family_nav",
            "family_heat_cap_R",
            "family_daily_stop_R",
            "family_weekly_stop_R",
            "active_strategy_ids",
            "paper_only_filtered",
        }
        assert set(strategy) >= {
            "account_id",
            "strategy_id",
            "family_id",
            "enabled",
            "live",
            "allocated_nav",
            "unit_risk_dollars",
            "max_heat_R",
            "strategy_heat_cap_R",
            "max_daily_loss_R",
            "max_weekly_loss_R",
            "risk_per_trade",
            "regime_overlays",
        }
        assert hash_config_payload({"b": 2, "a": 1}) == hash_config_payload({"a": 1, "b": 2})
        artifact = build_active_runtime_config_artifact(
            [
                ActiveRuntimeConfigRecord(
                    account_id="DU123",
                    config_scope="account",
                    scope_id="DU123",
                    runtime_env="backtest",
                    payload=account,
                )
            ]
        )
        assert artifact["records"][0]["payload"].keys() == account.keys()
        assert artifact["records"][0]["account_id"] == "DU123"
        assert artifact["records"][0]["runtime_env"] == "backtest"

    def test_active_runtime_config_producers_and_dashboard_route_are_wired(self):
        runtime = Path("apps/runtime/runtime.py").read_text(encoding="utf-8")
        factory = Path("libs/oms/services/factory.py").read_text(encoding="utf-8")
        momentum = Path("strategies/momentum/coordinator.py").read_text(encoding="utf-8")
        stock = Path("strategies/stock/coordinator.py").read_text(encoding="utf-8")
        swing = Path("strategies/swing/coordinator.py").read_text(encoding="utf-8")
        route = Path("apps/dashboard/src/app/api/runtime-config/route.ts").read_text(encoding="utf-8")

        assert "build_account_runtime_config" in runtime
        assert "config_scope=\"account\"" in runtime
        assert "account_id=account_id" in runtime
        assert "config_scope=\"oms\"" in factory
        assert "account_id=_adapter_account" in factory
        for src in (momentum, stock, swing):
            assert "build_family_runtime_config" in src
            assert "build_strategy_runtime_config" in src
            assert "config_scope=\"family\"" in src
            assert "config_scope=\"strategy\"" in src
            assert "account_id=active_account_id" in src
        assert "family_daily_stop_R=family_daily_stop_R" in stock
        assert "family_daily_stop_R=_STOCK_DIRECTIONAL_CAP_R" not in stock
        assert "FROM active_runtime_config" in route
        assert "WHERE runtime_env = $1" in route
        assert "AND account_id = $2" in route
        assert "degraded" in route

    def test_dashboard_runtime_env_helper_mirrors_runtime_priority(self):
        helper = "apps/dashboard/src/lib/active-config.ts"

        assert self._run_dashboard_ts_export(
            helper,
            "resolveDashboardRuntimeEnv",
            {"TRADING_MODE": "paper", "TRADING_ENV": "live", "SWING_TRADER_ENV": "dev"},
        ) == "paper"
        assert self._run_dashboard_ts_export(
            helper,
            "resolveDashboardRuntimeEnv",
            {"TRADING_ENV": "live", "SWING_TRADER_ENV": "paper"},
        ) == "live"
        assert self._run_dashboard_ts_export(
            helper,
            "resolveDashboardRuntimeEnv",
            {"SWING_TRADER_ENV": "paper", "ALGO_TRADER_ENV": "live"},
        ) == "paper"
        assert self._run_dashboard_ts_export(
            helper,
            "resolveDashboardRuntimeEnv",
            {"ALGO_TRADER_ENV": "live", "STOCK_TRADER_ENV": "paper"},
        ) == "live"
        assert self._run_dashboard_ts_export(
            helper,
            "resolveDashboardRuntimeEnv",
            {"STOCK_TRADER_ENV": "paper"},
        ) == "paper"
        assert self._run_dashboard_ts_export(
            helper,
            "resolveDashboardRuntimeEnv",
            {"RUNTIME_ENV": "live", "IB_PORT": "4001"},
        ) == "dev"
        assert self._run_dashboard_ts_export(
            helper,
            "resolveDashboardRuntimeEnv",
            {"DASHBOARD_RUNTIME_ENV": "live"},
        ) == "dev"

        env_route = self._run_dashboard_env_route({
            "TRADING_MODE": "live",
            "SWING_TRADER_ENV": "paper",
            "IB_PORT": "4002",
            "DASHBOARD_ACCOUNT_ID": "LIVE123",
        })
        assert env_route["mode"] == "live"
        assert env_route["account_id"] == "LIVE123"
        assert env_route["ib_port"] == 4002

    def test_dashboard_evidence_health_evaluator_surfaces_degradation(self):
        helper = "apps/dashboard/src/lib/evidence-health.ts"

        relay_down = self._run_dashboard_ts_export(
            helper,
            "evaluateEvidencePipelineHealth",
            {
                "relayUrl": "http://127.0.0.1:8000/health",
                "relayReachable": False,
                "relayError": "connection refused",
                "requiredBotIds": ["momentum"],
                "now": "2026-06-05T12:00:00.000Z",
            },
        )
        assert relay_down["status"] == "ERROR"
        assert relay_down["relay"]["status"] == "ERROR"
        assert relay_down["assistant"]["status"] == "UNKNOWN"

        missing_bot = self._run_dashboard_ts_export(
            helper,
            "evaluateEvidencePipelineHealth",
            {
                "relayUrl": "http://127.0.0.1:8000/health",
                "relayReachable": True,
                "relayPayload": {
                    "status": "ok",
                    "pending_events": 0,
                    "oldest_pending_age_seconds": 0,
                    "per_bot_pending": {},
                    "last_event_per_bot": {"momentum": "2026-06-05T11:59:00.000Z"},
                },
                "requiredBotIds": ["momentum", "stock"],
                "now": "2026-06-05T12:00:00.000Z",
            },
        )
        assert missing_bot["status"] == "ERROR"
        assert missing_bot["assistant"]["missing_bot_ids"] == ["stock"]

        backlog_and_stale = self._run_dashboard_ts_export(
            helper,
            "evaluateEvidencePipelineHealth",
            {
                "relayUrl": "http://127.0.0.1:8000/health",
                "relayReachable": True,
                "relayPayload": {
                    "status": "ok",
                    "pending_events": 501,
                    "oldest_pending_age_seconds": 700,
                    "per_bot_pending": {"momentum": 501},
                    "last_event_per_bot": {"momentum": "2026-06-05T11:45:00.000Z"},
                },
                "requiredBotIds": ["momentum"],
                "now": "2026-06-05T12:00:00.000Z",
                "backlogThreshold": 500,
                "relayStalePendingSeconds": 600,
                "assistantStaleEventSeconds": 600,
            },
        )
        assert backlog_and_stale["status"] == "WARNING"
        assert backlog_and_stale["relay"]["status"] == "WARNING"
        assert backlog_and_stale["assistant"]["stale_bot_ids"] == ["momentum"]

    def test_dashboard_health_routes_and_ui_surface_evidence_pipeline(self):
        health_route = Path("apps/dashboard/src/app/api/health/route.ts").read_text(
            encoding="utf-8"
        )
        live_route = Path("apps/dashboard/src/app/api/live/route.ts").read_text(
            encoding="utf-8"
        )
        types = Path("apps/dashboard/src/lib/types.ts").read_text(encoding="utf-8")
        health_component = Path("apps/dashboard/src/components/SystemHealth.tsx").read_text(
            encoding="utf-8"
        )

        assert "getEvidencePipelineHealth" in health_route
        assert "getEvidencePipelineHealth" in live_route
        assert "evidence: EvidencePipelineHealth | null" in types
        assert "Evidence Pipeline" in health_component
        assert "missing_bot_ids" in health_component

    def test_dashboard_active_config_api_behavior_scopes_runtime_env_and_account(self):
        active_rows = [
            {
                "account_id": "DU123",
                "config_scope": "account",
                "scope_id": "DU123",
                "runtime_env": "paper",
                "payload": {
                    "account_id": "DU123",
                    "heat_cap_R": 2.5,
                    "portfolio_daily_stop_R": 3.0,
                    "portfolio_weekly_stop_R": 5.0,
                    "global_standdown": False,
                    "account_urd": 200.0,
                },
            },
            {
                "account_id": "DU123",
                "config_scope": "family",
                "scope_id": "momentum",
                "runtime_env": "paper",
                "payload": {
                    "account_id": "DU123",
                    "family_id": "momentum",
                    "family_heat_cap_R": 10.0,
                    "family_daily_stop_R": 3.0,
                    "family_weekly_stop_R": 5.0,
                    "active_strategy_ids": ["NQ_REGIME"],
                },
            },
            {
                "account_id": "DU123",
                "config_scope": "strategy",
                "scope_id": "NQ_REGIME",
                "runtime_env": "paper",
                "payload": {
                    "account_id": "DU123",
                    "strategy_id": "NQ_REGIME",
                    "family_id": "momentum",
                    "risk_per_trade": 0.01,
                    "max_heat_R": 1.5,
                    "max_daily_loss_R": 2.0,
                    "max_weekly_loss_R": 5.0,
                    "allocated_nav": 10_000.0,
                    "unit_risk_dollars": 200.0,
                },
            },
            {
                "account_id": "DU999",
                "config_scope": "account",
                "scope_id": "DU999",
                "runtime_env": "paper",
                "applied_at": "2026-06-05T12:05:00.000Z",
                "payload": {
                    "account_id": "DU999",
                    "heat_cap_R": 99.0,
                    "portfolio_daily_stop_R": 99.0,
                    "portfolio_weekly_stop_R": 99.0,
                    "global_standdown": True,
                    "account_urd": 999.0,
                },
            },
            {
                "account_id": "DU999",
                "config_scope": "family",
                "scope_id": "momentum",
                "runtime_env": "paper",
                "applied_at": "2026-06-05T12:05:00.000Z",
                "payload": {
                    "account_id": "DU999",
                    "family_id": "momentum",
                    "family_heat_cap_R": 99.0,
                    "family_daily_stop_R": 99.0,
                    "family_weekly_stop_R": 99.0,
                    "active_strategy_ids": ["NQ_REGIME"],
                },
            },
            {
                "account_id": "DU999",
                "config_scope": "strategy",
                "scope_id": "NQ_REGIME",
                "runtime_env": "paper",
                "applied_at": "2026-06-05T12:05:00.000Z",
                "payload": {
                    "account_id": "DU999",
                    "strategy_id": "NQ_REGIME",
                    "family_id": "momentum",
                    "risk_per_trade": 0.99,
                    "max_heat_R": 99.0,
                    "max_daily_loss_R": 99.0,
                    "max_weekly_loss_R": 99.0,
                    "allocated_nav": 999_000.0,
                    "unit_risk_dollars": 999.0,
                },
            },
            {
                "account_id": "DU123",
                "config_scope": "account",
                "scope_id": "DU123",
                "runtime_env": "live",
                "applied_at": "2026-06-05T12:10:00.000Z",
                "payload": {
                    "account_id": "DU123",
                    "heat_cap_R": 77.0,
                    "portfolio_daily_stop_R": 77.0,
                    "portfolio_weekly_stop_R": 77.0,
                    "global_standdown": False,
                    "account_urd": 777.0,
                },
            },
        ]
        env = {"TRADING_MODE": "paper", "DASHBOARD_ACCOUNT_ID": "DU123"}

        live = self._run_dashboard_api_route(
            "apps/dashboard/src/app/api/live/route.ts",
            env,
            active_rows,
            url="http://localhost/api/live",
        )
        assert live["portfolio"]["heat_cap_R"] == pytest.approx(2.5)
        assert live["portfolio"]["portfolio_daily_stop_R"] == pytest.approx(3.0)
        assert live["portfolio"]["portfolio_weekly_stop_R"] == pytest.approx(5.0)
        assert live["portfolio"]["global_standdown"] is False
        assert live["portfolio"]["active_config_status"] == "fresh"
        assert live["strategies"][0]["active_risk_per_trade"] == pytest.approx(0.01)
        assert live["strategies"][0]["active_max_heat_R"] == pytest.approx(1.5)
        assert live["systemPnl"][0]["active_heat_cap_R"] == pytest.approx(10.0)

        portfolio = self._run_dashboard_api_route(
            "apps/dashboard/src/app/api/portfolio/route.ts",
            env,
            active_rows,
            url="http://localhost/api/portfolio",
        )
        assert portfolio["heat_cap_R"] == pytest.approx(2.5)
        assert portfolio["portfolio_daily_stop_R"] == pytest.approx(3.0)
        assert portfolio["portfolio_weekly_stop_R"] == pytest.approx(5.0)
        assert portfolio["global_standdown"] is False
        assert portfolio["active_config_status"] == "fresh"

        runtime_config = self._run_dashboard_api_route(
            "apps/dashboard/src/app/api/runtime-config/route.ts",
            env,
            active_rows,
            url="http://localhost/api/runtime-config",
        )
        assert runtime_config["status"] == "ok"
        assert {
            (row["account_id"], row["runtime_env"])
            for row in runtime_config["records"]
        } == {("DU123", "paper")}
        assert all(row["payload"]["account_id"] == "DU123" for row in runtime_config["records"])

    def test_dashboard_active_config_api_degrades_missing_account_required_keys(self):
        active_rows = [
            {
                "account_id": "DU123",
                "config_scope": "account",
                "scope_id": "DU123",
                "runtime_env": "paper",
                "payload": {
                    "account_id": "DU123",
                    "heat_cap_R": 2.5,
                    "account_urd": 200.0,
                },
            },
            {
                "account_id": "DU123",
                "config_scope": "family",
                "scope_id": "momentum",
                "runtime_env": "paper",
                "payload": {
                    "account_id": "DU123",
                    "family_id": "momentum",
                    "family_heat_cap_R": 10.0,
                    "family_daily_stop_R": 3.0,
                    "family_weekly_stop_R": 5.0,
                    "active_strategy_ids": ["NQ_REGIME"],
                },
            },
            {
                "account_id": "DU123",
                "config_scope": "strategy",
                "scope_id": "NQ_REGIME",
                "runtime_env": "paper",
                "payload": {
                    "account_id": "DU123",
                    "strategy_id": "NQ_REGIME",
                    "family_id": "momentum",
                    "risk_per_trade": 0.01,
                    "max_heat_R": 1.5,
                    "max_daily_loss_R": 2.0,
                    "max_weekly_loss_R": 5.0,
                },
            },
        ]
        env = {"TRADING_MODE": "paper", "DASHBOARD_ACCOUNT_ID": "DU123"}

        live = self._run_dashboard_api_route(
            "apps/dashboard/src/app/api/live/route.ts",
            env,
            active_rows,
            url="http://localhost/api/live",
        )
        assert live["portfolio"]["active_config_status"] == "missing"
        assert "missing account daily stop" in live["portfolio"]["active_config_warnings"]
        assert "missing account weekly stop" in live["portfolio"]["active_config_warnings"]
        assert "missing account global stand-down" in live["portfolio"]["active_config_warnings"]

        portfolio = self._run_dashboard_api_route(
            "apps/dashboard/src/app/api/portfolio/route.ts",
            env,
            active_rows,
            url="http://localhost/api/portfolio",
        )
        assert portfolio["active_config_status"] == "missing"
        assert "missing account daily stop" in portfolio["active_config_warnings"]
        assert "missing account weekly stop" in portfolio["active_config_warnings"]
        assert "missing account global stand-down" in portfolio["active_config_warnings"]

        runtime_config = self._run_dashboard_api_route(
            "apps/dashboard/src/app/api/runtime-config/route.ts",
            env,
            active_rows,
            url="http://localhost/api/runtime-config",
        )
        assert runtime_config["status"] == "degraded"
        assert any("missing account daily stop" in warning for warning in runtime_config["warnings"])
        assert any("missing account weekly stop" in warning for warning in runtime_config["warnings"])
        assert any("missing account global stand-down" in warning for warning in runtime_config["warnings"])

    def test_dashboard_active_config_queries_are_scoped_and_do_not_use_heat_cap_constant(self):
        live = Path("apps/dashboard/src/app/api/live/route.ts").read_text(encoding="utf-8")
        portfolio = Path("apps/dashboard/src/app/api/portfolio/route.ts").read_text(encoding="utf-8")
        runtime_config = Path("apps/dashboard/src/app/api/runtime-config/route.ts").read_text(
            encoding="utf-8"
        )
        header = Path("apps/dashboard/src/components/PortfolioHeader.tsx").read_text(
            encoding="utf-8"
        )
        strategy_card = Path("apps/dashboard/src/components/StrategyCard.tsx").read_text(
            encoding="utf-8"
        )
        system_group = Path("apps/dashboard/src/components/SystemGroup.tsx").read_text(
            encoding="utf-8"
        )
        types = Path("apps/dashboard/src/lib/types.ts").read_text(encoding="utf-8")

        for src in (live, portfolio):
            assert "FROM active_runtime_config" in src
            assert "runtime_env = $1" in src
            assert "account_id = $2" in src
            assert "scope_id = $2" in src
            assert "FROM v_active_runtime_config" not in src

        assert "FROM active_runtime_config" in runtime_config
        assert "account_id = $2" in runtime_config
        assert "FROM v_active_runtime_config" not in runtime_config
        assert "expires_at IS NOT NULL AND expires_at <= now()" in runtime_config
        assert "DEFAULT_PORTFOLIO_HEAT_CAP" not in header
        assert "DEFAULT_PORTFOLIO_HEAT_CAP" not in types
        assert "portfolio_daily_stop_R" in header
        assert "portfolio_weekly_stop_R" in header
        assert "global_standdown" in header
        assert "Daily Stop" in header
        assert "Weekly Stop" in header
        assert "Global Stand-down" in header
        assert "STRATEGY_CONFIG" not in strategy_card
        assert "STRATEGY_CONFIG" not in system_group
        assert "riskPct" not in types
        assert "maxHeatR" not in types
        assert "dailyStopR" not in types
        assert "active_risk_per_trade" in types
        assert "active_max_heat_R" in types
        assert "active_heat_cap_R" in types
        assert "config_scope IN ('family', 'strategy')" in live

    def test_active_runtime_config_backtest_artifact_writer(self, tmp_path):
        from libs.runtime.active_config import (
            ActiveRuntimeConfigRecord,
            build_account_runtime_config,
            write_active_runtime_config_artifact,
        )

        path = write_active_runtime_config_artifact(
            tmp_path,
            [
                ActiveRuntimeConfigRecord(
                    account_id="DU123",
                    config_scope="account",
                    scope_id="DU123",
                    runtime_env="backtest",
                    payload=build_account_runtime_config(
                        account_id="DU123",
                        heat_cap_R=2.5,
                        portfolio_daily_stop_R=3.0,
                        portfolio_weekly_stop_R=5.0,
                        global_standdown=False,
                        account_urd=200.0,
                    ),
                )
            ],
        )

        assert path.name == "active_runtime_config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["records"][0]["account_id"] == "DU123"
        assert payload["records"][0]["config_scope"] == "account"
        assert set(payload["records"][0]["payload"]) >= {"account_id", "heat_cap_R"}

    def test_assistant_brain_routes_forwarded_oms_events(self):
        src = (ASSISTANT_SRC / "trading_assistant/orchestrator/orchestrator_brain.py").read_text(
            encoding="utf-8"
        )
        for event_type in self.OMS_EVENT_TYPES:
            assert f'"{event_type}": _handle_oms_portfolio_event' in src

    def test_assistant_daily_rebuild_loads_forwarded_oms_events(self):
        src = (ASSISTANT_SRC / "trading_assistant/orchestrator/handlers.py").read_text(
            encoding="utf-8"
        )
        for event_type in self.OMS_EVENT_TYPES:
            assert f'"{event_type}"' in src

    def test_assistant_builder_and_prompt_include_oms_curated_files(self):
        builder = (ASSISTANT_SRC / "trading_assistant/skills/build_daily_metrics.py").read_text(
            encoding="utf-8"
        )
        prompt = (ASSISTANT_SRC / "trading_assistant/analysis/prompt_assembler.py").read_text(
            encoding="utf-8"
        )
        triage = (ASSISTANT_SRC / "trading_assistant/analysis/daily_triage.py").read_text(
            encoding="utf-8"
        )
        for filename in self.CURATED_FILES:
            assert filename in builder
            assert filename in prompt
            assert filename in triage

    def test_assistant_oms_event_ingestion_writes_raw_curated_and_prompt_visible_artifacts(
        self,
        tmp_path,
    ):
        self._add_trading_assistant_src_to_path()

        from trading_assistant.analysis.daily_triage import DailyTriage
        from trading_assistant.analysis import prompt_assembler
        from trading_assistant.orchestrator.adapters.vps_receiver import VPSReceiver
        from trading_assistant.orchestrator.db.queue import EventQueue
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain
        from trading_assistant.orchestrator.task_registry import TaskRegistry
        from trading_assistant.orchestrator.worker import Worker

        date = "2026-06-05"
        bot_id = "momentum"
        raw_dir = tmp_path / "raw"
        curated_dir = tmp_path / "curated"
        memory_dir = tmp_path / "memory"
        runs_dir = tmp_path / "runs"
        source_root = tmp_path / "src"
        for path in (raw_dir, curated_dir, memory_dir, runs_dir, source_root):
            path.mkdir(parents=True, exist_ok=True)

        event_payloads = {
            "risk_decision": {
                "date": date,
                "decision": "approved",
                "account_gate": "account_heat",
                "strategy_id": "NQ_REGIME",
                "symbol": "NQ",
            },
            "risk_denial": {
                "date": date,
                "reason": "account_daily_stop",
                "gateway_gate": "account",
                "strategy_id": "NQ_REGIME",
                "symbol": "NQ",
            },
            "risk_halt": {
                "date": date,
                "halt_reason": "account global_standdown",
                "source": "account",
                "strategy_id": "NQ_REGIME",
            },
            "reconciliation_alert": {
                "date": date,
                "status": "open",
                "discrepancies": [{"type": "missing_broker_fill"}],
            },
            "allocation_drift": {
                "date": date,
                "status": "exceeds_threshold",
                "drift_pct": 0.12,
            },
            "admin_correction": {
                "date": date,
                "action": "manual_repair",
                "status": "applied",
            },
            "inferred_fill": {
                "date": date,
                "lifecycle_action": "inferred_fill",
                "status": "inferred",
            },
            "allocation_snapshot": {
                "date": date,
                "family_id": "momentum",
                "allocation_pct": 0.35,
            },
            "position_snapshot": {
                "date": date,
                "positions": [{"symbol": "NQ", "qty": 1}],
            },
            "portfolio_snapshot": {
                "date": date,
                "heat_r": 1.1,
                "cash": 100000,
            },
            "family_daily_snapshot": {
                "date": date,
                "family_id": "momentum",
                "daily_pnl_r": 0.4,
            },
            "deployment": {
                "date": date,
                "deployment_id": "dep-1",
                "strategy_version": "v1",
            },
            "config_snapshot": {
                "date": date,
                "deployment_id": "dep-1",
                "config_version": "cfg-1",
            },
        }
        assert set(event_payloads) == set(self.OMS_EVENT_TYPES)

        relay_events = []
        for event_type, payload in event_payloads.items():
            relay_events.append({
                "event_id": f"oms-{event_type}",
                "bot_id": bot_id,
                "event_type": event_type,
                "exchange_timestamp": f"{date}T15:00:00Z",
                "payload": payload,
            })

        async def _ingest_through_receiver_brain_and_worker() -> tuple[int, int, int]:
            queue = EventQueue(db_path=str(tmp_path / "events.db"))
            registry = TaskRegistry(db_path=str(tmp_path / "tasks.db"))
            await queue.initialize()
            await registry.initialize()
            try:
                receiver = VPSReceiver(relay_url="http://relay", local_queue=queue)
                inserted, _event_ids = await receiver._store_events(relay_events)
                worker = Worker(
                    queue=queue,
                    registry=registry,
                    brain=OrchestratorBrain(),
                    raw_data_dir=raw_dir,
                )
                processed = await worker.process_batch(limit=len(relay_events))
                pending = await queue.count_pending()
                return inserted, processed, pending
            finally:
                await queue.close()
                await registry.close()

        inserted, processed, pending = asyncio.run(_ingest_through_receiver_brain_and_worker())
        assert inserted == len(self.OMS_EVENT_TYPES)
        assert processed == len(self.OMS_EVENT_TYPES)
        assert pending == 0

        for event_type in self.OMS_EVENT_TYPES:
            raw_path = raw_dir / date / bot_id / f"{event_type}.jsonl"
            assert raw_path.exists(), f"missing raw daily file for {event_type}"
            raw_record = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
            assert raw_record["event_type"] == event_type
            assert raw_record["date"] == date
            assert raw_record["bot_id"] == bot_id

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=MagicMock(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            source_root=source_root,
            bots=[bot_id],
            raw_data_dir=raw_dir,
        )
        handlers._rebuild_daily_curated_from_raw(date, [bot_id])

        bot_curated = curated_dir / date / bot_id
        for filename in self.CURATED_FILES:
            assert (bot_curated / filename).exists(), f"missing curated {filename}"

        risk_decisions = json.loads((bot_curated / "risk_decision_summary.json").read_text())
        assert risk_decisions["risk_decision_count"] == 1
        assert risk_decisions["risk_denial_count"] == 1
        assert risk_decisions["risk_halt_count"] == 1

        risk_denials = json.loads((bot_curated / "risk_denial_summary.json").read_text())
        assert risk_denials["total_denials"] == 1
        assert risk_denials["reason_counts"]["account_daily_stop"] == 1

        account_gate = json.loads((bot_curated / "account_gate_summary.json").read_text())
        assert account_gate["account_gate_event_count"] >= 1

        reconciliation = json.loads((bot_curated / "reconciliation_summary.json").read_text())
        assert reconciliation["event_counts"]["reconciliation_alert"] == 1
        assert reconciliation["event_counts"]["admin_correction"] == 1
        assert reconciliation["event_counts"]["inferred_fill"] == 1

        allocation = json.loads((bot_curated / "allocation_drift_summary.json").read_text())
        assert allocation["total_events"] == 1
        assert allocation["max_abs_drift_pct"] == 0.12

        portfolio_state = json.loads((bot_curated / "portfolio_state_summary.json").read_text())
        assert portfolio_state["event_counts"]["allocation_snapshot"] == 1
        assert portfolio_state["event_counts"]["position_snapshot"] == 1
        assert portfolio_state["event_counts"]["portfolio_snapshot"] == 1
        assert portfolio_state["event_counts"]["family_daily_snapshot"] == 1

        lineage = json.loads((bot_curated / "deployment_config_lineage.json").read_text())
        assert lineage["deployment_event_count"] == 1
        assert lineage["config_snapshot_count"] == 1
        assert lineage["deployment_ids"] == ["dep-1"]
        assert lineage["config_versions"] == ["cfg-1"]

        for filename in self.CURATED_FILES:
            assert filename in prompt_assembler._CURATED_FILES
            assert filename in DailyTriage._ALWAYS_VISIBLE_KEYS

    def test_capital_bootstrap_supports_live_filter(self):
        src = Path("libs/config/capital_bootstrap.py").read_text(encoding="utf-8")
        assert "live: bool = False" in src
        assert "registry.enabled_strategies(live=live)" in src

    def test_runtime_exposes_explicit_no_instrumentation_bypass(self):
        cli = Path("apps/runtime/cli.py").read_text(encoding="utf-8")
        runtime = Path("apps/runtime/runtime.py").read_text(encoding="utf-8")
        contracts = Path("strategies/contracts.py").read_text(encoding="utf-8")
        assert "--allow-no-instrumentation" in cli
        assert "allow_no_instrumentation" in runtime
        assert "require_instrumentation" in contracts

    def test_runtime_required_instrumentation_checks_relay_and_hmac(self, tmp_path, monkeypatch):
        from apps.runtime.runtime import RuntimeShell

        monkeypatch.delenv("INSTRUMENTATION_HMAC_SECRET", raising=False)
        shell = RuntimeShell(config_dir="config")
        checks = shell._instrumentation_readiness_checks(
            "stock",
            {
                "data_dir": str(tmp_path / "data"),
                "sidecar": {
                    "relay_url": "",
                    "hmac_secret_env": "INSTRUMENTATION_HMAC_SECRET",
                    "buffer_dir": str(tmp_path / "buffer"),
                },
            },
            require_instrumentation=True,
        )

        by_name = {check.name: check for check in checks}
        assert not by_name["instrumentation-relay:stock"].ok
        assert not by_name["instrumentation-hmac:stock"].ok
        assert by_name["instrumentation-data-dir:stock"].ok
        assert by_name["instrumentation-buffer-dir:stock"].ok

    def test_runtime_dev_instrumentation_checks_warn_not_fail(self, tmp_path, monkeypatch):
        from apps.runtime.runtime import RuntimeShell

        monkeypatch.delenv("INSTRUMENTATION_HMAC_SECRET", raising=False)
        shell = RuntimeShell(config_dir="config")
        checks = shell._instrumentation_readiness_checks(
            "swing",
            {
                "data_dir": str(tmp_path / "data"),
                "sidecar": {
                    "relay_url": "",
                    "hmac_secret_env": "INSTRUMENTATION_HMAC_SECRET",
                    "buffer_dir": str(tmp_path / "buffer"),
                },
            },
            require_instrumentation=False,
        )

        assert all(check.ok for check in checks)


# ===========================================================================
# Apr13RT-14 — Runtime async preflight checks
# ===========================================================================


class TestAsyncPreflight:
    """Tests for RuntimeShell._run_async_preflight()."""

    def _make_shell(self):
        from apps.runtime.runtime import RuntimeShell
        shell = RuntimeShell(config_dir="config")
        shell.registry = MagicMock()
        shell.registry.connection_groups = {}
        return shell

    @pytest.mark.asyncio
    async def test_preflight_catches_bad_coordinator_import(self):
        """Bad coordinator dotted path -> import check fails."""
        from apps.runtime import runtime as rt_mod

        shell = self._make_shell()
        original = rt_mod._FAMILY_COORDINATORS.copy()
        rt_mod._FAMILY_COORDINATORS["bad_family"] = "does.not.exist.BadCoordinator"
        try:
            checks = await shell._run_async_preflight(
                connect_ib=False,
                families={"bad_family"},
            )
            import_checks = [c for c in checks if c.name == "import:bad_family"]
            assert len(import_checks) == 1
            assert not import_checks[0].ok
        finally:
            rt_mod._FAMILY_COORDINATORS.clear()
            rt_mod._FAMILY_COORDINATORS.update(original)

    @pytest.mark.asyncio
    async def test_preflight_db_unreachable(self):
        """Mock asyncpg.connect to raise OSError -> database check fails."""
        shell = self._make_shell()

        mock_db_config = MagicMock()
        mock_db_config.to_dsn.return_value = "postgresql://localhost/test"

        with patch("libs.oms.persistence.db_config.DBConfig.from_env", return_value=mock_db_config), \
             patch("asyncpg.connect", side_effect=OSError("Connection refused")):
            checks = await shell._run_async_preflight(
                connect_ib=False,
                families=set(),
            )
            db_checks = [c for c in checks if c.name == "database"]
            assert len(db_checks) == 1
            assert not db_checks[0].ok

    @pytest.mark.asyncio
    async def test_preflight_ib_unreachable(self):
        """Mock asyncio.open_connection to raise -> ib-gateway check fails."""
        shell = self._make_shell()
        group_cfg = MagicMock()
        group_cfg.host = "127.0.0.1"
        group_cfg.port = 4002
        shell.registry.connection_groups = {"default": group_cfg}

        with patch("asyncio.open_connection",
                   side_effect=ConnectionRefusedError("refused")):
            checks = await shell._run_async_preflight(
                connect_ib=True,
                families=set(),
            )
            gw_checks = [c for c in checks if c.name.startswith("ib-gateway:")]
            assert len(gw_checks) == 1
            assert not gw_checks[0].ok

    @pytest.mark.asyncio
    async def test_preflight_flags_unresolved_stock_account_id(self):
        shell = self._make_shell()

        with patch("apps.runtime.runtime.get_environment", return_value="dev"), patch(
            "apps.runtime.runtime.validate_stock_readiness",
            return_value=(
                {},
                [
                    MagicMock(
                        check_name="stock-account-config:default",
                        detail="account_id is unresolved placeholder ${IB_ACCOUNT_ID}",
                    )
                ],
            ),
        ):
            checks = await shell._run_async_preflight(
                connect_ib=False,
                families={"stock"},
            )

        stock_checks = [c for c in checks if c.name == "stock-account-config:default"]
        assert len(stock_checks) == 1
        assert not stock_checks[0].ok

    @pytest.mark.asyncio
    async def test_preflight_flags_missing_stock_artifact(self):
        shell = self._make_shell()

        with patch("apps.runtime.runtime.get_environment", return_value="dev"), patch(
            "apps.runtime.runtime.validate_stock_readiness",
            return_value=(
                {},
                [
                    MagicMock(
                        check_name="stock-artifact-readiness:IARIC_v1",
                        detail="watchlist unavailable for 2026-04-24: missing file",
                    )
                ],
            ),
        ):
            checks = await shell._run_async_preflight(
                connect_ib=False,
                families={"stock"},
            )

        stock_checks = [c for c in checks if c.name == "stock-artifact-readiness:IARIC_v1"]
        assert len(stock_checks) == 1
        assert not stock_checks[0].ok


# ===========================================================================
# Apr10-7 / Apr13RT-12 — Paper equity scoping
# ===========================================================================


class TestPaperEquityScoping:
    """Tests for paper equity parameter wiring in multi-strategy OMS."""

    def test_multi_oms_signature_has_paper_equity_params(self):
        """build_multi_strategy_oms must accept paper_equity_pool, scope, initial."""
        from libs.oms.services.factory import build_multi_strategy_oms
        sig = inspect.signature(build_multi_strategy_oms)
        params = sig.parameters
        assert "paper_equity_pool" in params
        assert "paper_equity_scope" in params
        assert "paper_initial_equity" in params

    def test_wire_callbacks_multi_signature_has_paper_equity_params(self):
        """_wire_adapter_callbacks_multi must accept paper equity params."""
        from libs.oms.services.factory import _wire_adapter_callbacks_multi
        sig = inspect.signature(_wire_adapter_callbacks_multi)
        params = sig.parameters
        assert "paper_equity_pool" in params
        assert "paper_equity_scope" in params
        assert "paper_initial_equity" in params

    def test_swing_coordinator_uses_paper_equity(self):
        """Swing coordinator source must reference PaperEquityManager and paper_equity_pool."""
        src_path = Path("strategies/swing/coordinator.py")
        text = src_path.read_text(encoding="utf-8")
        assert "paper_equity_pool" in text
        assert "PaperEquityManager" in text
