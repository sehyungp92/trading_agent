"""SQLite-backed live order-management state."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crypto_trader.core.models import Fill, Side
from crypto_trader.core.runtime_types import ExecutionReport

FILL_STATUS_RECEIVED = "RECEIVED"
FILL_STATUS_UNRESOLVED = "UNRESOLVED"
FILL_STATUS_PROCESSING_FAILED = "PROCESSING_FAILED"
FILL_STATUS_STRATEGY_DISPATCHED = "STRATEGY_DISPATCHED"
FILL_STATUS_DISPATCHED = FILL_STATUS_STRATEGY_DISPATCHED
FILL_STATUS_COORDINATOR_APPLIED = "COORDINATOR_APPLIED"
FILL_STATUS_LIFECYCLE_APPLIED = "LIFECYCLE_APPLIED"
FILL_STATUS_FINALIZED = "FINALIZED"
FILL_STATUS_PROCESSED = "PROCESSED"
_LEGACY_FILL_STATUS_DISPATCHED = "DISPATCHED"
FILL_STRATEGY_DISPATCHED_STATUSES = frozenset({
    FILL_STATUS_STRATEGY_DISPATCHED,
    FILL_STATUS_COORDINATOR_APPLIED,
    FILL_STATUS_LIFECYCLE_APPLIED,
    FILL_STATUS_FINALIZED,
    FILL_STATUS_PROCESSED,
    _LEGACY_FILL_STATUS_DISPATCHED,
})
FILL_COORDINATOR_APPLIED_STATUSES = frozenset({
    FILL_STATUS_COORDINATOR_APPLIED,
    FILL_STATUS_LIFECYCLE_APPLIED,
    FILL_STATUS_FINALIZED,
    FILL_STATUS_PROCESSED,
})
FILL_LIFECYCLE_APPLIED_STATUSES = frozenset({
    FILL_STATUS_LIFECYCLE_APPLIED,
    FILL_STATUS_FINALIZED,
    FILL_STATUS_PROCESSED,
})
FILL_FINALIZED_STATUSES = frozenset({
    FILL_STATUS_FINALIZED,
    FILL_STATUS_PROCESSED,
})
FILL_STATUS_ORDER = {
    FILL_STATUS_RECEIVED: 10,
    FILL_STATUS_UNRESOLVED: 10,
    FILL_STATUS_PROCESSING_FAILED: 10,
    FILL_STATUS_STRATEGY_DISPATCHED: 20,
    _LEGACY_FILL_STATUS_DISPATCHED: 20,
    FILL_STATUS_COORDINATOR_APPLIED: 30,
    FILL_STATUS_LIFECYCLE_APPLIED: 40,
    FILL_STATUS_FINALIZED: 45,
    FILL_STATUS_PROCESSED: 50,
}


class OmsStore:
    """Durable OMS records for restart-safe order/fill ownership."""

    def __init__(self, state_dir: Path | str) -> None:
        path = Path(state_dir)
        if path.suffix:
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            self.path = path / "live_oms.sqlite3"
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def upsert_order(
        self,
        *,
        client_order_id: str,
        strategy_id: str,
        symbol: str,
        side: str,
        status: str,
        exchange_order_id: str = "",
        order_type: str = "",
        role: str = "",
        decision_id: str = "",
        position_instance_id: str = "",
        reduce_only: bool = False,
        oca_group: str | None = None,
        bracket_group: str | None = None,
        metadata: dict[str, Any] | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        ts = _iso(updated_at)
        self._conn.execute(
            """
            INSERT INTO orders (
                client_order_id, exchange_order_id, strategy_id, symbol, side,
                order_type, status, role, decision_id, position_instance_id,
                reduce_only, oca_group, bracket_group, metadata, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_order_id) DO UPDATE SET
                exchange_order_id=excluded.exchange_order_id,
                strategy_id=excluded.strategy_id,
                symbol=excluded.symbol,
                side=excluded.side,
                order_type=excluded.order_type,
                status=excluded.status,
                role=excluded.role,
                decision_id=excluded.decision_id,
                position_instance_id=excluded.position_instance_id,
                reduce_only=excluded.reduce_only,
                oca_group=excluded.oca_group,
                bracket_group=excluded.bracket_group,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (
                client_order_id,
                exchange_order_id,
                strategy_id,
                symbol,
                side,
                order_type,
                status,
                role,
                decision_id,
                position_instance_id,
                int(reduce_only),
                oca_group,
                bracket_group,
                json.dumps(metadata or {}, sort_keys=True),
                ts,
            ),
        )
        self._conn.commit()

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id=? OR exchange_order_id=?",
            (order_id, order_id),
        ).fetchone()
        return _row_to_dict(row)

    def get_strategy_for_order(self, order_id: str) -> str | None:
        """Return durable order ownership for any known client or exchange id."""
        record = self.get_order(order_id)
        if record is None:
            return None
        strategy_id = record.get("strategy_id")
        return str(strategy_id) if strategy_id else None

    def list_orders(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM orders ORDER BY updated_at, client_order_id"
        ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    def list_open_orders(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM orders
            WHERE status NOT IN ('FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED')
            ORDER BY updated_at, client_order_id
            """
        ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    def update_order_metadata(
        self,
        order_id: str,
        *,
        metadata_updates: dict[str, Any],
        status: str | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        """Merge metadata updates into an existing order without replacing ownership fields."""
        row = self._conn.execute(
            "SELECT client_order_id, metadata FROM orders WHERE client_order_id=? OR exchange_order_id=?",
            (order_id, order_id),
        ).fetchone()
        if row is None:
            return False

        metadata = json.loads(row["metadata"] or "{}")
        metadata.update(metadata_updates)
        if status is None:
            self._conn.execute(
                """
                UPDATE orders
                SET metadata=?, updated_at=?
                WHERE client_order_id=?
                """,
                (
                    json.dumps(metadata, sort_keys=True),
                    _iso(updated_at),
                    row["client_order_id"],
                ),
            )
        else:
            self._conn.execute(
                """
                UPDATE orders
                SET metadata=?, status=?, updated_at=?
                WHERE client_order_id=?
                """,
                (
                    json.dumps(metadata, sort_keys=True),
                    status,
                    _iso(updated_at),
                    row["client_order_id"],
                ),
            )
        self._conn.commit()
        return True

    def record_fill(
        self,
        *,
        fill_id: str,
        client_order_id: str,
        exchange_order_id: str,
        strategy_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        commission: float,
        timestamp: datetime,
        exchange_fill_id: str = "",
        received_at: datetime | None = None,
        raw: dict[str, Any] | None = None,
    ) -> bool:
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO fills (
                fill_id, client_order_id, exchange_order_id, strategy_id,
                symbol, side, qty, price, commission, timestamp,
                exchange_fill_id, received_at, status, processing_error,
                processed_at, updated_at, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill_id,
                client_order_id,
                exchange_order_id,
                strategy_id,
                symbol,
                side,
                qty,
                price,
                commission,
                timestamp.isoformat(),
                exchange_fill_id,
                _iso(received_at),
                FILL_STATUS_PROCESSED,
                "",
                _iso(None),
                _iso(None),
                json.dumps(raw or {}, sort_keys=True),
            ),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def record_received_fill(
        self,
        *,
        fill_id: str,
        client_order_id: str,
        exchange_order_id: str,
        strategy_id: str = "",
        symbol: str,
        side: str,
        qty: float,
        price: float,
        commission: float,
        timestamp: datetime,
        exchange_fill_id: str = "",
        received_at: datetime | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        """Durably store a seen exchange fill without marking it consumed."""
        now = _iso(None)
        payload = json.dumps(raw or {}, sort_keys=True)
        received_ts = _iso(received_at)
        existing = self._conn.execute(
            "SELECT status, received_at, processing_error FROM fills WHERE fill_id=?",
            (fill_id,),
        ).fetchone()
        if existing is None:
            self._conn.execute(
                """
                INSERT INTO fills (
                    fill_id, client_order_id, exchange_order_id, strategy_id,
                    symbol, side, qty, price, commission, timestamp,
                    exchange_fill_id, received_at, status, processing_error,
                    processed_at, updated_at, raw
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill_id,
                    client_order_id,
                    exchange_order_id,
                    strategy_id,
                    symbol,
                    side,
                    qty,
                    price,
                    commission,
                    timestamp.isoformat(),
                    exchange_fill_id,
                    received_ts,
                    FILL_STATUS_RECEIVED,
                    "",
                    "",
                    now,
                    payload,
                ),
            )
            self._conn.commit()
            return

        next_status = self._status_after_advance(
            str(existing["status"] or ""),
            FILL_STATUS_RECEIVED,
        )
        processing_error = (
            ""
            if next_status == FILL_STATUS_RECEIVED
            else str(existing["processing_error"] or "")
        )
        self._conn.execute(
            """
            UPDATE fills
            SET client_order_id=?,
                exchange_order_id=?,
                strategy_id=CASE WHEN ? != '' THEN ? ELSE strategy_id END,
                symbol=?,
                side=?,
                qty=?,
                price=?,
                commission=?,
                timestamp=?,
                exchange_fill_id=?,
                received_at=CASE WHEN received_at='' THEN ? ELSE received_at END,
                status=?,
                processing_error=?,
                updated_at=?,
                raw=?
            WHERE fill_id=?
            """,
            (
                client_order_id,
                exchange_order_id,
                strategy_id,
                strategy_id,
                symbol,
                side,
                qty,
                price,
                commission,
                timestamp.isoformat(),
                exchange_fill_id,
                received_ts,
                next_status,
                processing_error,
                now,
                payload,
                fill_id,
            ),
        )
        self._conn.commit()

    def has_fill(self, fill_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM fills WHERE fill_id=?",
            (fill_id,),
        ).fetchone()
        return row is not None

    def get_fill(self, fill_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM fills WHERE fill_id=?",
            (fill_id,),
        ).fetchone()
        return _row_to_dict(row)

    def list_fills(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM fills ORDER BY timestamp, fill_id"
        ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    def get_fill_status(self, fill_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT status FROM fills WHERE fill_id=?",
            (fill_id,),
        ).fetchone()
        return None if row is None else str(row["status"])

    def is_fill_processed(self, fill_id: str) -> bool:
        return self.get_fill_status(fill_id) == FILL_STATUS_PROCESSED

    def mark_fill_dispatched(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
    ) -> None:
        self.mark_fill_strategy_dispatched(fill_id, strategy_id=strategy_id)

    def mark_fill_strategy_dispatched(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
    ) -> None:
        self._advance_fill_status(
            fill_id,
            FILL_STATUS_STRATEGY_DISPATCHED,
            strategy_id=strategy_id,
            processing_error="",
        )

    def mark_fill_coordinator_applied(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
    ) -> None:
        self._advance_fill_status(
            fill_id,
            FILL_STATUS_COORDINATOR_APPLIED,
            strategy_id=strategy_id,
            processing_error="",
        )

    def mark_fill_lifecycle_applied(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
    ) -> None:
        self._advance_fill_status(
            fill_id,
            FILL_STATUS_LIFECYCLE_APPLIED,
            strategy_id=strategy_id,
            processing_error="",
        )

    def mark_fill_finalized(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
    ) -> None:
        self._advance_fill_status(
            fill_id,
            FILL_STATUS_FINALIZED,
            strategy_id=strategy_id,
            processing_error="",
        )

    def mark_fill_processed(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
        processed_at: datetime | None = None,
    ) -> None:
        self._advance_fill_status(
            fill_id,
            FILL_STATUS_PROCESSED,
            strategy_id=strategy_id,
            processing_error="",
            processed_at=processed_at or datetime.now(timezone.utc),
        )

    def mark_fill_unresolved(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
        reason: str = "",
    ) -> None:
        self._advance_fill_status(
            fill_id,
            FILL_STATUS_UNRESOLVED,
            strategy_id=strategy_id,
            processing_error=reason,
        )

    def mark_fill_processing_failed(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
        error: str = "",
    ) -> None:
        self._advance_fill_status(
            fill_id,
            FILL_STATUS_PROCESSING_FAILED,
            strategy_id=strategy_id,
            processing_error=error,
        )

    def record_fill_processing_error(
        self,
        fill_id: str,
        *,
        strategy_id: str = "",
        error: str = "",
    ) -> None:
        self._conn.execute(
            """
            UPDATE fills
            SET strategy_id=CASE WHEN ? != '' THEN ? ELSE strategy_id END,
                processing_error=?,
                updated_at=?
            WHERE fill_id=?
            """,
            (
                strategy_id,
                strategy_id,
                error,
                _iso(None),
                fill_id,
            ),
        )
        self._conn.commit()

    def _advance_fill_status(
        self,
        fill_id: str,
        status: str,
        *,
        strategy_id: str = "",
        processing_error: str | None = None,
        processed_at: datetime | None = None,
    ) -> None:
        self._advance_fill_status_uncommitted(
            fill_id,
            status,
            strategy_id=strategy_id,
            processing_error=processing_error,
            processed_at=processed_at,
        )
        self._conn.commit()

    def _advance_fill_status_uncommitted(
        self,
        fill_id: str,
        status: str,
        *,
        strategy_id: str = "",
        processing_error: str | None = None,
        processed_at: datetime | None = None,
    ) -> bool:
        row = self._conn.execute(
            "SELECT status FROM fills WHERE fill_id=?",
            (fill_id,),
        ).fetchone()
        if row is None:
            return False
        next_status = self._status_after_advance(str(row["status"] or ""), status)
        processed_at_text = _iso(processed_at) if processed_at is not None else ""
        self._conn.execute(
            """
            UPDATE fills
            SET status=?,
                strategy_id=CASE WHEN ? != '' THEN ? ELSE strategy_id END,
                processing_error=CASE WHEN ? IS NOT NULL THEN ? ELSE processing_error END,
                processed_at=CASE WHEN ? != '' THEN ? ELSE processed_at END,
                updated_at=?
            WHERE fill_id=?
            """,
            (
                next_status,
                strategy_id,
                strategy_id,
                processing_error,
                processing_error,
                processed_at_text,
                processed_at_text,
                _iso(None),
                fill_id,
            ),
        )
        return True

    def _status_after_advance(self, current_status: str, target_status: str) -> str:
        return (
            target_status
            if self._fill_status_rank(target_status) >= self._fill_status_rank(current_status)
            else current_status
        )

    def _fill_status_rank(self, status: str | None) -> int:
        return FILL_STATUS_ORDER.get(str(status or ""), 0)

    def get_watermark(self, name: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM watermarks WHERE name=?",
            (name,),
        ).fetchone()
        return None if row is None else str(row["value"])

    def set_watermark(self, name: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO watermarks (name, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at
            """,
            (name, value, _iso(None)),
        )
        self._conn.commit()

    def upsert_position(
        self,
        *,
        position_instance_id: str,
        strategy_id: str,
        symbol: str,
        direction: str,
        qty: float,
        avg_entry: float,
        status: str = "OPEN",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO positions (
                position_instance_id, strategy_id, symbol, direction, qty,
                avg_entry, status, metadata, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(position_instance_id) DO UPDATE SET
                strategy_id=excluded.strategy_id,
                symbol=excluded.symbol,
                direction=excluded.direction,
                qty=excluded.qty,
                avg_entry=excluded.avg_entry,
                status=excluded.status,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (
                position_instance_id,
                strategy_id,
                symbol,
                direction,
                qty,
                avg_entry,
                status,
                json.dumps(metadata or {}, sort_keys=True),
                _iso(None),
            ),
        )
        self._conn.commit()

    def list_open_positions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE status='OPEN' ORDER BY position_instance_id"
        ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    def upsert_strategy_snapshot(
        self,
        strategy_id: str,
        snapshot: dict[str, Any],
        *,
        updated_at: datetime | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO strategy_snapshots (strategy_id, snapshot, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(strategy_id) DO UPDATE SET
                snapshot=excluded.snapshot,
                updated_at=excluded.updated_at
            """,
            (strategy_id, json.dumps(snapshot, sort_keys=True), _iso(updated_at)),
        )
        self._conn.commit()

    def get_strategy_snapshot(self, strategy_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT snapshot FROM strategy_snapshots WHERE strategy_id=?",
            (strategy_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["snapshot"] or "{}")

    def upsert_lifecycle_entry(self, entry: Any) -> None:
        data = _plain(entry)
        self._conn.execute(
            """
            INSERT INTO lifecycle_entries (
                position_instance_id, strategy_id, symbol, direction, qty,
                avg_entry, entry_time, entry_commission, exit_commission,
                closed_qty, realized_price_pnl, funding_paid, metadata, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(position_instance_id) DO UPDATE SET
                strategy_id=excluded.strategy_id,
                symbol=excluded.symbol,
                direction=excluded.direction,
                qty=excluded.qty,
                avg_entry=excluded.avg_entry,
                entry_time=excluded.entry_time,
                entry_commission=excluded.entry_commission,
                exit_commission=excluded.exit_commission,
                closed_qty=excluded.closed_qty,
                realized_price_pnl=excluded.realized_price_pnl,
                funding_paid=excluded.funding_paid,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (
                data["position_instance_id"],
                data["strategy_id"],
                data["symbol"],
                data["direction"],
                data["qty"],
                data["avg_entry"],
                data["entry_time"],
                data.get("entry_commission", 0.0),
                data.get("exit_commission", 0.0),
                data.get("closed_qty", 0.0),
                data.get("realized_price_pnl", 0.0),
                data.get("funding_paid", 0.0),
                json.dumps(data.get("metadata", {}), sort_keys=True),
                _iso(None),
            ),
        )
        self._conn.commit()

    def replace_lifecycle_entries(self, entries: list[Any]) -> None:
        self._replace_lifecycle_entries_uncommitted(entries)
        self._conn.commit()

    def persist_lifecycle_phase(
        self,
        fill_id: str,
        entries: list[Any],
        *,
        strategy_id: str = "",
        closed_trade_event: tuple[datetime, dict[str, Any]] | None = None,
    ) -> None:
        """Atomically persist lifecycle state and advance the fill lifecycle phase."""
        with self._conn:
            self._replace_lifecycle_entries_uncommitted(entries)
            if closed_trade_event is not None:
                timestamp, payload = closed_trade_event
                self._conn.execute(
                    "INSERT INTO event_journal (stream, timestamp, payload) VALUES (?, ?, ?)",
                    (
                        "fill_lifecycle_closed_trade",
                        timestamp.isoformat(),
                        json.dumps(payload, sort_keys=True),
                    ),
                )
            advanced = self._advance_fill_status_uncommitted(
                fill_id,
                FILL_STATUS_LIFECYCLE_APPLIED,
                strategy_id=strategy_id,
                processing_error="",
            )
            if not advanced:
                raise KeyError(f"fill not found: {fill_id}")

    def _replace_lifecycle_entries_uncommitted(self, entries: list[Any]) -> None:
        self._conn.execute("DELETE FROM lifecycle_entries")
        for entry in entries:
            data = _plain(entry)
            self._conn.execute(
                """
                INSERT INTO lifecycle_entries (
                    position_instance_id, strategy_id, symbol, direction, qty,
                    avg_entry, entry_time, entry_commission, exit_commission,
                    closed_qty, realized_price_pnl, funding_paid, metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["position_instance_id"],
                    data["strategy_id"],
                    data["symbol"],
                    data["direction"],
                    data["qty"],
                    data["avg_entry"],
                    data["entry_time"],
                    data.get("entry_commission", 0.0),
                    data.get("exit_commission", 0.0),
                    data.get("closed_qty", 0.0),
                    data.get("realized_price_pnl", 0.0),
                    data.get("funding_paid", 0.0),
                    json.dumps(data.get("metadata", {}), sort_keys=True),
                    _iso(None),
                ),
            )

    def list_lifecycle_entries(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM lifecycle_entries ORDER BY position_instance_id"
        ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    def record_discrepancy(
        self,
        *,
        kind: str,
        description: str,
        symbol: str = "",
        strategy_id: str = "",
        severity: str = "error",
        status: str = "OPEN",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        existing = self._conn.execute(
            """
            SELECT id FROM reconciliation_discrepancies
            WHERE status != 'RESOLVED'
              AND kind=?
              AND symbol=?
              AND strategy_id=?
              AND description=?
              AND metadata=?
            ORDER BY created_at, id
            LIMIT 1
            """,
            (kind, symbol, strategy_id, description, metadata_json),
        ).fetchone()
        if existing is not None:
            return int(existing["id"])

        cur = self._conn.execute(
            """
            INSERT INTO reconciliation_discrepancies (
                severity, kind, symbol, strategy_id, description, status,
                metadata, created_at, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                severity,
                kind,
                symbol,
                strategy_id,
                description,
                status,
                metadata_json,
                _iso(None),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_unresolved_discrepancies(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM reconciliation_discrepancies
            WHERE status != 'RESOLVED'
            ORDER BY created_at, id
            """
        ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    def get_discrepancy(self, discrepancy_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM reconciliation_discrepancies WHERE id=?",
            (discrepancy_id,),
        ).fetchone()
        return _row_to_dict(row)

    def resolve_discrepancy(
        self,
        discrepancy_id: int,
        *,
        resolution: str,
        resolved_by: str = "",
        metadata: dict[str, Any] | None = None,
        resolved_at: datetime | None = None,
    ) -> bool:
        row = self._conn.execute(
            "SELECT metadata FROM reconciliation_discrepancies WHERE id=?",
            (discrepancy_id,),
        ).fetchone()
        if row is None:
            return False
        existing_metadata = json.loads(row["metadata"] or "{}")
        existing_metadata.update(metadata or {})
        existing_metadata["resolution"] = resolution
        if resolved_by:
            existing_metadata["resolved_by"] = resolved_by
        self._conn.execute(
            """
            UPDATE reconciliation_discrepancies
            SET status='RESOLVED', metadata=?, resolved_at=?
            WHERE id=?
            """,
            (
                json.dumps(existing_metadata, sort_keys=True),
                _iso(resolved_at),
                discrepancy_id,
            ),
        )
        self._conn.commit()
        return True

    def record_execution_report(self, report: ExecutionReport) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO execution_reports (
                report_id, kind, timestamp, symbol, client_order_id,
                exchange_order_id, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.report_id,
                report.kind.value,
                report.timestamp.isoformat(),
                report.symbol,
                report.client_order_id,
                report.exchange_order_id,
                json.dumps(report.to_dict(), sort_keys=True),
            ),
        )
        self._conn.commit()

    def append_event(
        self,
        stream: str,
        timestamp: datetime,
        payload: dict[str, Any],
    ) -> None:
        self._conn.execute(
            "INSERT INTO event_journal (stream, timestamp, payload) VALUES (?, ?, ?)",
            (stream, timestamp.isoformat(), json.dumps(payload, sort_keys=True)),
        )
        self._conn.commit()

    def record_admin_allocation_correction(
        self,
        allocation: dict[str, Any],
        *,
        corrected_by: str = "",
        reason: str = "",
        timestamp: datetime | None = None,
    ) -> None:
        """Audit an operator assignment of unknown exchange exposure."""
        payload = {
            "event_kind": "admin_allocation_correction",
            "position_instance_id": allocation.get("position_instance_id", ""),
            "strategy_id": allocation.get("strategy_id", ""),
            "symbol": allocation.get("symbol", ""),
            "corrected_by": corrected_by,
            "reason": reason,
            "allocation": _plain(allocation),
        }
        self.append_event("admin_allocation_correction", timestamp or datetime.now(timezone.utc), payload)

    def list_events(self, stream: str | None = None) -> list[dict[str, Any]]:
        if stream is None:
            rows = self._conn.execute(
                "SELECT * FROM event_journal ORDER BY id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM event_journal WHERE stream=? ORDER BY id",
                (stream,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );
            INSERT OR IGNORE INTO schema_version(version) VALUES (1);

            CREATE TABLE IF NOT EXISTS orders (
                client_order_id TEXT PRIMARY KEY,
                exchange_order_id TEXT NOT NULL DEFAULT '',
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT '',
                decision_id TEXT NOT NULL DEFAULT '',
                position_instance_id TEXT NOT NULL DEFAULT '',
                reduce_only INTEGER NOT NULL DEFAULT 0,
                oca_group TEXT,
                bracket_group TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_orders_exchange_oid
                ON orders(exchange_order_id);
            CREATE INDEX IF NOT EXISTS idx_orders_strategy_symbol
                ON orders(strategy_id, symbol);

            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                client_order_id TEXT NOT NULL,
                exchange_order_id TEXT NOT NULL DEFAULT '',
                exchange_fill_id TEXT NOT NULL DEFAULT '',
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                commission REAL NOT NULL,
                timestamp TEXT NOT NULL,
                received_at TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'PROCESSED',
                processing_error TEXT NOT NULL DEFAULT '',
                processed_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT '',
                raw TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_fills_order
                ON fills(client_order_id);
            CREATE INDEX IF NOT EXISTS idx_fills_timestamp
                ON fills(timestamp);

            CREATE TABLE IF NOT EXISTS positions (
                position_instance_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                qty REAL NOT NULL,
                avg_entry REAL NOT NULL,
                status TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS watermarks (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS strategy_snapshots (
                strategy_id TEXT PRIMARY KEY,
                snapshot TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lifecycle_entries (
                position_instance_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                qty REAL NOT NULL,
                avg_entry REAL NOT NULL,
                entry_time TEXT NOT NULL,
                entry_commission REAL NOT NULL DEFAULT 0,
                exit_commission REAL NOT NULL DEFAULT 0,
                closed_qty REAL NOT NULL DEFAULT 0,
                realized_price_pnl REAL NOT NULL DEFAULT 0,
                funding_paid REAL NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reconciliation_discrepancies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                severity TEXT NOT NULL,
                kind TEXT NOT NULL,
                symbol TEXT NOT NULL DEFAULT '',
                strategy_id TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_reconciliation_discrepancies_identity
                ON reconciliation_discrepancies(status, kind, symbol, strategy_id);

            CREATE TABLE IF NOT EXISTS execution_reports (
                report_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL DEFAULT '',
                client_order_id TEXT NOT NULL DEFAULT '',
                exchange_order_id TEXT NOT NULL DEFAULT '',
                payload TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS event_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stream TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        _add_column_if_missing(self._conn, "fills", "exchange_fill_id", "TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self._conn, "fills", "received_at", "TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self._conn, "fills", "status", "TEXT NOT NULL DEFAULT 'PROCESSED'")
        _add_column_if_missing(self._conn, "fills", "processing_error", "TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self._conn, "fills", "processed_at", "TEXT NOT NULL DEFAULT ''")
        _add_column_if_missing(self._conn, "fills", "updated_at", "TEXT NOT NULL DEFAULT ''")
        self._conn.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (2)")
        self._conn.commit()


def _iso(ts: datetime | None) -> str:
    return (ts or datetime.now(timezone.utc)).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in ("metadata", "raw", "snapshot", "payload"):
        if key in data:
            data[key] = json.loads(data[key] or "{}")
    if "reduce_only" in data:
        data["reduce_only"] = bool(data["reduce_only"])
    return data


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    ddl: str,
) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Side):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def fill_identity(fill: Fill) -> str:
    """Return a durable fill id, preferring exchange truth when available."""
    if fill.exchange_fill_id:
        return fill.exchange_fill_id
    exchange_oid = fill.exchange_order_id or fill.order_id
    return (
        f"{exchange_oid}:{int(fill.timestamp.timestamp() * 1000)}:"
        f"{fill.side.value}:{fill.qty:g}:{fill.fill_price:g}:{fill.commission:g}"
    )
