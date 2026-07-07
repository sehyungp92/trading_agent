"""OMS persistence repository."""
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import fields as dc_fields
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Optional

try:
    import asyncpg.exceptions as asyncpg_exceptions
except ImportError:  # pragma: no cover - asyncpg is installed in runtime
    asyncpg_exceptions = None  # type: ignore[assignment]

from ..models.fill import Fill
from ..models.instrument import Instrument
from ..models.instrument_registry import InstrumentRegistry
from ..models.order import (
    EntryPolicy,
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskContext,
)
from ..models.position import Position

logger = logging.getLogger(__name__)

QUEUE_CLAIM_TTL_SECONDS = 30
QUEUE_SUBMIT_INFLIGHT_TTL_SECONDS = 300


def _working_entry_status_values() -> tuple[str, ...]:
    return (
        OrderStatus.RISK_APPROVED.value,
        OrderStatus.QUEUED.value,
        OrderStatus.ROUTED.value,
        OrderStatus.ACKED.value,
        OrderStatus.WORKING.value,
        OrderStatus.PARTIALLY_FILLED.value,
    )


class OMSPersistenceInvariantError(RuntimeError):
    """Raised when persistence ordering breaks OMS invariants."""

    def __init__(
        self,
        *,
        operation: str,
        oms_order_id: str,
        detail: str,
        event_type: str | None = None,
        fill_id: str | None = None,
    ) -> None:
        parts = [f"{operation} failed for order {oms_order_id}", detail]
        if event_type:
            parts.append(f"event_type={event_type}")
        if fill_id:
            parts.append(f"fill_id={fill_id}")
        super().__init__(" | ".join(parts))
        self.operation = operation
        self.oms_order_id = oms_order_id
        self.detail = detail
        self.event_type = event_type
        self.fill_id = fill_id


class OMSRepository:
    """Persistence layer for OMS state. Event-sourcing pattern:
    1. Insert into order_events (append-only)
    2. Update orders (current state)
    """

    def __init__(self, pool):  # asyncpg pool
        self._pool = pool

    @asynccontextmanager
    async def _connection(self, conn=None) -> AsyncIterator[Any]:
        if conn is not None:
            yield conn
            return
        async with self._pool.acquire() as acquired:
            yield acquired

    @asynccontextmanager
    async def transaction(self, conn=None) -> AsyncIterator[Any]:
        async with self._connection(conn) as active_conn:
            async with active_conn.transaction():
                yield active_conn

    @staticmethod
    def _is_fk_violation(exc: Exception) -> bool:
        return bool(
            asyncpg_exceptions
            and isinstance(exc, asyncpg_exceptions.ForeignKeyViolationError)
        )

    async def save_order(self, order: OMSOrder, conn=None) -> None:
        """Upsert current order state."""
        async with self._connection(conn) as active_conn:
            await active_conn.execute(
                """
                INSERT INTO orders (
                    oms_order_id, client_order_id, strategy_id, account_id,
                    instrument_symbol, side, qty, order_type, limit_price, stop_price,
                    tif, role, status, broker, broker_order_id, perm_id, oca_group,
                    filled_qty, remaining_qty, avg_fill_price, reprice_count,
                    entry_policy, risk_context,
                    created_at, queued_at, queue_priority, queue_reason,
                    queue_attempt, queue_expires_at, queue_claimed_by,
                    queue_claimed_at, queue_claim_expires_at, dequeued_at,
                    queue_denial_reason,
                    submitted_at, acked_at, last_update_at,
                    retry_count, reject_reason
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22::jsonb,$23::jsonb,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35,$36,$37,$38,$39)
                ON CONFLICT (oms_order_id) DO UPDATE SET
                    status=$13, broker_order_id=$15, perm_id=$16,
                    filled_qty=$18, remaining_qty=$19, avg_fill_price=$20,
                    reprice_count=$21,
                    queued_at=$25, queue_priority=$26, queue_reason=$27,
                    queue_attempt=$28, queue_expires_at=$29,
                    queue_claimed_by=$30, queue_claimed_at=$31,
                    queue_claim_expires_at=$32, dequeued_at=$33,
                    queue_denial_reason=$34,
                    submitted_at=$35, acked_at=$36, last_update_at=$37,
                    retry_count=$38, reject_reason=$39
                """,
                order.oms_order_id,
                order.client_order_id,
                order.strategy_id,
                order.account_id,
                order.instrument.symbol if order.instrument else "",
                order.side.value,
                order.qty,
                order.order_type.value,
                order.limit_price,
                order.stop_price,
                order.tif,
                order.role.value,
                order.status.value,
                order.broker,
                order.broker_order_id,
                order.perm_id,
                order.oca_group,
                order.filled_qty,
                order.remaining_qty,
                order.avg_fill_price,
                order.reprice_count,
                json.dumps(order.entry_policy.__dict__) if order.entry_policy else None,
                json.dumps(order.risk_context.__dict__) if order.risk_context else None,
                order.created_at,
                order.queued_at,
                order.queue_priority,
                order.queue_reason,
                order.queue_attempt,
                order.queue_expires_at,
                order.queue_claimed_by,
                order.queue_claimed_at,
                order.queue_claim_expires_at,
                order.dequeued_at,
                order.queue_denial_reason,
                order.submitted_at,
                order.acked_at,
                order.last_update_at,
                order.retry_count,
                order.reject_reason,
            )

    async def save_event(
        self,
        oms_order_id: str,
        event_type: str,
        payload: dict,
        conn=None,
    ) -> None:
        """Append to order_events (immutable audit log)."""
        try:
            async with self._connection(conn) as active_conn:
                await active_conn.execute(
                    "INSERT INTO order_events (oms_order_id, event_type, payload) VALUES ($1, $2, $3::jsonb)",
                    oms_order_id,
                    event_type,
                    json.dumps(payload),
                )
        except Exception as exc:
            if self._is_fk_violation(exc):
                raise OMSPersistenceInvariantError(
                    operation="save_event",
                    oms_order_id=oms_order_id,
                    event_type=event_type,
                    detail="parent order row missing before event insert",
                ) from exc
            raise

    async def save_fill(self, fill: Fill, conn=None) -> bool:
        try:
            async with self._connection(conn) as active_conn:
                result = await active_conn.execute(
                    """
                    INSERT INTO fills (fill_id, oms_order_id, broker_fill_id, price, qty, fill_ts, fees)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (broker_fill_id) DO NOTHING
                    """,
                    fill.fill_id,
                    fill.oms_order_id,
                    fill.broker_fill_id,
                    fill.price,
                    fill.qty,
                    fill.timestamp,
                    fill.fees,
                )
                return result.rsplit(" ", 1)[-1] == "1"
        except Exception as exc:
            if self._is_fk_violation(exc):
                raise OMSPersistenceInvariantError(
                    operation="save_fill",
                    oms_order_id=fill.oms_order_id,
                    fill_id=fill.fill_id,
                    detail="parent order row missing before fill insert",
                ) from exc
            raise

    async def save_order_and_event(
        self,
        order: OMSOrder,
        event_type: str,
        payload: dict,
        conn=None,
    ) -> None:
        async with self.transaction(conn=conn) as active_conn:
            await self.save_order(order, conn=active_conn)
            await self.save_event(order.oms_order_id, event_type, payload, conn=active_conn)

    async def save_order_fill_and_event(
        self,
        order: OMSOrder,
        fill: Fill,
        event_type: str,
        payload: dict,
    ) -> bool:
        async with self.transaction() as conn:
            inserted = await self.save_fill(fill, conn=conn)
            if not inserted:
                return False
            await self.save_order(order, conn=conn)
            await self.save_event(order.oms_order_id, event_type, payload, conn=conn)
            return True

    async def fill_exists(self, broker_fill_id: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM fills WHERE broker_fill_id = $1", broker_fill_id
            )
        return row is not None

    async def get_order(self, oms_order_id: str) -> Optional[OMSOrder]:
        """Load order by ID. Returns None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM orders WHERE oms_order_id = $1", oms_order_id
            )
        if not row:
            return None
        return self._row_to_order(dict(row))

    async def get_order_id_by_client_order_id(
        self, strategy_id: str, client_order_id: str
    ) -> Optional[str]:
        """Look up oms_order_id by client_order_id for idempotency."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT oms_order_id FROM orders
                   WHERE strategy_id = $1 AND client_order_id = $2""",
                strategy_id,
                client_order_id,
            )
        return row["oms_order_id"] if row else None

    async def get_order_id_by_broker_order_id(
        self, broker_order_id: int
    ) -> Optional[str]:
        """Resolve an OMS order ID from a persisted broker order ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT oms_order_id FROM orders
                   WHERE broker_order_id = $1::text""",
                str(broker_order_id),
            )
        return row["oms_order_id"] if row else None

    async def get_pending_entry_risk_R(self, unit_risk_dollars: float) -> float:
        """Sum risk_R of working ENTRY orders. Includes PARTIALLY_FILLED scaled by remaining qty."""
        working_statuses = _working_entry_status_values()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT risk_context, qty, remaining_qty, status FROM orders
                   WHERE role = $1 AND status = ANY($2::text[])
                   AND risk_context IS NOT NULL""",
                OrderRole.ENTRY.value,
                list(working_statuses),
            )
        return self._sum_pending_risk(rows) / unit_risk_dollars if unit_risk_dollars > 0 else 0.0

    async def get_working_orders(
        self, strategy_id: str, instrument_symbol: str = None
    ) -> list[OMSOrder]:
        """Get all non-terminal orders for a strategy."""
        terminal = (
            OrderStatus.FILLED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.EXPIRED.value,
            OrderStatus.DONE.value,
        )
        async with self._pool.acquire() as conn:
            if instrument_symbol:
                rows = await conn.fetch(
                    """SELECT * FROM orders
                       WHERE strategy_id=$1 AND instrument_symbol=$2
                       AND status NOT IN ($3, $4, $5, $6, $7)""",
                    strategy_id,
                    instrument_symbol,
                    *terminal,
                )
            else:
                rows = await conn.fetch(
                    """SELECT * FROM orders
                       WHERE strategy_id=$1 AND status NOT IN ($2, $3, $4, $5, $6)""",
                    strategy_id,
                    *terminal,
                )
        return [self._row_to_order(dict(r)) for r in rows]

    async def count_working_orders(self, strategy_id: str) -> int:
        """Count non-terminal orders for a strategy."""
        terminal = (
            OrderStatus.FILLED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.EXPIRED.value,
            OrderStatus.DONE.value,
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT COUNT(*) as cnt FROM orders
                   WHERE strategy_id=$1 AND status NOT IN ($2,$3,$4,$5,$6)""",
                strategy_id,
                *terminal,
            )
        return row["cnt"] if row else 0

    async def get_positions(
        self, strategy_id: str, instrument_symbol: str = None
    ) -> list[Position]:
        async with self._pool.acquire() as conn:
            if instrument_symbol:
                rows = await conn.fetch(
                    "SELECT * FROM positions WHERE strategy_id=$1 AND instrument_symbol=$2",
                    strategy_id,
                    instrument_symbol,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM positions WHERE strategy_id=$1", strategy_id
                )
        return [self._row_to_position(dict(r)) for r in rows]

    async def save_position(self, position: Position) -> None:
        """Upsert position."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO positions
                    (account_id, instrument_symbol, strategy_id, net_qty, avg_price,
                     realized_pnl, unrealized_pnl, open_risk_dollars, open_risk_R, last_update_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now())
                ON CONFLICT (account_id, instrument_symbol, strategy_id) DO UPDATE SET
                    net_qty = EXCLUDED.net_qty,
                    avg_price = EXCLUDED.avg_price,
                    realized_pnl = EXCLUDED.realized_pnl,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    open_risk_dollars = EXCLUDED.open_risk_dollars,
                    open_risk_R = EXCLUDED.open_risk_R,
                    last_update_at = now()
                """,
                position.account_id,
                position.instrument_symbol,
                position.strategy_id,
                position.net_qty,
                position.avg_price,
                position.realized_pnl,
                position.unrealized_pnl,
                position.open_risk_dollars,
                position.open_risk_R,
            )

    async def get_all_working_orders(self) -> list[OMSOrder]:
        """Get all non-terminal orders across all strategies."""
        terminal = (
            OrderStatus.FILLED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.EXPIRED.value,
            OrderStatus.DONE.value,
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM orders
                   WHERE status NOT IN ($1, $2, $3, $4, $5)""",
                *terminal,
            )
        return [self._row_to_order(dict(r)) for r in rows]

    async def get_all_positions(self) -> list[Position]:
        """Get all positions across all strategies."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM positions")
        return [self._row_to_position(dict(r)) for r in rows]

    async def get_positions_for_strategies(
        self, strategy_ids: list[str],
    ) -> list[Position]:
        """Get positions for specific strategies only (family-scoped)."""
        if not strategy_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM positions WHERE strategy_id = ANY($1::text[])",
                strategy_ids,
            )
        return [self._row_to_position(dict(r)) for r in rows]

    async def get_pending_entry_risk_R_for_strategies(
        self, strategy_ids: list[str], unit_risk_dollars: float,
    ) -> float:
        """Sum risk_R of working ENTRY orders for specific strategies (family-scoped)."""
        if not strategy_ids or unit_risk_dollars <= 0:
            return 0.0
        working_statuses = _working_entry_status_values()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT risk_context, qty, remaining_qty, status FROM orders
                   WHERE role = $1 AND status = ANY($2::text[])
                   AND risk_context IS NOT NULL
                   AND strategy_id = ANY($3::text[])""",
                OrderRole.ENTRY.value,
                list(working_statuses),
                strategy_ids,
            )
        return self._sum_pending_risk(rows) / unit_risk_dollars

    async def mark_order_queued(
        self,
        order_id: str,
        priority: int,
        reason: str,
        queued_at: datetime,
        expires_at: datetime | None,
        conn=None,
    ) -> OMSOrder | None:
        """Persist a risk-approved order as queued and append an audit event."""
        async with self.transaction(conn=conn) as active_conn:
            row = await active_conn.fetchrow(
                """
                UPDATE orders
                SET status = $2,
                    queued_at = COALESCE(queued_at, $3),
                    queue_priority = $4,
                    queue_reason = $5,
                    queue_expires_at = $6,
                    queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = $3
                WHERE oms_order_id = $1
                  AND status IN ($7, $2)
                RETURNING *
                """,
                order_id,
                OrderStatus.QUEUED.value,
                queued_at,
                priority,
                reason,
                expires_at,
                OrderStatus.RISK_APPROVED.value,
            )
            if row is None:
                return None
            await self.save_event(
                order_id,
                "ORDER_QUEUED",
                {
                    "priority": priority,
                    "reason": reason,
                    "queued_at": queued_at.isoformat(),
                    "expires_at": expires_at.isoformat() if expires_at else None,
                },
                conn=active_conn,
            )
        return self._row_to_order(dict(row))

    async def claim_queued_orders(
        self,
        limit: int,
        claimant_id: str,
        now: datetime,
        conn=None,
    ) -> list[OMSOrder]:
        """Atomically claim ready queued orders for one drain worker."""
        if limit <= 0:
            return []
        claim_expires_at = now + timedelta(seconds=QUEUE_CLAIM_TTL_SECONDS)
        async with self.transaction(conn=conn) as active_conn:
            rows = await active_conn.fetch(
                """
                WITH ordered AS (
                    SELECT
                        oms_order_id,
                        queue_priority,
                        queued_at
                    FROM orders
                    WHERE status = $1
                      AND (
                          submitted_at IS NULL
                          OR queue_claimed_by IS NULL
                      )
                      AND (queue_claim_expires_at IS NULL OR queue_claim_expires_at < $2)
                      AND (
                          submitted_at IS NOT NULL
                          OR queue_expires_at IS NULL
                          OR queue_expires_at > $2
                      )
                    ORDER BY queue_priority ASC NULLS LAST, queued_at ASC
                    LIMIT $3
                    FOR UPDATE SKIP LOCKED
                ),
                candidates AS (
                    SELECT
                        oms_order_id,
                        row_number() OVER (
                            ORDER BY queue_priority ASC NULLS LAST, queued_at ASC
                        ) AS claim_order
                    FROM ordered
                )
                UPDATE orders o
                SET queue_claimed_by = $4,
                    queue_claimed_at = $2,
                    queue_claim_expires_at = $5,
                    queue_attempt = COALESCE(o.queue_attempt, 0) + 1,
                    last_update_at = $2
                FROM candidates
                WHERE o.oms_order_id = candidates.oms_order_id
                RETURNING o.*, candidates.claim_order
                """,
                OrderStatus.QUEUED.value,
                now,
                limit,
                claimant_id,
                claim_expires_at,
            )
        ordered = sorted((dict(row) for row in rows), key=lambda row: row["claim_order"])
        return [self._row_to_order(row) for row in ordered]

    async def release_queued_order(
        self,
        order_id: str,
        claimant_id: str,
        conn=None,
    ) -> None:
        async with self._connection(conn) as active_conn:
            await active_conn.execute(
                """
                UPDATE orders
                SET queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = now()
                WHERE oms_order_id = $1
                  AND status = $2
                  AND queue_claimed_by = $3
                  AND submitted_at IS NULL
                """,
                order_id,
                OrderStatus.QUEUED.value,
                claimant_id,
            )

    async def release_queued_claims(
        self,
        claimant_id: str,
        conn=None,
    ) -> int:
        """Release every queued-order claim owned by one router instance."""
        async with self._connection(conn) as active_conn:
            released = await active_conn.fetchval(
                """
                WITH released AS (
                    UPDATE orders
                    SET queue_claimed_by = NULL,
                        queue_claimed_at = NULL,
                        queue_claim_expires_at = NULL,
                        last_update_at = now()
                    WHERE status = $1
                      AND queue_claimed_by = $2
                      AND submitted_at IS NULL
                    RETURNING 1
                )
                SELECT COUNT(*) FROM released
                """,
                OrderStatus.QUEUED.value,
                claimant_id,
            )
        return int(released or 0)

    async def clear_queue_claim(self, order_id: str, conn=None) -> None:
        async with self._connection(conn) as active_conn:
            await active_conn.execute(
                """
                UPDATE orders
                SET queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = now()
                WHERE oms_order_id = $1
                """,
                order_id,
            )

    async def mark_queued_order_expired(
        self,
        order_id: str,
        reason: str,
        conn=None,
    ) -> OMSOrder | None:
        now = datetime.now(timezone.utc)
        async with self.transaction(conn=conn) as active_conn:
            row = await active_conn.fetchrow(
                """
                UPDATE orders
                SET status = $2,
                    queue_denial_reason = $3,
                    queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = $4
                WHERE oms_order_id = $1
                  AND status = $5
                  AND submitted_at IS NULL
                RETURNING *
                """,
                order_id,
                OrderStatus.EXPIRED.value,
                reason,
                now,
                OrderStatus.QUEUED.value,
            )
            if row is None:
                return None
            await self.save_event(
                order_id,
                "QUEUED_ORDER_EXPIRED",
                {"reason": reason, "expired_at": now.isoformat()},
                conn=active_conn,
            )
        return self._row_to_order(dict(row))

    async def mark_queued_order_dequeued(
        self,
        order_id: str,
        claimant_id: str,
        dequeued_at: datetime,
        conn=None,
    ) -> OMSOrder | None:
        async with self.transaction(conn=conn) as active_conn:
            row = await active_conn.fetchrow(
                """
                UPDATE orders
                SET status = $2,
                    dequeued_at = $3,
                    queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = $3
                WHERE oms_order_id = $1
                  AND status = $4
                  AND queue_claimed_by = $5
                RETURNING *
                """,
                order_id,
                OrderStatus.RISK_APPROVED.value,
                dequeued_at,
                OrderStatus.QUEUED.value,
                claimant_id,
            )
            if row is None:
                return None
            await self.save_event(
                order_id,
                "QUEUED_ORDER_DEQUEUED",
                {"claimant_id": claimant_id, "dequeued_at": dequeued_at.isoformat()},
                conn=active_conn,
            )
        return self._row_to_order(dict(row))

    async def mark_queued_order_submit_started(
        self,
        order_id: str,
        claimant_id: str,
        started_at: datetime,
        conn=None,
    ) -> OMSOrder | None:
        """Mark a claimed queued order as broker-submit in-flight without hiding it."""
        submit_claim_expires_at = started_at + timedelta(
            seconds=QUEUE_SUBMIT_INFLIGHT_TTL_SECONDS
        )
        async with self.transaction(conn=conn) as active_conn:
            row = await active_conn.fetchrow(
                """
                UPDATE orders
                SET dequeued_at = COALESCE(dequeued_at, $3),
                    submitted_at = COALESCE(submitted_at, $3),
                    queue_claim_expires_at = $5,
                    last_update_at = $3
                WHERE oms_order_id = $1
                  AND status = $2
                  AND queue_claimed_by = $4
                RETURNING *
                """,
                order_id,
                OrderStatus.QUEUED.value,
                started_at,
                claimant_id,
                submit_claim_expires_at,
            )
            if row is None:
                return None
            await self.save_event(
                order_id,
                "QUEUED_ORDER_SUBMIT_STARTED",
                {"claimant_id": claimant_id, "started_at": started_at.isoformat()},
                conn=active_conn,
            )
        return self._row_to_order(dict(row))

    async def mark_queued_order_submitted(
        self,
        order_id: str,
        claimant_id: str,
        broker_order_id: int | str | None,
        perm_id: int | str | None,
        submitted_at: datetime,
        dequeued_at: datetime,
        conn=None,
    ) -> OMSOrder | None:
        """Atomically persist queued-order broker mapping and ROUTED status."""
        if broker_order_id in (None, ""):
            return None
        async with self.transaction(conn=conn) as active_conn:
            row = await active_conn.fetchrow(
                """
                UPDATE orders
                SET status = $2,
                    broker_order_id = $3,
                    perm_id = $4,
                    dequeued_at = COALESCE(dequeued_at, $6),
                    submitted_at = $5,
                    queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = $5
                WHERE oms_order_id = $1
                  AND status = $7
                  AND queue_claimed_by = $8
                  AND broker_order_id IS NULL
                RETURNING *
                """,
                order_id,
                OrderStatus.ROUTED.value,
                str(broker_order_id) if broker_order_id is not None else None,
                int(perm_id) if perm_id not in (None, "") else None,
                submitted_at,
                dequeued_at,
                OrderStatus.QUEUED.value,
                claimant_id,
            )
            if row is None:
                return None
            await self.save_event(
                order_id,
                "QUEUED_ORDER_SUBMITTED",
                {
                    "claimant_id": claimant_id,
                    "submitted_at": submitted_at.isoformat(),
                    "broker_order_id": str(broker_order_id)
                    if broker_order_id is not None
                    else None,
                    "perm_id": int(perm_id) if perm_id not in (None, "") else None,
                },
                conn=active_conn,
            )
        return self._row_to_order(dict(row))

    async def mark_queued_order_denied(
        self,
        order_id: str,
        claimant_id: str,
        reason: str,
        conn=None,
    ) -> OMSOrder | None:
        now = datetime.now(timezone.utc)
        async with self.transaction(conn=conn) as active_conn:
            row = await active_conn.fetchrow(
                """
                UPDATE orders
                SET status = $2,
                    reject_reason = $3,
                    queue_denial_reason = $3,
                    queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = $4
                WHERE oms_order_id = $1
                  AND status = $5
                  AND queue_claimed_by = $6
                RETURNING *
                """,
                order_id,
                OrderStatus.REJECTED.value,
                reason,
                now,
                OrderStatus.QUEUED.value,
                claimant_id,
            )
            if row is None:
                return None
            await self.save_event(
                order_id,
                "QUEUED_ORDER_DENIED",
                {
                    "claimant_id": claimant_id,
                    "reason": reason,
                    "denied_at": now.isoformat(),
                },
                conn=active_conn,
            )
        return self._row_to_order(dict(row))

    async def expire_due_queued_orders(self, now: datetime | None = None) -> list[OMSOrder]:
        now = now or datetime.now(timezone.utc)
        async with self.transaction() as conn:
            rows = await conn.fetch(
                """
                UPDATE orders
                SET status = $2,
                    queue_denial_reason = 'queue TTL expired',
                    queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = $3
                WHERE status = $1
                  AND queue_expires_at IS NOT NULL
                  AND queue_expires_at <= $3
                  AND submitted_at IS NULL
                RETURNING *
                """,
                OrderStatus.QUEUED.value,
                OrderStatus.EXPIRED.value,
                now,
            )
            for row in rows:
                await self.save_event(
                    row["oms_order_id"],
                    "QUEUED_ORDER_EXPIRED",
                    {"reason": "queue TTL expired", "expired_at": now.isoformat()},
                    conn=conn,
                )
        return [self._row_to_order(dict(row)) for row in rows]

    async def recover_inflight_queued_orders(
        self,
        now: datetime,
        reason: str,
        conn=None,
    ) -> list[OMSOrder]:
        """Recover queue-drain rows stranded in non-QUEUED states after a crash."""
        recovered_rows: list[dict] = []
        async with self.transaction(conn=conn) as active_conn:
            expired = await active_conn.fetch(
                """
                UPDATE orders
                SET status = $2,
                    queue_denial_reason = 'queue TTL expired during in-flight recovery',
                    queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = $3
                WHERE status = ANY($1::text[])
                  AND queued_at IS NOT NULL
                  AND broker_order_id IS NULL
                  AND submitted_at IS NULL
                  AND queue_expires_at IS NOT NULL
                  AND queue_expires_at <= $3
                RETURNING *
                """,
                [OrderStatus.RISK_APPROVED.value, OrderStatus.ROUTED.value],
                OrderStatus.EXPIRED.value,
                now,
            )
            for row in expired:
                await self.save_event(
                    row["oms_order_id"],
                    "QUEUED_ORDER_EXPIRED",
                    {
                        "reason": "queue TTL expired during in-flight recovery",
                        "expired_at": now.isoformat(),
                    },
                    conn=active_conn,
                )
                recovered_rows.append(dict(row))

            requeued = await active_conn.fetch(
                """
                UPDATE orders
                SET status = $2,
                    queue_reason = CASE
                        WHEN queue_reason IS NULL OR queue_reason = '' THEN $4
                        ELSE queue_reason
                    END,
                    queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = $3
                WHERE status = ANY($1::text[])
                  AND queued_at IS NOT NULL
                  AND broker_order_id IS NULL
                  AND (
                      submitted_at IS NOT NULL
                      OR queue_expires_at IS NULL
                      OR queue_expires_at > $3
                  )
                RETURNING *
                """,
                [OrderStatus.RISK_APPROVED.value, OrderStatus.ROUTED.value],
                OrderStatus.QUEUED.value,
                now,
                reason,
            )
            for row in requeued:
                await self.save_event(
                    row["oms_order_id"],
                    "QUEUED_ORDER_RECOVERED",
                    {
                        "reason": reason,
                        "recovered_at": now.isoformat(),
                    },
                    conn=active_conn,
                )
                recovered_rows.append(dict(row))

            submit_inflight = await active_conn.fetch(
                """
                UPDATE orders
                SET queue_reason = CASE
                        WHEN queue_reason IS NULL OR queue_reason = '' THEN $3
                        ELSE queue_reason
                    END,
                    queue_claimed_by = NULL,
                    queue_claimed_at = NULL,
                    queue_claim_expires_at = NULL,
                    last_update_at = $2
                WHERE status = $1
                  AND queued_at IS NOT NULL
                  AND submitted_at IS NOT NULL
                  AND broker_order_id IS NULL
                  AND queue_claimed_by IS NOT NULL
                  AND (
                      queue_claim_expires_at IS NULL
                      OR queue_claim_expires_at <= $2
                  )
                RETURNING *
                """,
                OrderStatus.QUEUED.value,
                now,
                reason,
            )
            for row in submit_inflight:
                await self.save_event(
                    row["oms_order_id"],
                    "QUEUED_ORDER_RECOVERED",
                    {
                        "reason": reason,
                        "recovered_at": now.isoformat(),
                        "submit_inflight": True,
                    },
                    conn=active_conn,
                )
                recovered_rows.append(dict(row))
        return [self._row_to_order(row) for row in recovered_rows]

    async def get_queued_order_summary(
        self,
        account_id: str | None = None,
        family_id: str | None = None,
    ) -> dict[str, Any]:
        del family_id  # orders do not currently carry family_id directly.
        if account_id:
            query = """
                SELECT COUNT(*) AS queued_count, MIN(queued_at) AS oldest_queued_at
                FROM orders
                WHERE status = $1 AND account_id = $2
            """
            args = (OrderStatus.QUEUED.value, account_id)
        else:
            query = """
                SELECT COUNT(*) AS queued_count, MIN(queued_at) AS oldest_queued_at
                FROM orders
                WHERE status = $1
            """
            args = (OrderStatus.QUEUED.value,)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
        oldest = row["oldest_queued_at"] if row else None
        now = datetime.now(timezone.utc)
        return {
            "queued_count": int(row["queued_count"] or 0) if row else 0,
            "oldest_queued_at": oldest,
            "oldest_queued_age_seconds": (
                (now - oldest).total_seconds() if oldest else None
            ),
        }

    @staticmethod
    def _sum_pending_risk(rows) -> float:
        """Sum risk_dollars from pending entry order rows."""
        total = 0.0
        for row in rows:
            values = dict(row)
            rc = values.get("risk_context")
            if not rc:
                continue
            data = json.loads(rc) if isinstance(rc, str) else rc
            risk = data.get("risk_dollars", 0.0)
            if values["status"] == OrderStatus.PARTIALLY_FILLED.value:
                qty = values.get("qty") or 1
                remaining = values.get("remaining_qty") or 0
                risk = risk * (remaining / qty) if qty > 0 else 0.0
            total += risk
        return total

    def _row_to_order(self, row: dict) -> OMSOrder:
        """Convert DB row to OMSOrder."""
        entry_policy = None
        if row.get("entry_policy"):
            ep = row["entry_policy"]
            ep_data = ep if isinstance(ep, dict) else json.loads(ep)
            ep_keys = {f.name for f in dc_fields(EntryPolicy)}
            entry_policy = EntryPolicy(**{k: v for k, v in ep_data.items() if k in ep_keys})

        risk_context = None
        if row.get("risk_context"):
            rc = row["risk_context"]
            rc_data = rc if isinstance(rc, dict) else json.loads(rc)
            rc_keys = {f.name for f in dc_fields(RiskContext)}
            risk_context = RiskContext(**{k: v for k, v in rc_data.items() if k in rc_keys})

        # Look up instrument from registry, fall back to minimal stub
        instrument = None
        if row.get("instrument_symbol"):
            instrument = InstrumentRegistry.get(row["instrument_symbol"])
            if not instrument:
                logger.warning(f"Unknown instrument {row['instrument_symbol']}, using stub")
                instrument = Instrument(
                    symbol=row["instrument_symbol"],
                    root=row["instrument_symbol"],
                    venue="",
                    tick_size=0.01,
                    tick_value=0.01,
                    multiplier=1.0,
                )

        return OMSOrder(
            oms_order_id=row["oms_order_id"],
            client_order_id=row.get("client_order_id") or "",
            strategy_id=row["strategy_id"],
            account_id=row.get("account_id") or "",
            instrument=instrument,
            side=OrderSide(row["side"]),
            qty=row["qty"],
            order_type=OrderType(row["order_type"]),
            limit_price=row.get("limit_price"),
            stop_price=row.get("stop_price"),
            tif=row.get("tif") or "DAY",
            role=OrderRole(row["role"]),
            entry_policy=entry_policy,
            risk_context=risk_context,
            broker=row.get("broker") or "IBKR",
            broker_order_id=row.get("broker_order_id"),
            perm_id=row.get("perm_id"),
            oca_group=row.get("oca_group") or "",
            status=OrderStatus(row["status"]),
            created_at=row.get("created_at"),
            queued_at=row.get("queued_at"),
            queue_priority=row.get("queue_priority"),
            queue_reason=row.get("queue_reason") or "",
            queue_attempt=row.get("queue_attempt") or 0,
            queue_expires_at=row.get("queue_expires_at"),
            queue_claimed_by=row.get("queue_claimed_by") or "",
            queue_claimed_at=row.get("queue_claimed_at"),
            queue_claim_expires_at=row.get("queue_claim_expires_at"),
            dequeued_at=row.get("dequeued_at"),
            queue_denial_reason=row.get("queue_denial_reason") or "",
            submitted_at=row.get("submitted_at"),
            acked_at=row.get("acked_at"),
            last_update_at=row.get("last_update_at"),
            filled_qty=row.get("filled_qty") or 0.0,
            remaining_qty=row.get("remaining_qty") or 0.0,
            avg_fill_price=row.get("avg_fill_price") or 0.0,
            reprice_count=row.get("reprice_count") or 0,
            retry_count=row.get("retry_count") or 0,
            reject_reason=row.get("reject_reason") or "",
        )

    def _row_to_position(self, row: dict) -> Position:
        return Position(
            account_id=row["account_id"],
            instrument_symbol=row["instrument_symbol"],
            strategy_id=row["strategy_id"],
            net_qty=row.get("net_qty") or 0.0,
            avg_price=row.get("avg_price") or 0.0,
            realized_pnl=row.get("realized_pnl") or 0.0,
            unrealized_pnl=row.get("unrealized_pnl") or 0.0,
            open_risk_dollars=row.get("open_risk_dollars") or 0.0,
            open_risk_R=row.get("open_risk_R") or 0.0,
            last_update_at=row.get("last_update_at"),
        )
