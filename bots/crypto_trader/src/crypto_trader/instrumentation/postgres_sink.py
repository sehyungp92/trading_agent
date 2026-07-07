"""PostgreSQL event sink for instrumentation output."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
    canonical_event_envelope,
)

log = structlog.get_logger()

_INSTRUMENTATION_EVENT_INSERT_SQL = """
                    INSERT INTO instrumentation_events (
                        event_id, logical_event_id, event_type, bot_id,
                        family_id, portfolio_id, account_alias, strategy_id,
                        symbol, exchange_timestamp, local_timestamp, payload, lineage
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s::jsonb, %s::jsonb
                    )
                    ON CONFLICT (event_id) DO NOTHING
                    """


def _has_explicit_economics(event: InstrumentedTradeEvent) -> bool:
    return any((
        event.price_pnl_gross != 0.0,
        event.total_fees != 0.0,
        event.realized_pnl_net != 0.0,
        event.funding_paid != 0.0,
    ))


def _event_realized_net_pnl(event: InstrumentedTradeEvent) -> float:
    if _has_explicit_economics(event):
        return event.realized_pnl_net
    return event.pnl


class PostgresSink:
    """Writes instrumentation events to PostgreSQL.

    Implements the Sink protocol (6 methods) for auto-dispatch via EventEmitter,
    plus 2 direct methods (write_equity, upsert_positions) called from engine.

    All methods swallow exceptions — never blocks the engine.
    """

    def __init__(
        self,
        dsn: str,
        *,
        error_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(dsn, min_size=1, max_size=3)
        self._error_callback = error_callback
        self._in_error_callback = False

    # ------------------------------------------------------------------
    # Sink protocol methods (called via EventEmitter.add_sink)
    # ------------------------------------------------------------------

    def write_trade(self, event: InstrumentedTradeEvent) -> None:
        """INSERT trade, idempotent via ON CONFLICT DO NOTHING."""
        try:
            net_pnl = _event_realized_net_pnl(event)
            confluences = json.dumps(event.confluences) if event.confluences else "[]"
            market_ctx = (
                json.dumps(event.market_context.to_dict())
                if event.market_context
                else None
            )
            strategy_id = event.metadata.strategy_id if event.metadata else "unknown"
            confirmation_type = event.entry_signal or None

            with self._pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO trades (
                        trade_id, strategy_id, symbol, direction,
                        entry_time, exit_time, entry_price, exit_price,
                        position_size, pnl, net_pnl, r_multiple,
                        commission, funding_paid, setup_grade, exit_reason,
                        confirmation_type, entry_method, confluences,
                        mae_r, mfe_r, exit_efficiency, market_context
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s::jsonb,
                        %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (trade_id) DO NOTHING
                    """,
                    (
                        event.trade_id,
                        strategy_id,
                        event.pair,
                        event.side,
                        event.entry_time,
                        event.exit_time,
                        event.entry_price,
                        event.exit_price,
                        event.position_size,
                        event.pnl,
                        net_pnl,
                        event.r_multiple,
                        event.commission,
                        event.funding_paid,
                        event.setup_grade,
                        event.exit_reason,
                        confirmation_type,
                        event.entry_method,
                        confluences,
                        event.mae_r,
                        event.mfe_r,
                        event.exit_efficiency,
                        market_ctx,
                    ),
                )
        except Exception as exc:
            log.exception("postgres_sink.write_trade_failed")
            self._emit_error(
                exc,
                message="failed to write typed trade row",
                recovery_action="continue_with_generic_event",
                event_type="trade",
            )
        self.write_event("trade", event)

    def write_daily(self, event: DailySnapshot) -> None:
        """UPSERT daily snapshot."""
        try:
            per_strategy = json.dumps(
                event.per_strategy_summary if event.per_strategy_summary else {}
            )
            with self._pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO daily_snapshots (
                        trade_date, total_trades, win_count, loss_count,
                        gross_pnl, net_pnl, max_drawdown_pct,
                        sharpe_rolling_30d, sortino_rolling_30d,
                        per_strategy_summary
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (trade_date) DO UPDATE SET
                        total_trades = EXCLUDED.total_trades,
                        win_count = EXCLUDED.win_count,
                        loss_count = EXCLUDED.loss_count,
                        gross_pnl = EXCLUDED.gross_pnl,
                        net_pnl = EXCLUDED.net_pnl,
                        max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                        sharpe_rolling_30d = EXCLUDED.sharpe_rolling_30d,
                        sortino_rolling_30d = EXCLUDED.sortino_rolling_30d,
                        per_strategy_summary = EXCLUDED.per_strategy_summary
                    """,
                    (
                        event.date,
                        event.total_trades,
                        event.win_count,
                        event.loss_count,
                        event.gross_pnl,
                        event.net_pnl,
                        event.max_drawdown_pct,
                        event.sharpe_rolling_30d,
                        event.sortino_rolling_30d,
                        per_strategy,
                    ),
                )
        except Exception as exc:
            log.exception("postgres_sink.write_daily_failed")
            self._emit_error(
                exc,
                message="failed to write typed daily snapshot row",
                recovery_action="continue_with_generic_event",
                event_type="daily_snapshot",
            )
        self.write_event("daily_snapshot", event)

    def write_health_report(self, event: HealthReportSnapshot) -> None:
        """INSERT health snapshot."""
        try:
            report = event.report or {}
            assessment = report.get("assessment", "unknown")
            uptime_sec = report.get("uptime_sec")
            alerts = json.dumps(report.get("alerts", []))
            report_json = json.dumps(report, default=str)

            with self._pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO health_snapshots (
                        timestamp, assessment, uptime_sec, alerts, report
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        event.timestamp,
                        assessment,
                        uptime_sec,
                        alerts,
                        report_json,
                    ),
                )
        except Exception as exc:
            log.exception("postgres_sink.write_health_report_failed")
            self._emit_error(
                exc,
                message="failed to write typed health snapshot row",
                recovery_action="continue_with_generic_event",
                event_type="heartbeat",
            )
        self.write_event("heartbeat", event)

    def write_missed(self, event: MissedOpportunityEvent) -> None:
        """Persist missed opportunities to the generic assistant event table."""
        self.write_event("missed_opportunity", event)

    def write_error(self, event: ErrorEvent) -> None:
        """Persist errors to the generic assistant event table."""
        self.write_event("error", event)

    def write_funnel(self, event: PipelineFunnelSnapshot) -> None:
        """Persist funnels to the generic assistant event table."""
        self.write_event("pipeline_funnel", event)

    def write_event(self, event_type: str, event) -> None:
        """Best-effort generic event persistence for assistant telemetry."""
        try:
            row = self._instrumentation_event_row(event_type, event)
            if row is None:
                return
            with self._pool.connection() as conn:
                conn.execute(_INSTRUMENTATION_EVENT_INSERT_SQL, row)
        except Exception as exc:
            log.exception("postgres_sink.write_event_failed", event_type=event_type)
            self._emit_error(
                exc,
                message=f"failed to write canonical {event_type} event",
                recovery_action="continue_without_postgres",
                event_type=event_type,
            )

    def write_events_batch(self, event_type: str, events: list[Any]) -> None:
        """Batch homogeneous generic canonical events idempotently."""
        rows = [
            row for row in (
                self._instrumentation_event_row(event_type, event)
                for event in events
            )
            if row is not None
        ]
        if not rows:
            return
        try:
            with self._pool.connection() as conn:
                executemany = getattr(conn, "executemany", None)
                if callable(executemany):
                    executemany(_INSTRUMENTATION_EVENT_INSERT_SQL, rows)
                else:
                    for row in rows:
                        conn.execute(_INSTRUMENTATION_EVENT_INSERT_SQL, row)
        except Exception as exc:
            log.exception("postgres_sink.write_events_batch_failed", event_type=event_type)
            self._emit_error(
                exc,
                message=f"failed to batch canonical {event_type} events",
                recovery_action="continue_without_postgres",
                event_type=event_type,
            )

    # ------------------------------------------------------------------
    # Direct methods (called from engine, NOT part of Sink protocol)
    # ------------------------------------------------------------------

    def write_equity(self, equity: float, timestamp: datetime) -> None:
        """Insert equity snapshot. Called every 5 min (~288/day)."""
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    "INSERT INTO equity_snapshots (timestamp, equity) VALUES (%s, %s)",
                    (timestamp, equity),
                )
        except Exception as exc:
            log.exception("postgres_sink.write_equity_failed")
            self._emit_error(
                exc,
                message="failed to write equity snapshot",
                recovery_action="continue_without_postgres",
                event_type="equity_snapshot",
            )

    def upsert_positions(self, positions: list[dict[str, Any]]) -> None:
        """Full-sync open positions: DELETE all then INSERT current.

        Max ~9 rows (PortfolioConfig.max_total_positions), so this is fast.
        """
        try:
            with self._pool.connection() as conn:
                with conn.transaction():
                    conn.execute("DELETE FROM positions")
                    for pos in positions:
                        conn.execute(
                            """
                            INSERT INTO positions (
                                strategy_id, symbol, direction, qty, avg_entry,
                                unrealized_pnl, risk_r, stop_price, entry_time
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                pos.get("strategy_id", "unknown"),
                                pos["symbol"],
                                pos["direction"],
                                pos["qty"],
                                pos["avg_entry"],
                                pos.get("unrealized_pnl", 0.0),
                                pos.get("risk_r", 0.0),
                                pos.get("stop_price"),
                                pos.get("entry_time"),
                            ),
                        )
        except Exception as exc:
            log.exception("postgres_sink.upsert_positions_failed")
            self._emit_error(
                exc,
                message="failed to upsert open positions",
                recovery_action="continue_without_postgres",
                event_type="position_snapshot",
            )

    def upsert_strategy_position_allocations(self, allocations: list[dict[str, Any]]) -> None:
        """Full-sync strategy ownership allocations into the additive read model."""
        try:
            with self._pool.connection() as conn:
                with conn.transaction():
                    conn.execute("DELETE FROM strategy_position_allocations")
                    for allocation in allocations:
                        conn.execute(
                            """
                            INSERT INTO strategy_position_allocations (
                                position_instance_id, strategy_id, symbol, direction,
                                allocated_qty, avg_entry, risk_r, entry_time,
                                status, confidence, source, entry_order_ids,
                                entry_fill_ids, exit_order_ids, exit_fill_ids,
                                metadata, last_update_at
                            ) VALUES (
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s::jsonb,
                                %s::jsonb, %s::jsonb, %s::jsonb,
                                %s::jsonb, %s
                            )
                            ON CONFLICT (position_instance_id) DO UPDATE SET
                                strategy_id=EXCLUDED.strategy_id,
                                symbol=EXCLUDED.symbol,
                                direction=EXCLUDED.direction,
                                allocated_qty=EXCLUDED.allocated_qty,
                                avg_entry=EXCLUDED.avg_entry,
                                risk_r=EXCLUDED.risk_r,
                                entry_time=EXCLUDED.entry_time,
                                status=EXCLUDED.status,
                                confidence=EXCLUDED.confidence,
                                source=EXCLUDED.source,
                                entry_order_ids=EXCLUDED.entry_order_ids,
                                entry_fill_ids=EXCLUDED.entry_fill_ids,
                                exit_order_ids=EXCLUDED.exit_order_ids,
                                exit_fill_ids=EXCLUDED.exit_fill_ids,
                                metadata=EXCLUDED.metadata,
                                last_update_at=EXCLUDED.last_update_at
                            """,
                            (
                                allocation["position_instance_id"],
                                allocation["strategy_id"],
                                allocation["symbol"],
                                allocation["direction"],
                                allocation["allocated_qty"],
                                allocation.get("avg_entry", 0.0),
                                allocation.get("risk_r", allocation.get("open_risk_R", 0.0)),
                                allocation.get("entry_time"),
                                allocation.get("status", "OPEN"),
                                allocation.get("confidence", "unknown"),
                                allocation.get("source", "unknown"),
                                json.dumps(allocation.get("entry_order_ids", [])),
                                json.dumps(allocation.get("entry_fill_ids", [])),
                                json.dumps(allocation.get("exit_order_ids", [])),
                                json.dumps(allocation.get("exit_fill_ids", [])),
                                json.dumps(allocation.get("metadata", {}), default=str),
                                datetime.now(timezone.utc),
                            ),
                        )
        except Exception as exc:
            log.exception("postgres_sink.upsert_strategy_position_allocations_failed")
            self._emit_error(
                exc,
                message="failed to upsert strategy position allocations",
                recovery_action="continue_without_postgres",
                event_type="position_allocation_snapshot",
            )

    def upsert_exchange_positions(self, positions: list[dict[str, Any]]) -> None:
        """Full-sync exchange net positions separately from strategy ownership."""
        try:
            with self._pool.connection() as conn:
                with conn.transaction():
                    conn.execute("DELETE FROM exchange_positions")
                    for position in positions:
                        conn.execute(
                            """
                            INSERT INTO exchange_positions (
                                symbol, direction, qty, avg_entry,
                                unrealized_pnl, liquidation_price, observed_at,
                                metadata, last_update_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                            ON CONFLICT (symbol) DO UPDATE SET
                                direction=EXCLUDED.direction,
                                qty=EXCLUDED.qty,
                                avg_entry=EXCLUDED.avg_entry,
                                unrealized_pnl=EXCLUDED.unrealized_pnl,
                                liquidation_price=EXCLUDED.liquidation_price,
                                observed_at=EXCLUDED.observed_at,
                                metadata=EXCLUDED.metadata,
                                last_update_at=EXCLUDED.last_update_at
                            """,
                            (
                                position["symbol"],
                                position["direction"],
                                position["qty"],
                                position.get("avg_entry", 0.0),
                                position.get("unrealized_pnl", 0.0),
                                position.get("liquidation_price"),
                                position.get("observed_at"),
                                json.dumps(position.get("metadata", {}), default=str),
                                datetime.now(timezone.utc),
                            ),
                        )
        except Exception as exc:
            log.exception("postgres_sink.upsert_exchange_positions_failed")
            self._emit_error(
                exc,
                message="failed to upsert exchange net positions",
                recovery_action="continue_without_postgres",
                event_type="exchange_position_snapshot",
            )

    def close(self) -> None:
        """Close connection pool."""
        try:
            self._pool.close()
        except Exception:
            log.exception("postgres_sink.close_failed")

    def _instrumentation_event_row(
        self,
        event_type: str,
        event: Any,
    ) -> tuple[Any, ...] | None:
        raw_payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        raw_metadata = (
            raw_payload.get("metadata")
            if isinstance(raw_payload.get("metadata"), dict)
            else {}
        )
        payload = canonical_event_envelope(
            event_type,
            raw_payload,
            bot_id=str(raw_payload.get("bot_id") or raw_metadata.get("bot_id") or ""),
            source={"sink": "postgres"},
        )
        metadata = (
            payload.get("payload", {}).get("metadata", {})
            if isinstance(payload.get("payload"), dict)
            else {}
        )
        lineage = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
        event_id = payload.get("event_id")
        if not event_id:
            return None
        return (
            event_id,
            payload.get("logical_event_id"),
            payload.get("event_type") or event_type,
            payload.get("bot_id") or metadata.get("bot_id") or "",
            payload.get("family_id") or metadata.get("family_id") or lineage.get("family_id"),
            payload.get("portfolio_id") or metadata.get("portfolio_id") or lineage.get("portfolio_id"),
            payload.get("account_alias") or metadata.get("account_alias") or lineage.get("account_alias"),
            payload.get("strategy_id"),
            payload.get("symbol") or payload.get("payload", {}).get("symbol"),
            payload.get("exchange_timestamp"),
            payload.get("local_timestamp") or metadata.get("local_timestamp"),
            json.dumps(payload, default=str),
            json.dumps(lineage, default=str),
        )

    def _emit_error(
        self,
        exc: Exception,
        *,
        message: str,
        recovery_action: str,
        event_type: str,
        severity: str = "medium",
    ) -> None:
        if self._error_callback is None or self._in_error_callback:
            return
        self._in_error_callback = True
        try:
            self._error_callback({
                "component": "postgres_sink",
                "error_type": type(exc).__name__,
                "message": f"{message}: {exc}",
                "severity": severity,
                "recovery_action": recovery_action,
                "event_type": event_type,
            })
        except Exception:
            log.exception("postgres_sink.error_callback_failed", event_type=event_type)
        finally:
            self._in_error_callback = False
