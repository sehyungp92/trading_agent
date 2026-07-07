from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import asyncpg

from backtests.shared.parity.calibration_report import (
    REQUIRED_CALIBRATION_FAMILIES,
    REQUIRED_CALIBRATION_STRATEGY_IDS,
    validate_calibration_report,
)


SLIPPAGE_TICKS_TOLERANCE = 0.25
COMMISSION_TOLERANCE = 0.01
RATE_TOLERANCE = 0.01


@dataclass(frozen=True)
class FillMetricRow:
    strategy_id: str
    symbol: str
    side: str
    order_type: str
    tif: str
    order_qty: float
    fill_qty: float
    reference_price: float
    fill_price: float
    commission: float
    submit_ts: datetime | None
    fill_ts: datetime | None
    ioc_rejected: bool = False


async def generate_report(
    *,
    dsn: str,
    output_dir: Path,
    days: int = 30,
    as_of: datetime | None = None,
    backtest_fills_json: Path | None = None,
    allow_incomplete: bool = False,
) -> tuple[Path, Path]:
    if backtest_fills_json is None and not allow_incomplete:
        raise ValueError("backtest_fills_json is required unless allow_incomplete=True")
    as_of = as_of or datetime.now(timezone.utc)
    start = as_of - timedelta(days=days)
    broker_rows, trade_counts = await _load_broker_rows(dsn, start=start, end=as_of)
    backtest_rows = _load_backtest_rows(backtest_fills_json) if backtest_fills_json else []
    report = _build_report(
        broker_rows=broker_rows,
        backtest_rows=backtest_rows,
        trade_counts=trade_counts,
        start=start,
        end=as_of,
        days=days,
        has_backtest_source=backtest_fills_json is not None,
    )
    if not allow_incomplete:
        validate_calibration_report(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = as_of.date().isoformat()
    json_path = output_dir / f"parity_calibration_{stamp}.json"
    md_path = output_dir / f"parity_calibration_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


async def _load_broker_rows(
    dsn: str,
    *,
    start: datetime,
    end: datetime,
) -> tuple[list[FillMetricRow], dict[str, int]]:
    conn = await asyncpg.connect(dsn)
    try:
        fills = await conn.fetch(
            """
            SELECT
                o.strategy_id,
                o.instrument_symbol,
                o.side,
                o.order_type,
                COALESCE(o.tif, '') AS tif,
                o.qty AS order_qty,
                o.limit_price,
                o.stop_price,
                o.created_at,
                f.price,
                f.qty AS fill_qty,
                COALESCE(f.fees, 0) AS fees,
                f.fill_ts
            FROM fills f
            JOIN orders o ON o.oms_order_id = f.oms_order_id
            WHERE f.fill_ts >= $1 AND f.fill_ts < $2
            ORDER BY f.fill_ts
            """,
            start,
            end,
        )
        rejects = await conn.fetch(
            """
            SELECT
                COALESCE(o.strategy_id, oe.strategy_id, '') AS strategy_id,
                COALESCE(o.instrument_symbol, '') AS instrument_symbol,
                COALESCE(o.side, '') AS side,
                COALESCE(o.order_type, '') AS order_type,
                COALESCE(o.tif, '') AS tif,
                COALESCE(o.qty, 0) AS order_qty,
                COALESCE(o.limit_price, o.stop_price, 0) AS reference_price,
                o.created_at,
                oe.event_ts
            FROM order_events oe
            LEFT JOIN orders o ON o.oms_order_id = oe.oms_order_id
            WHERE oe.event_ts >= $1
              AND oe.event_ts < $2
              AND oe.event_type IN ('ORDER_REJECTED', 'ORDER_EXPIRED')
              AND COALESCE(o.tif, '') = 'IOC'
            ORDER BY oe.event_ts
            """,
            start,
            end,
        )
        trade_count_rows = await conn.fetch(
            """
            SELECT strategy_id, count(*) AS completed_trades
            FROM trades
            WHERE entry_ts >= $1 AND entry_ts < $2
            GROUP BY strategy_id
            """,
            start,
            end,
        )
    finally:
        await conn.close()

    rows = [
        FillMetricRow(
            strategy_id=row["strategy_id"],
            symbol=row["instrument_symbol"],
            side=row["side"],
            order_type=row["order_type"],
            tif=row["tif"],
            order_qty=float(row["order_qty"]),
            fill_qty=float(row["fill_qty"]),
            reference_price=_reference_price(row),
            fill_price=float(row["price"]),
            commission=float(row["fees"]),
            submit_ts=row["created_at"],
            fill_ts=row["fill_ts"],
        )
        for row in fills
    ]
    rows.extend(
        FillMetricRow(
            strategy_id=row["strategy_id"],
            symbol=row["instrument_symbol"],
            side=row["side"],
            order_type=row["order_type"],
            tif=row["tif"],
            order_qty=float(row["order_qty"]),
            fill_qty=0.0,
            reference_price=float(row["reference_price"] or 0.0),
            fill_price=0.0,
            commission=0.0,
            submit_ts=row["created_at"],
            fill_ts=row["event_ts"],
            ioc_rejected=True,
        )
        for row in rejects
    )
    trade_counts = {
        row["strategy_id"]: int(row["completed_trades"])
        for row in trade_count_rows
    }
    return rows, trade_counts


def _load_backtest_rows(path: Path) -> list[FillMetricRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("fills", payload) if isinstance(payload, dict) else payload
    rows: list[FillMetricRow] = []
    for item in records:
        rows.append(
            FillMetricRow(
                strategy_id=str(item["strategy_id"]),
                symbol=str(item.get("symbol", "")),
                side=str(item.get("side", "")),
                order_type=str(item.get("order_type", "")),
                tif=str(item.get("tif", "")),
                order_qty=float(item.get("order_qty", item.get("qty", 0))),
                fill_qty=float(item.get("fill_qty", item.get("qty", 0))),
                reference_price=float(item.get("reference_price", item.get("fill_price", 0))),
                fill_price=float(item.get("fill_price", 0)),
                commission=float(item.get("commission", 0)),
                submit_ts=_parse_dt(item.get("submit_ts")),
                fill_ts=_parse_dt(item.get("fill_ts")),
                ioc_rejected=bool(item.get("ioc_rejected", False)),
            )
        )
    return rows


def _build_report(
    *,
    broker_rows: list[FillMetricRow],
    backtest_rows: list[FillMetricRow],
    trade_counts: dict[str, int],
    start: datetime,
    end: datetime,
    days: int,
    has_backtest_source: bool,
) -> dict[str, Any]:
    expected_strategy_ids = set(REQUIRED_CALIBRATION_STRATEGY_IDS)
    broker_rows = [row for row in broker_rows if row.strategy_id in expected_strategy_ids]
    backtest_rows = [row for row in backtest_rows if row.strategy_id in expected_strategy_ids]
    trade_counts = {
        strategy_id: count
        for strategy_id, count in trade_counts.items()
        if strategy_id in expected_strategy_ids
    }
    broker_by_strategy = _group_by_strategy(broker_rows)
    backtest_by_strategy = _group_by_strategy(backtest_rows)
    strategy_results = []
    for strategy_id in REQUIRED_CALIBRATION_STRATEGY_IDS:
        broker_metrics = _metrics(broker_by_strategy[strategy_id])
        replay_metrics = _metrics(backtest_by_strategy.get(strategy_id, []))
        tolerance_results = _tolerances(broker_metrics, replay_metrics, has_backtest_source)
        strategy_results.append(
            {
                "strategy_id": strategy_id,
                "broker_fill_sample_count": broker_metrics["broker_fill_sample_count"],
                "broker_completed_trade_count": trade_counts.get(strategy_id, 0),
                "backtest_fill_sample_count": replay_metrics["broker_fill_sample_count"],
                "broker_metrics": broker_metrics,
                "backtest_metrics": replay_metrics,
                "tolerance_results": tolerance_results,
                "within_tolerance": bool(tolerance_results) and all(tolerance_results.values()),
            }
        )

    return {
        "schema_version": 1,
        "report_type": "broker_backed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"days": days, "start": start.isoformat(), "end": end.isoformat()},
        "source_tables": ["orders", "trades", "fills", "order_events"],
        "expected_family_ids": list(REQUIRED_CALIBRATION_FAMILIES),
        "expected_strategy_ids": list(REQUIRED_CALIBRATION_STRATEGY_IDS),
        "broker_fill_sample_count": sum(1 for row in broker_rows if not row.ioc_rejected),
        "broker_completed_trade_count": sum(trade_counts.values()),
        "within_tolerance": bool(strategy_results) and all(row["within_tolerance"] for row in strategy_results),
        "strategy_results": strategy_results,
    }


def _metrics(rows: list[FillMetricRow]) -> dict[str, float | int]:
    filled = [row for row in rows if not row.ioc_rejected]
    ioc = [row for row in rows if row.tif == "IOC" or row.ioc_rejected]
    slippage = [_slippage_ticks(row) for row in filled if row.reference_price > 0]
    fill_latencies = [
        (row.fill_ts - row.submit_ts).total_seconds()
        for row in filled
        if row.fill_ts is not None and row.submit_ts is not None
    ]
    return {
        "broker_fill_sample_count": len(filled),
        "mean_slippage_ticks": _safe_mean(slippage),
        "mean_commission": _safe_mean([row.commission for row in filled]),
        "partial_fill_rate": _rate([row.fill_qty < row.order_qty for row in filled if row.order_qty > 0]),
        "ioc_reject_rate": _rate([row.ioc_rejected for row in ioc]),
        "mean_time_to_fill_seconds": _safe_mean(fill_latencies),
    }


def _tolerances(
    broker_metrics: dict[str, float | int],
    replay_metrics: dict[str, float | int],
    has_backtest_source: bool,
) -> dict[str, bool]:
    if not has_backtest_source or replay_metrics["broker_fill_sample_count"] <= 0:
        return {
            "backtest_source_present": False,
            "slippage_mean_abs_diff": False,
            "commission_abs_diff": False,
            "partial_fill_rate_abs_diff": False,
            "ioc_reject_rate_abs_diff": False,
        }
    return {
        "backtest_source_present": True,
        "slippage_mean_abs_diff": abs(
            float(broker_metrics["mean_slippage_ticks"]) - float(replay_metrics["mean_slippage_ticks"])
        )
        <= SLIPPAGE_TICKS_TOLERANCE,
        "commission_abs_diff": abs(
            float(broker_metrics["mean_commission"]) - float(replay_metrics["mean_commission"])
        )
        <= COMMISSION_TOLERANCE,
        "partial_fill_rate_abs_diff": abs(
            float(broker_metrics["partial_fill_rate"]) - float(replay_metrics["partial_fill_rate"])
        )
        <= RATE_TOLERANCE,
        "ioc_reject_rate_abs_diff": abs(
            float(broker_metrics["ioc_reject_rate"]) - float(replay_metrics["ioc_reject_rate"])
        )
        <= RATE_TOLERANCE,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live/Backtest Parity Calibration Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Report type: {report['report_type']}",
        f"Within tolerance: {str(report['within_tolerance']).lower()}",
        f"Broker fill sample count: {report['broker_fill_sample_count']}",
        "",
        "| Strategy | Broker fills | Backtest fills | Within tolerance |",
        "|---|---:|---:|---|",
    ]
    for row in report["strategy_results"]:
        lines.append(
            "| {strategy_id} | {broker_fill_sample_count} | {backtest_fill_sample_count} | {within_tolerance} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def _group_by_strategy(rows: list[FillMetricRow]) -> dict[str, list[FillMetricRow]]:
    grouped: dict[str, list[FillMetricRow]] = defaultdict(list)
    for row in rows:
        if row.strategy_id:
            grouped[row.strategy_id].append(row)
    return grouped


def _reference_price(row: asyncpg.Record) -> float:
    for key in ("limit_price", "stop_price", "price"):
        value = row[key]
        if value is not None:
            return float(value)
    return 0.0


def _slippage_ticks(row: FillMetricRow) -> float:
    tick_size = 0.25 if row.symbol in {"MNQ", "NQ"} else 0.01
    if row.side.upper() == "SELL":
        return (row.reference_price - row.fill_price) / tick_size
    return (row.fill_price - row.reference_price) / tick_size


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _rate(values: list[bool]) -> float:
    return float(sum(1 for value in values if value) / len(values)) if values else 0.0


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate broker-backed parity calibration report.")
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--backtest-fills-json", type=Path, help="Deterministic replay fill export to compare with broker fills")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a diagnostic report without replay fills; the production validator will reject it",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dsn = os.environ.get("PARITY_POSTGRES_DSN")
    if not dsn:
        raise SystemExit("PARITY_POSTGRES_DSN is required for broker-backed calibration")
    if args.backtest_fills_json is None and not args.allow_incomplete:
        raise SystemExit("--backtest-fills-json is required unless --allow-incomplete is set")
    json_path, md_path = asyncio.run(
        generate_report(
            dsn=dsn,
            output_dir=Path(args.output_dir),
            days=args.days,
            backtest_fills_json=args.backtest_fills_json,
            allow_incomplete=args.allow_incomplete,
        )
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
