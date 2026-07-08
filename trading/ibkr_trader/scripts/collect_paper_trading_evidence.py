from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.oms.persistence.db_config import DBConfig

_WATCHDOG_PROBLEM_PATTERN = re.compile(
    r"stale|disconnected|internal error|db query failed|timed out|problem",
    flags=re.IGNORECASE,
)


async def _collect_snapshot(hours: int) -> dict[str, Any]:
    db_config = DBConfig.from_env()
    if db_config is None:
        raise RuntimeError("Database configuration not found. Set DATABASE_URL or DB_HOST/DB_* environment variables.")

    pool = await asyncpg.create_pool(dsn=db_config.to_dsn(), min_size=1, max_size=2)
    try:
        strategy_rows = await pool.fetch(
            """
            SELECT
                strategy_id,
                health_status,
                heartbeat_age_sec,
                bar_age_sec,
                last_decision_code,
                last_decision_details,
                last_seen_bar_ts,
                last_error,
                last_error_ts
            FROM v_strategy_health
            ORDER BY strategy_id
            """
        )
        adapter_rows = await pool.fetch(
            """
            SELECT
                adapter_id,
                health_status,
                heartbeat_age_sec,
                disconnect_count_24h,
                last_error_code,
                last_error_message
            FROM v_adapter_health
            ORDER BY adapter_id
            """
        )
        trade_rows = await pool.fetch(
            """
            SELECT
                strategy_id,
                COUNT(*)::int AS trade_count,
                COALESCE(SUM(realized_usd), 0)::float8 AS realized_usd,
                COALESCE(SUM(CASE WHEN realized_usd > 0 THEN 1 ELSE 0 END), 0)::int AS winning_trades,
                COALESCE(SUM(CASE WHEN realized_usd < 0 THEN 1 ELSE 0 END), 0)::int AS losing_trades
            FROM trades
            WHERE entry_ts >= now() - make_interval(hours => $1::int)
            GROUP BY strategy_id
            ORDER BY strategy_id
            """,
            hours,
        )
    finally:
        await pool.close()

    strategies = [_normalize_record(row) for row in strategy_rows]
    adapters = [_normalize_record(row) for row in adapter_rows]
    trades = [_normalize_record(row) for row in trade_rows]

    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": hours,
        "strategies": strategies,
        "adapters": adapters,
        "trades": trades,
        "summary": {
            "all_strategy_heartbeats_ok": all(row["health_status"] == "OK" for row in strategies),
            "all_adapters_ok": all(row["health_status"] == "OK" for row in adapters),
            "all_decision_codes_present": all(bool(row.get("last_decision_code")) for row in strategies),
            "strategies_with_recent_errors": [row["strategy_id"] for row in strategies if row.get("last_error")],
        },
    }


def _normalize_record(row: asyncpg.Record | dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    for key, value in list(payload.items()):
        if isinstance(value, datetime):
            payload[key] = value.isoformat()
        elif hasattr(value, "items"):
            payload[key] = dict(value)
    return payload


def _scan_watchdog_logs(paths: list[Path]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            findings.append({
                "path": str(path),
                "problem": "missing_file",
                "line": None,
                "text": "",
            })
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if _WATCHDOG_PROBLEM_PATTERN.search(line):
                findings.append({
                    "path": str(path),
                    "problem": "watchdog_problem_line",
                    "line": line_no,
                    "text": line.strip(),
                })
    return findings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect paper-trading evidence from heartbeat and watchdog surfaces.")
    parser.add_argument("--hours", type=int, default=6, help="Rolling trade window to summarize.")
    parser.add_argument(
        "--watchdog-log",
        action="append",
        default=[],
        help="Optional watchdog log path(s) to scan for stale/false-positive alert lines.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


async def _main() -> int:
    args = _build_parser().parse_args()
    snapshot = await _collect_snapshot(max(1, args.hours))
    log_paths = [Path(item) for item in args.watchdog_log]
    snapshot["watchdog_findings"] = _scan_watchdog_logs(log_paths)
    snapshot["summary"]["watchdog_problem_count"] = len(snapshot["watchdog_findings"])

    indent = 2 if args.pretty or args.output else None
    payload = json.dumps(snapshot, indent=indent, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + ("\n" if not payload.endswith("\n") else ""), encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
