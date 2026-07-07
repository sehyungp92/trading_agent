"""Health check coroutines for the watchdog.

Each check returns list[CheckResult]. Checks never raise -- they catch
exceptions internally and return an error CheckResult instead.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    key: str            # cooldown key, e.g. "heartbeat:Helix_v40"
    check_name: str     # human-readable check name
    detail: str         # description of the problem or OK state
    is_problem: bool


# ---------------------------------------------------------------------------
# 1. Heartbeat staleness (market-hours gated)
# ---------------------------------------------------------------------------

async def check_heartbeats(
    pool: asyncpg.Pool,
    config: dict,
    active_families: set[str],
    strategy_family_map: dict[str, str],
) -> list[CheckResult]:
    """Check v_strategy_health for stale heartbeats."""
    threshold = config.get("checks", {}).get("heartbeat", {}).get("stale_threshold_sec", 180)
    results: list[CheckResult] = []
    try:
        rows = await pool.fetch("SELECT * FROM v_strategy_health")
        stale_rows = []
        for row in rows:
            sid = row["strategy_id"]
            family = strategy_family_map.get(sid)
            # Skip disabled / unknown strategies and inactive families.
            if not family or family not in active_families:
                continue
            age = row["heartbeat_age_sec"]
            status = row["health_status"]
            is_stale = age is not None and float(age) > threshold
            if is_stale:
                stale_rows.append((sid, int(age), status))
            else:
                results.append(CheckResult(
                    key=f"heartbeat:{sid}",
                    check_name="Heartbeat",
                    detail=f"{sid} OK ({int(age or 0)}s)",
                    is_problem=False,
                ))
        if len(stale_rows) >= 3:
            ages = [age for _, age, _ in stale_rows]
            sids = ", ".join(sid for sid, _, _ in stale_rows)
            results.append(CheckResult(
                key="heartbeat:systemic",
                check_name="Heartbeat",
                detail=(
                    f"{len(stale_rows)} strategies stale "
                    f"({min(ages)}-{max(ages)}s, threshold {threshold}s): {sids}"
                ),
                is_problem=True,
            ))
        else:
            results.append(CheckResult(
                key="heartbeat:systemic",
                check_name="Heartbeat",
                detail="Systemic heartbeat OK",
                is_problem=False,
            ))
            for sid, age, status in stale_rows:
                results.append(CheckResult(
                    key=f"heartbeat:{sid}",
                    check_name="Heartbeat",
                    detail=f"{sid} -- stale for {age}s (threshold {threshold}s), status={status}",
                    is_problem=True,
                ))
    except Exception as exc:
        results.append(CheckResult(
            key="heartbeat:__db_error__",
            check_name="Heartbeat",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 2. Adapter health (not market-hours gated)
# ---------------------------------------------------------------------------

async def check_adapters(pool: asyncpg.Pool, config: dict) -> list[CheckResult]:
    """Check v_adapter_health for disconnected or stale adapters."""
    results: list[CheckResult] = []
    try:
        rows = await pool.fetch("SELECT * FROM v_adapter_health")
        for row in rows:
            aid = row["adapter_id"]
            status = row["health_status"]
            is_problem = status != "OK"
            detail = f"{aid} -- {status}"
            if is_problem and row.get("last_error_message"):
                detail += f" ({row['last_error_message']})"
            results.append(CheckResult(
                key=f"adapter:{aid}",
                check_name="Adapter",
                detail=detail,
                is_problem=is_problem,
            ))
    except Exception as exc:
        results.append(CheckResult(
            key="adapter:__db_error__",
            check_name="Adapter",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 3. Active halts
# ---------------------------------------------------------------------------

async def check_halts(pool: asyncpg.Pool, config: dict) -> list[CheckResult]:
    """Check v_active_halts for any current halts."""
    results: list[CheckResult] = []
    try:
        rows = await pool.fetch(
            "SELECT halt_level, entity, halt_reason, last_update_at FROM v_active_halts"
        )
        for row in rows:
            level = row["halt_level"]
            entity = row["entity"]
            reason = row["halt_reason"] or "unknown"
            results.append(CheckResult(
                key=f"halt:{level}:{entity}",
                check_name="Halt",
                detail=f"{level} halt on {entity} -- {reason}",
                is_problem=True,
            ))
        if not rows:
            results.append(CheckResult(
                key="halt:none",
                check_name="Halt",
                detail="No active halts",
                is_problem=False,
            ))
    except Exception as exc:
        results.append(CheckResult(
            key="halt:__db_error__",
            check_name="Halt",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 4. IB Gateway TCP probe
# ---------------------------------------------------------------------------

async def check_ib_gateway(config: dict) -> list[CheckResult]:
    """TCP connect to IB Gateway port."""
    gw_cfg = config.get("checks", {}).get("ib_gateway", {})
    host = gw_cfg.get("host", "127.0.0.1")
    port = gw_cfg.get("port", 4002)
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=5.0
        )
        writer.close()
        await writer.wait_closed()
        return [CheckResult(
            key="ib_gateway:connectivity",
            check_name="IB Gateway",
            detail=f"Connected to {host}:{port}",
            is_problem=False,
        )]
    except Exception as exc:
        return [CheckResult(
            key="ib_gateway:connectivity",
            check_name="IB Gateway",
            detail=f"Cannot reach {host}:{port} -- {exc}",
            is_problem=True,
        )]


# ---------------------------------------------------------------------------
# 5. Daily P&L drawdown
# ---------------------------------------------------------------------------

async def check_daily_pnl(pool: asyncpg.Pool, config: dict) -> list[CheckResult]:
    """Check per-strategy and portfolio-level drawdown thresholds."""
    pnl_cfg = config.get("checks", {}).get("daily_pnl", {})
    strat_threshold = pnl_cfg.get("strategy_drawdown_r", -3.0)
    port_threshold = pnl_cfg.get("portfolio_drawdown_r", -5.0)
    results: list[CheckResult] = []
    try:
        # Per-strategy
        rows = await pool.fetch(
            "SELECT strategy_id, family_id, daily_realized_r "
            "FROM risk_daily_strategy WHERE trade_date = CURRENT_DATE"
        )
        for row in rows:
            sid = row["strategy_id"]
            pnl_r = float(row["daily_realized_r"])
            if pnl_r <= strat_threshold:
                results.append(CheckResult(
                    key=f"pnl:{sid}",
                    check_name="Daily P&L",
                    detail=f"{sid} at {pnl_r:+.1f}R (threshold {strat_threshold}R)",
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key=f"pnl:{sid}",
                    check_name="Daily P&L",
                    detail=f"{sid} at {pnl_r:+.1f}R",
                    is_problem=False,
                ))

        # Portfolio-level
        port_rows = await pool.fetch(
            "SELECT family_id, daily_realized_r "
            "FROM risk_daily_portfolio WHERE trade_date = CURRENT_DATE"
        )
        for row in port_rows:
            fid = row["family_id"]
            pnl_r = float(row["daily_realized_r"])
            if pnl_r <= port_threshold:
                results.append(CheckResult(
                    key=f"pnl:portfolio:{fid}",
                    check_name="Daily P&L",
                    detail=f"Portfolio {fid} at {pnl_r:+.1f}R (threshold {port_threshold}R)",
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key=f"pnl:portfolio:{fid}",
                    check_name="Daily P&L",
                    detail=f"Portfolio {fid} at {pnl_r:+.1f}R",
                    is_problem=False,
                ))
    except Exception as exc:
        results.append(CheckResult(
            key="pnl:__db_error__",
            check_name="Daily P&L",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 6. Relay health
# ---------------------------------------------------------------------------

async def check_relay(
    session: aiohttp.ClientSession, config: dict
) -> list[CheckResult]:
    """HTTP GET to relay /health endpoint."""
    relay_cfg = config.get("checks", {}).get("relay", {})
    url = relay_cfg.get("url", "http://127.0.0.1:8000/health")
    backlog_threshold = relay_cfg.get("backlog_threshold", 500)
    stale_threshold = relay_cfg.get("oldest_pending_age_sec", 600)
    results: list[CheckResult] = []
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                results.append(CheckResult(
                    key="relay:reachable",
                    check_name="Relay",
                    detail=f"HTTP {resp.status} from {url}",
                    is_problem=True,
                ))
                return results
            results.append(CheckResult(
                key="relay:reachable",
                check_name="Relay",
                detail="Relay reachable",
                is_problem=False,
            ))
            body = await resp.json()
            pending = body.get("pending_events", 0)
            if pending > backlog_threshold:
                results.append(CheckResult(
                    key="relay:backlog",
                    check_name="Relay",
                    detail=f"Backlog: {pending} pending (threshold {backlog_threshold})",
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key="relay:backlog",
                    check_name="Relay",
                    detail=f"Backlog OK ({pending} pending)",
                    is_problem=False,
                ))
            oldest_age = body.get("oldest_pending_age_seconds", 0)
            if oldest_age > stale_threshold:
                results.append(CheckResult(
                    key="relay:stale_pending",
                    check_name="Relay",
                    detail=f"Oldest pending event: {int(oldest_age)}s (threshold {stale_threshold}s)",
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key="relay:stale_pending",
                    check_name="Relay",
                    detail=f"Oldest pending: {int(oldest_age)}s",
                    is_problem=False,
                ))
    except Exception as exc:
        results.append(CheckResult(
            key="relay:reachable",
            check_name="Relay",
            detail=f"Cannot reach relay: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 7. Recent errors in strategy_state
# ---------------------------------------------------------------------------

async def check_errors(pool: asyncpg.Pool, config: dict) -> list[CheckResult]:
    """Check strategy_state for recent errors."""
    max_age = config.get("checks", {}).get("errors", {}).get("max_age_minutes", 30)
    results: list[CheckResult] = []
    try:
        rows = await pool.fetch(
            "SELECT strategy_id, last_error, last_error_ts "
            "FROM strategy_state "
            "WHERE last_error IS NOT NULL "
            "  AND last_error_ts > now() - make_interval(mins := $1)",
            max_age,
        )
        for row in rows:
            sid = row["strategy_id"]
            err = (row["last_error"] or "")[:120]
            results.append(CheckResult(
                key=f"error:{sid}",
                check_name="Error",
                detail=f"{sid} -- {err}",
                is_problem=True,
            ))
        if not rows:
            results.append(CheckResult(
                key="error:none",
                check_name="Error",
                detail="No recent errors",
                is_problem=False,
            ))
    except Exception as exc:
        results.append(CheckResult(
            key="error:__db_error__",
            check_name="Error",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 8. Data freshness (market-hours gated)
# ---------------------------------------------------------------------------

async def check_data_freshness(
    pool: asyncpg.Pool,
    config: dict,
    active_families: set[str],
    strategy_family_map: dict[str, str],
) -> list[CheckResult]:
    """Alert when engines stop receiving bar data during market hours."""
    thresholds = config.get("checks", {}).get("data_freshness", {}).get("thresholds", {})
    results: list[CheckResult] = []
    try:
        rows = await pool.fetch(
            "SELECT strategy_id, last_decision_code, last_seen_bar_ts, bar_age_sec, "
            "       last_decision_details, health_status "
            "FROM v_strategy_health WHERE health_status != 'STALE'"
        )
        stale_count = 0
        for row in rows:
            sid = row["strategy_id"]
            family = strategy_family_map.get(sid)
            if not family or family not in active_families:
                continue
            threshold = thresholds.get(family, 5400)
            bar_age = row["bar_age_sec"]
            decision = row["last_decision_code"]
            details = row["last_decision_details"] or {}

            if bar_age is not None and bar_age > threshold:
                stale_count += 1
                farm_info = ""
                if isinstance(details, dict):
                    farms = details.get("ib_farm_status", {})
                    broken = [f for f, s in farms.items() if s != "OK"] if isinstance(farms, dict) else []
                    if broken:
                        farm_info = f" (farms: {', '.join(broken)})"
                results.append(CheckResult(
                    key=f"data_fresh:{sid}",
                    check_name="DataFreshness",
                    detail=f"{sid} -- bar data stale ({int(bar_age)}s, threshold {threshold}s){farm_info}",
                    is_problem=True,
                ))
            elif bar_age is None and decision == "IDLE":
                results.append(CheckResult(
                    key=f"data_fresh:{sid}",
                    check_name="DataFreshness",
                    detail=f"{sid} -- no bar data received (decision={decision})",
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key=f"data_fresh:{sid}",
                    check_name="DataFreshness",
                    detail=f"{sid} -- OK (decision={decision})",
                    is_problem=False,
                ))
    except Exception as exc:
        results.append(CheckResult(
            key="data_fresh:__db_error__",
            check_name="DataFreshness",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
    return results


# ---------------------------------------------------------------------------
# 9. Liveness detection (processing-level health)
# ---------------------------------------------------------------------------

async def check_liveness(
    pool: asyncpg.Pool,
    config: dict,
    active_families: set[str],
    strategy_family_map: dict[str, str],
    prev_bars: dict[str, int],
    stalled_counts: dict[str, int],
) -> list[CheckResult]:
    """Detect silent failures: stalled engines and per-symbol data stalls.

    Uses monotonic ``bars_processed`` counters and per-symbol freshness
    timestamps from the ``liveness`` key inside ``last_decision_details``
    JSONB column of ``strategy_state``.

    ``prev_bars`` and ``stalled_counts`` are mutable dicts maintained across
    cycles by the caller (main loop).  They reset on watchdog restart, which
    prevents false-positive alerts on deploy.
    """
    liveness_cfg = config.get("checks", {}).get("liveness", {})
    symbol_thresholds = liveness_cfg.get("symbol_stale_thresholds", {})
    max_stalled = liveness_cfg.get("stalled_cycles", 3)
    results: list[CheckResult] = []

    try:
        rows = await pool.fetch(
            "SELECT strategy_id, last_decision_details, last_decision_code, "
            "       last_seen_bar_ts "
            "FROM strategy_state "
            "WHERE last_decision_details IS NOT NULL"
        )
    except Exception as exc:
        results.append(CheckResult(
            key="liveness:__db_error__",
            check_name="Liveness",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
        return results

    now = datetime.now(timezone.utc)
    for row in rows:
        sid = row["strategy_id"]
        family = strategy_family_map.get(sid)
        if not family or family not in active_families:
            continue

        raw_details = row["last_decision_details"]
        if isinstance(raw_details, dict):
            details = raw_details
        elif isinstance(raw_details, str):
            try:
                details = json.loads(raw_details)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            continue

        liveness = details.get("liveness")
        if not isinstance(liveness, dict):
            continue  # engine not yet updated -- graceful skip

        decision = row["last_decision_code"] or "UNKNOWN"

        # --- bars_processed monotonic counter check ---
        bars = liveness.get("bars_processed", 0)
        if isinstance(bars, (int, float)):
            bars = int(bars)
            if sid in prev_bars:
                if bars == prev_bars[sid]:
                    stalled_counts[sid] = stalled_counts.get(sid, 0) + 1
                    if stalled_counts[sid] >= max_stalled:
                        results.append(CheckResult(
                            key=f"liveness:stalled:{sid}",
                            check_name="Liveness",
                            detail=(
                                f"{sid} -- engine stalled. bars_processed={bars} "
                                f"unchanged for {stalled_counts[sid]} cycles. "
                                f"Last decision: {decision}. "
                                f"Check asyncio task health."
                            ),
                            is_problem=True,
                        ))
                else:
                    stalled_counts[sid] = 0
                    results.append(CheckResult(
                        key=f"liveness:stalled:{sid}",
                        check_name="Liveness",
                        detail=f"{sid} -- OK (bars_processed={bars}, decision={decision})",
                        is_problem=False,
                    ))
            # Store current value for next cycle comparison
            prev_bars[sid] = bars

        # --- Per-symbol freshness check ---
        sym_freshness = liveness.get("symbol_freshness")
        threshold = symbol_thresholds.get(family, 5400)

        # Overlay special case: check rebalance date instead of symbol freshness
        rebalance_date = liveness.get("last_rebalance_date")
        if rebalance_date is not None:
            today_str = now.strftime("%Y-%m-%d")
            if rebalance_date != today_str:
                results.append(CheckResult(
                    key=f"liveness:rebalance:{sid}",
                    check_name="Liveness",
                    detail=(
                        f"{sid} -- rebalance not run today "
                        f"(last: {rebalance_date}, today: {today_str})"
                    ),
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key=f"liveness:rebalance:{sid}",
                    check_name="Liveness",
                    detail=f"{sid} -- rebalance OK (date={rebalance_date})",
                    is_problem=False,
                ))
            continue  # skip symbol_freshness for overlay

        if isinstance(sym_freshness, dict) and sym_freshness:
            for sym, ts_str in sym_freshness.items():
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_sec = (now - ts).total_seconds()
                except (ValueError, TypeError):
                    continue

                if age_sec > threshold:
                    results.append(CheckResult(
                        key=f"liveness:sym:{sid}:{sym}",
                        check_name="Liveness",
                        detail=(
                            f"{sid} -- {sym} bar stale "
                            f"({int(age_sec / 60)}min, threshold {threshold // 60}min). "
                            f"bars_processed={bars}. "
                            f"Check IBKR data subscriptions."
                        ),
                        is_problem=True,
                    ))
                else:
                    results.append(CheckResult(
                        key=f"liveness:sym:{sid}:{sym}",
                        check_name="Liveness",
                        detail=f"{sid}:{sym} -- OK ({int(age_sec)}s old)",
                        is_problem=False,
                    ))

        # --- OMS execution health check ---
        oms_health = details.get("oms_health")
        if isinstance(oms_health, dict):
            consec = oms_health.get("consecutive_denials", 0)
            if consec > 5:
                submitted = oms_health.get("submitted", 0)
                denied = oms_health.get("denied", 0)
                results.append(CheckResult(
                    key=f"liveness:oms:{sid}",
                    check_name="Liveness",
                    detail=(
                        f"{sid} -- execution blocked. "
                        f"{consec} consecutive intent denials "
                        f"(submitted={submitted}, denied={denied}). "
                        f"Check portfolio rules."
                    ),
                    is_problem=True,
                ))
            else:
                results.append(CheckResult(
                    key=f"liveness:oms:{sid}",
                    check_name="Liveness",
                    detail=f"{sid} -- OMS OK",
                    is_problem=False,
                ))

    return results


# ---------------------------------------------------------------------------
# 10. Daily quiet-day classifier (one CheckResult per active strategy).
# ---------------------------------------------------------------------------

def classify_daily_activity(
    bars: int | None,
    trades: int | None,
    denials: int | None,
    last_bar_ts: datetime | None,
    family_disconnect_count: int,
    now: datetime,
    session_start_threshold_hours: float = 8.0,
) -> str:
    """Return one of ACTIVE / NORMAL_QUIET / BLOCKED / DEAD / BROKER_DOWN.

    Pure function so the test suite can drive it without DB fixtures.
    Inputs default to 0 / None when unknown:
      * ``bars``, ``trades``, ``denials`` are aggregates from
        ``v_daily_strategy_activity``.
      * ``last_bar_ts`` is the most recent bar timestamp the engine has
        emitted today.
      * ``family_disconnect_count`` is ``adapter_state.disconnect_count_24h``
        for the family's adapter.

    Decision tree (first match wins):

      1. ACTIVE          -- trades > 0 (informational)
      2. BROKER_DOWN     -- bars == 0 AND family_disconnect_count > 0
      3. DEAD            -- bars == 0
                         OR last_bar_ts older than session_start_threshold
      4. BLOCKED         -- bars > 0 AND trades == 0 AND denials > 0
      5. NORMAL_QUIET    -- everything else (engine alive, no signal)
    """
    bars_v = bars or 0
    trades_v = trades or 0
    denials_v = denials or 0

    if trades_v > 0:
        return "ACTIVE"

    bar_is_stale = (
        last_bar_ts is not None
        and (now - last_bar_ts).total_seconds() > session_start_threshold_hours * 3600
    )

    if bars_v == 0:
        if family_disconnect_count > 0:
            return "BROKER_DOWN"
        return "DEAD"

    if last_bar_ts is None or bar_is_stale:
        if family_disconnect_count > 0:
            return "BROKER_DOWN"
        return "DEAD"

    if denials_v > 0:
        return "BLOCKED"

    return "NORMAL_QUIET"


async def check_daily_classification(
    pool: asyncpg.Pool,
    config: dict,
    active_families: set[str],
    strategy_family_map: dict[str, str],
) -> list[CheckResult]:
    """Run the quiet-day classifier for the current trading day.

    Reads from ``v_daily_strategy_activity`` (which aggregates over
    ``strategy_heartbeat_history``) plus ``adapter_state`` for the
    BROKER_DOWN signal. Emits one CheckResult per strategy in an active
    family. Non-NORMAL classifications are flagged ``is_problem=True``;
    NORMAL_QUIET and ACTIVE are not problems.
    """
    cfg = config.get("checks", {}).get("daily_classification", {})
    session_threshold_h = float(cfg.get("session_start_threshold_hours", 8.0))
    results: list[CheckResult] = []

    try:
        rows = await pool.fetch(
            "SELECT strategy_id, family_id, bars, denials, trades, last_bar_ts "
            "FROM v_daily_strategy_activity "
            "WHERE day = (now() AT TIME ZONE 'UTC')::date"
        )
    except Exception as exc:
        results.append(CheckResult(
            key="daily_class:__db_error__",
            check_name="DailyClassification",
            detail=f"DB query failed: {exc}",
            is_problem=True,
        ))
        return results

    # Pull adapter disconnect counts once (small table, ~3 rows).
    family_disconnects: dict[str, int] = {}
    try:
        adapter_rows = await pool.fetch(
            "SELECT adapter_id, disconnect_count_24h FROM adapter_state"
        )
        for ar in adapter_rows:
            family_disconnects[ar["adapter_id"]] = int(ar["disconnect_count_24h"] or 0)
    except Exception as exc:
        logger.warning("DailyClassification: adapter_state read failed: %s", exc)

    seen: set[str] = set()
    now = datetime.now(timezone.utc)

    for row in rows:
        sid = row["strategy_id"]
        family = row["family_id"] or strategy_family_map.get(sid)
        if not family or family not in active_families:
            continue

        seen.add(sid)
        disconnects = family_disconnects.get(family, 0)
        last_bar_ts = row["last_bar_ts"]
        if last_bar_ts is not None and last_bar_ts.tzinfo is None:
            last_bar_ts = last_bar_ts.replace(tzinfo=timezone.utc)

        label = classify_daily_activity(
            bars=row["bars"],
            trades=row["trades"],
            denials=row["denials"],
            last_bar_ts=last_bar_ts,
            family_disconnect_count=disconnects,
            now=now,
            session_start_threshold_hours=session_threshold_h,
        )
        is_problem = label not in ("ACTIVE", "NORMAL_QUIET")
        results.append(CheckResult(
            key=f"daily_class:{sid}",
            check_name="DailyClassification",
            detail=(
                f"{sid} -- {label} "
                f"(bars={row['bars'] or 0}, trades={row['trades'] or 0}, "
                f"denials={row['denials'] or 0})"
            ),
            is_problem=is_problem,
        ))

    # Strategies in active families with NO row in v_daily_strategy_activity
    # are silently DEAD: the snapshot writer never captured a single
    # heartbeat for them today. Surface explicitly so they don't slip past.
    for sid, family in strategy_family_map.items():
        if family in active_families and sid not in seen:
            results.append(CheckResult(
                key=f"daily_class:{sid}",
                check_name="DailyClassification",
                detail=f"{sid} -- DEAD (no heartbeat history captured today)",
                is_problem=True,
            ))

    return results
