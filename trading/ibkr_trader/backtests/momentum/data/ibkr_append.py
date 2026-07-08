"""Append recent NQ data to momentum backtest parquet files using IBKR.

Downloads NQ quarterly contracts for the gap period, handles Panama-style
forward adjustment at contract roll points, and appends to existing files.

Only NQ_5m and NQ_1d need updating -- 15m/30m/1h/4h are resampled from 5m
by the engine pipeline.

Requires IB Gateway or TWS running.

Usage: python -m backtests.momentum.data.ibkr_append [--port 7496]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "raw"

PACING_DELAY = 12
PACING_VIOLATION_SLEEP = 65
MAX_RETRIES = 4

# (symbol, suffix, ibkr_bar_size, use_rth)
DOWNLOADS = [
    ("NQ", "5m", "5 mins", False),
    ("NQ", "15m", "15 mins", False),
    ("NQ", "1d", "1 day", True),
    ("ES", "1d", "1 day", True),
]

# Panama alias mapping
PANAMA_ALIASES = {
    "NQ_5m.parquet": "NQ_5m_panama.parquet",
    "NQ_15m.parquet": "NQ_15m_panama.parquet",
    "NQ_1d.parquet": "NQ_daily_panama.parquet",
}


# ---------------------------------------------------------------------------
# Contract generation
# ---------------------------------------------------------------------------

def _contracts_for_gap(symbol: str, gap_start: date, gap_end: date) -> list[dict]:
    """Return quarterly contracts whose active period overlaps the gap.

    Supports NQ and ES (both CME quarterly futures with same roll schedule).
    Roll date = Monday of expiry week (3rd Friday - 4 days).
    """
    quarters = [("03", "H"), ("06", "M"), ("09", "U"), ("12", "Z")]
    all_contracts = []

    for y in range(gap_start.year - 1, gap_end.year + 2):
        for month_str, code in quarters:
            month = int(month_str)
            first_day = date(y, month, 1)
            days_to_friday = (4 - first_day.weekday()) % 7
            third_friday = first_day + timedelta(days=days_to_friday, weeks=2)
            roll_date = third_friday - timedelta(days=4)

            all_contracts.append({
                "yyyymm": f"{y}{month_str}",
                "expiry": third_friday,
                "roll_date": roll_date,
                "code": code,
                "symbol": symbol,
                "local_sym": f"{symbol}{code}{y % 10}",
            })

    all_contracts.sort(key=lambda c: c["expiry"])

    relevant = []
    for i, c in enumerate(all_contracts):
        prev_roll = all_contracts[i - 1]["roll_date"] if i > 0 else date(2000, 1, 1)
        active_start = prev_roll
        active_end = c["roll_date"]
        if active_end > gap_start and active_start <= gap_end:
            relevant.append(c)

    return relevant


# ---------------------------------------------------------------------------
# IBKR helpers
# ---------------------------------------------------------------------------

async def _connect(host: str, port: int, client_id: int):
    from ib_async import IB
    ib = IB()
    await ib.connectAsync(host, port, clientId=client_id, timeout=60)
    logger.info("Connected to %s:%d (clientId=%d)", host, port, client_id)
    return ib


async def _request(ib, contract, end_str: str, duration: str,
                   bar_size: str, use_rth: bool) -> list:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            bars = await asyncio.wait_for(
                ib.reqHistoricalDataAsync(
                    contract, endDateTime=end_str, durationStr=duration,
                    barSizeSetting=bar_size, whatToShow="TRADES",
                    useRTH=use_rth, formatDate=2, timeout=0,
                ),
                timeout=120,
            )
            return bars or []
        except Exception as e:
            msg = str(e).lower()
            if "pacing" in msg or "162" in msg:
                logger.warning("Pacing violation, waiting %ds...", PACING_VIOLATION_SLEEP)
                await asyncio.sleep(PACING_VIOLATION_SLEEP)
            elif "no data" in msg:
                return []
            elif attempt < MAX_RETRIES:
                logger.warning("Error (attempt %d): %s", attempt, e)
                await asyncio.sleep(attempt * 10)
            else:
                logger.error("Failed after %d attempts: %s", MAX_RETRIES, e)
                return []
    return []


def _bars_to_df(bars) -> pd.DataFrame:
    records = [{
        "time": b.date if isinstance(b.date, datetime) else pd.Timestamp(b.date),
        "open": b.open, "high": b.high, "low": b.low,
        "close": b.close, "volume": int(b.volume),
    } for b in bars]
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def _add_rth_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_RTH column: True for 09:30-15:59 ET on weekdays."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    idx_et = df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    df["is_RTH"] = (minutes >= 570) & (minutes < 960) & (idx_et.weekday < 5)
    return df


# ---------------------------------------------------------------------------
# Per-contract download
# ---------------------------------------------------------------------------

async def _download_contract(ib, spec: dict, bar_size: str, use_rth: bool,
                              gap_days: int) -> pd.DataFrame | None:
    """Download historical bars for one NQ quarterly contract."""
    from ib_async import Future

    sym = spec.get("symbol", "NQ")
    fut = Future(symbol=sym, exchange="CME", tradingClass=sym,
                 lastTradeDateOrContractMonth=spec["yyyymm"],
                 includeExpired=True)
    qualified = await ib.qualifyContractsAsync(fut)
    if not qualified:
        logger.warning("Could not qualify %s", spec["local_sym"])
        return None
    fut = qualified[0]
    fut.includeExpired = True
    logger.info("Qualified %s (conId=%d)", fut.localSymbol, fut.conId)

    # For expired contracts, use expiry as end date; otherwise current time
    is_expired = spec["expiry"] < date.today()
    if is_expired:
        end_str = spec["expiry"].strftime("%Y%m%d") + " 23:59:59 UTC"
    else:
        end_str = ""

    duration = f"{min(gap_days + 10, 180)} D"
    bars = await _request(ib, fut, end_str, duration, bar_size, use_rth)
    if not bars:
        return None

    df = _bars_to_df(bars)
    logger.info("  %s: %d bars (%s to %s)", spec["local_sym"], len(df),
                df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))
    return df


# ---------------------------------------------------------------------------
# Append logic
# ---------------------------------------------------------------------------

async def append_timeframe(ib, symbol: str, suffix: str, bar_size: str,
                            use_rth: bool, target_end: date) -> int:
    """Download and append incremental data for one symbol+timeframe."""
    fname = f"{symbol}_{suffix}.parquet"
    primary_path = DATA_DIR / fname
    if not primary_path.exists():
        logger.warning("SKIP %s -- not found", fname)
        return 0

    existing = pd.read_parquet(primary_path, engine="pyarrow")
    if not isinstance(existing.index, pd.DatetimeIndex):
        existing.index = pd.to_datetime(existing.index, utc=True)

    last_ts = existing.index[-1]
    gap_days = (target_end - last_ts.date()).days

    if gap_days <= 0:
        logger.info("%s: up to date (last=%s)", fname, last_ts)
        return 0

    logger.info("%s: %d rows, last=%s, gap=%d days to %s",
                fname, len(existing), last_ts, gap_days, target_end)

    # Find contracts spanning the gap
    contracts = _contracts_for_gap(symbol, last_ts.date(), target_end)
    if not contracts:
        logger.error("No contracts found for %s gap period", symbol)
        return 0
    logger.info("Contracts: %s", ", ".join(c["local_sym"] for c in contracts))

    # Download each contract
    contract_data: dict[str, pd.DataFrame] = {}
    for spec in contracts:
        logger.info("Downloading %s (%s)...", spec["local_sym"], suffix)
        df = await _download_contract(ib, spec, bar_size, use_rth, gap_days)
        if df is not None and len(df) > 0:
            contract_data[spec["yyyymm"]] = df
        await asyncio.sleep(PACING_DELAY)

    if not contract_data:
        logger.warning("%s: no data downloaded", fname)
        return 0

    # Stitch segments with forward adjustment at roll boundaries
    contracts.sort(key=lambda c: c["expiry"])
    segments: list[pd.DataFrame] = []
    cumulative_adj = 0.0

    for i, spec in enumerate(contracts):
        yyyymm = spec["yyyymm"]
        if yyyymm not in contract_data:
            continue
        df = contract_data[yyyymm]
        roll_ts = pd.Timestamp(
            datetime.combine(spec["roll_date"], datetime.min.time()), tz="UTC")

        # Segment: older contracts up to their roll date, newest from prev roll
        if i < len(contracts) - 1:
            seg = df[df.index < roll_ts]
        else:
            if i > 0:
                prev_roll_ts = pd.Timestamp(
                    datetime.combine(contracts[i - 1]["roll_date"],
                                     datetime.min.time()), tz="UTC")
                seg = df[df.index >= prev_roll_ts]
            else:
                seg = df

        if len(seg) == 0:
            continue

        # Compute forward adjustment at this roll boundary
        if i > 0:
            prev_yyyymm = contracts[i - 1]["yyyymm"]
            prev_roll_ts = pd.Timestamp(
                datetime.combine(contracts[i - 1]["roll_date"],
                                 datetime.min.time()), tz="UTC")

            if prev_yyyymm in contract_data:
                prev_df = contract_data[prev_yyyymm]
                old_before = prev_df[prev_df.index < prev_roll_ts]
                new_after = df[df.index >= prev_roll_ts]

                if len(old_before) > 0 and len(new_after) > 0:
                    gap = new_after.iloc[0]["open"] - old_before.iloc[-1]["close"]
                    tick = 0.25 if symbol == "NQ" else 0.25  # CME futures tick
                    gap = round(gap / tick) * tick
                    cumulative_adj += gap
                    logger.info("Roll %s -> %s: gap=%.2f, cumulative=%.2f",
                                prev_yyyymm, yyyymm, gap, cumulative_adj)

        # Apply forward adjustment to match existing price levels
        if cumulative_adj != 0:
            seg = seg.copy()
            for col in ["open", "high", "low", "close"]:
                seg[col] -= cumulative_adj

        segments.append(seg)

    if not segments:
        logger.warning("%s: no valid segments", fname)
        return 0

    new_data = pd.concat(segments).sort_index()
    new_data = new_data[~new_data.index.duplicated(keep="last")]
    new_data = new_data[new_data.index > last_ts]

    if new_data.empty:
        logger.info("%s: no new bars after %s", fname, last_ts)
        return 0

    # Schema: add is_RTH if existing has it
    if "is_RTH" in existing.columns and "is_RTH" not in new_data.columns:
        new_data = _add_rth_flag(new_data)

    # Match columns and dtypes
    new_data = new_data[[c for c in existing.columns if c in new_data.columns]]
    if "volume" in new_data.columns:
        new_data["volume"] = new_data["volume"].astype(existing["volume"].dtype)

    # Verify continuity at join point
    join_gap = abs(new_data.iloc[0]["open"] - existing.iloc[-1]["close"])
    join_pct = join_gap / existing.iloc[-1]["close"] * 100
    logger.info("%s: join gap=%.2f pts (%.2f%%)", fname, join_gap, join_pct)
    if join_pct > 3:
        logger.warning("LARGE gap at join -- verify Panama adjustment!")

    # Append and save
    combined = pd.concat([existing, new_data])
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    combined.to_parquet(primary_path, engine="pyarrow", index=True)
    rows_added = len(combined) - len(existing)
    logger.info("%s: +%d rows -> %d total (%s to %s)",
                fname, rows_added, len(combined),
                combined.index[0], combined.index[-1])

    # Update panama alias
    alias_name = PANAMA_ALIASES.get(fname)
    if alias_name:
        alias_path = DATA_DIR / alias_name
        if alias_path.exists():
            combined.to_parquet(alias_path, engine="pyarrow", index=True)
            logger.info("  Updated %s", alias_name)

    return rows_added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(port: int = 7496, client_id: int = 106,
               target_end: str = "2026-03-27"):
    target = date.fromisoformat(target_end)
    ib = await _connect("127.0.0.1", port, client_id)

    total = 0
    try:
        for symbol, suffix, bar_size, use_rth in DOWNLOADS:
            total += await append_timeframe(ib, symbol, suffix, bar_size,
                                            use_rth, target)
            await asyncio.sleep(PACING_DELAY)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    logger.info("Done. Total rows added: %d", total)

    # Verification
    logger.info("--- Verification ---")
    for symbol, suffix, _, _ in DOWNLOADS:
        fname = f"{symbol}_{suffix}.parquet"
        path = DATA_DIR / fname
        if path.exists():
            df = pd.read_parquet(path, engine="pyarrow")
            logger.info("  %s: %d rows, %s -> %s",
                        fname, len(df), df.index[0], df.index[-1])

            if len(df) > 1:
                diffs = df.index.to_series().diff().dropna()
                threshold = (pd.Timedelta(hours=72) if "m" in suffix
                             else pd.Timedelta(days=5))
                large_gaps = diffs[diffs > threshold]
                if len(large_gaps) > 0:
                    logger.warning("  %s: %d gaps > threshold:",
                                   fname, len(large_gaps))
                    for ts, gap in large_gaps.tail(3).items():
                        logger.warning("    %s -- gap of %s",
                                       ts.strftime("%Y-%m-%d"), gap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append IBKR NQ/ES data")
    parser.add_argument("--port", type=int, default=7496)
    parser.add_argument("--client-id", type=int, default=106)
    parser.add_argument("--target-end", default="2026-03-27")
    args = parser.parse_args()
    asyncio.run(main(port=args.port, client_id=args.client_id,
                     target_end=args.target_end))
