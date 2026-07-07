"""Scalp-family IBKR downloader built on the shared historical-data core."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtests.shared.data.ibkr.alignment import check_symbol_alignment, format_alignment_result
from backtests.shared.data.ibkr.bars import connect_ib, download_contract_bars
from backtests.shared.data.ibkr.contracts import (
    FuturesContractSpec,
    active_contract,
    generate_quarterly_contracts,
    roll_schedule,
)
from backtests.shared.data.ibkr.models import ConnectionSettings, DownloadResult
from backtests.shared.data.ibkr.pacing import RequestPacer
from backtests.shared.data.ibkr.stitch import stitch_panama
from backtests.shared.data.ibkr.store import (
    add_rth_flag,
    detect_large_gaps,
    merge_frames,
    read_parquet_if_exists,
    resample_ohlcv,
    rich_bar_path,
    write_compatibility_bars,
    write_manifest,
    write_parquet_atomic,
)
from backtests.shared.data.ibkr.ticks import download_tick_windows, session_windows

logger = logging.getLogger(__name__)

SCALP_SYMBOLS = ("NQ", "ES")
SCALP_REQUIRED_TICK_SYMBOLS = ("NQ",)
SCALP_TIMEFRAMES = ("1m", "5m", "1h", "4h", "1d")
SCALP_DIRECT_TIMEFRAMES = ("1m", "5m", "1h", "1d")
SCALP_DERIVED_TIMEFRAMES = {
    "5m": ("1m", "5min"),
    "1h": ("1m", "1h"),
    "4h": ("1m", "4h"),
}
SCALP_BID_ASK_TIMEFRAMES = ("1m",)


async def download_scalp_data(
    *,
    output_dir: Path = Path("data/raw"),
    symbols: list[str] | None = None,
    years: int = 2,
    latest: bool = False,
    dry_run: bool = False,
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 107,
    include_micro: bool = False,
    include_bid_ask: bool = True,
    tick_mode: str = "recent-gaps",
    tick_symbols: list[str] | None = None,
    tick_days: int = 5,
    merge_ticks: bool = False,
) -> list[DownloadResult]:
    """Download scalp requirements and emit both rich and compatibility files."""
    output_dir = Path(output_dir)
    requested_symbols = [symbol.upper() for symbol in (symbols or list(SCALP_SYMBOLS))]
    if include_micro and "MNQ" not in requested_symbols:
        requested_symbols.append("MNQ")
    requested_tick_symbols = _normalize_tick_symbols(tick_symbols, requested_symbols)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years)
    settings = ConnectionSettings(host=host, port=port, client_id=client_id)
    pacer = RequestPacer()
    results: list[DownloadResult] = []
    ib = None
    if not dry_run:
        ib = await connect_ib(settings)

    manifest_runs: list[dict[str, object]] = []
    try:
        for symbol in requested_symbols:
            contracts = generate_quarterly_contracts(symbol, start=start, end=end, years=years)
            for timeframe in SCALP_DIRECT_TIMEFRAMES:
                results.extend(
                    await _download_symbol_timeframe(
                        ib,
                        symbol=symbol,
                        contracts=contracts,
                        timeframe=timeframe,
                        what_to_show="TRADES",
                        output_dir=output_dir,
                        start=start,
                        end=end,
                        pacer=pacer,
                        latest=latest,
                        dry_run=dry_run,
                    )
                )
            results.extend(
                await _derive_symbol_timeframes(
                    symbol=symbol,
                    output_dir=output_dir,
                    dry_run=dry_run,
                )
            )
            if not dry_run:
                for alignment in check_symbol_alignment(output_dir, symbol):
                    message = format_alignment_result(alignment)
                    if not alignment.ok:
                        logger.warning(message)
                    results.append(
                        DownloadResult(
                            symbol=symbol,
                            timeframe=alignment.target_timeframe,
                            what_to_show="ALIGNMENT",
                            rows=alignment.compared_rows,
                            messages=[message],
                        )
                    )
            if include_bid_ask:
                for timeframe in SCALP_BID_ASK_TIMEFRAMES:
                    results.extend(
                        await _download_symbol_timeframe(
                            ib,
                            symbol=symbol,
                            contracts=contracts,
                            timeframe=timeframe,
                            what_to_show="BID_ASK",
                            output_dir=output_dir,
                            start=start,
                            end=end,
                            pacer=pacer,
                            latest=latest,
                            dry_run=dry_run,
                        )
                    )
            if tick_mode != "none" and symbol in requested_tick_symbols:
                results.extend(
                    await _download_symbol_ticks(
                        ib,
                        symbol=symbol,
                        contracts=contracts,
                        output_dir=output_dir,
                        end=end,
                        tick_days=tick_days,
                        pacer=pacer,
                        dry_run=dry_run,
                        merge_ticks=merge_ticks,
                    )
                )

            manifest_runs.append(
                {
                    "symbol": symbol,
                    "contracts": [contract.yyyymm for contract in contracts],
                    "timeframes": list(SCALP_TIMEFRAMES),
                    "bid_ask_timeframes": list(SCALP_BID_ASK_TIMEFRAMES if include_bid_ask else ()),
                    "tick_mode": tick_mode,
                    "tick_symbols": requested_tick_symbols,
                }
            )
    finally:
        if ib is not None:
            ib.disconnect()

    if not dry_run:
        write_manifest(
            output_dir / "scalp" / "meta" / "scalp_download_manifest.json",
            {
                "source": "IBKR",
                "years": years,
                "latest": latest,
                "runs": manifest_runs,
            },
        )
    return results


def _normalize_tick_symbols(tick_symbols: list[str] | None, requested_symbols: list[str]) -> list[str]:
    requested = {symbol.upper() for symbol in requested_symbols}
    if tick_symbols is None:
        candidates = SCALP_REQUIRED_TICK_SYMBOLS
    else:
        candidates = tuple(symbol.upper() for symbol in tick_symbols)
    normalized: list[str] = []
    for symbol in candidates:
        if symbol in requested and symbol not in normalized:
            normalized.append(symbol)
    return normalized


async def _download_symbol_timeframe(
    ib,
    *,
    symbol: str,
    contracts: list[FuturesContractSpec],
    timeframe: str,
    what_to_show: str,
    output_dir: Path,
    start: datetime,
    end: datetime,
    pacer: RequestPacer,
    latest: bool,
    dry_run: bool,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    contract_frames: dict[str, pd.DataFrame] = {}
    ordered = sorted(contracts, key=lambda contract: contract.expiry)
    for idx, contract in enumerate(ordered):
        previous_roll = ordered[idx - 1].roll_date if idx > 0 else start.date()
        request_start = max(start, datetime.combine(previous_roll, time.min, tzinfo=timezone.utc) - timedelta(days=7))
        request_end = min(end, datetime.combine(contract.expiry, time.max, tzinfo=timezone.utc) + timedelta(days=1))
        if request_end <= request_start:
            continue
        path = rich_bar_path(output_dir, symbol, contract.yyyymm, timeframe, what_to_show)
        result = await download_contract_bars(
            ib,
            contract,
            timeframe=timeframe,
            start=request_start,
            end=request_end,
            output_path=path,
            what_to_show=what_to_show,
            use_rth=timeframe == "1d",
            pacer=pacer,
            dry_run=dry_run,
            latest_only=latest,
        )
        results.append(result)
        if not dry_run:
            frame = read_parquet_if_exists(path)
            if not frame.empty:
                contract_frames[contract.yyyymm] = frame

    if dry_run:
        return results

    emit_flat_compatibility = timeframe in {"1m", "1d", "daily"}
    if what_to_show.upper() == "TRADES":
        continuous = stitch_panama(contract_frames, roll_schedule(ordered), tick_size=ordered[0].tick_size if ordered else 0.25)
        if not continuous.empty:
            continuous = add_rth_flag(continuous)
            paths: list[Path] = []
            if emit_flat_compatibility:
                paths.extend(write_compatibility_bars(continuous, output_dir, symbol, timeframe))
            continuous_folder = "continuous" if emit_flat_compatibility else "continuous_direct"
            continuous_path = output_dir / "scalp" / symbol / continuous_folder / f"{symbol}_{timeframe}_panama.parquet"
            write_parquet_atomic(continuous, continuous_path)
            paths.append(continuous_path)
            gaps = detect_large_gaps(continuous, timeframe)
            output_kind = "compatibility" if emit_flat_compatibility else "direct reference"
            results.append(
                DownloadResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    what_to_show=what_to_show,
                    rows=len(continuous),
                    start=continuous.index[0].to_pydatetime(),
                    end=continuous.index[-1].to_pydatetime(),
                    paths=paths,
                    messages=[f"{symbol} {timeframe} {output_kind}: {len(continuous)} rows, {len(gaps)} large gaps"],
                )
            )
    else:
        merged = merge_frames(*contract_frames.values())
        if not merged.empty:
            path = output_dir / f"{symbol}_{timeframe}_{what_to_show.lower()}.parquet"
            write_parquet_atomic(merged, path)
            results.append(
                DownloadResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    what_to_show=what_to_show,
                    rows=len(merged),
                    start=merged.index[0].to_pydatetime(),
                    end=merged.index[-1].to_pydatetime(),
                    paths=[path],
                    messages=[f"{symbol} {timeframe} {what_to_show}: {len(merged)} rows"],
                )
            )
    return results


async def _derive_symbol_timeframes(
    *,
    symbol: str,
    output_dir: Path,
    dry_run: bool,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for target_timeframe, (source_timeframe, rule) in SCALP_DERIVED_TIMEFRAMES.items():
        target_paths = [output_dir / f"{symbol}_{target_timeframe}.parquet"]
        if dry_run:
            results.append(
                DownloadResult(
                    symbol=symbol,
                    timeframe=target_timeframe,
                    what_to_show="TRADES",
                    dry_run=True,
                    paths=target_paths,
                    messages=[f"{symbol} {target_timeframe}: derived from {source_timeframe}"],
                )
            )
            continue

        source = read_parquet_if_exists(output_dir / f"{symbol}_{source_timeframe}.parquet")
        derived = resample_ohlcv(source, rule)
        if derived.empty:
            results.append(
                DownloadResult(
                    symbol=symbol,
                    timeframe=target_timeframe,
                    what_to_show="TRADES",
                    rows=0,
                    paths=target_paths,
                    messages=[f"{symbol} {target_timeframe}: source {source_timeframe} unavailable"],
                )
            )
            continue
        derived = add_rth_flag(derived)
        paths = write_compatibility_bars(derived, output_dir, symbol, target_timeframe)
        continuous_path = output_dir / "scalp" / symbol / "continuous" / f"{symbol}_{target_timeframe}_panama.parquet"
        write_parquet_atomic(derived, continuous_path)
        paths.append(continuous_path)
        results.append(
            DownloadResult(
                symbol=symbol,
                timeframe=target_timeframe,
                what_to_show="TRADES",
                rows=len(derived),
                start=derived.index[0].to_pydatetime(),
                end=derived.index[-1].to_pydatetime(),
                paths=paths,
                messages=[f"{symbol} {target_timeframe}: derived {len(derived)} rows from {source_timeframe}"],
            )
        )
    return results


async def _download_symbol_ticks(
    ib,
    *,
    symbol: str,
    contracts: list[FuturesContractSpec],
    output_dir: Path,
    end: datetime,
    tick_days: int,
    pacer: RequestPacer,
    dry_run: bool,
    merge_ticks: bool,
) -> list[DownloadResult]:
    current = active_contract(contracts, as_of=end)
    if current is None:
        return []
    windows = session_windows(end=end, days=tick_days)
    results: list[DownloadResult] = []
    windows_by_contract: dict[str, tuple[FuturesContractSpec, list[tuple[datetime, datetime]]]] = {}
    for start, window_end in windows:
        contract = active_contract(contracts, as_of=window_end) or current
        _existing_contract, grouped_windows = windows_by_contract.setdefault(contract.yyyymm, (contract, []))
        grouped_windows.append((start, window_end))
    for contract, grouped_windows in windows_by_contract.values():
        for tick_type in ("TRADES", "BID_ASK"):
            merge_path = output_dir / f"{symbol}_ticks.parquet" if merge_ticks and tick_type == "TRADES" else None
            results.append(
                await download_tick_windows(
                    ib,
                    contract,
                    grouped_windows,
                    output_root=output_dir,
                    tick_type=tick_type,
                    pacer=pacer,
                    dry_run=dry_run,
                    merge_output_path=merge_path,
                )
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download scalp IBKR data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--symbols", default="NQ,ES")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=107)
    parser.add_argument("--include-micro", action="store_true")
    parser.add_argument("--skip-bid-ask", action="store_true")
    parser.add_argument("--ticks", choices=["none", "recent-gaps"], default="recent-gaps")
    parser.add_argument(
        "--tick-symbols",
        default="NQ",
        help="Comma-separated symbols for tick downloads; use NQ,ES to include optional ES ticks.",
    )
    parser.add_argument("--tick-days", type=int, default=5)
    parser.add_argument("--merge-ticks", action="store_true")
    parser.add_argument(
        "--strict-alignment",
        action="store_true",
        help="Exit non-zero if derived 5m/1h/4h files do not align with the 1m source.",
    )
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    tick_symbols = [item.strip().upper() for item in args.tick_symbols.split(",") if item.strip()]
    results = await download_scalp_data(
        output_dir=args.output_dir,
        symbols=symbols,
        years=args.years,
        latest=args.latest,
        dry_run=args.dry_run,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        include_micro=args.include_micro,
        include_bid_ask=not args.skip_bid_ask,
        tick_mode=args.ticks,
        tick_symbols=tick_symbols,
        tick_days=args.tick_days,
        merge_ticks=args.merge_ticks,
    )
    for result in results:
        prefix = "DRY-RUN" if result.dry_run else "OK"
        detail = "; ".join(result.messages) if result.messages else f"{result.rows} rows"
        print(f"{prefix} {detail}")
    failed_alignment = _failed_alignment_messages(results)
    if args.strict_alignment and failed_alignment:
        for message in failed_alignment:
            print(f"ERROR alignment failed: {message}")
        return 1
    return 0


def _failed_alignment_messages(results: list[DownloadResult]) -> list[str]:
    return [
        message
        for result in results
        if result.what_to_show == "ALIGNMENT"
        for message in result.messages
        if not message.startswith("OK ")
    ]


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
