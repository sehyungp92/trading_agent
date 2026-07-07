"""Central IBKR sync command for all backtest data families."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from .alignment import check_symbol_alignment, format_alignment_result
from .bars import connect_ib, download_historical_bars, download_physical_futures_panama_bars
from .models import BarDownloadRequest, ConnectionSettings, DownloadResult
from .pacing import RequestPacer
from .requirements import family_bar_requirements

logger = logging.getLogger(__name__)


async def sync_families(
    *,
    families: list[str],
    years: int = 2,
    latest: bool = False,
    dry_run: bool = False,
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 107,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    normalized = [family.lower() for family in families]
    if "all" in normalized:
        normalized = ["scalp", "momentum", "swing", "stock"]

    if "scalp" in normalized:
        from backtests.scalp.data.downloader import download_scalp_data

        results.extend(
            await download_scalp_data(
                output_dir=Path("data/raw"),
                years=years,
                latest=latest,
                dry_run=dry_run,
                host=host,
                port=port,
                client_id=client_id,
            )
        )
        normalized = [family for family in normalized if family != "scalp"]

    if not normalized:
        return results

    derive_momentum = "momentum" in normalized
    settings = ConnectionSettings(host=host, port=port, client_id=client_id)
    pacer = RequestPacer()
    ib = None
    if not dry_run:
        ib = await connect_ib(settings)

    try:
        for family in normalized:
            requirements = list(family_bar_requirements(family, years=years))
            logger.info("[%s] %d symbols to sync", family, len(requirements))
            for i, requirement in enumerate(requirements, 1):
                end = datetime.now(timezone.utc)
                request = BarDownloadRequest(
                    symbol=requirement.symbol,
                    timeframe=requirement.timeframe,
                    sec_type=requirement.sec_type,
                    exchange=requirement.exchange,
                    trading_class=requirement.trading_class,
                    primary_exchange=requirement.primary_exchange,
                    what_to_show=requirement.what_to_show,
                    use_rth=requirement.use_rth,
                    duration=requirement.duration,
                    end=end,
                    output_dir=requirement.output_dir,
                    family=requirement.family,
                )
                output_path = requirement.output_dir / f"{requirement.symbol}_{requirement.timeframe}.parquet"
                if dry_run:
                    result = await _download_requirement(
                        None,
                        request,
                        output_path=output_path,
                        pacer=pacer,
                        dry_run=True,
                        latest_only=latest,
                    )
                else:
                    logger.info(
                        "[%s] (%d/%d) %s %s %s",
                        family, i, len(requirements),
                        requirement.symbol, requirement.timeframe,
                        requirement.what_to_show,
                    )
                    result = await _download_requirement(
                        ib,
                        request,
                        output_path=output_path,
                        pacer=pacer,
                        dry_run=False,
                        latest_only=latest,
                    )
                if isinstance(result, DownloadResult):
                    if not dry_run and result.rows:
                        logger.info(
                            "  -> %s %s: %d rows [%s .. %s]",
                            result.symbol, result.timeframe,
                            result.rows, result.start, result.end,
                        )
                    results.append(result)
    finally:
        if ib is not None:
            ib.disconnect()
    if derive_momentum:
        results.extend(_derive_momentum_compatibility_files(dry_run=dry_run))
    return results


async def _download_requirement(
    ib,
    request: BarDownloadRequest,
    *,
    output_path: Path,
    pacer: RequestPacer,
    dry_run: bool,
    latest_only: bool,
) -> DownloadResult:
    if request.family == "momentum" and request.sec_type.upper() == "FUT":
        return await download_physical_futures_panama_bars(
            ib,
            request,
            output_path=output_path,
            pacer=pacer,
            dry_run=dry_run,
            latest_only=latest_only,
        )
    return await download_historical_bars(
        ib,
        request,
        output_path=output_path,
        pacer=pacer,
        dry_run=dry_run,
        latest_only=latest_only,
    )


def _derive_momentum_compatibility_files(*, dry_run: bool) -> list[DownloadResult]:
    output_dir = Path("backtests/momentum/data/raw")
    symbols = ("NQ", "MNQ")
    targets = ("15m", "30m", "1h", "4h", "1d")
    results: list[DownloadResult] = []
    if dry_run:
        for symbol in symbols:
            results.append(
                DownloadResult(
                    symbol=symbol,
                    timeframe="derived",
                    dry_run=True,
                    messages=[f"{symbol} momentum: derive {','.join(targets)} from 5m"],
                    paths=[output_dir / f"{symbol}_{target}.parquet" for target in targets],
                )
            )
        return results

    from backtests.momentum.data.downloader import derive_aligned_momentum_timeframes

    for symbol in symbols:
        try:
            paths = derive_aligned_momentum_timeframes(symbol, output_dir, targets=targets)
        except FileNotFoundError as exc:
            results.append(
                DownloadResult(
                    symbol=symbol,
                    timeframe="derived",
                    what_to_show="TRADES",
                    messages=[str(exc)],
                )
            )
            continue
        results.append(
            DownloadResult(
                symbol=symbol,
                timeframe="derived",
                what_to_show="TRADES",
                paths=list(paths.values()),
                messages=[f"{symbol} momentum: derived {','.join(paths)} from 5m"],
            )
        )
    return results


def _split_cli_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def _split_timeframes(value: str) -> tuple[str, ...]:
    normalized = []
    for item in _split_cli_list(value):
        normalized.append("1d" if item == "1" else item)
    return tuple(normalized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared IBKR data sync.")
    subparsers = parser.add_subparsers(dest="command")
    sync_parser = subparsers.add_parser("sync", help="Sync one or more strategy data families")
    sync_parser.add_argument("--families", default="scalp", help="Comma-separated family list or all")
    sync_parser.add_argument("--years", type=int, default=2)
    sync_parser.add_argument("--latest", action="store_true", help="Prefer append/update behavior where supported")
    sync_parser.add_argument("--dry-run", action="store_true")
    sync_parser.add_argument("--host", default="127.0.0.1")
    sync_parser.add_argument("--port", type=int, default=4002)
    sync_parser.add_argument("--client-id", type=int, default=107)
    align_parser = subparsers.add_parser("check-alignment", help="Check cross-timeframe bar alignment")
    align_parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    align_parser.add_argument("--symbol", default="NQ")
    align_parser.add_argument("--base", default="1m")
    align_parser.add_argument("--targets", default="5m,1h,4h")
    align_parser.add_argument("--price-tolerance", type=float, default=1e-9)
    align_parser.add_argument("--volume-tolerance", type=float, default=0.0)
    momentum_align = subparsers.add_parser(
        "check-momentum-alignment",
        help="Check momentum compatibility files against strategy 5m-derived bars",
    )
    momentum_align.add_argument("--data-dir", type=Path, default=Path("backtests/momentum/data/raw"))
    momentum_align.add_argument("--symbols", default="NQ,MNQ")
    momentum_align.add_argument("--targets", default="15m,30m,1h,4h,1d")
    repair_parser = subparsers.add_parser(
        "repair-momentum-alignment",
        help="Derive momentum compatibility files from canonical 5m data",
    )
    repair_parser.add_argument("--data-dir", type=Path, default=Path("backtests/momentum/data/raw"))
    repair_parser.add_argument("--symbols", default="NQ,MNQ")
    repair_parser.add_argument("--targets", default="15m,30m,1h,4h,1d")
    repair_parser.add_argument("--write", action="store_true", help="Write derived files; default only prints the plan")
    repair_parser.add_argument("--no-backup", action="store_true", help="Do not preserve existing files as *_direct.parquet")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check-alignment":
        results = check_symbol_alignment(
            args.data_dir,
            args.symbol,
            base_timeframe=args.base,
            target_timeframes=_split_timeframes(args.targets),
            price_tolerance=args.price_tolerance,
            volume_tolerance=args.volume_tolerance,
        )
        for result in results:
            print(format_alignment_result(result))
        return 0 if all(result.ok for result in results) else 1
    if args.command == "check-momentum-alignment":
        from backtests.momentum.data.downloader import check_aligned_momentum_timeframes

        symbols = [item.upper() for item in _split_cli_list(args.symbols)]
        targets = _split_timeframes(args.targets)
        all_results = []
        for symbol in symbols:
            all_results.extend(check_aligned_momentum_timeframes(symbol, args.data_dir, targets=targets))
        for result in all_results:
            print(format_alignment_result(result))
        return 0 if all(result.ok for result in all_results) else 1
    if args.command == "repair-momentum-alignment":
        symbols = [item.upper() for item in _split_cli_list(args.symbols)]
        targets = _split_timeframes(args.targets)
        if not args.write:
            for symbol in symbols:
                print(f"DRY-RUN {symbol} momentum: derive {','.join(targets)} from 5m in {args.data_dir}")
            return 0

        from backtests.momentum.data.downloader import derive_aligned_momentum_timeframes

        for symbol in symbols:
            paths = derive_aligned_momentum_timeframes(
                symbol,
                args.data_dir,
                targets=targets,
                backup_existing=not args.no_backup,
            )
            print(f"OK {symbol} momentum: derived {','.join(paths)} from 5m")
        return 0
    if args.command != "sync":
        parser.print_help()
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    families = _split_cli_list(args.families)
    results = await sync_families(
        families=families,
        years=args.years,
        latest=args.latest,
        dry_run=args.dry_run,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    )
    for result in results:
        prefix = "DRY-RUN" if result.dry_run else "OK"
        detail = "; ".join(result.messages) if result.messages else f"{result.rows} rows"
        print(f"{prefix} {detail}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
