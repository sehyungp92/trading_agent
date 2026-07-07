"""Command-line interface for trading_assistant_data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .authority import finalize_slice_manifests
from .bundle_builder import (
    audit_coverage,
    build_bundle,
    export_filesystem,
    export_single_slice_coverage,
)
from .env import load_local_env
from .hygiene import clean_stale_cme_bid_ask_aliases
from .importer import import_reference_snapshot
from .io import write_json
from .legacy_compare import compare_legacy_source_requests
from .manifests import load_market_manifest
from .normalization import (
    normalize_all,
    normalize_crypto,
    normalize_krx_daily,
    normalize_krx_intraday,
    normalize_reference_trading_bars,
    normalize_trading_seed_data,
    normalize_us_equity_stock_raw,
)
from .repo import resolve_repo_root
from .reproduction import reproduce_data_bundle
from .source_refresh import sync_ibkr_from_source_requests, sync_kis_from_source_requests
from .source_requests import declare_source_requests
from .sources.hyperliquid.sync import sync_hyperliquid
from .validation import report_path, validate_bundle, validate_market_manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        payload = args.func(args)
        _emit(args, payload)
        return 0
    except Exception as exc:
        payload = {"ok": False, "error": str(exc)}
        _emit(args, payload, stderr=True)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading_assistant_data")
    parser.add_argument("--repo-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command")

    import_ref = sub.add_parser("import-reference", help="Copy raw reference snapshots into data/imported")
    _common(import_ref)
    import_ref.add_argument("--snapshot", required=True)
    import_ref.add_argument("--references-root", type=Path, required=True)
    import_ref.set_defaults(func=_cmd_import_reference)

    normalize = sub.add_parser("normalize", help="Normalize imported snapshots into canonical parquet")
    _common(normalize)
    normalize.add_argument("--snapshot", default="2026-05-30")
    normalize.add_argument("--all", action="store_true")
    normalize.add_argument("--crypto", action="store_true")
    normalize.add_argument("--ibkr", action="store_true")
    normalize.add_argument("--krx-intraday", action="store_true")
    normalize.add_argument("--krx-intraday-symbols", default="")
    normalize.add_argument("--krx-intraday-timeframes", default="")
    normalize.add_argument("--krx-daily", action="store_true")
    normalize.add_argument("--us-equity", action="store_true")
    normalize.add_argument("--trading-seeds", action="store_true")
    normalize.set_defaults(func=_cmd_normalize)

    source_requests = sub.add_parser(
        "declare-source-requests",
        help="Map imported legacy parquet files to reproducible source requests",
    )
    _common(source_requests)
    source_requests.add_argument("--snapshot", default="2026-05-30")
    source_requests.add_argument("--output-root", type=Path, default=None)
    source_requests.set_defaults(func=_cmd_declare_source_requests)

    sync = sub.add_parser("sync", help="Source-specific live sync entry points")
    sync_sub = sync.add_subparsers(dest="source", required=True)
    for name in ("ibkr", "hyperliquid", "kis"):
        child = sync_sub.add_parser(name)
        _common(child)
        child.add_argument("--families", default="")
        child.add_argument("--symbols", default="")
        child.add_argument("--symbols-file", type=Path, default=None)
        child.add_argument("--intervals", default="")
        child.add_argument("--start", default="")
        child.add_argument("--end", default="")
        child.add_argument("--years", type=int, default=0)
        child.add_argument("--lookback-days", type=int, default=30)
        child.add_argument("--overlap-bars", type=int, default=200)
        child.add_argument("--source-request-manifest", type=Path, default=None)
        child.add_argument("--max-requests", type=int, default=None)
        if name == "ibkr":
            child.add_argument(
                "--ibkr-coverage-mode",
                choices=["full-legacy", "retention-covered"],
                default="full-legacy",
                help=(
                    "full-legacy requires archived pre-retention CME contract evidence; "
                    "retention-covered trims futures requests to currently resolvable TWS contracts"
                ),
            )
            child.add_argument("--ibkr-contract-probe", type=Path, default=None)
        child.add_argument("--latest", action="store_true")
        child.add_argument("--funding", action="store_true")
        child.add_argument("--daily", action="store_true")
        child.add_argument("--intraday", action="store_true")
        child.set_defaults(func=_cmd_sync)

    clean = sub.add_parser("clean-canonical", help="Remove known stale canonical aliases")
    _common(clean)
    clean.add_argument("--stale-cme-bid-ask", action="store_true")
    clean.set_defaults(func=_cmd_clean_canonical)

    validate_slice = sub.add_parser("validate-slice", help="Validate one MarketDataManifest")
    _common(validate_slice)
    validate_slice.add_argument("--manifest", type=Path, required=True)
    validate_slice.set_defaults(func=_cmd_validate_slice)

    finalize = sub.add_parser("finalize-slices", help="Finalize slice manifests against a committed data snapshot")
    _common(finalize)
    finalize.add_argument("--run-month", required=True)
    finalize.add_argument("--slice-manifest", action="append", type=Path, default=[])
    finalize.add_argument("--requirements-file", type=Path, default=None)
    finalize.add_argument("--data-commit-sha", default="")
    finalize.set_defaults(func=_cmd_finalize_slices)

    build = sub.add_parser("build-bundle", help="Build a monthly DataBundleManifest")
    _common(build)
    build.add_argument("--run-month", required=True)
    build.add_argument("--bot-id", required=True)
    build.add_argument("--strategy-id", required=True)
    build.add_argument("--slice-manifest", action="append", type=Path, default=[])
    build.add_argument("--requirements-file", type=Path, default=None)
    build.set_defaults(func=_cmd_build_bundle)

    export = sub.add_parser("export-filesystem", help="Write FileSystemParquetAdapter exports")
    _common(export)
    export.add_argument("--run-month", required=True)
    export.add_argument("--bundle-manifest", type=Path, default=None)
    export.set_defaults(func=_cmd_export_filesystem)

    export_cov = sub.add_parser("export-single-slice-coverage", help="Write default monthly coverage manifest")
    _common(export_cov)
    export_cov.add_argument("--run-month", required=True)
    export_cov.add_argument("--bot-id", required=True)
    export_cov.add_argument("--strategy-id", required=True)
    export_cov.add_argument("--bundle-manifest", type=Path, default=None)
    export_cov.set_defaults(func=_cmd_export_single_slice)

    audit = sub.add_parser("audit-coverage", help="Summarize monthly slice authority")
    _common(audit)
    audit.add_argument("--run-month", required=True)
    audit.set_defaults(func=_cmd_audit_coverage)

    reproduce = sub.add_parser("reproduce-bundle", help="Recompute committed bundle facts")
    _common(reproduce)
    reproduce.add_argument("--bundle-manifest", type=Path, required=True)
    reproduce.add_argument("--artifact-root", type=Path, default=None)
    reproduce.set_defaults(func=_cmd_reproduce_bundle)

    compare_legacy = sub.add_parser(
        "compare-legacy-source",
        help="Compare legacy parquet files with refreshed source-owned canonical slices",
    )
    _common(compare_legacy)
    compare_legacy.add_argument("--source-request-manifest", type=Path, default=None)
    compare_legacy.add_argument("--families", default="")
    compare_legacy.add_argument("--symbols", default="")
    compare_legacy.add_argument("--intervals", default="")
    compare_legacy.add_argument("--latest-only", action="store_true")
    compare_legacy.add_argument("--artifact-root", type=Path, default=None)
    compare_legacy.set_defaults(func=_cmd_compare_legacy_source)
    return parser


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")


def _repo(args: argparse.Namespace) -> Path:
    repo_root = resolve_repo_root(args.repo_root)
    load_local_env(repo_root)
    return repo_root


def _cmd_import_reference(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    payload = import_reference_snapshot(
        repo_root=repo_root,
        snapshot=args.snapshot,
        references_root=args.references_root,
        dry_run=args.dry_run,
    )
    _write_report(repo_root, "import-reference", payload)
    return payload


def _cmd_normalize(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    selected = args.all or not (
        args.crypto
        or args.ibkr
        or args.krx_intraday
        or args.krx_daily
        or args.us_equity
        or args.trading_seeds
    )
    if selected:
        payload = normalize_all(repo_root, snapshot=args.snapshot, dry_run=args.dry_run)
    else:
        reports = []
        if args.crypto:
            reports.append(normalize_crypto(repo_root, snapshot=args.snapshot, dry_run=args.dry_run))
        if args.ibkr:
            reports.append(normalize_reference_trading_bars(repo_root, snapshot=args.snapshot, dry_run=args.dry_run))
        if args.krx_intraday:
            reports.append(
                normalize_krx_intraday(
                    repo_root,
                    snapshot=args.snapshot,
                    dry_run=args.dry_run,
                    symbols=_split_csv(args.krx_intraday_symbols) or None,
                    timeframes=_split_csv(args.krx_intraday_timeframes) or None,
                )
            )
        if args.krx_daily:
            reports.append(normalize_krx_daily(repo_root, snapshot=args.snapshot, dry_run=args.dry_run))
        if args.us_equity:
            reports.append(normalize_us_equity_stock_raw(repo_root, snapshot=args.snapshot, dry_run=args.dry_run))
        if args.trading_seeds:
            reports.append(normalize_trading_seed_data(repo_root, snapshot=args.snapshot, dry_run=args.dry_run))
        payload = {"snapshot": args.snapshot, "dry_run": args.dry_run, "reports": reports}
    _write_report(repo_root, "normalize", payload)
    _raise_on_report_errors(payload)
    return payload


def _cmd_declare_source_requests(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    artifact_root = args.output_root or (
        repo_root / "data" / "source_requests" / f"reference_snapshot_{args.snapshot}"
    )
    payload = declare_source_requests(
        repo_root=repo_root,
        snapshot=args.snapshot,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    _write_report(
        repo_root,
        "declare-source-requests",
        {
            key: payload[key]
            for key in (
                "schema_version",
                "snapshot",
                "snapshot_root",
                "legacy_file_count",
                "request_count",
                "unclassified_count",
                "unclassified",
                "source_kind_counts",
                "source_counts",
                "legacy_family_counts",
                "market_counts",
            )
        }
        | {
            "dry_run": args.dry_run,
            "artifact_root": str(artifact_root),
            "source_request_manifest": str(artifact_root / "source_request_manifest.json"),
        },
    )
    if payload["unclassified_count"]:
        preview = "; ".join(payload["unclassified"][:10])
        suffix = (
            f"; ... ({payload['unclassified_count']} total unclassified)"
            if payload["unclassified_count"] > 10
            else ""
        )
        raise RuntimeError(f"unclassified legacy parquet files: {preview}{suffix}")
    return payload


def _cmd_sync(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    symbols = _split_csv(args.symbols)
    if args.symbols_file:
        symbols.extend(
            line.strip()
            for line in args.symbols_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    intervals = _split_csv(args.intervals)
    if args.source == "hyperliquid":
        payload = sync_hyperliquid(
            repo_root=repo_root,
            symbols=symbols or None,
            intervals=intervals or None,
            start=args.start or None,
            end=args.end or None,
            years=args.years,
            lookback_days=args.lookback_days,
            latest=args.latest,
            funding=args.funding,
            overlap_bars=args.overlap_bars,
            dry_run=args.dry_run,
        )
        _write_report(repo_root, "sync-hyperliquid", payload)
        return payload
    if args.source == "ibkr":
        payload = sync_ibkr_from_source_requests(
            repo_root=repo_root,
            source_request_manifest=args.source_request_manifest,
            families=_split_csv(args.families),
            symbols=symbols,
            intervals=intervals,
            max_requests=args.max_requests,
            coverage_mode=args.ibkr_coverage_mode,
            contract_probe_path=args.ibkr_contract_probe,
            dry_run=args.dry_run,
        )
        _write_report(repo_root, "sync-ibkr", payload)
        if payload.get("status") == "failed":
            raise RuntimeError(f"IBKR sync failed for {payload.get('failure_count', 0)} requests")
        return payload
    if args.source == "kis":
        payload = sync_kis_from_source_requests(
            repo_root=repo_root,
            source_request_manifest=args.source_request_manifest,
            families=_split_csv(args.families),
            symbols=symbols,
            intervals=intervals,
            max_requests=args.max_requests,
            dry_run=args.dry_run,
        )
        _write_report(repo_root, "sync-kis", payload)
        if payload.get("status") == "failed":
            raise RuntimeError(f"KIS sync failed for {payload.get('failure_count', 0)} requests")
        return payload
    raise RuntimeError(f"unsupported sync source: {args.source}")


def _cmd_clean_canonical(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    if not args.stale_cme_bid_ask:
        raise RuntimeError("select a cleanup target")
    payload = clean_stale_cme_bid_ask_aliases(repo_root=repo_root, dry_run=args.dry_run)
    _write_report(repo_root, "clean-canonical", payload)
    return payload


def _cmd_validate_slice(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    manifest = load_market_manifest(args.manifest)
    report = validate_market_manifest(manifest).to_dict()
    payload = {"manifest": str(args.manifest), **report}
    _write_report(repo_root, "validate-slice", payload)
    if not report["valid"]:
        raise RuntimeError("; ".join(report["errors"]))
    return payload


def _cmd_finalize_slices(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    result = finalize_slice_manifests(
        repo_root=repo_root,
        run_month=args.run_month,
        slice_manifest_paths=args.slice_manifest or None,
        requirements_path=args.requirements_file,
        data_commit_sha=args.data_commit_sha or None,
        dry_run=args.dry_run,
    )
    payload = result.to_dict()
    _write_report(repo_root, "finalize-slices", payload)
    if payload["skipped_count"]:
        raise RuntimeError("; ".join(item["reason"] for item in payload["skipped"][:10]))
    return payload


def _cmd_build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    result = build_bundle(
        repo_root=repo_root,
        run_month=args.run_month,
        bot_id=args.bot_id,
        strategy_id=args.strategy_id,
        slice_manifest_paths=args.slice_manifest or None,
        requirements_path=args.requirements_file,
        dry_run=args.dry_run,
    )
    report = validate_bundle(result.bundle).to_dict()
    payload = {
        "bundle_path": str(result.bundle_path),
        "slice_index_path": str(result.slice_index_path),
        "bundle_checksum": result.bundle.bundle_checksum,
        "status": result.bundle.status.value,
        "dry_run": args.dry_run,
        "validation": report,
    }
    _write_report(repo_root, "build-bundle", payload)
    if not report["valid"]:
        raise RuntimeError(result.bundle.diagnostics_only_reason or "; ".join(report["errors"]))
    return payload


def _cmd_export_filesystem(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    payload = export_filesystem(
        repo_root=repo_root,
        run_month=args.run_month,
        bundle_manifest_path=args.bundle_manifest,
        dry_run=args.dry_run,
    )
    _write_report(repo_root, "export-filesystem", payload)
    return payload


def _cmd_export_single_slice(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    payload = export_single_slice_coverage(
        repo_root=repo_root,
        run_month=args.run_month,
        bot_id=args.bot_id,
        strategy_id=args.strategy_id,
        bundle_manifest_path=args.bundle_manifest,
        dry_run=args.dry_run,
    )
    _write_report(repo_root, "export-single-slice-coverage", payload)
    return payload


def _cmd_audit_coverage(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    payload = audit_coverage(repo_root, run_month=args.run_month)
    _write_report(repo_root, "audit-coverage", payload)
    return payload


def _cmd_reproduce_bundle(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    artifact_root = args.artifact_root or report_path(repo_root, "reproduce-bundle").parent
    payload = reproduce_data_bundle(
        repo_root=repo_root,
        bundle_manifest_path=args.bundle_manifest,
        artifact_root=artifact_root,
    )
    if not payload["ok"]:
        raise RuntimeError("data reproduction failed")
    return payload


def _cmd_compare_legacy_source(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = _repo(args)
    payload = compare_legacy_source_requests(
        repo_root=repo_root,
        source_request_manifest=args.source_request_manifest,
        families=_split_csv(args.families),
        symbols=_split_csv(args.symbols),
        intervals=_split_csv(args.intervals),
        latest_only=args.latest_only,
        artifact_root=args.artifact_root,
    )
    _write_report(repo_root, "compare-legacy-source", payload)
    if not payload["ok"]:
        blocked = payload.get("blocked_count", 0)
        failed = payload.get("failed_count", 0)
        raise RuntimeError(f"legacy/source comparison did not pass: blocked={blocked}, failed={failed}")
    return payload


def _write_report(repo_root: Path, command: str, payload: Any) -> None:
    write_json(report_path(repo_root, command), payload)


def _raise_on_report_errors(payload: dict[str, Any]) -> None:
    errors: list[str] = []
    for report in payload.get("reports", []):
        errors.extend(str(error) for error in report.get("errors", []))
    if errors:
        preview = "; ".join(errors[:10])
        suffix = f"; ... ({len(errors)} total errors)" if len(errors) > 10 else ""
        raise RuntimeError(preview + suffix)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _emit(args: argparse.Namespace, payload: Any, *, stderr: bool = False) -> None:
    stream = sys.stderr if stderr else sys.stdout
    if getattr(args, "json_output", False):
        print(json.dumps(payload, indent=2, sort_keys=True, default=str), file=stream)
    else:
        if isinstance(payload, dict):
            ok = payload.get("ok", True)
            status = payload.get("status") or payload.get("name") or args.command
            print(f"{'OK' if ok is not False else 'ERROR'} {status}", file=stream)
            if payload.get("error"):
                print(payload["error"], file=stream)
            elif payload.get("bundle_path"):
                print(payload["bundle_path"], file=stream)
            elif payload.get("manifest_paths"):
                print(f"slice manifests: {len(payload['manifest_paths'])}", file=stream)
        else:
            print(payload, file=stream)


if __name__ == "__main__":
    raise SystemExit(main())
