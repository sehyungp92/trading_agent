"""Run read-only source refreshes from declared source-request manifests."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .repo import git_commit_sha
from .manifests import load_market_manifest
from .slices import is_authoritative_slice_manifest
from .sources.ibkr.cme_nq_read_only import CmeNqRefreshRequest, IBKRCmeNqReadOnlyAdapter
from .sources.ibkr.live_read_only import IBAsyncHistoricalBarProvider, _contract_specs_for_request
from .sources.ibkr.us_equity_read_only import IBKRUsEquityReadOnlyAdapter, UsEquityRefreshRequest
from .sources.kis.krx_read_only import KISKrxReadOnlyAdapter, KisApiKrxProvider, KrxRefreshRequest

IBKR_COVERAGE_FULL_LEGACY = "full-legacy"
IBKR_COVERAGE_RETENTION_COVERED = "retention-covered"
IBKR_COVERAGE_MODES = frozenset({IBKR_COVERAGE_FULL_LEGACY, IBKR_COVERAGE_RETENTION_COVERED})
DEFAULT_IBKR_CONTRACT_PROBE = (
    "data/validation_reports/approval_run/2026-06-01/ibkr-contract-resolution-probe.json"
)
STRATEGY_DATA_FAMILY_BY_LEGACY_FAMILY = {
    "k_stock_kis_intraday": "k_stock_olr_kalcb",
    "k_stock_krx_lrs": "k_stock_olr_kalcb",
    "trading_momentum": "trading_momentum_family",
    "trading_swing": "trading_swing_family",
    "trading_stock": "trading_stock",
}


def sync_ibkr_from_source_requests(
    *,
    repo_root: Path,
    source_request_manifest: Path | None = None,
    families: list[str] | None = None,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    max_requests: int | None = None,
    coverage_mode: str = IBKR_COVERAGE_FULL_LEGACY,
    contract_probe_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    requests = _select_requests(
        repo_root=repo_root,
        source_request_manifest=source_request_manifest,
        source="ibkr",
        families=families,
        symbols=symbols,
        intervals=intervals,
        max_requests=max_requests,
    )
    requests, coverage_report, preflight_failures = _prepare_ibkr_coverage_requests(
        repo_root=repo_root,
        requests=requests,
        coverage_mode=coverage_mode,
        contract_probe_path=contract_probe_path,
        dry_run=dry_run,
    )
    if dry_run:
        return _planned_payload(
            "ibkr",
            requests,
            dry_run=True,
            coverage_mode=coverage_mode,
            coverage_report=coverage_report,
            preflight_failures=preflight_failures,
        )
    if preflight_failures:
        return _complete_payload(
            "ibkr",
            requests,
            [],
            preflight_failures,
            coverage_mode=coverage_mode,
            coverage_report=coverage_report,
        )
    _require_network_write_enabled(source="ibkr", request_count=len(requests))
    provider = IBAsyncHistoricalBarProvider()
    results = []
    failures = []
    cme_adapter = IBKRCmeNqReadOnlyAdapter(provider)
    us_adapter = IBKRUsEquityReadOnlyAdapter(provider)
    for source_request in requests:
        try:
            if source_request["source_kind"] == "ibkr_cme_futures_historical_bars":
                results.append(
                    cme_adapter.refresh_historical_bars(
                        repo_root=repo_root,
                        request=_cme_request(source_request, repo_root),
                    ).to_dict()
                )
            elif source_request["source_kind"] == "ibkr_us_equity_historical_bars":
                results.append(
                    us_adapter.refresh_historical_bars(
                        repo_root=repo_root,
                        request=_us_equity_request(source_request, repo_root),
                    ).to_dict()
                )
            else:
                raise ValueError(
                    f"unsupported IBKR source request kind: {source_request['source_kind']}"
                )
        except Exception as exc:
            failures.append({**_request_summary(source_request), "error": str(exc)})
    return _complete_payload(
        "ibkr",
        requests,
        results,
        failures,
        coverage_mode=coverage_mode,
        coverage_report=coverage_report,
    )


def sync_kis_from_source_requests(
    *,
    repo_root: Path,
    source_request_manifest: Path | None = None,
    families: list[str] | None = None,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    max_requests: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    requests = _select_requests(
        repo_root=repo_root,
        source_request_manifest=source_request_manifest,
        source="kis",
        families=families,
        symbols=symbols,
        intervals=intervals,
        max_requests=max_requests,
    )
    requests = _filter_kis_declared_strategy_requirements(repo_root, requests, families)
    requests, coverage_report, preflight_failures = _prepare_kis_incremental_requests(
        repo_root=repo_root,
        requests=requests,
    )
    if dry_run:
        return _planned_payload(
            "kis",
            requests,
            dry_run=True,
            coverage_report=coverage_report,
            preflight_failures=preflight_failures,
        )
    if preflight_failures:
        return _complete_payload(
            "kis",
            requests,
            [],
            preflight_failures,
            coverage_report=coverage_report,
        )
    skipped = [_kis_skipped_result(item) for item in requests if item.get("sync_action") == "skip"]
    active_requests = [item for item in requests if item.get("sync_action") != "skip"]
    if not active_requests:
        return _complete_payload(
            "kis",
            requests,
            skipped,
            [],
            coverage_report=coverage_report,
        )
    _require_network_write_enabled(source="kis", request_count=len(active_requests))
    adapter = KISKrxReadOnlyAdapter(KisApiKrxProvider())
    results = list(skipped)
    failures = []
    progress_path = _kis_progress_path(repo_root)
    for source_request in requests:
        if source_request.get("sync_action") == "skip":
            continue
        _append_progress(
            progress_path,
            {
                "event": "start",
                "source": "kis",
                **_request_summary(source_request),
                "at": _pulled_at(),
            },
        )
        try:
            result = (
                adapter.refresh_historical_bars(
                    repo_root=repo_root,
                    request=_krx_request(source_request, repo_root),
                ).to_dict()
            )
            results.append(result)
            _append_progress(progress_path, {"event": "complete", **result, "at": _pulled_at()})
        except Exception as exc:
            failure = {**_request_summary(source_request), "error": str(exc)}
            failures.append(failure)
            _append_progress(progress_path, {"event": "failed", **failure, "at": _pulled_at()})
    return _complete_payload("kis", requests, results, failures, coverage_report=coverage_report)


def _select_requests(
    *,
    repo_root: Path,
    source_request_manifest: Path | None,
    source: str,
    families: list[str] | None,
    symbols: list[str] | None,
    intervals: list[str] | None,
    max_requests: int | None,
) -> list[dict[str, Any]]:
    manifest_path = source_request_manifest or (
        Path(repo_root)
        / "data"
        / "source_requests"
        / "reference_snapshot_2026-05-30"
        / "source_request_manifest.json"
    )
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    family_set = {item.strip() for item in families or [] if item.strip()}
    symbol_set = {item.strip().upper() for item in symbols or [] if item.strip()}
    interval_set = {item.strip() for item in intervals or [] if item.strip()}
    selected = []
    for item in payload.get("requests", []):
        if item.get("source") != source:
            continue
        if family_set and not {
            str(item.get("legacy_family") or ""),
            _strategy_data_family(item),
        }.intersection(family_set):
            continue
        if symbol_set and str(item.get("symbol", "")).upper() not in symbol_set:
            continue
        if interval_set and item.get("timeframe") not in interval_set:
            continue
        selected.append(item)
    selected.sort(key=lambda item: (item["legacy_family"], item["symbol"], item["timeframe"], item["legacy_path"]))
    if max_requests is not None:
        selected = selected[:max_requests]
    return selected


def _prepare_kis_incremental_requests(
    *,
    repo_root: Path,
    requests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    max_days = int(os.getenv("KIS_MAX_LIVE_REFRESH_DAYS", "45"))
    overlap_days = int(os.getenv("KIS_INCREMENTAL_OVERLAP_DAYS", "7"))
    allow_full_history = os.getenv("KIS_ALLOW_FULL_HISTORY_REFRESH", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    coverage_report: dict[str, Any] = {
        "coverage_mode": "kis_incremental_or_skip_existing_authority",
        "max_live_refresh_days": max_days,
        "overlap_days": overlap_days,
        "adjustments": [],
    }
    prepared: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_request in requests:
        if source_request.get("source_kind") != "kis_krx_intraday_bars":
            prepared.append(source_request)
            continue
        request = source_request["download_request"]
        start = _parse_ts(request["start"])
        end = _parse_ts(request["end"])
        latest = _latest_authoritative_kis_end(repo_root, source_request)
        copied = deepcopy(source_request)
        if latest is not None and latest >= end:
            copied["sync_action"] = "skip"
            copied["coverage_note"] = "existing authoritative KIS slice covers requested window"
            coverage_report["adjustments"].append(
                {
                    **_request_summary(copied),
                    "sync_action": "skip",
                    "existing_end": latest.isoformat(),
                }
            )
            prepared.append(copied)
            continue
        if latest is not None:
            incremental_start = max(start, latest - timedelta(days=overlap_days))
            copied["download_request"]["start"] = incremental_start.isoformat()
            copied["coverage_note"] = (
                "incremental KIS refresh from existing authoritative coverage "
                f"with {overlap_days} day overlap"
            )
            coverage_report["adjustments"].append(
                {
                    **_request_summary(source_request),
                    "sync_action": "incremental",
                    "original_start": start.isoformat(),
                    "incremental_start": incremental_start.isoformat(),
                    "existing_end": latest.isoformat(),
                }
            )
            prepared.append(copied)
            continue
        days = max(0, (end - start).days)
        if days > max_days and not allow_full_history:
            failures.append(
                {
                    **_request_summary(source_request),
                    "error": (
                        "KIS live sync refused an unseeded full-history intraday rebuild. "
                        "Import/normalize the archived k_stock KIS updater parquet first, "
                        "then run live KIS as an incremental append/repair, or set "
                        "KIS_ALLOW_FULL_HISTORY_REFRESH=true intentionally."
                    ),
                    "requested_days": days,
                    "max_live_refresh_days": max_days,
                }
            )
            continue
        prepared.append(copied)
    return prepared, coverage_report, failures


def _filter_kis_declared_strategy_requirements(
    repo_root: Path,
    requests: list[dict[str, Any]],
    families: list[str] | None,
) -> list[dict[str, Any]]:
    family_set = {item.strip() for item in families or [] if item.strip()}
    if not family_set.intersection({"k_stock_kis_intraday", "k_stock_olr_kalcb"}):
        return requests
    declared = _declared_k_stock_kis_requirements(repo_root)
    if not declared:
        return requests
    selected = [
        item
        for item in requests
        if (str(item.get("symbol", "")).zfill(6), str(item.get("timeframe", ""))) in declared
    ]
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in selected:
        key = (str(item.get("symbol", "")).zfill(6), str(item.get("timeframe", "")))
        current = deduped.get(key)
        if current is None or _request_end(item) > _request_end(current):
            deduped[key] = item
    return [deduped[key] for key in sorted(deduped)]


def _declared_k_stock_kis_requirements(repo_root: Path) -> set[tuple[str, str]]:
    path = Path(repo_root) / "data" / "requirements" / "strategies" / "k_stock" / "portfolio.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(item.get("symbol", "")).zfill(6), str(item.get("timeframe", "")))
        for item in payload.get("requirements", [])
        if item.get("source") == "kis"
    }


def _request_end(source_request: dict[str, Any]) -> datetime:
    return _parse_ts(str(source_request.get("download_request", {}).get("end", "")))


def _latest_authoritative_kis_end(repo_root: Path, source_request: dict[str, Any]) -> datetime | None:
    symbol = str(source_request.get("symbol", "")).zfill(6)
    timeframe = str(source_request.get("timeframe", ""))
    family = _strategy_data_family(source_request)
    root = (
        Path(repo_root)
        / "data"
        / "manifests"
        / "slices"
        / "kis"
        / "krx_equity"
        / symbol
        / timeframe
    )
    if not root.exists():
        return None
    ends = []
    for path in root.glob("*.market_data_manifest.json"):
        try:
            manifest = load_market_manifest(path)
        except (OSError, ValueError):
            continue
        if not is_authoritative_slice_manifest(manifest):
            continue
        if manifest.lineage.get("strategy_data_family", "") != family:
            continue
        ends.append(manifest.end_ts)
    return max(ends) if ends else None


def _kis_skipped_result(source_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": source_request.get("coverage_note", "existing authoritative coverage"),
        **_request_summary(source_request),
    }


def _prepare_ibkr_coverage_requests(
    *,
    repo_root: Path,
    requests: list[dict[str, Any]],
    coverage_mode: str,
    contract_probe_path: Path | None,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if coverage_mode not in IBKR_COVERAGE_MODES:
        raise ValueError(
            "unsupported IBKR coverage mode: "
            f"{coverage_mode}; expected one of {sorted(IBKR_COVERAGE_MODES)}"
        )
    probe = _load_ibkr_contract_probe(repo_root, contract_probe_path)
    coverage_report = {
        "coverage_mode": coverage_mode,
        "contract_probe_path": str(_contract_probe_path(repo_root, contract_probe_path)),
        "probe_loaded": bool(probe),
        "adjustments": [],
    }
    prepared: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_request in requests:
        if source_request.get("source_kind") != "ibkr_cme_futures_historical_bars":
            prepared.append(source_request)
            continue
        request = source_request["download_request"]
        symbol = str(request.get("symbol", "")).upper()
        if not probe:
            if dry_run and coverage_mode == IBKR_COVERAGE_FULL_LEGACY:
                prepared.append(source_request)
                coverage_report["adjustments"].append(
                    {
                        **_request_summary(source_request),
                        "coverage_note": (
                            "full-legacy dry-run did not evaluate live TWS retention; "
                            "non-dry-run requires a contract-resolution probe or archived evidence"
                        ),
                    }
                )
                continue
            failures.append(
                {
                    **_request_summary(source_request),
                    "error": "IBKR CME futures sync requires a contract-resolution probe",
                }
            )
            continue
        available = _available_probe_contracts(probe, symbol)
        if not available:
            failures.append(
                {
                    **_request_summary(source_request),
                    "error": f"IBKR contract-resolution probe has no contracts for {symbol}",
                }
            )
            continue
        missing = _missing_required_contracts(source_request, available)
        if coverage_mode == IBKR_COVERAGE_FULL_LEGACY:
            if missing:
                failures.append(
                    {
                        **_request_summary(source_request),
                        "error": (
                            "full legacy IBKR reproduction requires archived raw payloads "
                            "and contract metadata for pre-retention CME contracts; "
                            f"live TWS probe is missing {', '.join(missing[:8])}"
                        ),
                        "missing_contracts": missing,
                        "earliest_live_contract": available[0]["local_symbol"],
                        "earliest_live_expiry": available[0]["last_trade_date_or_contract_month"],
                    }
                )
            else:
                prepared.append(_with_contract_chain(source_request, available))
            continue
        adjusted = _retention_covered_request(source_request, available)
        if adjusted is None:
            failures.append(
                {
                    **_request_summary(source_request),
                    "error": "source request ends before live TWS retention-covered cutoff",
                    "earliest_live_contract": available[0]["local_symbol"],
                    "earliest_live_expiry": available[0]["last_trade_date_or_contract_month"],
                }
            )
            continue
        prepared.append(adjusted)
        original = source_request["download_request"]
        updated = adjusted["download_request"]
        if original.get("start") != updated.get("start") or original.get("contract_chain") != updated.get("contract_chain"):
            coverage_report["adjustments"].append(
                {
                    **_request_summary(source_request),
                    "original_start": original.get("start"),
                    "retention_start": updated.get("start"),
                    "contract_chain": updated.get("contract_chain", []),
                    "coverage_note": updated.get("coverage_note", ""),
                }
            )
    return prepared, coverage_report, failures


def _load_ibkr_contract_probe(repo_root: Path, path: Path | None) -> dict[str, Any]:
    probe_path = _contract_probe_path(repo_root, path)
    if not probe_path.exists():
        return {}
    return json.loads(probe_path.read_text(encoding="utf-8"))


def _contract_probe_path(repo_root: Path, path: Path | None) -> Path:
    if path is not None:
        return path if path.is_absolute() else Path(repo_root) / path
    return Path(repo_root) / DEFAULT_IBKR_CONTRACT_PROBE


def _available_probe_contracts(probe: dict[str, Any], symbol: str) -> list[dict[str, str]]:
    contracts = probe.get("roots", {}).get(symbol, {}).get("contracts", [])
    available = [
        {
            "local_symbol": str(item.get("local_symbol", "")).upper(),
            "last_trade_date_or_contract_month": str(
                item.get("last_trade_date_or_contract_month", "")
            ),
            "con_id": str(item.get("con_id", "")),
        }
        for item in contracts
        if str(item.get("local_symbol", "")).strip()
    ]
    return sorted(available, key=lambda item: item["last_trade_date_or_contract_month"])


def _missing_required_contracts(
    source_request: dict[str, Any],
    available: list[dict[str, str]],
) -> list[str]:
    available_symbols = {item["local_symbol"] for item in available}
    return [
        spec["local_symbol"]
        for spec in _critical_contract_specs(source_request)
        if spec["local_symbol"] not in available_symbols
    ]


def _critical_contract_specs(source_request: dict[str, Any]) -> list[dict[str, Any]]:
    request = _cme_request(source_request, Path("."))
    return [spec for spec in _contract_specs_for_request(request) if spec["critical"]]


def _with_contract_chain(
    source_request: dict[str, Any],
    available: list[dict[str, str]],
) -> dict[str, Any]:
    copied = deepcopy(source_request)
    copied["download_request"]["contract_chain"] = [
        item["local_symbol"] for item in available
    ]
    return copied


def _retention_covered_request(
    source_request: dict[str, Any],
    available: list[dict[str, str]],
) -> dict[str, Any] | None:
    original = source_request["download_request"]
    original_start = _parse_ts(original["start"])
    original_end = _parse_ts(original["end"])
    cutoff = _retention_start_for_first_contract(available[0])
    start = max(original_start, cutoff)
    if start > original_end:
        return None
    copied = deepcopy(source_request)
    request = copied["download_request"]
    request["start"] = start.isoformat()
    request["coverage_mode"] = IBKR_COVERAGE_RETENTION_COVERED
    request["retention_cutoff"] = cutoff.isoformat()
    request["contract_chain"] = [
        item["local_symbol"]
        for item in _contracts_overlapping_window(available, start, original_end)
    ]
    request["coverage_note"] = (
        "retention-covered live TWS lane; older legacy contracts require archived "
        "IBKR raw payloads and contract metadata"
    )
    copied["authority_status"] = "retention_covered_request_ready"
    return copied


def _retention_start_for_first_contract(contract: dict[str, str]) -> datetime:
    expiry = _parse_contract_expiry(contract["last_trade_date_or_contract_month"])
    previous_expiry = _previous_quarter_expiry(expiry)
    previous_roll = previous_expiry - timedelta(days=4)
    return datetime.combine(previous_roll, datetime.min.time(), tzinfo=UTC)


def _contracts_overlapping_window(
    contracts: list[dict[str, str]],
    start: datetime,
    end: datetime,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for item in contracts:
        expiry = _parse_contract_expiry(item["last_trade_date_or_contract_month"])
        roll = datetime.combine(expiry - timedelta(days=4), datetime.min.time(), tzinfo=UTC)
        if roll < start - timedelta(days=120):
            continue
        if datetime.combine(expiry.replace(day=1), datetime.min.time(), tzinfo=UTC) > end + timedelta(days=120):
            continue
        selected.append(item)
    return selected


def _parse_contract_expiry(value: str) -> date:
    if len(value) < 8:
        raise ValueError(f"contract expiry must be YYYYMMDD: {value}")
    return datetime.strptime(value[:8], "%Y%m%d").date()


def _previous_quarter_expiry(expiry: date) -> date:
    previous_month = {3: 12, 6: 3, 9: 6, 12: 9}[expiry.month]
    previous_year = expiry.year - 1 if previous_month == 12 else expiry.year
    return _third_friday(previous_year, previous_month)


def _third_friday(year: int, month: int) -> date:
    current = date(year, month, 1)
    first_friday_offset = (4 - current.weekday()) % 7
    return current + timedelta(days=first_friday_offset + 14)


def _cme_request(source_request: dict[str, Any], repo_root: Path) -> CmeNqRefreshRequest:
    request = source_request["download_request"]
    return CmeNqRefreshRequest(
        symbol=request["symbol"],
        timeframe=request["timeframe"],
        start=_parse_ts(request["start"]),
        end=_parse_ts(request["end"]),
        exchange=request.get("exchange", "CME"),
        sec_type=request.get("sec_type", "FUT"),
        currency=request.get("currency", "USD"),
        what_to_show=request.get("what_to_show", "TRADES"),
        use_rth=bool(request.get("use_rth", False)),
        roll_policy=request.get("roll_policy", ""),
        contract_chain=tuple(request.get("contract_chain", ())),
        pulled_at_utc=_pulled_at(),
        source_version=git_commit_sha(repo_root) or ("0" * 40),
        strategy_data_family=_strategy_data_family(source_request),
        source_request_id=str(source_request.get("request_id") or ""),
    )


def _us_equity_request(source_request: dict[str, Any], repo_root: Path) -> UsEquityRefreshRequest:
    request = source_request["download_request"]
    return UsEquityRefreshRequest(
        symbol=request["symbol"],
        timeframe=request["timeframe"],
        start=_parse_ts(request["start"]),
        end=_parse_ts(request["end"]),
        exchange=request.get("exchange", "SMART"),
        primary_exchange=request.get("primary_exchange", ""),
        sec_type=request.get("sec_type", "STK"),
        currency=request.get("currency", "USD"),
        what_to_show=request.get("what_to_show", "TRADES"),
        use_rth=bool(request.get("use_rth", True)),
        pulled_at_utc=_pulled_at(),
        source_version=git_commit_sha(repo_root) or ("0" * 40),
        strategy_data_family=_strategy_data_family(source_request),
        source_request_id=str(source_request.get("request_id") or ""),
    )


def _krx_request(source_request: dict[str, Any], repo_root: Path) -> KrxRefreshRequest:
    request = source_request["download_request"]
    return KrxRefreshRequest(
        symbol=request["fid_input_iscd"],
        timeframe=request["timeframe"],
        start=_parse_ts(request["start"]),
        end=_parse_ts(request["end"]),
        market_code=request.get("fid_cond_mrkt_div_code", "J"),
        calendar_holidays_path=str(Path(repo_root) / "data" / "calendars" / "krx_holidays.yaml"),
        pulled_at_utc=_pulled_at(),
        source_version=git_commit_sha(repo_root) or ("0" * 40),
        strategy_data_family=_strategy_data_family(source_request),
        source_request_id=str(source_request.get("request_id") or ""),
    )


def _planned_payload(
    source: str,
    requests: list[dict[str, Any]],
    *,
    dry_run: bool,
    coverage_mode: str = "",
    coverage_report: dict[str, Any] | None = None,
    preflight_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "source": source,
        "status": "planned",
        "dry_run": dry_run,
        "request_count": len(requests),
        "requests": [_request_summary(item) for item in requests],
    }
    if coverage_mode:
        payload["coverage_mode"] = coverage_mode
    if coverage_report is not None:
        payload["coverage_report"] = coverage_report
    if preflight_failures:
        payload["preflight_failures"] = preflight_failures
        payload["preflight_failure_count"] = len(preflight_failures)
    return payload


def _complete_payload(
    source: str,
    requests: list[dict[str, Any]],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    coverage_mode: str = "",
    coverage_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "source": source,
        "status": "complete" if not failures else "failed",
        "ok": not failures,
        "dry_run": False,
        "request_count": len(requests),
        "result_count": len(results),
        "failure_count": len(failures),
        "requests": [_request_summary(item) for item in requests],
        "results": results,
        "failures": failures,
    }
    if coverage_mode:
        payload["coverage_mode"] = coverage_mode
    if coverage_report is not None:
        payload["coverage_report"] = coverage_report
    return payload


def _kis_progress_path(repo_root: Path) -> Path:
    configured = os.getenv("KIS_SYNC_PROGRESS_PATH", "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(repo_root) / path
    return (
        Path(repo_root)
        / "data"
        / "validation_reports"
        / "source_refresh"
        / "kis_sync_progress.jsonl"
    )


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _request_summary(item: dict[str, Any]) -> dict[str, Any]:
    request = item.get("download_request", {})
    return {
        "request_id": item.get("request_id"),
        "legacy_family": item.get("legacy_family"),
        "strategy_data_family": _strategy_data_family(item),
        "legacy_path": item.get("legacy_path"),
        "source_kind": item.get("source_kind"),
        "symbol": item.get("symbol"),
        "timeframe": item.get("timeframe"),
        "start": request.get("start"),
        "end": request.get("end"),
        "source_endpoint": item.get("source_endpoint"),
    }


def _strategy_data_family(source_request: dict[str, Any]) -> str:
    explicit = str(source_request.get("strategy_data_family") or "").strip()
    if explicit:
        return explicit
    legacy_family = str(source_request.get("legacy_family") or "").strip()
    return STRATEGY_DATA_FAMILY_BY_LEGACY_FAMILY.get(legacy_family, legacy_family)


def _require_network_write_enabled(*, source: str, request_count: int) -> None:
    if request_count <= 0:
        raise RuntimeError(f"no {source} source requests selected")
    if os.getenv("TA_SOURCE_REFRESH_ALLOW_NETWORK", "").strip().lower() not in {"1", "true", "yes"}:
        raise RuntimeError("TA_SOURCE_REFRESH_ALLOW_NETWORK must be true for non-dry-run sync")
    if os.getenv("TA_SOURCE_REFRESH_ALLOW_WRITE", "").strip().lower() not in {"1", "true", "yes"}:
        raise RuntimeError("TA_SOURCE_REFRESH_ALLOW_WRITE must be true for non-dry-run sync")


def _parse_ts(value: str) -> datetime:
    if not value:
        raise ValueError("source request missing start/end timestamp")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()


def _pulled_at() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
