"""Compare refreshed source-owned slices against legacy parquet files."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .checksums import canonical_json_sha256, parquet_content_checksum
from .io import read_json
from .manifests import load_market_manifest


def compare_legacy_source_requests(
    *,
    repo_root: Path,
    source_request_manifest: Path | None = None,
    families: list[str] | None = None,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    latest_only: bool = False,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    source_request_manifest = source_request_manifest or (
        repo_root
        / "data"
        / "source_requests"
        / "reference_snapshot_2026-05-30"
        / "source_request_manifest.json"
    )
    artifact_root = artifact_root or repo_root / "data" / "validation_reports" / "legacy_source_compare"
    artifact_root.mkdir(parents=True, exist_ok=True)
    requests = _select_requests(
        source_request_manifest,
        families=families,
        symbols=symbols,
        intervals=intervals,
        latest_only=latest_only,
    )
    index = _slice_index(repo_root)
    comparisons = [
        _compare_one(repo_root=repo_root, source_request=item, index=index) for item in requests
    ]
    ok = bool(comparisons) and all(item["status"] == "pass" for item in comparisons)
    payload = {
        "ok": ok,
        "status": "pass" if ok else "blocked",
        "source_request_manifest": str(source_request_manifest),
        "request_count": len(requests),
        "passed_count": sum(1 for item in comparisons if item["status"] == "pass"),
        "blocked_count": sum(1 for item in comparisons if item["status"] == "blocked"),
        "failed_count": sum(1 for item in comparisons if item["status"] == "fail"),
        "comparisons": comparisons,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    report_path = artifact_root / "legacy_source_compare_report.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload


def _select_requests(
    manifest_path: Path,
    *,
    families: list[str] | None,
    symbols: list[str] | None,
    intervals: list[str] | None,
    latest_only: bool,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    family_set = {item.strip() for item in families or [] if item.strip()}
    symbol_set = {item.strip().upper() for item in symbols or [] if item.strip()}
    interval_set = {item.strip() for item in intervals or [] if item.strip()}
    selected = []
    for item in payload.get("requests", []):
        if family_set and item.get("legacy_family") not in family_set:
            continue
        if symbol_set and str(item.get("symbol", "")).upper() not in symbol_set:
            continue
        if interval_set and item.get("timeframe") not in interval_set:
            continue
        selected.append(item)
    if latest_only:
        selected = _latest_requests(selected)
    return sorted(selected, key=lambda item: item["legacy_path"])


def _latest_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in requests:
        key = (
            str(item.get("legacy_family", "")),
            str(item.get("source", "")),
            str(item.get("market", "")),
            str(item.get("symbol", "")),
            str(item.get("timeframe", "")),
            str(item.get("source_kind", "")),
            str(item.get("data_kind", "")),
        )
        previous = latest.get(key)
        if previous is None or _latest_sort_key(item) > _latest_sort_key(previous):
            latest[key] = item
    return list(latest.values())


def _latest_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    bounds = item.get("legacy_time_bounds") or {}
    return (str(bounds.get("end") or ""), str(item.get("legacy_path") or ""))


def _compare_one(
    *,
    repo_root: Path,
    source_request: dict[str, Any],
    index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    legacy_path = repo_root / source_request["legacy_path"]
    if not legacy_path.exists():
        return _blocked(source_request, "legacy parquet missing")
    if _table_request(source_request):
        return _compare_table_export(
            repo_root=repo_root,
            source_request=source_request,
            legacy_path=legacy_path,
            index=index,
        )
    legacy = _legacy_frame(legacy_path, source_request)
    if legacy.empty:
        return _blocked(source_request, "legacy parquet has no comparable rows")
    candidates = _matching_index_entries(source_request, legacy, index, repo_root)
    if not candidates:
        return _blocked(
            source_request,
            "no source-owned canonical slice covers the legacy request window",
        )
    canonical_paths = [
        repo_root / path
        for candidate in candidates
        for path in candidate.get("canonical_paths", [])
    ]
    canonical_frames = [pd.read_parquet(path) for path in canonical_paths if path.exists()]
    canonical = (
        pd.concat(canonical_frames, ignore_index=True)
        if canonical_frames
        else pd.DataFrame()
    )
    if canonical.empty:
        return _blocked(source_request, "matching canonical slice has no readable rows")
    canonical = _filter_canonical_to_legacy_window(source_request, legacy, canonical)
    return _comparison_payload(
        source_request,
        legacy_path,
        legacy,
        canonical,
        candidates,
        repo_root=repo_root,
    )


def _compare_table_export(
    *,
    repo_root: Path,
    source_request: dict[str, Any],
    legacy_path: Path,
    index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    legacy = pd.read_parquet(legacy_path)
    if legacy.empty and source_request.get("data_kind") == "fx_rates":
        return {
            "status": "pass",
            "request_id": source_request["request_id"],
            "legacy_path": source_request["legacy_path"],
            "source_kind": source_request["source_kind"],
            "symbol": source_request["symbol"],
            "timeframe": source_request["timeframe"],
            "exact_checksum_expected": True,
            "deterministic_explanation": "Empty LRS fx-rate table is preserved as an empty optional research input.",
            "accepted_difference": {"accepted": False, "reason": "exact_empty_table"},
            "matched_manifest_ids": [],
            "matched_manifest_count": 0,
            "matched_manifest_id": "",
            "legacy_rows": 0,
            "canonical_rows": 0,
            "common_timestamp_rows": 0,
            "row_count_match": True,
            "timestamp_coverage_match": True,
            "value_mismatches": [],
            "ohlcv_mismatches": [],
            "legacy_checksum": parquet_content_checksum(legacy_path),
            "transformed_comparison_checksum": canonical_json_sha256(
                {
                    "legacy_path": str(legacy_path),
                    "legacy_checksum": parquet_content_checksum(legacy_path),
                    "canonical_row_count": 0,
                    "empty_optional_table": True,
                }
            ),
        }
    candidates = _matching_table_entries(source_request, index, repo_root)
    if not candidates:
        return _blocked(source_request, "no source-owned canonical table covers the legacy export")
    canonical_paths = [
        repo_root / path
        for candidate in candidates
        for path in candidate.get("canonical_paths", [])
    ]
    canonical_frames = [pd.read_parquet(path) for path in canonical_paths if path.exists()]
    canonical = (
        pd.concat(canonical_frames, ignore_index=True)
        if canonical_frames
        else pd.DataFrame()
    )
    if canonical.empty:
        return _blocked(source_request, "matching canonical table has no readable rows")
    legacy_cmp = _comparable_table_frame(legacy)
    canonical_cmp = _comparable_table_frame(canonical)
    legacy_checksum = _frame_records_checksum(legacy_cmp)
    canonical_checksum = _frame_records_checksum(canonical_cmp)
    row_count_match = len(legacy_cmp) == len(canonical_cmp)
    content_match = legacy_checksum == canonical_checksum
    status = "pass" if row_count_match and content_match else "fail"
    return {
        "status": status,
        "request_id": source_request["request_id"],
        "legacy_path": source_request["legacy_path"],
        "source_kind": source_request["source_kind"],
        "symbol": source_request["symbol"],
        "timeframe": source_request["timeframe"],
        "exact_checksum_expected": True,
        "deterministic_explanation": "Exact row parity is expected for LRS static research-table exports.",
        "accepted_difference": {"accepted": False, "reason": "exact_parity_required"},
        "matched_manifest_ids": [candidate.get("manifest_id") for candidate in candidates],
        "matched_manifest_count": len(candidates),
        "matched_manifest_id": candidates[0].get("manifest_id") if candidates else "",
        "legacy_rows": len(legacy_cmp),
        "canonical_rows": len(canonical_cmp),
        "common_timestamp_rows": 0,
        "row_count_match": row_count_match,
        "timestamp_coverage_match": True,
        "value_mismatches": [] if content_match else [{"column": "*", "rows": -1}],
        "ohlcv_mismatches": [] if content_match else [{"column": "*", "rows": -1}],
        "legacy_checksum": parquet_content_checksum(legacy_path),
        "legacy_content_checksum": legacy_checksum,
        "canonical_content_checksum": canonical_checksum,
        "transformed_comparison_checksum": canonical_json_sha256(
            {
                "legacy_path": str(legacy_path),
                "legacy_checksum": parquet_content_checksum(legacy_path),
                "canonical_row_count": len(canonical_cmp),
                "content_match": content_match,
            }
        ),
    }


def _legacy_frame(path: Path, source_request: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    timestamp_col = next(
        (
            column
            for column in ("timestamp_utc", "timestamp", "time", "date", "datetime", "ts")
            if column in frame.columns
        ),
        None,
    )
    if timestamp_col is None:
        if isinstance(frame.index, pd.DatetimeIndex):
            out = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp_utc"})
        else:
            return pd.DataFrame()
    else:
        out = frame.copy()
        if timestamp_col == "ts":
            out["timestamp_utc"] = pd.to_datetime(out[timestamp_col], unit="ms", utc=True)
        else:
            out["timestamp_utc"] = pd.to_datetime(out[timestamp_col], utc=True)
        if timestamp_col == "date":
            out["trading_date"] = out[timestamp_col].astype(str).str.slice(0, 10)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    if _funding_request(source_request):
        rate_col = next(
            (
                column
                for column in ("rate", "fundingRate", "funding_rate")
                if column in out.columns
            ),
            None,
        )
        if rate_col is None:
            return pd.DataFrame()
        out["timestamp_utc"] = out["timestamp_utc"].dt.floor("h")
        out = out.rename(columns={rate_col: "rate"})
        return out.loc[:, ["timestamp_utc", "rate"]].sort_values("timestamp_utc").reset_index(drop=True)
    if _flow_request(source_request):
        flow_columns = _value_columns(source_request)
        missing_flow = [column for column in flow_columns if column not in out.columns]
        if missing_flow:
            return pd.DataFrame()
        columns = ["timestamp_utc", *flow_columns]
        if "trading_date" in out.columns:
            columns.insert(1, "trading_date")
        return out.loc[:, columns].sort_values("timestamp_utc").reset_index(drop=True)
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = out.rename(columns=rename)
    columns = ["timestamp_utc", "open", "high", "low", "close", "volume"]
    missing = [column for column in columns if column not in out.columns]
    if missing:
        return pd.DataFrame()
    if "trading_date" in out.columns:
        columns.insert(1, "trading_date")
    return out.loc[:, columns].sort_values("timestamp_utc").reset_index(drop=True)


def _canonical_frame(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    if "rate" in out.columns:
        out["timestamp_utc"] = out["timestamp_utc"].dt.floor("h")
    return out.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def _slice_index(repo_root: Path) -> dict[str, dict[str, Any]]:
    path = repo_root / "data" / "manifests" / "slices" / "slice_index.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return {str(item.get("manifest_id")): item for item in payload.get("slices", [])}


def _matching_index_entries(
    source_request: dict[str, Any],
    legacy: pd.DataFrame,
    index: dict[str, dict[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    market = _canonical_market(source_request["market"])
    path_fragments = [
        (
            f"/{source_request['source']}/{market}/"
            f"{str(source_request['symbol']).upper()}/{timeframe}/"
        )
        for timeframe in _candidate_timeframes(source_request)
    ]
    matches: list[dict[str, Any]] = []
    exact_source_request_matches: list[dict[str, Any]] = []
    for entry in index.values():
        manifest_text = str(entry.get("manifest_path", "")).replace("\\", "/")
        if not any(fragment in f"/{manifest_text}" for fragment in path_fragments):
            continue
        if not _entry_matches_strategy_family(source_request, entry):
            continue
        manifest_path = repo_root / manifest_text
        if not manifest_path.exists():
            continue
        manifest = load_market_manifest(manifest_path)
        if manifest.source != source_request["source"]:
            continue
        if manifest.market != market:
            continue
        if manifest.symbol.upper() != str(source_request["symbol"]).upper():
            continue
        if manifest.timeframe not in _candidate_timeframes(source_request):
            continue
        if _manifest_overlaps_legacy_window(manifest, source_request, legacy):
            matches.append(entry)
            if _manifest_matches_source_request_id(manifest, source_request):
                exact_source_request_matches.append(entry)
    selected = exact_source_request_matches or matches
    return sorted(selected, key=lambda item: str(item.get("manifest_path", "")))


def _matching_table_entries(
    source_request: dict[str, Any],
    index: dict[str, dict[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    market = _canonical_market(source_request["market"])
    path_fragment = (
        f"/{source_request['source']}/{market}/"
        f"{str(source_request['symbol']).upper()}/{source_request['timeframe']}/"
    )
    matches: list[dict[str, Any]] = []
    for entry in index.values():
        manifest_text = str(entry.get("manifest_path", "")).replace("\\", "/")
        if path_fragment not in f"/{manifest_text}":
            continue
        if not _entry_matches_strategy_family(source_request, entry):
            continue
        manifest_path = repo_root / manifest_text
        if not manifest_path.exists():
            continue
        manifest = load_market_manifest(manifest_path)
        if manifest.source != source_request["source"]:
            continue
        if manifest.market != market:
            continue
        if manifest.symbol.upper() != str(source_request["symbol"]).upper():
            continue
        if manifest.timeframe != source_request["timeframe"]:
            continue
        matches.append(entry)
    return sorted(matches, key=lambda item: str(item.get("manifest_path", "")))


def _entry_matches_strategy_family(source_request: dict[str, Any], entry: dict[str, Any]) -> bool:
    expected = {
        "crypto_trader": {"crypto_trader_portfolio", "crypto_trader"},
        "k_stock_kis_intraday": {"k_stock_olr_kalcb"},
        "k_stock_krx_lrs": {"k_stock_olr_kalcb"},
        "trading_momentum": {"trading_momentum_family", "trading_momentum"},
        "trading_stock": {"trading_stock_family", "trading_stock"},
        "trading_swing": {"trading_swing_family", "trading_swing"},
    }.get(str(source_request.get("legacy_family") or ""))
    if not expected:
        return True
    family = str(entry.get("strategy_data_family") or "")
    if not family:
        return str(source_request.get("legacy_family") or "") not in {
            "k_stock_kis_intraday",
            "k_stock_krx_lrs",
        }
    return family in expected


def _comparable_table_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    metadata_columns = {
        "source_file",
        "source_row_hash",
        "source",
        "market",
        "timeframe",
        "kind",
    }
    keep = [column for column in out.columns if str(column) not in metadata_columns]
    out = out.loc[:, keep]
    out.columns = [str(column) for column in out.columns]
    return out.reindex(sorted(out.columns), axis=1)


def _frame_records_checksum(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
    records = (
        normalized.fillna("<NA>")
        .astype(str)
        .sort_values(list(normalized.columns))
        .to_dict("records")
    )
    return canonical_json_sha256({"columns": list(normalized.columns), "records": records})


def _manifest_matches_source_request_id(
    manifest: Any,
    source_request: dict[str, Any],
) -> bool:
    return (
        str((manifest.lineage or {}).get("source_request_id") or "")
        == str(source_request.get("request_id") or "")
    )


def _comparison_payload(
    source_request: dict[str, Any],
    legacy_path: Path,
    legacy: pd.DataFrame,
    canonical: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    legacy_aligned, canonical_aligned, join_key = _aligned_comparison_frames(
        source_request,
        legacy,
        canonical,
    )
    common = legacy_aligned.merge(
        canonical_aligned,
        on=join_key,
        suffixes=("_legacy", "_canonical"),
    )
    value_columns = _value_columns(source_request)
    mismatches = []
    for column in value_columns:
        left = f"{column}_legacy"
        right = f"{column}_canonical"
        if left not in common.columns or right not in common.columns:
            continue
        diff = (common[left].astype(float) - common[right].astype(float)).abs()
        if bool((diff > 1e-9).any()):
            mismatches.append({"column": column, "rows": int((diff > 1e-9).sum())})
    row_count_match = len(legacy_aligned) == len(canonical_aligned)
    timestamp_match = len(common) == len(legacy_aligned) == len(canonical_aligned)
    exact_checksum_expected = _exact_checksum_expected(source_request)
    non_exact_acceptance = _non_exact_acceptance(
        source_request=source_request,
        legacy_rows=len(legacy_aligned),
        canonical_rows=len(canonical_aligned),
        common_rows=len(common),
        mismatches=mismatches,
        candidates=candidates,
        repo_root=repo_root,
    )
    status = (
        "pass"
        if (
            row_count_match
            and timestamp_match
            and not mismatches
        )
        or (not exact_checksum_expected and non_exact_acceptance["accepted"])
        else "fail"
    )
    transformed_checksum = canonical_json_sha256(
        {
            "legacy_path": str(legacy_path),
            "legacy_checksum": parquet_content_checksum(legacy_path),
            "canonical_row_count": len(canonical_aligned),
            "common_row_count": len(common),
            "mismatches": mismatches,
        }
    )
    return {
        "status": status,
        "request_id": source_request["request_id"],
        "legacy_path": source_request["legacy_path"],
        "source_kind": source_request["source_kind"],
        "symbol": source_request["symbol"],
        "timeframe": source_request["timeframe"],
        "exact_checksum_expected": exact_checksum_expected,
        "deterministic_explanation": _explanation(source_request),
        "accepted_difference": non_exact_acceptance,
        "matched_manifest_ids": [candidate.get("manifest_id") for candidate in candidates],
        "matched_manifest_count": len(candidates),
        "matched_manifest_id": candidates[0].get("manifest_id") if candidates else "",
        "legacy_rows": len(legacy_aligned),
        "canonical_rows": len(canonical_aligned),
        "common_timestamp_rows": len(common),
        "row_count_match": row_count_match,
        "timestamp_coverage_match": timestamp_match,
        "value_mismatches": mismatches,
        "ohlcv_mismatches": mismatches,
        "legacy_checksum": parquet_content_checksum(legacy_path),
        "transformed_comparison_checksum": transformed_checksum,
    }


def _blocked(source_request: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "request_id": source_request.get("request_id"),
        "legacy_path": source_request.get("legacy_path"),
        "source_kind": source_request.get("source_kind"),
        "symbol": source_request.get("symbol"),
        "timeframe": source_request.get("timeframe"),
        "reason": reason,
    }


def _canonical_market(market: str) -> str:
    return {"crypto_derivatives": "crypto_perp"}.get(str(market), str(market))


def _daily_request(source_request: dict[str, Any]) -> bool:
    timeframe = str(source_request.get("timeframe") or "").lower()
    return timeframe in {"1d", "daily", "1d_flow"} or timeframe.startswith("1d_daily")


def _funding_request(source_request: dict[str, Any]) -> bool:
    return str(source_request.get("source_kind") or "") == "hyperliquid_funding"


def _flow_request(source_request: dict[str, Any]) -> bool:
    return str(source_request.get("source_kind") or "") == "lrs_krx_flow_export"


def _table_request(source_request: dict[str, Any]) -> bool:
    return str(source_request.get("source_kind") or "") == "lrs_krx_table_export"


def _candidate_timeframes(source_request: dict[str, Any]) -> list[str]:
    timeframe = str(source_request.get("timeframe") or "")
    if _flow_request(source_request) and timeframe == "1d_flow":
        data_kind = str(source_request.get("data_kind") or "")
        if data_kind == "daily_foreign_flow":
            return ["1d_daily_foreign_flow"]
        if data_kind == "daily_institutional_flow":
            return ["1d_daily_institutional_flow"]
        return ["1d_daily_flow"]
    return [timeframe]


def _value_columns(source_request: dict[str, Any]) -> list[str]:
    if _funding_request(source_request):
        return ["rate"]
    if _flow_request(source_request):
        data_kind = str(source_request.get("data_kind") or "")
        if data_kind == "daily_foreign_flow":
            return ["foreign_net"]
        if data_kind == "daily_institutional_flow":
            return ["institutional_net"]
        return ["foreign_net", "inst_net"]
    return ["open", "high", "low", "close", "volume"]


def _manifest_overlaps_legacy_window(
    manifest: Any,
    source_request: dict[str, Any],
    legacy: pd.DataFrame,
) -> bool:
    legacy_start = legacy["timestamp_utc"].min().to_pydatetime()
    legacy_end = legacy["timestamp_utc"].max().to_pydatetime()
    if _daily_request(source_request):
        legacy_start_date = legacy_start.date()
        legacy_end_date = legacy_end.date()
        if source_request.get("market") == "krx_equity":
            legacy_start_date -= timedelta(days=1)
            legacy_end_date -= timedelta(days=1)
        return (
            manifest.end_ts.date() >= legacy_start_date
            and manifest.start_ts.date() <= legacy_end_date
        )
    return manifest.end_ts >= legacy_start and manifest.start_ts <= legacy_end


def _filter_canonical_to_legacy_window(
    source_request: dict[str, Any],
    legacy: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    if _daily_request(source_request):
        legacy_dates = _frame_trading_dates(legacy)
        canonical_dates = _frame_trading_dates(canonical)
        return canonical[
            (canonical_dates >= legacy_dates.min()) & (canonical_dates <= legacy_dates.max())
        ].reset_index(drop=True)
    return canonical[
        (canonical["timestamp_utc"] >= legacy["timestamp_utc"].min())
        & (canonical["timestamp_utc"] <= legacy["timestamp_utc"].max())
    ].reset_index(drop=True)


def _aligned_comparison_frames(
    source_request: dict[str, Any],
    legacy: pd.DataFrame,
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not _daily_request(source_request):
        return legacy, canonical, "timestamp_utc"
    legacy_aligned = legacy.copy()
    canonical_aligned = canonical.copy()
    legacy_aligned["trading_date"] = _frame_trading_dates(legacy_aligned).astype(str)
    canonical_aligned["trading_date"] = _frame_trading_dates(canonical_aligned).astype(str)
    return legacy_aligned, canonical_aligned, "trading_date"


def _frame_trading_dates(frame: pd.DataFrame) -> pd.Series:
    if "trading_date" in frame.columns:
        return pd.to_datetime(frame["trading_date"].astype(str), errors="coerce").dt.date
    if "timestamp_exchange" in frame.columns:
        text = frame["timestamp_exchange"].astype(str)
        if bool(text.str.match(r"^\d{4}-\d{2}-\d{2}").all()):
            return pd.to_datetime(text.str.slice(0, 10), errors="coerce").dt.date
    return pd.to_datetime(frame["timestamp_utc"], utc=True).dt.date


def _explanation(source_request: dict[str, Any]) -> str:
    if _is_hyperliquid_candle_request(source_request):
        return (
            "Exact file checksum parity is not required for refreshed Hyperliquid "
            "candles because the exchange can revise a small number of historical "
            "OHLCV rows between the imported snapshot and a live source refresh. "
            "The approval comparison requires complete timestamp coverage and "
            "keeps value mismatches within a narrow deterministic tolerance."
        )
    if source_request["market"] == "cme_futures":
        return (
            "Exact file checksum parity is not required for CME continuous futures when "
            "the refreshed source path writes source-contract-tagged physical-contract "
            "bars and a deterministic Panama stitch. Legacy files may include older "
            "pre-retention history or an older continuous-adjustment convention; the "
            "approval comparison requires refreshed coverage to be source-contract and "
            "conId backed, roll-policy tagged, Panama-checksummed, and timestamp-aligned "
            "inside the retention-covered overlap."
        )
    if _is_sparse_ibkr_us_equity_request(source_request):
        return (
            "Exact file checksum parity is not required for IBKR useRTH=False US-equity "
            "trade bars because extended-hours trade bars are sparse and may be revised "
            "between pulls. The approval comparison requires successful source refresh, "
            "qualified contract lineage, legal timestamps, no multi-day coverage holes, "
            "and near-complete timestamp/value overlap with row-count differences reported."
        )
    if _is_kis_krx_intraday_request(source_request):
        return (
            "Exact file checksum parity is not required for refreshed KIS intraday bars when "
            "a newer source-owned pull corrects a bounded number of historical rows. The "
            "approval comparison requires complete timestamp coverage and keeps value "
            "mismatches within a narrow deterministic tolerance."
        )
    if _flow_request(source_request):
        return "Exact exchange-date and KRX investor-flow parity is expected for LRS flow exports."
    return "Exact timestamp and OHLCV parity is expected for source-owned bars."


def _exact_checksum_expected(source_request: dict[str, Any]) -> bool:
    if _is_hyperliquid_candle_request(source_request):
        return False
    if source_request["market"] == "cme_futures":
        return False
    if _is_sparse_ibkr_us_equity_request(source_request):
        return False
    if _is_kis_krx_intraday_request(source_request):
        return False
    return True


def _is_sparse_ibkr_us_equity_request(source_request: dict[str, Any]) -> bool:
    request = source_request.get("download_request", {})
    return (
        source_request.get("source_kind") == "ibkr_us_equity_historical_bars"
        and source_request.get("market") == "us_equity"
        and request.get("use_rth") is False
    )


def _is_hyperliquid_candle_request(source_request: dict[str, Any]) -> bool:
    return (
        source_request.get("source_kind") == "hyperliquid_candles"
        and source_request.get("source") == "hyperliquid"
        and source_request.get("market") == "crypto_perp"
    )


def _is_kis_krx_intraday_request(source_request: dict[str, Any]) -> bool:
    return (
        source_request.get("source_kind") == "kis_krx_intraday_bars"
        and source_request.get("source") == "kis"
        and source_request.get("market") == "krx_equity"
    )


def _non_exact_acceptance(
    *,
    source_request: dict[str, Any],
    legacy_rows: int,
    canonical_rows: int,
    common_rows: int,
    mismatches: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    if _exact_checksum_expected(source_request):
        return {"accepted": False, "reason": "exact_parity_required"}
    denominator = max(legacy_rows, canonical_rows, 1)
    timestamp_overlap_ratio = common_rows / denominator
    mismatch_rows = max((int(item.get("rows") or 0) for item in mismatches), default=0)
    value_mismatch_ratio = mismatch_rows / max(common_rows, 1)
    if source_request.get("market") == "cme_futures":
        cme_acceptance = _cme_physical_chain_panama_acceptance(
            source_request=source_request,
            candidates=candidates,
            repo_root=repo_root,
            legacy_rows=legacy_rows,
            canonical_rows=canonical_rows,
            common_rows=common_rows,
            timestamp_overlap_ratio=timestamp_overlap_ratio,
            value_mismatch_ratio=value_mismatch_ratio,
        )
        if cme_acceptance is not None:
            return cme_acceptance
    timestamp_threshold = 1.0 if _is_hyperliquid_candle_request(source_request) else 0.999
    if _is_hyperliquid_candle_request(source_request):
        value_threshold = max(0.005, min(0.01, 10 / max(common_rows, 1)))
    else:
        value_threshold = 0.001
    accepted = (
        timestamp_overlap_ratio >= timestamp_threshold
        and value_mismatch_ratio <= value_threshold
    )
    if _is_sparse_ibkr_us_equity_request(source_request):
        reason = "sparse_extended_hours_source_revisions_within_tolerance"
    elif _is_kis_krx_intraday_request(source_request):
        reason = "kis_intraday_source_revisions_within_tolerance"
    elif _is_hyperliquid_candle_request(source_request):
        reason = "hyperliquid_refresh_revisions_within_tolerance"
    else:
        reason = "non_exact_source_transform_within_tolerance"
    return {
        "accepted": accepted,
        "reason": reason if accepted else "non_exact_difference_exceeds_tolerance",
        "timestamp_overlap_ratio": timestamp_overlap_ratio,
        "value_mismatch_ratio": value_mismatch_ratio,
        "timestamp_overlap_threshold": timestamp_threshold,
        "value_mismatch_threshold": value_threshold,
    }


def _cme_physical_chain_panama_acceptance(
    *,
    source_request: dict[str, Any],
    candidates: list[dict[str, Any]],
    repo_root: Path,
    legacy_rows: int,
    canonical_rows: int,
    common_rows: int,
    timestamp_overlap_ratio: float,
    value_mismatch_ratio: float,
) -> dict[str, Any] | None:
    request_policy = str(
        (source_request.get("download_request") or {}).get("continuous_contract_policy") or ""
    )
    if request_policy != "ibkr_physical_contract_chain_panama_v1":
        return None
    manifests = _candidate_manifests(repo_root, candidates)
    if not manifests:
        return {
            "accepted": False,
            "reason": "cme_physical_chain_panama_manifest_missing",
            "timestamp_overlap_ratio": timestamp_overlap_ratio,
            "value_mismatch_ratio": value_mismatch_ratio,
            "timestamp_overlap_threshold": 1.0,
            "value_mismatch_threshold": 1.0,
        }
    lineage_items = [manifest.lineage or {} for manifest in manifests]
    missing_evidence = [
        field
        for field in (
            "roll_policy",
            "contract_chain_checksum",
            "continuous_construction_checksum",
            "source_contract_coverage",
            "source_conid_coverage",
        )
        if not all(str(lineage.get(field) or "").strip() for lineage in lineage_items)
    ]
    if missing_evidence:
        return {
            "accepted": False,
            "reason": "cme_physical_chain_panama_lineage_incomplete",
            "missing_evidence": missing_evidence,
            "timestamp_overlap_ratio": timestamp_overlap_ratio,
            "value_mismatch_ratio": value_mismatch_ratio,
            "timestamp_overlap_threshold": 1.0,
            "value_mismatch_threshold": 1.0,
        }
    coverage_ok = all(
        lineage.get("source_contract_coverage") == "all_rows"
        and lineage.get("source_conid_coverage") == "all_rows"
        for lineage in lineage_items
    )
    usable_ok = all(bool(manifest.usable_for_authoritative_validation) for manifest in manifests)
    canonical_overlap_ratio = common_rows / max(canonical_rows, 1)
    canonical_overlap_threshold = 0.999
    accepted = (
        usable_ok
        and coverage_ok
        and canonical_rows > 0
        and common_rows > 0
        and canonical_overlap_ratio >= canonical_overlap_threshold
    )
    legacy_coverage_gap = legacy_rows > common_rows
    reason = (
        "cme_physical_chain_panama_retention_cutoff_accepted"
        if accepted and legacy_coverage_gap
        else "cme_physical_chain_panama_legacy_continuous_difference_accepted"
        if accepted
        else "cme_physical_chain_panama_overlap_or_authority_failed"
    )
    return {
        "accepted": accepted,
        "reason": reason,
        "timestamp_overlap_ratio": timestamp_overlap_ratio,
        "canonical_timestamp_overlap_ratio": canonical_overlap_ratio,
        "canonical_timestamp_overlap_threshold": canonical_overlap_threshold,
        "value_mismatch_ratio": value_mismatch_ratio,
        "value_mismatch_threshold": 1.0,
        "retention_covered_subset": legacy_coverage_gap,
        "physical_chain_authority": "ibkr_physical_contract_chain_panama_v1",
        "manifest_authoritative": usable_ok,
        "source_contract_coverage": "all_rows" if coverage_ok else "incomplete",
        "contract_chain_checksums": sorted(
            {str(lineage.get("contract_chain_checksum") or "") for lineage in lineage_items}
        ),
        "continuous_construction_checksums": sorted(
            {str(lineage.get("continuous_construction_checksum") or "") for lineage in lineage_items}
        ),
        "roll_policies": sorted({str(lineage.get("roll_policy") or "") for lineage in lineage_items}),
    }


def _candidate_manifests(repo_root: Path, candidates: list[dict[str, Any]]) -> list[Any]:
    manifests = []
    for candidate in candidates:
        manifest_path = repo_root / str(candidate.get("manifest_path", ""))
        if not manifest_path.exists():
            continue
        manifests.append(load_market_manifest(manifest_path))
    return manifests
