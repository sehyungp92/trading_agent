"""Artifact-producing data bundle reproduction checks."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .calendars.core import expected_bars
from .calendars.cme import CALENDAR_ID as CME_CALENDAR_ID
from .calendars.cme import calendar_definition as cme_calendar_definition
from .calendars.crypto import calendar_definition as crypto_calendar_definition
from .calendars.krx import (
    CALENDAR_ID as KRX_CALENDAR_ID,
    KIS_INTRADAY_CALENDAR_ID,
    calendar_definition as krx_calendar_definition,
    kis_intraday_calendar_definition,
)
from .calendars.us_equities import CALENDAR_ID as US_EQUITIES_CALENDAR_ID
from .calendars.us_equities import calendar_definition as us_equities_calendar_definition
from .checksums import parquet_content_checksum
from .manifests import DataBundleManifest, MarketDataManifest, load_bundle_manifest, load_market_manifest


def reproduce_data_bundle(
    *,
    repo_root: Path,
    bundle_manifest_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Recompute committed bundle facts and emit a durable reproduction report."""

    repo_root = Path(repo_root).resolve()
    bundle_manifest_path = _resolve(bundle_manifest_path, repo_root)
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle_manifest(bundle_manifest_path)
    slice_index_path = bundle_manifest_path.with_name("slice_index.json")
    slice_index = json.loads(slice_index_path.read_text(encoding="utf-8"))
    index_by_manifest = {
        str(item.get("manifest_id")): item for item in slice_index.get("slices", [])
    }

    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "bundle_is_authoritative",
            bundle.usable_for_authoritative_validation,
            [] if bundle.usable_for_authoritative_validation else bundle.authoritative_contract_errors(),
        )
    )
    recomputed_bundle_checksum = _bundle_checksum(bundle)
    checks.append(
        _check(
            "bundle_checksum_matches",
            recomputed_bundle_checksum == bundle.bundle_checksum,
            _mismatch(
                recomputed_bundle_checksum,
                bundle.bundle_checksum,
                "recomputed bundle checksum",
            ),
        )
    )
    checks.append(
        _check(
            "slice_index_bundle_checksum_matches",
            slice_index.get("bundle_checksum") == bundle.bundle_checksum,
            _mismatch(slice_index.get("bundle_checksum"), bundle.bundle_checksum, "slice index checksum"),
        )
    )

    slice_reports: list[dict[str, Any]] = []
    for item in bundle.slice_manifests:
        slice_report, slice_checks = _reproduce_slice(
            repo_root=repo_root,
            bundle=bundle,
            item=item,
            index_entry=index_by_manifest.get(item.manifest_id, {}),
        )
        slice_reports.append(slice_report)
        checks.extend(slice_checks)

    ok = all(check["passed"] for check in checks)
    report_path = artifact_root / "data_reproduction_report.json"
    payload = {
        "ok": ok,
        "bundle_manifest_path": str(bundle_manifest_path),
        "bundle_id": bundle.bundle_id,
        "bundle_checksum": bundle.bundle_checksum,
        "recomputed_bundle_checksum": recomputed_bundle_checksum,
        "slice_count": len(bundle.slice_manifests),
        "calendars": bundle.calendars,
        "fee_model_version": bundle.fee_model_version,
        "slippage_model_version": bundle.slippage_model_version,
        "adjustment_policy": bundle.adjustment_policy,
        "checks": checks,
        "slices": slice_reports,
        "report_path": str(report_path),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def _reproduce_slice(
    *,
    repo_root: Path,
    bundle: DataBundleManifest,
    item: Any,
    index_entry: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = repo_root / item.manifest_path
    manifest = load_market_manifest(manifest_path)
    canonical_paths = [repo_root / path for path in index_entry.get("canonical_paths", [])]
    frames = [pd.read_parquet(path) for path in canonical_paths if path.exists()]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    actual_bars = int(len(frame))
    timestamp_column = "timestamp_utc" if "timestamp_utc" in frame.columns else "timestamp"
    start_ts = pd.Timestamp(frame[timestamp_column].min()).to_pydatetime() if actual_bars else None
    end_ts = pd.Timestamp(frame[timestamp_column].max()).to_pydatetime() if actual_bars else None
    calendar = _calendar_for(manifest)
    expected_count = expected_bars(calendar, manifest.timeframe, manifest.start_ts, manifest.end_ts)
    canonical_checksums = [parquet_content_checksum(path) for path in canonical_paths]
    manifest_checksum_match = len(canonical_checksums) == 1 and canonical_checksums[0] == manifest.checksum

    prefix = f"slice:{item.manifest_id}"
    checks = [
        _check(
            f"{prefix}:manifest_fields_match_bundle",
            _slice_manifest_matches(item, manifest),
            ["bundle slice metadata does not match MarketDataManifest"]
            if not _slice_manifest_matches(item, manifest) else [],
        ),
        _check(
            f"{prefix}:slice_index_paths_exist",
            bool(canonical_paths) and all(path.exists() for path in canonical_paths),
            [f"missing canonical path: {path}" for path in canonical_paths if not path.exists()],
        ),
        _check(
            f"{prefix}:checksum_matches_canonical_parquet",
            manifest_checksum_match,
            _mismatch(canonical_checksums, manifest.checksum, "canonical parquet checksum"),
        ),
        _check(
            f"{prefix}:bar_count_matches_manifest",
            actual_bars == manifest.actual_bars,
            _mismatch(actual_bars, manifest.actual_bars, "actual bars"),
        ),
        _check(
            f"{prefix}:expected_bars_match_calendar",
            expected_count == manifest.expected_bars,
            _mismatch(expected_count, manifest.expected_bars, "calendar expected bars"),
        ),
        _check(
            f"{prefix}:coverage_matches",
            manifest.expected_bars > 0
            and manifest.actual_bars / manifest.expected_bars == manifest.coverage_ratio,
            ["coverage ratio does not match actual/expected bars"],
        ),
        _check(
            f"{prefix}:policy_versions_match_bundle",
            manifest.fee_model_version == bundle.fee_model_version
            and manifest.slippage_model_version == bundle.slippage_model_version
            and _adjustment_policy_matches_bundle(
                bundle_policy=bundle.adjustment_policy,
                slice_policy=manifest.adjustment_policy,
            ),
            ["fee/slippage/adjustment policy differs between slice and bundle"],
        ),
        _check(
            f"{prefix}:calendar_matches_bundle",
            manifest.session_calendar in bundle.calendars and item.calendar == manifest.session_calendar,
            ["slice calendar differs from bundle calendars"],
        ),
    ]
    report = {
        "manifest_id": item.manifest_id,
        "manifest_path": str(manifest_path),
        "canonical_paths": [str(path) for path in canonical_paths],
        "source": manifest.source,
        "market": manifest.market,
        "symbol": manifest.symbol,
        "timeframe": manifest.timeframe,
        "start_ts": manifest.start_ts.isoformat(),
        "end_ts": manifest.end_ts.isoformat(),
        "observed_start_ts": start_ts.isoformat() if start_ts else "",
        "observed_end_ts": end_ts.isoformat() if end_ts else "",
        "expected_bars": manifest.expected_bars,
        "recomputed_expected_bars": expected_count,
        "actual_bars": manifest.actual_bars,
        "recomputed_actual_bars": actual_bars,
        "coverage_ratio": manifest.coverage_ratio,
        "checksum": manifest.checksum,
        "recomputed_canonical_checksums": canonical_checksums,
        "calendar": manifest.session_calendar,
        "fee_model_version": manifest.fee_model_version,
        "slippage_model_version": manifest.slippage_model_version,
        "adjustment_policy": manifest.adjustment_policy,
    }
    return report, checks


def _adjustment_policy_matches_bundle(*, bundle_policy: str, slice_policy: str) -> bool:
    if bundle_policy == slice_policy:
        return True
    prefix = "mixed_adjustment_policy:"
    if not bundle_policy.startswith(prefix):
        return False
    allowed = {
        item.strip()
        for item in bundle_policy.removeprefix(prefix).split(",")
        if item.strip()
    }
    return slice_policy in allowed


def _slice_manifest_matches(item: Any, manifest: MarketDataManifest) -> bool:
    return (
        item.manifest_id == manifest.manifest_id
        and item.source == manifest.source
        and item.market == manifest.market
        and item.symbol == manifest.symbol
        and item.timeframe == manifest.timeframe
        and item.checksum == manifest.checksum
        and item.calendar == manifest.session_calendar
        and item.authoritative == manifest.usable_for_authoritative_validation
    )


def _bundle_checksum(bundle: DataBundleManifest) -> str:
    raw = "|".join(
        [
            bundle.data_repo_commit_sha,
            bundle.fee_model_version,
            bundle.slippage_model_version,
            bundle.adjustment_policy,
            *[
                "|".join([item.manifest_id, item.symbol, item.timeframe, item.checksum])
                for item in bundle.slice_manifests
            ],
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _calendar_for(manifest: MarketDataManifest):
    if manifest.session_calendar == "crypto_utc_24_7_v1":
        return crypto_calendar_definition()
    if manifest.session_calendar == CME_CALENDAR_ID:
        return cme_calendar_definition()
    if manifest.session_calendar == US_EQUITIES_CALENDAR_ID:
        return us_equities_calendar_definition()
    if manifest.session_calendar == KRX_CALENDAR_ID:
        holidays = _krx_holidays_path(manifest)
        return krx_calendar_definition(holidays if holidays.exists() else None)
    if manifest.session_calendar == KIS_INTRADAY_CALENDAR_ID:
        holidays = _krx_holidays_path(manifest)
        return kis_intraday_calendar_definition(holidays if holidays.exists() else None)
    raise ValueError(f"unsupported reproduction calendar: {manifest.session_calendar}")


def _krx_holidays_path(manifest: MarketDataManifest) -> Path:
    raw = str(manifest.lineage.get("calendar_holidays_path", "") if manifest.lineage else "")
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2] / "data" / "calendars" / "krx_holidays.yaml"


def _resolve(path: Path, repo_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def _check(name: str, passed: bool, errors: list[Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "errors": [] if passed else [str(error) for error in errors],
    }


def _mismatch(actual: Any, expected: Any, label: str) -> list[str]:
    return [] if actual == expected else [f"{label} mismatch: {actual!r} != {expected!r}"]
