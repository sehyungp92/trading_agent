"""Recreate committed data-bundle evidence and emit reproduction reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from trading_assistant_backtest.contract_models import DataBundleManifest, DataBundleStatus
from trading_assistant_backtest.paths import monorepo_root, workspace_root

SCOPE_GLOBS = {
    "crypto_trader_portfolio": (
        "*/crypto_portfolio/phased_optimizer/data_bundle_manifest.json",
    ),
    "k_stock_olr_kalcb": (
        "*/k_stock_olr_kalcb/portfolio/data_bundle_manifest.json",
    ),
    "trading_stock_family": (
        "*/trading_stock_family/portfolio/data_bundle_manifest.json",
    ),
    "trading_momentum_family": (
        "*/trading_momentum_family/portfolio/data_bundle_manifest.json",
    ),
    "trading_swing_family": (
        "*/trading_swing_family/portfolio/data_bundle_manifest.json",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce authoritative data bundles and emit data_reproduction_report.json."
    )
    parser.add_argument("--agent-root", type=Path, default=_default_agent_root())
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--bundle-manifest", action="append", type=Path, default=[])
    parser.add_argument(
        "--scope",
        choices=(*SCOPE_GLOBS.keys(), "all"),
        default="all",
    )
    args = parser.parse_args(argv)

    agent_root = Path(args.agent_root).resolve()
    artifact_root = Path(args.artifact_root or _default_artifact_root(agent_root)).resolve()
    bundle_paths = (
        [Path(path).resolve() for path in args.bundle_manifest]
        if args.bundle_manifest
        else _default_bundle_paths(agent_root, args.scope)
    )
    reports = [
        reproduce_data_bundle(
            bundle_manifest_path=path,
            data_repo_root=workspace_root(agent_root, "trading_assistant_data"),
            artifact_root=artifact_root / _report_slug(path),
        )
        for path in bundle_paths
    ]
    summary = {
        "status": "pass" if reports and all(report["ok"] for report in reports) else "fail",
        "report_count": len(reports),
        "reports": reports,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary["status"] == "pass" else 1


def reproduce_data_bundle(
    *,
    bundle_manifest_path: Path,
    data_repo_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    bundle_manifest_path = Path(bundle_manifest_path).resolve()
    data_repo_root = Path(data_repo_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    raw_bundle = _read_json(bundle_manifest_path)
    bundle = DataBundleManifest.model_validate(raw_bundle)
    recomputed_bundle = DataBundleManifest.model_validate(
        {**raw_bundle, "bundle_checksum": "", "bundle_id": ""}
    )
    slice_index_path = bundle_manifest_path.with_name("slice_index.json")
    slice_index = _read_json(slice_index_path)
    index_by_manifest_id = {
        str(item.get("manifest_id") or ""): item
        for item in slice_index.get("slices", [])
        if isinstance(item, dict)
    }

    checks = [
        _check(
            "bundle_is_authoritative",
            bundle.status == DataBundleStatus.AUTHORITATIVE,
            [] if bundle.status == DataBundleStatus.AUTHORITATIVE else [bundle.status.value],
        ),
        _check(
            "bundle_checksum_matches",
            bundle.bundle_checksum == recomputed_bundle.bundle_checksum,
            _error_if(
                bundle.bundle_checksum != recomputed_bundle.bundle_checksum,
                f"{bundle.bundle_checksum} != {recomputed_bundle.bundle_checksum}",
            ),
        ),
        _check(
            "slice_index_bundle_checksum_matches",
            slice_index.get("bundle_checksum") == bundle.bundle_checksum,
            _error_if(
                slice_index.get("bundle_checksum") != bundle.bundle_checksum,
                "slice_index bundle_checksum does not match data bundle",
            ),
        ),
    ]
    slice_reports: list[dict[str, Any]] = []
    for bundle_slice in bundle.slice_manifests:
        manifest_path = _resolve_data_path(data_repo_root, bundle_slice.manifest_path)
        manifest = _read_json(manifest_path)
        index_entry = index_by_manifest_id.get(bundle_slice.manifest_id, {})
        canonical_paths = [
            _resolve_data_path(data_repo_root, path)
            for path in index_entry.get("canonical_paths", [])
        ]
        slice_report, slice_checks = _reproduce_slice(
            bundle=bundle,
            bundle_slice=bundle_slice.model_dump(mode="json"),
            manifest=manifest,
            manifest_path=manifest_path,
            canonical_paths=canonical_paths,
        )
        slice_reports.append(slice_report)
        checks.extend(slice_checks)

    report_path = artifact_root / "data_reproduction_report.json"
    report = {
        "adjustment_policy": bundle.adjustment_policy,
        "bundle_checksum": bundle.bundle_checksum,
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_path": str(bundle_manifest_path),
        "calendars": bundle.calendars,
        "checks": checks,
        "fee_model_version": bundle.fee_model_version,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "ok": all(check["passed"] for check in checks),
        "recomputed_bundle_checksum": recomputed_bundle.bundle_checksum,
        "report_path": str(report_path),
        "slice_count": len(slice_reports),
        "slices": slice_reports,
        "slippage_model_version": bundle.slippage_model_version,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _reproduce_slice(
    *,
    bundle: DataBundleManifest,
    bundle_slice: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    canonical_paths: list[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_id = str(bundle_slice.get("manifest_id") or "")
    existing_paths = [path for path in canonical_paths if path.exists()]
    checks = [
        _check(
            f"slice:{manifest_id}:manifest_fields_match_bundle",
            _manifest_fields_match_bundle(bundle_slice, manifest),
            _manifest_field_errors(bundle_slice, manifest),
        ),
        _check(
            f"slice:{manifest_id}:slice_index_paths_exist",
            bool(canonical_paths) and len(existing_paths) == len(canonical_paths),
            [
                str(path)
                for path in canonical_paths
                if not path.exists()
            ]
            or ([] if canonical_paths else ["slice_index has no canonical paths"]),
        ),
    ]

    frames = [pd.read_parquet(path) for path in existing_paths]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    timestamp_column = _timestamp_column(frame)
    observed_start = _timestamp_value(frame[timestamp_column].min()) if timestamp_column else ""
    observed_end = _timestamp_value(frame[timestamp_column].max()) if timestamp_column else ""
    checksums = [_parquet_content_checksum(path) for path in existing_paths]
    actual_bars = int(len(frame))
    expected_bars = int(manifest.get("expected_bars") or 0)
    manifest_actual_bars = int(manifest.get("actual_bars") or 0)
    manifest_checksum = str(manifest.get("checksum") or "")
    checksum_matches = manifest_checksum in checksums if checksums else False

    checks.extend(
        [
            _check(
                f"slice:{manifest_id}:checksum_matches_canonical_parquet",
                checksum_matches,
                _error_if(not checksum_matches, "manifest checksum not found in canonical files"),
            ),
            _check(
                f"slice:{manifest_id}:bar_count_matches_manifest",
                actual_bars == manifest_actual_bars,
                _error_if(
                    actual_bars != manifest_actual_bars,
                    f"{actual_bars} != {manifest_actual_bars}",
                ),
            ),
            _check(
                f"slice:{manifest_id}:expected_bars_match_calendar",
                expected_bars == int(bundle_slice.get("expected_bars") or expected_bars),
                [],
            ),
            _check(
                f"slice:{manifest_id}:coverage_matches",
                float(manifest.get("coverage_ratio") or 0.0) >= 1.0,
                _error_if(
                    float(manifest.get("coverage_ratio") or 0.0) < 1.0,
                    f"coverage_ratio={manifest.get('coverage_ratio')}",
                ),
            ),
            _check(
                f"slice:{manifest_id}:policy_versions_match_bundle",
                _policy_versions_match_bundle(bundle, manifest),
                _policy_version_errors(bundle, manifest),
            ),
            _check(
                f"slice:{manifest_id}:calendar_matches_bundle",
                str(manifest.get("session_calendar") or "") in set(bundle.calendars),
                _error_if(
                    str(manifest.get("session_calendar") or "") not in set(bundle.calendars),
                    "slice calendar not present in bundle calendars",
                ),
            ),
        ]
    )
    report = {
        "actual_bars": manifest_actual_bars,
        "adjustment_policy": manifest.get("adjustment_policy", ""),
        "calendar": manifest.get("session_calendar", ""),
        "canonical_paths": [str(path) for path in existing_paths],
        "checksum": manifest_checksum,
        "coverage_ratio": manifest.get("coverage_ratio", 0.0),
        "end_ts": _timestamp_value(manifest.get("end_ts", "")),
        "expected_bars": expected_bars,
        "fee_model_version": manifest.get("fee_model_version", ""),
        "manifest_id": manifest_id,
        "manifest_path": str(manifest_path),
        "market": manifest.get("market", ""),
        "observed_end_ts": observed_end,
        "observed_start_ts": observed_start,
        "recomputed_actual_bars": actual_bars,
        "recomputed_canonical_checksums": checksums,
        "recomputed_expected_bars": expected_bars,
        "slippage_model_version": manifest.get("slippage_model_version", ""),
        "source": manifest.get("source", ""),
        "start_ts": _timestamp_value(manifest.get("start_ts", "")),
        "symbol": manifest.get("symbol", ""),
        "timeframe": manifest.get("timeframe", ""),
    }
    return report, checks


def _manifest_fields_match_bundle(bundle_slice: dict[str, Any], manifest: dict[str, Any]) -> bool:
    return not _manifest_field_errors(bundle_slice, manifest)


def _manifest_field_errors(bundle_slice: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for bundle_key, manifest_key in (
        ("manifest_id", "manifest_id"),
        ("source", "source"),
        ("market", "market"),
        ("symbol", "symbol"),
        ("timeframe", "timeframe"),
        ("checksum", "checksum"),
        ("calendar", "session_calendar"),
    ):
        if str(bundle_slice.get(bundle_key) or "") != str(manifest.get(manifest_key) or ""):
            errors.append(f"{bundle_key}/{manifest_key} mismatch")
    if bundle_slice.get("authoritative") != manifest.get("usable_for_authoritative_validation"):
        errors.append("authoritative flag mismatch")
    return errors


def _policy_versions_match_bundle(bundle: DataBundleManifest, manifest: dict[str, Any]) -> bool:
    return not _policy_version_errors(bundle, manifest)


def _policy_version_errors(bundle: DataBundleManifest, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in ("fee_model_version", "slippage_model_version"):
        if str(getattr(bundle, field_name) or "") != str(manifest.get(field_name) or ""):
            errors.append(f"{field_name} mismatch")
    if not _adjustment_policy_matches_bundle(
        bundle_policy=str(bundle.adjustment_policy or ""),
        slice_policy=str(manifest.get("adjustment_policy") or ""),
    ):
        errors.append("adjustment_policy mismatch")
    return errors


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


def _timestamp_column(frame: pd.DataFrame) -> str:
    for candidate in ("timestamp_utc", "timestamp", "datetime", "date"):
        if candidate in frame.columns:
            return candidate
    return ""


def _timestamp_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _check(name: str, passed: bool, errors: list[str]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "errors": errors}


def _error_if(condition: bool, message: str) -> list[str]:
    return [message] if condition else []


def _resolve_data_path(data_repo_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else data_repo_root / candidate


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_content_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_sha256_file(path).encode("ascii"))
    try:
        schema = pq.read_schema(path)
        digest.update(str(schema).encode("utf-8"))
        metadata = schema.metadata or {}
        for key in sorted(metadata):
            digest.update(key)
            digest.update(metadata[key])
    except Exception:
        digest.update(b"<schema-unavailable>")
    return digest.hexdigest()


def _default_bundle_paths(agent_root: Path, scope: str) -> list[Path]:
    data_bundle_root = (
        workspace_root(agent_root, "trading_assistant_data")
        / "data"
        / "bundles"
        / "monthly"
    )
    scopes = SCOPE_GLOBS.keys() if scope == "all" else (scope,)
    paths: list[Path] = []
    for scope_id in scopes:
        for pattern in SCOPE_GLOBS[scope_id]:
            paths.extend(data_bundle_root.glob(pattern))
    return sorted(paths)


def _report_slug(bundle_path: Path) -> str:
    parts = Path(bundle_path).parts
    if len(parts) < 4:
        return Path(bundle_path).parent.name
    strategy_id = parts[-2]
    bot_id = parts[-3]
    run_month = parts[-4]
    base = f"{bot_id}_{strategy_id}"
    if bot_id == "crypto_portfolio" and strategy_id in {
        "btc_1m",
        "full_crypto_bars",
        "phased_optimizer",
    }:
        return base
    return f"{base}_{run_month}"


def _default_artifact_root(agent_root: Path) -> Path:
    return (
        workspace_root(agent_root, "trading_assistant_backtest")
        / "artifacts"
        / "validation"
        / "data_reproduction"
    )


def _default_agent_root() -> Path:
    return monorepo_root()


if __name__ == "__main__":
    raise SystemExit(main())
