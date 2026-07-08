"""Approval-only telemetry conformance checks."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.paths import monorepo_root, normalize_workspace_path

SCHEMA_VERSION = "approval_telemetry_conformance_report_v1"
REQUIRED_ENVELOPE_FIELDS = ("event_id", "bot_id", "event_type")
DROP_COUNT_FIELDS = (
    "dropped_count",
    "dropped_events",
    "quarantined_count",
    "rejected_count",
    "invalid_count",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an approval-only telemetry conformance report."
    )
    parser.add_argument("--agent-root", type=Path, default=monorepo_root())
    parser.add_argument("--scope", default="trading_stock_family")
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--telemetry-manifest", type=Path, default=None)
    parser.add_argument("--scheduled-shadow-report", type=Path, default=None)
    parser.add_argument("--relay-ingest-evidence", type=Path, default=None)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)

    report = run_telemetry_conformance_check(
        agent_root=args.agent_root,
        scope_id=args.scope,
        artifact_root=args.artifact_root,
        telemetry_manifest_path=args.telemetry_manifest,
        scheduled_shadow_report_path=args.scheduled_shadow_report,
        relay_ingest_evidence_path=args.relay_ingest_evidence,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] or not args.require_pass else 1


def run_telemetry_conformance_check(
    *,
    agent_root: Path,
    scope_id: str,
    artifact_root: Path | None = None,
    telemetry_manifest_path: Path | None = None,
    scheduled_shadow_report_path: Path | None = None,
    relay_ingest_evidence_path: Path | None = None,
) -> dict[str, Any]:
    agent_root = Path(agent_root).resolve()
    root = (
        normalize_workspace_path(agent_root, artifact_root)
        if artifact_root is not None
        else agent_root / "artifacts" / "validation" / "telemetry_conformance" / scope_id
    )
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "telemetry_conformance_report.json"
    scheduled_shadow = _read_json(
        normalize_workspace_path(agent_root, scheduled_shadow_report_path)
        if scheduled_shadow_report_path is not None
        else Path()
    )
    resolved_telemetry_manifest = (
        normalize_workspace_path(agent_root, telemetry_manifest_path)
        if telemetry_manifest_path is not None
        else None
    )
    resolved_relay = _relay_path(
        agent_root=agent_root,
        explicit=relay_ingest_evidence_path,
        scheduled_shadow=scheduled_shadow,
    )

    checks = [
        _telemetry_manifest_check(resolved_telemetry_manifest),
        _scheduled_shadow_reference_check(scheduled_shadow_report_path, scheduled_shadow),
        _relay_ingest_check(resolved_relay),
    ]
    blockers = [
        f"{check['name']}: {error}"
        for check in checks
        if not check["passed"]
        for error in check["errors"]
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "scope_id": scope_id,
        "ok": not blockers,
        "checks": checks,
        "blockers": blockers,
        "telemetry_manifest_path": _path_text(resolved_telemetry_manifest),
        "scheduled_shadow_report_path": str(scheduled_shadow_report_path or ""),
        "relay_ingest_evidence_path": _path_text(resolved_relay),
        "runtime_non_crashing": True,
        "artifact_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _telemetry_manifest_check(path: Path | None) -> dict[str, Any]:
    payload = _read_json(path)
    errors: list[str] = []
    evidence = [str(path)] if path is not None else []
    if path is None or not path.exists():
        errors.append("telemetry_manifest.json missing")
    elif not payload:
        errors.append("telemetry_manifest.json missing or malformed")
    else:
        schema = str(payload.get("manifest_version") or payload.get("schema_version") or "")
        if schema and schema != "telemetry_manifest_v1":
            errors.append(f"telemetry manifest schema is {schema!r}")
        _count_value(
            payload.get("total_events"),
            label="telemetry manifest total_events",
            errors=errors,
        )
        missing_counts = payload.get("missing_field_counts")
        if isinstance(missing_counts, dict):
            bad = {}
            for field, count in missing_counts.items():
                value = _count_value(
                    count,
                    label=f"telemetry manifest missing_field_counts.{field}",
                    errors=errors,
                )
                if value > 0 and str(field) in REQUIRED_ENVELOPE_FIELDS:
                    bad[str(field)] = value
            if bad:
                errors.append(f"required envelope fields missing in telemetry manifest: {bad}")
    return _check("telemetry_manifest_conformant", not errors, errors, evidence)


def _scheduled_shadow_reference_check(
    path: Path | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    evidence = [str(path)] if path else []
    if path is None:
        errors.append("scheduled shadow report missing")
    elif not payload:
        errors.append("scheduled shadow report missing or malformed")
    elif payload.get("ok") is not True:
        errors.append("scheduled shadow report is not ok")
    return _check("scheduled_shadow_telemetry_reference", not errors, errors, evidence)


def _relay_ingest_check(path: Path | None) -> dict[str, Any]:
    payload = _read_json(path)
    errors: list[str] = []
    evidence = [str(path)] if path is not None else []
    if path is None or not path.exists() or not path.is_file():
        errors.append("relay ingest evidence missing")
        return _check("relay_ingest_no_drops_or_quarantines", False, errors, evidence)
    if not payload:
        errors.append("relay ingest evidence missing or malformed")
        return _check("relay_ingest_no_drops_or_quarantines", False, errors, evidence)
    if payload.get("ok") is False:
        errors.append("relay ingest evidence is not ok")
    for field in DROP_COUNT_FIELDS:
        count = _count_value(
            payload.get(field),
            label=f"relay ingest {field}",
            errors=errors,
        )
        if count > 0:
            errors.append(f"relay ingest {field} is {count}")
    for label in ("classification_counts", "dropped_field_counts", "unknown_field_counts"):
        counts = payload.get(label)
        if isinstance(counts, dict):
            for key, value in counts.items():
                count = _count_value(
                    value,
                    label=f"relay ingest {label}.{key}",
                    errors=errors,
                )
                if count <= 0:
                    continue
                key_text = str(key).lower()
                if label == "classification_counts" and key_text not in {
                    "enqueued",
                    "duplicate",
                }:
                    errors.append(f"relay ingest classification {key} count is {count}")
                elif label != "classification_counts":
                    errors.append(f"relay ingest {label}.{key} count is {count}")
    for index, event in enumerate(_sample_events(payload)):
        if not isinstance(event, dict):
            errors.append(f"sample_events[{index}] is not an object")
            continue
        missing = [field for field in REQUIRED_ENVELOPE_FIELDS if not str(event.get(field) or "")]
        if missing:
            errors.append(f"sample_events[{index}] missing required fields: {', '.join(missing)}")
    return _check("relay_ingest_no_drops_or_quarantines", not errors, errors, evidence)


def _relay_path(
    *,
    agent_root: Path,
    explicit: Path | None,
    scheduled_shadow: dict[str, Any],
) -> Path | None:
    if explicit is not None:
        return normalize_workspace_path(agent_root, explicit)
    raw = str(scheduled_shadow.get("relay_ingest_evidence_path") or "")
    return normalize_workspace_path(agent_root, raw) if raw else None


def _path_text(path: Path | None) -> str:
    return str(path) if path is not None else ""


def _sample_events(payload: dict[str, Any]) -> list[Any]:
    for key in ("sample_events", "events", "accepted_events"):
        value = payload.get(key)
        if isinstance(value, list):
            return value[:20]
    return []


def _count_value(value: Any, *, label: str, errors: list[str]) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        errors.append(f"{label} must be an integer count; found {value!r}")
        return 0
    try:
        count = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be an integer count; found {value!r}")
        return 0
    if count < 0:
        errors.append(f"{label} cannot be negative")
    return count


def _check(
    name: str,
    passed: bool,
    errors: list[str],
    evidence_paths: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "errors": [] if passed else errors,
        "evidence_paths": evidence_paths,
    }


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
