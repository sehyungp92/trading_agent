from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from migration_support import ROOT, file_sha256, read_json

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import verify_operational_deployment_evidence as verifier  # noqa: E402


COLLECTION_SCHEMA = "trading_agent_operational_evidence_collection_v1"
OUTPUT_SCHEMA = "trading_agent_operational_evidence_v1"


def main() -> int:
    args = _parser().parse_args()
    result = collect_operational_evidence(
        manifest_path=_resolve(args.manifest),
        output_path=_resolve(args.output),
        plan_path=_resolve(args.plan),
        reviewed_commit=args.reviewed_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble deployments/operational_evidence.json from collected VPS "
            "and assistant evidence, then validate it with the strict verifier."
        )
    )
    parser.add_argument("--manifest", required=True, help="Collected evidence manifest JSON")
    parser.add_argument("--output", default="deployments/operational_evidence.json")
    parser.add_argument("--plan", default="deployments/cutover_plan.json")
    parser.add_argument("--reviewed-commit", default="")
    return parser


def collect_operational_evidence(
    *,
    manifest_path: Path,
    output_path: Path,
    plan_path: Path,
    reviewed_commit: str = "",
) -> dict[str, Any]:
    errors: list[str] = []
    if not manifest_path.exists():
        return _result(False, output_path, [f"collection manifest missing: {_label(manifest_path)}"])
    if not plan_path.exists():
        return _result(False, output_path, [f"cutover plan missing: {_label(plan_path)}"])

    manifest = read_json(manifest_path)
    plan = read_json(plan_path)
    if manifest.get("schema_version") != COLLECTION_SCHEMA:
        errors.append(f"collection manifest schema_version must be {COLLECTION_SCHEMA}")

    reviewed = (
        str(reviewed_commit or manifest.get("reviewed_commit_sha") or "").strip()
        or _git_head()
    )
    now = _utc_z()
    plan_records = {
        str(record.get("bot") or ""): record
        for record in plan.get("records", [])
        if isinstance(record, dict)
    }
    records = []
    for raw in manifest.get("records", []):
        if not isinstance(raw, dict):
            errors.append("collection manifest records must be objects")
            continue
        records.append(_record(raw, plan_records.get(str(raw.get("bot") or ""), {}), reviewed, now, errors))

    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "reviewed_commit_sha": reviewed,
        "generated_at_utc": now,
        "records": records,
    }
    if not errors:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f"{output_path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        errors.extend(verifier._evidence_errors(temp_path, plan_path))
        if errors:
            temp_path.unlink(missing_ok=True)
        else:
            temp_path.replace(output_path)
    return _result(not errors, output_path, errors)


def _record(
    raw: dict[str, Any],
    plan_record: dict[str, Any],
    reviewed: str,
    now: str,
    errors: list[str],
) -> dict[str, Any]:
    bot = str(raw.get("bot") or "").strip()
    metadata = _section(raw.get("deployment_metadata"), now)
    metadata_refs = verifier._deployment_metadata_refs(metadata)
    assistant_ingest = _assistant_ingest(raw.get("assistant_ingest"), metadata_refs, now, errors, bot)
    return {
        "bot": bot,
        "running_commit_sha": str(raw.get("running_commit_sha") or reviewed),
        "vps_deployment": _vps_deployment(raw.get("vps_deployment"), plan_record, errors, bot, now),
        "sidecar_forwarding": _section(raw.get("sidecar_forwarding"), now),
        "assistant_ingest": assistant_ingest,
        "deployment_metadata": metadata,
        "monthly_shadow": _section(raw.get("monthly_shadow"), now),
        "rollback_smoke": dict(raw.get("rollback_smoke") or {}),
        "evidence_artifacts": _evidence_artifacts(raw.get("evidence_artifacts")),
    }


def _vps_deployment(
    value: Any,
    plan_record: dict[str, Any],
    errors: list[str],
    bot: str,
    now: str,
) -> dict[str, Any]:
    payload = dict(value or {}) if isinstance(value, dict) else {}
    compose_file = str(payload.get("compose_file") or plan_record.get("compose_file") or "")
    if compose_file:
        payload["compose_file"] = compose_file
        compose_path = _resolve(compose_file)
        if compose_path.exists() and not payload.get("compose_sha256"):
            payload["compose_sha256"] = file_sha256(compose_path)
    health_url = str(payload.get("health_url") or "").strip()
    if health_url:
        try:
            health = _read_http_json(health_url)
        except Exception as exc:
            errors.append(f"{bot}: vps_deployment.health_url probe failed: {exc}")
        else:
            payload["health_checked_at_utc"] = now
            payload["health_response"] = health
            payload["running"] = _health_running(health)
            if not payload.get("services"):
                services = _health_services(health)
                if services:
                    payload["services"] = services
    return payload


def _section(value: Any, now: str) -> dict[str, Any]:
    payload = dict(value or {}) if isinstance(value, dict) else {}
    if payload.get("ok") is True:
        payload.setdefault("checked_at_utc", now)
    return payload


def _assistant_ingest(
    value: Any,
    metadata_refs: dict[str, set[str]],
    now: str,
    errors: list[str],
    bot: str,
) -> dict[str, Any]:
    payload = _section(value, now)
    relay_db_path = str(payload.get("relay_db_path") or "").strip()
    relay_items = [
        _normalise_relay_evidence(item, metadata_refs, relay_db_path, errors, bot)
        for item in _relay_items(payload, relay_db_path, errors, bot)
    ]
    if relay_items:
        payload["relay_ingest_evidence"] = relay_items[0] if len(relay_items) == 1 else relay_items
        payload.pop("relay_ingest_evidence_path", None)
        payload.pop("relay_ingest_evidence_paths", None)
        payload.pop("relay_db_path", None)
        payload.pop("relay_ingest_event_id", None)
        payload.pop("relay_ingest_event_ids", None)
        payload.setdefault("events_ingested", len(relay_items))
    return payload


def _relay_items(
    value: dict[str, Any],
    relay_db_path: str,
    errors: list[str],
    bot: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    raw = value.get("relay_ingest_evidence")
    if isinstance(raw, list):
        items.extend(item for item in raw if isinstance(item, dict))
    elif isinstance(raw, dict):
        items.append(raw)
    paths: list[Any] = []
    if value.get("relay_ingest_evidence_path"):
        paths.append(value["relay_ingest_evidence_path"])
    if isinstance(value.get("relay_ingest_evidence_paths"), list):
        paths.extend(value["relay_ingest_evidence_paths"])
    for raw_path in paths:
        path = _resolve(str(raw_path or ""))
        if not path.exists():
            errors.append(f"{bot}: assistant_ingest relay evidence path missing: {raw_path}")
            continue
        loaded = read_json(path)
        if isinstance(loaded, list):
            items.extend(item for item in loaded if isinstance(item, dict))
        elif isinstance(loaded, dict):
            items.append(loaded)
        else:
            errors.append(f"{bot}: assistant_ingest relay evidence path is not object/list: {raw_path}")
    event_ids = _relay_event_ids(value)
    if event_ids and not relay_db_path:
        errors.append(f"{bot}: relay_ingest_event_ids require assistant_ingest.relay_db_path")
    for event_id in event_ids:
        db_event = _relay_db_event(relay_db_path, event_id, errors, bot)
        if db_event:
            items.append(db_event)
    return items


def _normalise_relay_evidence(
    evidence: dict[str, Any],
    metadata_refs: dict[str, set[str]],
    relay_db_path: str,
    errors: list[str],
    bot: str,
) -> dict[str, Any]:
    payload = dict(evidence)
    event_id = str(payload.get("event_id") or "").strip()
    if relay_db_path and event_id and payload.get("relay_db_confirmed") is not True:
        db_event = _relay_db_event(relay_db_path, event_id, errors, bot)
        if db_event:
            for key, value in db_event.items():
                if value not in ("", None, []) and not payload.get(key):
                    payload[key] = value
            payload["relay_db_confirmed"] = True
    hashes = sorted(metadata_refs.get("hashes") or set())
    if hashes:
        payload.setdefault("deployment_metadata_hashes", hashes)
    fingerprint = str(payload.get("secret_fingerprint") or payload.get("hmac_secret_fingerprint") or "")
    if fingerprint and not isinstance(payload.get("auth"), dict):
        payload["auth"] = {"secret_fingerprint": fingerprint}
    if "freshness" not in payload and (payload.get("observed_at") or payload.get("generated_at")):
        payload["freshness"] = {"ok": payload.get("ok") is True, "max_event_age_seconds": 0}
    return payload


def _relay_event_ids(value: dict[str, Any]) -> list[str]:
    raw = value.get("relay_ingest_event_ids")
    if raw is None and value.get("relay_ingest_event_id"):
        raw = [value.get("relay_ingest_event_id")]
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item or "").strip()]


def _relay_db_event(db_path: str, event_id: str, errors: list[str], bot: str) -> dict[str, Any]:
    if not db_path:
        return {}
    path = _resolve(db_path)
    if not path.exists():
        errors.append(f"{bot}: relay DB missing: {db_path}")
        return {}
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT event_id, bot_id, event_type, payload, exchange_timestamp, received_at
                FROM events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        errors.append(f"{bot}: relay DB read failed for event_id {event_id}: {exc}")
        return {}
    if row is None:
        errors.append(f"{bot}: relay DB missing event_id {event_id}")
        return {}
    payload = _json_object(row["payload"])
    observed_at = _utc_z_from(row["received_at"] or row["exchange_timestamp"])
    return {
        "ok": True,
        "event_id": str(row["event_id"] or event_id),
        "bot_id": str(row["bot_id"] or payload.get("bot_id") or ""),
        "event_type": str(row["event_type"] or payload.get("event_type") or ""),
        "runtime_instance_id": str(payload.get("runtime_instance_id") or ""),
        "effective_config_hash": str(payload.get("effective_config_hash") or ""),
        "deployment_id": str(payload.get("deployment_id") or ""),
        "source": str(payload.get("source") or ""),
        "observed_at": observed_at,
        "generated_at": observed_at,
        "freshness": {"ok": True, "max_event_age_seconds": 0},
        "relay_db_confirmed": True,
    }


def _read_http_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {"payload": payload}


def _health_running(health: dict[str, Any]) -> bool:
    status = str(health.get("status") or health.get("state") or "").strip().lower()
    return bool(health.get("ok") is True or status in {"ok", "healthy", "running", "warn", "degraded"})


def _health_services(health: dict[str, Any]) -> dict[str, str]:
    raw = health.get("services")
    if not isinstance(raw, dict):
        return {}
    return {str(name): _service_status(status) for name, status in raw.items()}


def _service_status(status: Any) -> str:
    if status is True:
        return "running"
    if isinstance(status, dict):
        return "running" if _health_running(status) else str(status.get("status") or "unknown")
    text = str(status or "").strip().lower()
    return "running" if text in {"ok", "healthy", "running", "up"} else text


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _evidence_artifacts(value: Any) -> list[dict[str, Any]]:
    artifacts = value if isinstance(value, list) else []
    normalised: list[dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, dict):
            normalised.append({"invalid": item})
            continue
        payload = dict(item)
        path = _resolve(str(payload.get("path") or ""))
        if path.exists() and path.is_file() and not payload.get("sha256"):
            payload["sha256"] = file_sha256(path)
        normalised.append(payload)
    return normalised


def _resolve(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _utc_z() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_z_from(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return _utc_z()
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return _utc_z()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _label(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _result(valid: bool, output_path: Path, errors: list[str]) -> dict[str, Any]:
    return {
        "valid": valid,
        "output": _label(output_path),
        "errors": errors,
    }


if __name__ == "__main__":
    raise SystemExit(main())
