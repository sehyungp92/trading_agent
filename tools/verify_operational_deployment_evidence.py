from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from migration_support import ROOT, file_sha256, read_json

CONTRACTS_SRC = ROOT / "packages" / "trading_contracts" / "src"
if str(CONTRACTS_SRC) not in sys.path:
    sys.path.insert(0, str(CONTRACTS_SRC))

from trading_contracts.relay_evidence import validate_relay_ingest_evidence  # noqa: E402


BOT_BRIDGES = {
    "ibkr": {
        "trading_swing_family",
        "trading_momentum_family",
        "trading_stock_family",
    },
    "crypto": {
        "crypto_trend_v1",
        "crypto_momentum_v1",
        "crypto_breakout_v1",
    },
    "k_stock": {"k_stock_olr_kalcb"},
}
RELAY_BOT_IDS = {
    "ibkr": {"ibkr", "swing_multi_01", "stock_trader", "momentum_nq_01"},
    "crypto": {"crypto", "paper_bot_01"},
    "k_stock": {"k_stock", "k_stock_trader"},
}


def main() -> int:
    args = _parser().parse_args()
    evidence_path = ROOT / args.evidence
    plan_path = ROOT / args.plan
    errors = _evidence_errors(evidence_path, plan_path)
    result = {
        "valid": not errors,
        "evidence": args.evidence,
        "plan": args.plan,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify operational VPS deployment evidence.")
    parser.add_argument("--evidence", default="deployments/operational_evidence.json")
    parser.add_argument("--plan", default="deployments/cutover_plan.json")
    return parser


def _evidence_errors(evidence_path: Path, plan_path: Path) -> list[str]:
    if not evidence_path.exists():
        return [
            f"missing operational evidence {evidence_path.relative_to(ROOT).as_posix()}",
            "collect VPS and local assistant evidence after deployment before marking completion",
        ]
    if not plan_path.exists():
        return [f"missing cutover plan {plan_path.relative_to(ROOT).as_posix()}"]

    evidence = read_json(evidence_path)
    plan = read_json(plan_path)
    errors: list[str] = []
    if evidence.get("schema_version") != "trading_agent_operational_evidence_v1":
        errors.append("operational evidence schema_version must be trading_agent_operational_evidence_v1")
    reviewed_commit = str(evidence.get("reviewed_commit_sha") or "")
    if not _is_full_hash(reviewed_commit):
        errors.append("operational evidence reviewed_commit_sha must be a full git object id")
    if not _is_utc_z(str(evidence.get("generated_at_utc") or "")):
        errors.append("operational evidence generated_at_utc must be UTC ending in Z")

    plan_records = {
        str(record.get("bot") or ""): record
        for record in plan.get("records", [])
        if isinstance(record, dict)
    }
    records = {
        str(record.get("bot") or ""): record
        for record in evidence.get("records", [])
        if isinstance(record, dict)
    }
    for bot in sorted(BOT_BRIDGES):
        record = records.get(bot)
        if not isinstance(record, dict):
            errors.append(f"{bot}: missing operational evidence record")
            continue
        errors.extend(_record_errors(bot, record, reviewed_commit, plan_records.get(bot, {})))
    extra = sorted(set(records) - set(BOT_BRIDGES))
    if extra:
        errors.append(f"unexpected operational evidence bot record(s): {extra}")
    return errors


def _record_errors(
    bot: str,
    record: dict[str, Any],
    reviewed_commit: str,
    plan_record: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    running_commit = str(record.get("running_commit_sha") or "")
    if running_commit != reviewed_commit:
        errors.append(f"{bot}: running_commit_sha must match reviewed_commit_sha")
    metadata_refs = _deployment_metadata_refs(record.get("deployment_metadata"))
    errors.extend(_vps_deployment_errors(bot, record.get("vps_deployment"), plan_record))
    sidecar_forwarding = record.get("sidecar_forwarding")
    errors.extend(_boolean_section_errors(bot, "sidecar_forwarding", sidecar_forwarding))
    if bot == "crypto":
        errors.extend(_crypto_sidecar_policy_errors(sidecar_forwarding))
    errors.extend(_assistant_ingest_errors(bot, record.get("assistant_ingest"), metadata_refs))
    errors.extend(_metadata_errors(bot, record.get("deployment_metadata")))
    errors.extend(_monthly_shadow_errors(bot, record.get("monthly_shadow")))
    errors.extend(_rollback_smoke_errors(bot, record.get("rollback_smoke")))
    errors.extend(_evidence_artifact_errors(bot, record.get("evidence_artifacts")))
    return errors


def _vps_deployment_errors(bot: str, value: Any, plan_record: dict[str, Any]) -> list[str]:
    if not isinstance(value, dict):
        return [f"{bot}: missing vps_deployment"]
    errors: list[str] = []
    if not str(value.get("host_id") or "").strip():
        errors.append(f"{bot}: vps_deployment.host_id missing")
    if value.get("running") is not True:
        errors.append(f"{bot}: vps_deployment.running must be true")
    compose_file = str(value.get("compose_file") or "")
    if compose_file != str(plan_record.get("compose_file") or ""):
        errors.append(f"{bot}: vps_deployment.compose_file must match cutover plan")
    compose_hash = str(value.get("compose_sha256") or "")
    if compose_file:
        path = ROOT / compose_file
        if not path.exists():
            errors.append(f"{bot}: compose_file does not exist: {compose_file}")
        elif compose_hash != file_sha256(path):
            errors.append(f"{bot}: vps_deployment.compose_sha256 mismatch")
    services = value.get("services")
    if not isinstance(services, dict) or not services:
        errors.append(f"{bot}: vps_deployment.services missing")
    else:
        for service_name, service_status in services.items():
            if service_status != "running":
                errors.append(f"{bot}: service {service_name} must be running")
    return errors


def _boolean_section_errors(bot: str, section: str, value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [f"{bot}: missing {section}"]
    errors: list[str] = []
    if value.get("ok") is not True:
        errors.append(f"{bot}: {section}.ok must be true")
    if not _is_utc_z(str(value.get("checked_at_utc") or "")):
        errors.append(f"{bot}: {section}.checked_at_utc must be UTC ending in Z")
    if not str(value.get("evidence_ref") or "").strip():
        errors.append(f"{bot}: {section}.evidence_ref missing")
    return errors


def _crypto_sidecar_policy_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    policy = value.get("runtime_policy") or value.get("sidecar_runtime_policy")
    if not isinstance(policy, dict):
        return ["crypto: sidecar_forwarding.runtime_policy missing"]
    errors: list[str] = []
    if not isinstance(policy.get("thresholds"), dict):
        errors.append("crypto: sidecar_forwarding.runtime_policy.thresholds missing")
    standdown_required = policy.get("standdown_required") is True or policy.get("ok") is False
    incident_action = str(policy.get("incident_action") or "").strip()
    open_position_action = str(policy.get("open_position_action") or "").strip()
    if standdown_required:
        if incident_action != "cancel_working_entry_orders":
            errors.append(
                "crypto: sidecar_forwarding.runtime_policy.incident_action must be cancel_working_entry_orders"
            )
        if open_position_action != "hold_existing_positions":
            errors.append(
                "crypto: sidecar_forwarding.runtime_policy.open_position_action must be hold_existing_positions"
            )
    else:
        if not incident_action:
            errors.append("crypto: sidecar_forwarding.runtime_policy.incident_action missing")
        if not open_position_action:
            errors.append("crypto: sidecar_forwarding.runtime_policy.open_position_action missing")
    return errors


def _assistant_ingest_errors(
    bot: str,
    value: Any,
    metadata_refs: dict[str, set[str]],
) -> list[str]:
    errors = _boolean_section_errors(bot, "assistant_ingest", value)
    if not isinstance(value, dict):
        return errors
    if int(value.get("events_ingested") or 0) <= 0:
        errors.append(f"{bot}: assistant_ingest.events_ingested must be positive")
    relay_items, relay_errors = _relay_ingest_evidence_items(bot, value)
    errors.extend(relay_errors)
    if relay_items and not all(metadata_refs.values()):
        errors.append(f"{bot}: deployment metadata refs unavailable for relay evidence validation")
    for index, evidence in enumerate(relay_items):
        label = "relay_ingest_evidence" if len(relay_items) == 1 else f"relay_ingest_evidence[{index}]"
        if not isinstance(evidence, dict):
            errors.append(f"{bot}: assistant_ingest.{label} must be an object")
            continue
        expected_bot = _expected_relay_bot_id(bot, evidence)
        errors.extend(
            f"{bot}: assistant_ingest.{label}: {error}"
            for error in validate_relay_ingest_evidence(
                evidence,
                expected_bot_id=expected_bot,
                deployment_ids=metadata_refs["deployment_ids"],
                runtime_instance_ids=metadata_refs["runtime_instance_ids"],
                deployment_metadata_hashes=metadata_refs["hashes"],
            )
        )
    return errors


def _metadata_errors(bot: str, value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [f"{bot}: missing deployment_metadata"]
    errors: list[str] = []
    if value.get("ok") is not True:
        errors.append(f"{bot}: deployment_metadata.ok must be true")
    bridge_ids = set(str(item) for item in value.get("bridge_ids", []))
    missing = sorted(BOT_BRIDGES[bot] - bridge_ids)
    extra = sorted(bridge_ids - BOT_BRIDGES[bot])
    if missing:
        errors.append(f"{bot}: deployment_metadata.bridge_ids missing {missing}")
    if extra:
        errors.append(f"{bot}: deployment_metadata.bridge_ids contains unexpected {extra}")
    reports = value.get("install_report_paths")
    if not isinstance(reports, list) or not reports:
        errors.append(f"{bot}: deployment_metadata.install_report_paths missing")
    return errors


def _monthly_shadow_errors(bot: str, value: Any) -> list[str]:
    errors = _boolean_section_errors(bot, "monthly_shadow", value)
    if isinstance(value, dict) and value.get("uses_real_metadata") is not True:
        errors.append(f"{bot}: monthly_shadow.uses_real_metadata must be true")
    return errors


def _rollback_smoke_errors(bot: str, value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [f"{bot}: missing rollback_smoke"]
    errors: list[str] = []
    if value.get("returncode") != 0:
        errors.append(f"{bot}: rollback_smoke.returncode must be 0")
    if value.get("side_effect_scope") != "no_live_orders":
        errors.append(f"{bot}: rollback_smoke.side_effect_scope must be no_live_orders")
    if not _is_utc_z(str(value.get("executed_at_utc") or "")):
        errors.append(f"{bot}: rollback_smoke.executed_at_utc must be UTC ending in Z")
    command = value.get("command")
    if not isinstance(command, list) or not command:
        errors.append(f"{bot}: rollback_smoke.command missing")
    elif "config" in {str(part) for part in command} and "run" not in {str(part) for part in command}:
        errors.append(f"{bot}: rollback_smoke.command cannot be compose config only")
    return errors


def _evidence_artifact_errors(bot: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return [f"{bot}: evidence_artifacts missing"]
    errors: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{bot}: evidence_artifacts[{index}] must be an object")
            continue
        raw_path = str(item.get("path") or "")
        expected = str(item.get("sha256") or "")
        if not raw_path:
            errors.append(f"{bot}: evidence_artifacts[{index}].path missing")
            continue
        path = ROOT / raw_path
        if not path.exists() or not path.is_file():
            errors.append(f"{bot}: evidence artifact missing: {raw_path}")
            continue
        if expected != file_sha256(path):
            errors.append(f"{bot}: evidence artifact hash mismatch: {raw_path}")
    return errors


def _relay_ingest_evidence_items(
    bot: str,
    value: dict[str, Any],
) -> tuple[list[Any], list[str]]:
    items: list[Any] = []
    errors: list[str] = []
    raw = value.get("relay_ingest_evidence")
    if isinstance(raw, list):
        items.extend(raw)
    elif raw is not None:
        items.append(raw)

    raw_paths: list[Any] = []
    if value.get("relay_ingest_evidence_path"):
        raw_paths.append(value.get("relay_ingest_evidence_path"))
    paths = value.get("relay_ingest_evidence_paths")
    if isinstance(paths, list):
        raw_paths.extend(paths)
    for raw_path in raw_paths:
        path = _resolve_evidence_path(str(raw_path or ""))
        if path is None or not path.exists() or not path.is_file():
            errors.append(f"{bot}: assistant_ingest relay evidence path missing: {raw_path}")
            continue
        try:
            payload = read_json(path)
        except Exception as exc:
            errors.append(f"{bot}: assistant_ingest relay evidence path malformed: {raw_path}: {exc}")
            continue
        if isinstance(payload, list):
            items.extend(payload)
        else:
            items.append(payload)
    if not items:
        errors.append(f"{bot}: assistant_ingest.relay_ingest_evidence missing")
    return items, errors


def _deployment_metadata_refs(value: Any) -> dict[str, set[str]]:
    refs = {"deployment_ids": set(), "runtime_instance_ids": set(), "hashes": set()}
    if not isinstance(value, dict):
        return refs
    reports = value.get("install_report_paths")
    if not isinstance(reports, list):
        return refs
    for raw_report in reports:
        report_path = _resolve_evidence_path(str(raw_report or ""))
        if report_path is None or not report_path.exists() or not report_path.is_file():
            continue
        try:
            report = read_json(report_path)
        except Exception:
            continue
        for key in ("metadata_path", "installed_path"):
            metadata_path = _resolve_evidence_path(str(report.get(key) or ""))
            if metadata_path is None or not metadata_path.exists() or not metadata_path.is_file():
                continue
            try:
                metadata = read_json(metadata_path)
            except Exception:
                continue
            if metadata.get("deployment_id"):
                refs["deployment_ids"].add(str(metadata["deployment_id"]))
            if metadata.get("runtime_instance_id"):
                refs["runtime_instance_ids"].add(str(metadata["runtime_instance_id"]))
            refs["hashes"].add(hashlib.sha256(metadata_path.read_bytes()).hexdigest())
    return refs


def _resolve_evidence_path(raw_path: str) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def _expected_relay_bot_id(bot: str, evidence: dict[str, Any]) -> str:
    evidence_bot = str(evidence.get("bot_id") or "").strip()
    return evidence_bot if evidence_bot in RELAY_BOT_IDS.get(bot, {bot}) else bot


def _is_full_hash(value: str) -> bool:
    return len(value) in {40, 64} and all(char in "0123456789abcdefABCDEF" for char in value)


def _is_utc_z(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00").astimezone(UTC)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
