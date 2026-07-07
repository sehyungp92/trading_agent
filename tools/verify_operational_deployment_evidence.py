from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from migration_support import ROOT, file_sha256, read_json


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
    errors.extend(_vps_deployment_errors(bot, record.get("vps_deployment"), plan_record))
    errors.extend(_boolean_section_errors(bot, "sidecar_forwarding", record.get("sidecar_forwarding")))
    errors.extend(_assistant_ingest_errors(bot, record.get("assistant_ingest")))
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


def _assistant_ingest_errors(bot: str, value: Any) -> list[str]:
    errors = _boolean_section_errors(bot, "assistant_ingest", value)
    if isinstance(value, dict) and int(value.get("events_ingested") or 0) <= 0:
        errors.append(f"{bot}: assistant_ingest.events_ingested must be positive")
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
