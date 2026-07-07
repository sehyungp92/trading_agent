from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from migration_support import ROOT, file_sha256, read_json


PLACEHOLDER_TOKENS = ("previous-production", "placeholder", "todo", "example")
ROLLBACK_OPERATIONAL_MARKERS = {
    "ibkr": ("ibkr-trading-runtime", "preflight"),
    "crypto": ("crypto-trader", "status"),
    "k_stock": ("k-stock-preflight",),
}


def main() -> int:
    args = _parser().parse_args()
    path = ROOT / args.plan
    errors = _plan_errors(path)
    result = {"valid": not errors, "plan": args.plan, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify per-bot cutover and rollback evidence.")
    parser.add_argument("--plan", default="deployments/cutover_plan.json")
    return parser


def _plan_errors(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing cutover plan {path.relative_to(ROOT).as_posix()}"]
    plan = read_json(path)
    records = plan.get("records")
    if plan.get("schema_version") != "trading_agent_cutover_plan_v2":
        return ["cutover plan schema_version must be trading_agent_cutover_plan_v2"]
    if not isinstance(records, list) or not records:
        return ["cutover plan records must be a non-empty list"]
    errors: list[str] = []
    for record in records:
        errors.extend(_record_errors(record if isinstance(record, dict) else {}))
    return errors


def _record_errors(record: dict[str, Any]) -> list[str]:
    bot = str(record.get("bot") or "<missing>")
    errors: list[str] = []
    for field in ("candidate_image", "compose_file", "compose_sha256", "live_config", "live_config_hash", "first_mode"):
        if not str(record.get(field) or "").strip():
            errors.append(f"{bot}: missing {field}")
    errors.extend(_placeholder_errors(bot, record))
    errors.extend(_hash_errors(bot, record, "compose_file", "compose_sha256"))
    errors.extend(_hash_errors(bot, record, "live_config", "live_config_hash"))
    previous = record.get("previous_state")
    rollback = record.get("rollback")
    if not isinstance(previous, dict):
        errors.append(f"{bot}: missing previous_state")
        previous = {}
    if not isinstance(rollback, dict):
        errors.append(f"{bot}: missing rollback")
        rollback = {}
    errors.extend(_previous_state_errors(bot, previous, record))
    errors.extend(_rollback_errors(bot, rollback, previous))
    errors.extend(_startup_evidence_errors(bot, record.get("startup_evidence")))
    return errors


def _previous_state_errors(bot: str, previous: dict[str, Any], record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("image", "compose_file", "compose_sha256", "live_config_hashes"):
        if not previous.get(field):
            errors.append(f"{bot}: previous_state missing {field}")
    errors.extend(_hash_errors(bot, previous, "compose_file", "compose_sha256", prefix="previous_state"))
    live_hashes = previous.get("live_config_hashes")
    if not isinstance(live_hashes, dict) or record.get("live_config") not in live_hashes:
        errors.append(f"{bot}: previous_state live_config_hashes missing current live config")
    elif live_hashes[record["live_config"]] != record.get("live_config_hash"):
        errors.append(f"{bot}: previous_state live config hash does not match current live config")
    return errors


def _rollback_errors(bot: str, rollback: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("image", "compose_file", "compose_sha256", "live_config_hashes", "restore_command", "restore_test"):
        if not rollback.get(field):
            errors.append(f"{bot}: rollback missing {field}")
    if rollback.get("image") != previous.get("image"):
        errors.append(f"{bot}: rollback image must match recorded previous_state image")
    if rollback.get("live_config_hashes") != previous.get("live_config_hashes"):
        errors.append(f"{bot}: rollback live_config_hashes must match previous_state")
    errors.extend(_hash_errors(bot, rollback, "compose_file", "compose_sha256", prefix="rollback"))
    restore_test = rollback.get("restore_test")
    if isinstance(restore_test, dict):
        errors.extend(_restore_test_errors(bot, restore_test))
    return errors


def _restore_test_errors(bot: str, restore_test: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if restore_test.get("kind") != "operational_restore_smoke":
        errors.append(f"{bot}: rollback restore_test kind must be operational_restore_smoke")
    if restore_test.get("returncode") != 0:
        errors.append(f"{bot}: rollback restore_test did not pass")
    command = restore_test.get("command")
    if not isinstance(command, list) or not command:
        errors.append(f"{bot}: rollback restore_test missing command")
        command = []
    joined = " ".join(str(part) for part in command)
    if _is_compose_config_only(command):
        errors.append(f"{bot}: rollback restore_test must run a bot restore smoke, not only compose config")
    if " run " not in f" {joined} " and " exec " not in f" {joined} ":
        errors.append(f"{bot}: rollback restore_test must run or exec a service command")
    markers = ROLLBACK_OPERATIONAL_MARKERS.get(bot, ())
    missing_markers = [marker for marker in markers if marker not in joined]
    if missing_markers:
        errors.append(f"{bot}: rollback restore_test command missing operational marker(s): {missing_markers}")

    evidence = restore_test.get("evidence")
    if not isinstance(evidence, dict):
        return [*errors, f"{bot}: rollback restore_test missing evidence"]
    if evidence.get("side_effect_scope") != "no_live_orders":
        errors.append(f"{bot}: rollback restore_test evidence side_effect_scope must be no_live_orders")
    executed_at = str(evidence.get("executed_at_utc") or "")
    if not _is_utc_z(executed_at):
        errors.append(f"{bot}: rollback restore_test evidence executed_at_utc must be UTC ending in Z")
    assertions = evidence.get("assertions")
    if not isinstance(assertions, list) or len([item for item in assertions if str(item or "").strip()]) < 3:
        errors.append(f"{bot}: rollback restore_test evidence must include at least three assertions")
    return errors


def _is_compose_config_only(command: list[Any]) -> bool:
    parts = [str(part) for part in command]
    return "compose" in parts and "config" in parts and "run" not in parts and "exec" not in parts


def _is_utc_z(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00").astimezone(UTC)
    except ValueError:
        return False
    return True


def _startup_evidence_errors(bot: str, evidence: Any) -> list[str]:
    if not isinstance(evidence, dict):
        return [f"{bot}: missing startup_evidence"]
    errors: list[str] = []
    for field in ("dependency_report", "runtime_metadata_root", "image_build_status"):
        if not str(evidence.get(field) or "").strip():
            errors.append(f"{bot}: startup_evidence missing {field}")
    for field in ("dependency_report", "runtime_metadata_root"):
        raw = evidence.get(field)
        if raw and not (ROOT / str(raw)).exists():
            errors.append(f"{bot}: startup_evidence path does not exist: {raw}")
    if evidence.get("image_build_status") != "pass":
        errors.append(f"{bot}: startup_evidence image_build_status must be pass")
    return errors


def _hash_errors(
    bot: str,
    record: dict[str, Any],
    path_field: str,
    hash_field: str,
    *,
    prefix: str = "",
) -> list[str]:
    raw_path = record.get(path_field)
    expected = str(record.get(hash_field) or "")
    label = f"{prefix}.{path_field}" if prefix else path_field
    if not raw_path:
        return []
    path = ROOT / str(raw_path)
    if not path.exists():
        return [f"{bot}: {label} does not exist: {raw_path}"]
    actual = file_sha256(path)
    if actual != expected:
        return [f"{bot}: {hash_field} mismatch for {raw_path}"]
    return []


def _placeholder_errors(bot: str, payload: Any) -> list[str]:
    text = json.dumps(payload, sort_keys=True).lower()
    return [
        f"{bot}: cutover plan contains placeholder token {token}"
        for token in PLACEHOLDER_TOKENS
        if token in text
    ]


if __name__ == "__main__":
    raise SystemExit(main())
