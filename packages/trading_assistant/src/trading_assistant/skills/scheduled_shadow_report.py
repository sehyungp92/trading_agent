"""Production scheduled-shadow cycle report writer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_contracts.relay_evidence import validate_relay_ingest_evidence

from trading_assistant.paths import monorepo_root
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult

SCHEDULED_SHADOW_SCHEMA_VERSION = "scheduled_shadow_cycle_report_v1"

SCOPE_BRIDGES = {
    "trading_stock_family": ["trading_stock_family"],
    "trading_momentum_family": ["trading_momentum_family"],
    "trading_swing_family": ["trading_swing_family"],
    "k_stock_olr_kalcb": ["k_stock_olr_kalcb"],
    "crypto_trader_portfolio": [
        "crypto_trend_v1",
        "crypto_momentum_v1",
        "crypto_breakout_v1",
    ],
}

SCOPE_BOTS = {
    "trading_stock_family": "ibkr",
    "trading_momentum_family": "ibkr",
    "trading_swing_family": "ibkr",
    "k_stock_olr_kalcb": "k_stock",
    "crypto_trader_portfolio": "crypto",
}

BOT_ALIASES = {
    "trading": "ibkr",
    "ibkr": "ibkr",
    "crypto": "crypto",
    "crypto_trader": "crypto",
    "k_stock": "k_stock",
    "k_stock_trader": "k_stock",
}


def write_scheduled_shadow_cycle_report(
    *,
    result: MonthlyValidationResult,
    monthly_validation_result_path: Path,
    deployment_metadata_install_report_paths: list[Path],
    operational_evidence_path: Path,
    relay_ingest_evidence_path: Path | None,
    learning_sufficiency_manifest_path: Path,
    optimizer_run_manifest_path: Path,
    approval_evidence_mode: bool,
    adoption_disabled: bool,
    scope_id: str = "",
    bridge_ids: list[str] | None = None,
    bot_id: str = "",
    vps_host_id: str = "",
    assistant_host_id: str = "local",
    output_root: Path | None = None,
) -> Path:
    """Write a scheduled shadow report consumed by approval evidence.

    The report is intentionally emitted even when incomplete so downstream
    approval checks can show concrete blockers instead of a missing file.
    """

    resolved_scope = scope_id or result.strategy_id
    resolved_bridge_ids = bridge_ids or SCOPE_BRIDGES.get(resolved_scope, [resolved_scope])
    resolved_bot_id = bot_id or SCOPE_BOTS.get(resolved_scope) or BOT_ALIASES.get(
        result.bot_id,
        result.bot_id,
    )
    root = output_root or (
        monorepo_root() / "artifacts" / "validation" / "scheduled_shadow"
    )
    report_dir = Path(root) / resolved_scope / result.run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "scheduled_shadow_cycle_report.json"

    blockers: list[str] = []
    if not approval_evidence_mode:
        blockers.append("approval_evidence_mode is not enabled")
    if not adoption_disabled:
        blockers.append("adoption must remain disabled for scheduled shadow evidence")

    _require_existing("monthly_validation_result_path", monthly_validation_result_path, blockers)
    _require_existing("operational_evidence_path", operational_evidence_path, blockers)
    _require_existing(
        "learning_sufficiency_manifest_path",
        learning_sufficiency_manifest_path,
        blockers,
    )
    _require_existing("optimizer_run_manifest_path", optimizer_run_manifest_path, blockers)
    if relay_ingest_evidence_path is None:
        blockers.append("relay_ingest_evidence_path missing")
    else:
        _require_existing("relay_ingest_evidence_path", relay_ingest_evidence_path, blockers)

    metadata_ok = _deployment_metadata_reports_ok(
        deployment_metadata_install_report_paths,
        blockers,
    )
    metadata_refs = _deployment_metadata_refs(deployment_metadata_install_report_paths)
    if relay_ingest_evidence_path is not None and Path(relay_ingest_evidence_path).exists():
        _relay_ingest_evidence_ok(
            relay_ingest_evidence_path,
            bot_id=resolved_bot_id,
            metadata_refs=metadata_refs,
            blockers=blockers,
        )
    _optimizer_manifest_ok(optimizer_run_manifest_path, blockers)

    payload = {
        "schema_version": SCHEDULED_SHADOW_SCHEMA_VERSION,
        "scope_id": resolved_scope,
        "bridge_ids": resolved_bridge_ids,
        "run_id": result.run_id,
        "run_month": result.run_month,
        "bot_id": resolved_bot_id,
        "vps_host_id": vps_host_id,
        "assistant_host_id": assistant_host_id,
        "monthly_validation_result_path": str(monthly_validation_result_path),
        "deployment_metadata_install_report_paths": [
            str(path) for path in deployment_metadata_install_report_paths
        ],
        "operational_evidence_path": str(operational_evidence_path),
        "relay_ingest_evidence_path": str(relay_ingest_evidence_path or ""),
        "learning_sufficiency_manifest_path": str(learning_sufficiency_manifest_path),
        "optimizer_run_manifest_path": str(optimizer_run_manifest_path),
        "approval_evidence_mode": bool(approval_evidence_mode),
        "uses_live_vps_metadata": metadata_ok,
        "adoption_disabled": bool(adoption_disabled),
        "source_kind": "monthly_validation_shadow",
        "ok": not blockers,
        "blockers": blockers,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00",
            "Z",
        ),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _require_existing(label: str, path: Path, blockers: list[str]) -> None:
    if not Path(path).exists():
        blockers.append(f"{label} missing: {path}")


def _deployment_metadata_reports_ok(paths: list[Path], blockers: list[str]) -> bool:
    if not paths:
        blockers.append("deployment_metadata_install_report_paths missing")
        return False
    ok = True
    for path in paths:
        report = _read_json(path)
        if not report:
            blockers.append(f"deployment metadata install report missing or malformed: {path}")
            ok = False
            continue
        if report.get("ok") is not True or report.get("installed") is not True:
            blockers.append(f"deployment metadata install report did not install cleanly: {path}")
            ok = False
    return ok


def _optimizer_manifest_ok(path: Path, blockers: list[str]) -> None:
    manifest = _read_json(path)
    if not manifest:
        blockers.append(f"optimizer manifest missing or malformed: {path}")
        return
    if manifest.get("approval_evidence_mode") is not True:
        blockers.append("optimizer manifest approval_evidence_mode is not true")
    if manifest.get("approval_grade_optimizer_run") is not True:
        blockers.append("optimizer manifest is not approval-grade")
    if manifest.get("smoke_mode") is not False:
        blockers.append("optimizer manifest is smoke mode")


def _deployment_metadata_refs(paths: list[Path]) -> dict[str, set[str]]:
    refs = {"deployment_ids": set(), "runtime_instance_ids": set(), "hashes": set()}
    for path in paths:
        report = _read_json(path)
        for key in ("metadata_path", "installed_path"):
            raw_path = str(report.get(key) or "").strip()
            if not raw_path:
                continue
            metadata_path = Path(raw_path)
            if not metadata_path.exists() or not metadata_path.is_file():
                continue
            metadata = _read_json(metadata_path)
            if metadata.get("deployment_id"):
                refs["deployment_ids"].add(str(metadata["deployment_id"]))
            if metadata.get("runtime_instance_id"):
                refs["runtime_instance_ids"].add(str(metadata["runtime_instance_id"]))
            refs["hashes"].add(_sha256_file(metadata_path))
    return refs


def _relay_ingest_evidence_ok(
    path: Path,
    *,
    bot_id: str,
    metadata_refs: dict[str, set[str]],
    blockers: list[str],
) -> None:
    evidence = _read_json(path)
    if not evidence:
        blockers.append(f"relay ingest evidence missing or malformed: {path}")
        return
    blockers.extend(
        validate_relay_ingest_evidence(
            evidence,
            expected_bot_id=bot_id,
            deployment_ids=metadata_refs["deployment_ids"],
            runtime_instance_ids=metadata_refs["runtime_instance_ids"],
            deployment_metadata_hashes=metadata_refs["hashes"],
        )
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
