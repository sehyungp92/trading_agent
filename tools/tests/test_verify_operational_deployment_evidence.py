from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import verify_operational_deployment_evidence as verifier  # noqa: E402


NOW = "2026-07-08T12:00:00Z"
COMMIT = "a" * 40


def test_operational_evidence_verifier_accepts_strong_relay_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence_path, plan_path, _payload = _write_bundle(tmp_path, monkeypatch)

    assert verifier._evidence_errors(evidence_path, plan_path) == []


def test_operational_evidence_verifier_rejects_weak_relay_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence_path, plan_path, payload = _write_bundle(tmp_path, monkeypatch)
    relay = payload["records"][0]["assistant_ingest"]["relay_ingest_evidence"]
    missing_hash = dict(relay)
    missing_hash["event_id"] = "relay-heartbeat-missing-metadata-hash"
    missing_hash.pop("deployment_metadata_hash")
    relay.update(
        {
            "bot_id": "wrong_bot",
            "deployment_id": "unlinked-deployment",
            "runtime_instance_id": "unlinked-runtime",
            "deployment_metadata_hash": "0" * 64,
            "auth": {"secret_fingerprint": "change-me"},
            "freshness": {"ok": True, "max_event_age_seconds": 999999999},
        }
    )
    payload["records"][0]["assistant_ingest"]["relay_ingest_evidence"] = [relay, missing_hash]
    _write_json(evidence_path, payload)

    errors = verifier._evidence_errors(evidence_path, plan_path)

    assert any("bot_id 'wrong_bot' does not match 'ibkr'" in error for error in errors)
    assert any("placeholder HMAC secret fingerprint" in error for error in errors)
    assert any("deployment_id is not linked" in error for error in errors)
    assert any("runtime_instance_id is not linked" in error for error in errors)
    assert any("deployment metadata hash mismatch" in error for error in errors)
    assert any("missing deployment metadata hash" in error for error in errors)
    assert any("max_event_age_seconds is stale" in error for error in errors)


def test_operational_evidence_verifier_rejects_missing_crypto_sidecar_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence_path, plan_path, payload = _write_bundle(tmp_path, monkeypatch)
    crypto = next(record for record in payload["records"] if record["bot"] == "crypto")
    crypto["sidecar_forwarding"].pop("runtime_policy")
    _write_json(evidence_path, payload)

    errors = verifier._evidence_errors(evidence_path, plan_path)

    assert "crypto: sidecar_forwarding.runtime_policy missing" in errors


def test_operational_evidence_verifier_rejects_unsafe_crypto_incident_action(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence_path, plan_path, payload = _write_bundle(tmp_path, monkeypatch)
    crypto = next(record for record in payload["records"] if record["bot"] == "crypto")
    crypto["sidecar_forwarding"]["runtime_policy"] = {
        "ok": False,
        "standdown_required": True,
        "incident_action": "block_entries_only",
        "open_position_action": "hold_existing_positions",
        "thresholds": {"consecutive_send_failures": 3},
    }
    _write_json(evidence_path, payload)

    errors = verifier._evidence_errors(evidence_path, plan_path)

    assert any("incident_action must be cancel_working_entry_orders" in error for error in errors)


def _write_bundle(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict[str, Any]]:
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    plan_records = []
    evidence_records = []
    for bot in ("ibkr", "crypto", "k_stock"):
        compose = deployments / f"{bot}.compose.yml"
        compose.write_text(f"services:\n  {bot}: {{}}\n", encoding="utf-8")
        plan_records.append(
            {"bot": bot, "compose_file": _rel(compose, tmp_path), "first_mode": "paper"}
        )
        evidence_records.append(_record(tmp_path, bot, compose))

    plan_path = deployments / "cutover_plan.json"
    evidence_path = deployments / "operational_evidence.json"
    _write_json(plan_path, {"schema_version": "trading_agent_cutover_plan_v2", "records": plan_records})
    payload = {
        "schema_version": "trading_agent_operational_evidence_v1",
        "reviewed_commit_sha": COMMIT,
        "generated_at_utc": NOW,
        "records": evidence_records,
    }
    _write_json(evidence_path, payload)
    return evidence_path, plan_path, payload


def _record(root: Path, bot: str, compose: Path) -> dict[str, Any]:
    metadata = root / "metadata" / bot / "deployment_metadata.json"
    deployment_id = f"{bot}-deployment"
    runtime_instance_id = f"{bot}-runtime"
    _write_json(
        metadata,
        {"deployment_id": deployment_id, "runtime_instance_id": runtime_instance_id},
    )
    install_report = root / "reports" / bot / "deployment_metadata_install_report.json"
    _write_json(
        install_report,
        {
            "metadata_path": _rel(metadata, root),
            "installed_path": _rel(metadata, root),
            "ok": True,
            "installed": True,
        },
    )
    artifact = root / "artifacts" / bot / "evidence.json"
    _write_json(artifact, {"ok": True})
    return {
        "bot": bot,
        "running_commit_sha": COMMIT,
        "vps_deployment": {
            "host_id": f"{bot}-vps",
            "running": True,
            "compose_file": _rel(compose, root),
            "compose_sha256": verifier.file_sha256(compose),
            "services": {f"{bot}-service": "running"},
        },
        "sidecar_forwarding": (
            {**_ok_section(), "runtime_policy": _crypto_sidecar_runtime_policy()}
            if bot == "crypto"
            else _ok_section()
        ),
        "assistant_ingest": {
            **_ok_section(),
            "events_ingested": 1,
            "relay_ingest_evidence": _relay_evidence(bot, deployment_id, runtime_instance_id, metadata),
        },
        "deployment_metadata": {
            "ok": True,
            "bridge_ids": sorted(verifier.BOT_BRIDGES[bot]),
            "install_report_paths": [_rel(install_report, root)],
        },
        "monthly_shadow": {**_ok_section(), "uses_real_metadata": True},
        "rollback_smoke": {
            "returncode": 0,
            "side_effect_scope": "no_live_orders",
            "executed_at_utc": NOW,
            "command": ["docker", "compose", "run", f"{bot}-preflight"],
        },
        "evidence_artifacts": [{"path": _rel(artifact, root), "sha256": verifier.file_sha256(artifact)}],
    }


def _relay_evidence(
    bot: str,
    deployment_id: str,
    runtime_instance_id: str,
    metadata: Path,
) -> dict[str, Any]:
    bot_ids = {"ibkr": "ibkr", "crypto": "paper_bot_01", "k_stock": "k_stock_trader"}
    return {
        "ok": True,
        "bot_id": bot_ids[bot],
        "event_id": f"relay-heartbeat-{bot}",
        "effective_config_hash": "b" * 64,
        "deployment_id": deployment_id,
        "runtime_instance_id": runtime_instance_id,
        "deployment_metadata_hash": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "auth": {"secret_fingerprint": "hmac-sha256:abcdef1234567890"},
        "freshness": {"ok": True, "max_event_age_seconds": 60},
        "generated_at": NOW,
    }


def _ok_section() -> dict[str, Any]:
    return {"ok": True, "checked_at_utc": NOW, "evidence_ref": "unit-test"}


def _crypto_sidecar_runtime_policy() -> dict[str, Any]:
    return {
        "ok": True,
        "standdown_required": False,
        "standdown_reason": "",
        "incident_action": "none",
        "open_position_action": "none",
        "thresholds": {
            "consecutive_send_failures": 3,
            "buffered_event_count": 100,
            "oldest_buffered_event_age_seconds": 300,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
