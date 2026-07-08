from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import collect_operational_evidence as collector  # noqa: E402
import verify_operational_deployment_evidence as verifier  # noqa: E402


NOW = "2026-07-08T12:00:00Z"
COMMIT = "a" * 40


def test_collect_operational_evidence_writes_verifier_conformant_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, plan_path = _write_collection_bundle(tmp_path, monkeypatch)
    output_path = tmp_path / "deployments" / "operational_evidence.json"

    result = collector.collect_operational_evidence(
        manifest_path=manifest_path,
        output_path=output_path,
        plan_path=plan_path,
        reviewed_commit=COMMIT,
    )

    assert result == {"valid": True, "output": "deployments/operational_evidence.json", "errors": []}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    first = payload["records"][0]
    assert first["vps_deployment"]["compose_sha256"]
    assert first["vps_deployment"]["health_checked_at_utc"].endswith("Z")
    assert first["evidence_artifacts"][0]["sha256"]
    assert first["assistant_ingest"]["relay_ingest_evidence"]["deployment_metadata_hashes"]
    assert verifier._evidence_errors(output_path, plan_path) == []


def test_collect_operational_evidence_refuses_placeholder_relay_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, plan_path = _write_collection_bundle(tmp_path, monkeypatch)
    relay_path = tmp_path / "relay" / "crypto.json"
    relay = json.loads(relay_path.read_text(encoding="utf-8"))
    relay["auth"] = {"secret_fingerprint": "change-me"}
    relay_path.write_text(json.dumps(relay), encoding="utf-8")
    output_path = tmp_path / "deployments" / "operational_evidence.json"

    result = collector.collect_operational_evidence(
        manifest_path=manifest_path,
        output_path=output_path,
        plan_path=plan_path,
        reviewed_commit=COMMIT,
    )

    assert result["valid"] is False
    assert not output_path.exists()
    assert any("placeholder HMAC secret fingerprint" in error for error in result["errors"])


def test_collect_operational_evidence_confirms_relay_event_from_assistant_db(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, plan_path = _write_collection_bundle(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crypto = next(record for record in manifest["records"] if record["bot"] == "crypto")
    relay_path = tmp_path / crypto["assistant_ingest"].pop("relay_ingest_evidence_path")
    relay = json.loads(relay_path.read_text(encoding="utf-8"))
    db_path = tmp_path / "relay" / "assistant.db"
    _write_relay_db_event(db_path, relay)
    crypto["assistant_ingest"]["relay_db_path"] = _rel(db_path, tmp_path)
    crypto["assistant_ingest"]["relay_ingest_evidence"] = {
        "ok": True,
        "event_id": relay["event_id"],
        "auth": relay["auth"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    output_path = tmp_path / "deployments" / "operational_evidence.json"

    result = collector.collect_operational_evidence(
        manifest_path=manifest_path,
        output_path=output_path,
        plan_path=plan_path,
        reviewed_commit=COMMIT,
    )

    assert result["valid"] is True
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    crypto_out = next(record for record in payload["records"] if record["bot"] == "crypto")
    relay_out = crypto_out["assistant_ingest"]["relay_ingest_evidence"]
    assert relay_out["relay_db_confirmed"] is True
    assert relay_out["deployment_id"] == relay["deployment_id"]
    assert relay_out["runtime_instance_id"] == relay["runtime_instance_id"]


def test_collect_operational_evidence_fails_when_relay_db_event_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, plan_path = _write_collection_bundle(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crypto = next(record for record in manifest["records"] if record["bot"] == "crypto")
    relay_path = tmp_path / crypto["assistant_ingest"].pop("relay_ingest_evidence_path")
    relay = json.loads(relay_path.read_text(encoding="utf-8"))
    db_path = tmp_path / "relay" / "assistant.db"
    _create_relay_db(db_path)
    crypto["assistant_ingest"]["relay_db_path"] = _rel(db_path, tmp_path)
    crypto["assistant_ingest"]["relay_ingest_evidence"] = {
        "ok": True,
        "event_id": relay["event_id"],
        "auth": relay["auth"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    output_path = tmp_path / "deployments" / "operational_evidence.json"

    result = collector.collect_operational_evidence(
        manifest_path=manifest_path,
        output_path=output_path,
        plan_path=plan_path,
        reviewed_commit=COMMIT,
    )

    assert result["valid"] is False
    assert not output_path.exists()
    assert any("relay DB missing event_id" in error for error in result["errors"])


def _write_collection_bundle(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(
        collector,
        "_read_http_json",
        lambda _url: {"status": "ok", "services": {"runtime": "running"}},
    )
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    plan_records = []
    collection_records = []
    for bot in ("ibkr", "crypto", "k_stock"):
        compose = deployments / f"{bot}.compose.yml"
        compose.write_text(f"services:\n  {bot}: {{}}\n", encoding="utf-8")
        plan_records.append(
            {"bot": bot, "compose_file": _rel(compose, tmp_path), "first_mode": "paper"}
        )
        collection_records.append(_collection_record(tmp_path, bot, compose))

    plan_path = deployments / "cutover_plan.json"
    manifest_path = deployments / "operational_evidence.collection.json"
    _write_json(plan_path, {"schema_version": "trading_agent_cutover_plan_v2", "records": plan_records})
    _write_json(
        manifest_path,
        {
            "schema_version": collector.COLLECTION_SCHEMA,
            "reviewed_commit_sha": COMMIT,
            "records": collection_records,
        },
    )
    return manifest_path, plan_path


def _collection_record(root: Path, bot: str, compose: Path) -> dict[str, Any]:
    metadata = root / "metadata" / bot / "deployment_metadata.json"
    deployment_id = f"{bot}-deployment"
    runtime_instance_id = f"{bot}-runtime"
    _write_json(metadata, {"deployment_id": deployment_id, "runtime_instance_id": runtime_instance_id})
    install_report = root / "reports" / bot / "deployment_metadata_install_report.json"
    _write_json(
        install_report,
        {"metadata_path": _rel(metadata, root), "installed_path": _rel(metadata, root), "ok": True, "installed": True},
    )
    relay = root / "relay" / f"{bot}.json"
    _write_json(relay, _relay_evidence(bot, deployment_id, runtime_instance_id))
    artifact = root / "artifacts" / bot / "evidence.json"
    _write_json(artifact, {"ok": True})
    return {
        "bot": bot,
        "vps_deployment": {
            "host_id": f"{bot}-vps",
            "running": True,
            "compose_file": _rel(compose, root),
            "health_url": f"https://health.invalid/{bot}/health",
            "services": {f"{bot}-service": "running"},
        },
        "sidecar_forwarding": (
            {**_ok_section(), "runtime_policy": _crypto_sidecar_runtime_policy()}
            if bot == "crypto"
            else _ok_section()
        ),
        "assistant_ingest": {
            **_ok_section(),
            "relay_ingest_evidence_path": _rel(relay, root),
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
        "evidence_artifacts": [{"path": _rel(artifact, root)}],
    }


def _relay_evidence(bot: str, deployment_id: str, runtime_instance_id: str) -> dict[str, Any]:
    bot_ids = {"ibkr": "ibkr", "crypto": "paper_bot_01", "k_stock": "k_stock_trader"}
    return {
        "ok": True,
        "bot_id": bot_ids[bot],
        "event_id": f"relay-heartbeat-{bot}",
        "effective_config_hash": "b" * 64,
        "deployment_id": deployment_id,
        "runtime_instance_id": runtime_instance_id,
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
            "buffered_event_count": 500,
            "oldest_buffered_event_age_seconds": 900,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _create_relay_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                bot_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                exchange_timestamp TEXT NOT NULL,
                received_at TEXT NOT NULL,
                acked INTEGER DEFAULT 0,
                priority INTEGER DEFAULT 3
            );
            """
        )


def _write_relay_db_event(path: Path, relay: dict[str, Any]) -> None:
    _create_relay_db(path)
    stored_payload = {
        "bot_id": relay["bot_id"],
        "event_type": "deployment_start",
        "runtime_instance_id": relay["runtime_instance_id"],
        "effective_config_hash": relay["effective_config_hash"],
        "deployment_id": relay["deployment_id"],
        "source": "unit-test",
    }
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO events
                (event_id, bot_id, event_type, payload, exchange_timestamp, received_at, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relay["event_id"],
                relay["bot_id"],
                "deployment_start",
                json.dumps(stored_payload, sort_keys=True),
                NOW,
                NOW,
                0,
            ),
        )


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
