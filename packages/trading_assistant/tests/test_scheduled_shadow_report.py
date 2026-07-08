from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from trading_assistant.schemas.monthly_validation import (
    MonthlyValidationResult,
    MonthlyValidationStatus,
)
from trading_assistant.skills.scheduled_shadow_report import (
    write_scheduled_shadow_cycle_report,
)


def test_scheduled_shadow_report_writes_production_grade_cycle(tmp_path: Path) -> None:
    paths = _evidence_paths(tmp_path)
    result = _monthly_result()

    report_path = write_scheduled_shadow_cycle_report(
        result=result,
        monthly_validation_result_path=paths["monthly_result"],
        deployment_metadata_install_report_paths=[paths["install_report"]],
        operational_evidence_path=paths["operational"],
        relay_ingest_evidence_path=paths["relay"],
        learning_sufficiency_manifest_path=paths["learning"],
        optimizer_run_manifest_path=paths["optimizer"],
        approval_evidence_mode=True,
        adoption_disabled=True,
        output_root=tmp_path / "scheduled_shadow",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["uses_live_vps_metadata"] is True
    assert report["adoption_disabled"] is True
    assert report["bot_id"] == "ibkr"
    assert report["source_kind"] == "monthly_validation_shadow"
    assert report["deployment_metadata_install_report_paths"] == [str(paths["install_report"])]


def test_scheduled_shadow_report_blocks_missing_install_report(tmp_path: Path) -> None:
    paths = _evidence_paths(tmp_path)
    paths["install_report"].unlink()

    report_path = write_scheduled_shadow_cycle_report(
        result=_monthly_result(),
        monthly_validation_result_path=paths["monthly_result"],
        deployment_metadata_install_report_paths=[paths["install_report"]],
        operational_evidence_path=paths["operational"],
        relay_ingest_evidence_path=paths["relay"],
        learning_sufficiency_manifest_path=paths["learning"],
        optimizer_run_manifest_path=paths["optimizer"],
        approval_evidence_mode=True,
        adoption_disabled=True,
        output_root=tmp_path / "scheduled_shadow",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["uses_live_vps_metadata"] is False
    assert any("install report missing or malformed" in item for item in report["blockers"])


def test_scheduled_shadow_report_blocks_placeholder_relay_evidence(tmp_path: Path) -> None:
    paths = _evidence_paths(tmp_path)
    paths["relay"].write_text(
        json.dumps(
            {
                "ok": True,
                "bot_id": "ibkr",
                "event_id": "relay-heartbeat-1",
                "effective_config_hash": "b" * 64,
                "deployment_id": "deployment-1",
                "runtime_instance_id": "ibkr-paper-vps-1",
                "deployment_metadata_hash": "a" * 64,
                "auth": {"secret_fingerprint": "change-me"},
                "freshness": {"ok": True, "max_event_age_seconds": 10},
                "generated_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    report_path = write_scheduled_shadow_cycle_report(
        result=_monthly_result(),
        monthly_validation_result_path=paths["monthly_result"],
        deployment_metadata_install_report_paths=[paths["install_report"]],
        operational_evidence_path=paths["operational"],
        relay_ingest_evidence_path=paths["relay"],
        learning_sufficiency_manifest_path=paths["learning"],
        optimizer_run_manifest_path=paths["optimizer"],
        approval_evidence_mode=True,
        adoption_disabled=True,
        output_root=tmp_path / "scheduled_shadow",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert any("placeholder HMAC secret fingerprint" in item for item in report["blockers"])


def test_scheduled_shadow_report_blocks_malformed_optimizer_manifest(tmp_path: Path) -> None:
    paths = _evidence_paths(tmp_path)
    paths["optimizer"].write_text("{not-json", encoding="utf-8")

    report_path = write_scheduled_shadow_cycle_report(
        result=_monthly_result(),
        monthly_validation_result_path=paths["monthly_result"],
        deployment_metadata_install_report_paths=[paths["install_report"]],
        operational_evidence_path=paths["operational"],
        relay_ingest_evidence_path=paths["relay"],
        learning_sufficiency_manifest_path=paths["learning"],
        optimizer_run_manifest_path=paths["optimizer"],
        approval_evidence_mode=True,
        adoption_disabled=True,
        output_root=tmp_path / "scheduled_shadow",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert any("optimizer manifest missing or malformed" in item for item in report["blockers"])


def _monthly_result() -> MonthlyValidationResult:
    return MonthlyValidationResult(
        run_id="monthly-trading-trading_stock_family-2026-05",
        run_month="2026-05",
        bot_id="trading",
        strategy_id="trading_stock_family",
        status=MonthlyValidationStatus.WATCH,
    )


def _evidence_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "monthly_result": tmp_path / "monthly_validation_result.json",
        "install_report": tmp_path / "deployment_metadata_install_report.json",
        "operational": tmp_path / "operational_evidence.json",
        "relay": tmp_path / "relay_ingest_evidence.json",
        "learning": tmp_path / "learning_sufficiency_manifest.json",
        "optimizer": tmp_path / "optimizer_run_manifest.json",
    }
    paths["monthly_result"].write_text("{}", encoding="utf-8")
    paths["install_report"].write_text(
        json.dumps({"ok": True, "installed": True}),
        encoding="utf-8",
    )
    paths["operational"].write_text("{}", encoding="utf-8")
    paths["relay"].write_text(
        json.dumps(
            {
                "ok": True,
                "bot_id": "ibkr",
                "event_id": "relay-heartbeat-1",
                "effective_config_hash": "b" * 64,
                "deployment_id": "deployment-1",
                "runtime_instance_id": "ibkr-paper-vps-1",
                "deployment_metadata_hash": "a" * 64,
                "auth": {"secret_fingerprint": "hmac-sha256:abcdef1234567890"},
                "freshness": {"ok": True, "max_event_age_seconds": 10},
                "generated_at": datetime.now(UTC).isoformat(),
                "classification_counts": {"enqueued": 1, "duplicate": 0},
            }
        ),
        encoding="utf-8",
    )
    paths["learning"].write_text("{}", encoding="utf-8")
    paths["optimizer"].write_text(
        json.dumps(
            {
                "approval_evidence_mode": True,
                "approval_grade_optimizer_run": True,
                "smoke_mode": False,
            }
        ),
        encoding="utf-8",
    )
    return paths
