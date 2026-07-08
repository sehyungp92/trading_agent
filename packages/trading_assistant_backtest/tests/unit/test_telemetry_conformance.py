from __future__ import annotations

import json
from pathlib import Path

from trading_assistant_backtest.validation.telemetry_conformance import (
    run_telemetry_conformance_check,
)


def test_telemetry_conformance_passes_clean_relay_evidence(tmp_path: Path) -> None:
    telemetry, scheduled, relay = _write_inputs(tmp_path)

    report = run_telemetry_conformance_check(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "out",
        telemetry_manifest_path=telemetry,
        scheduled_shadow_report_path=scheduled,
        relay_ingest_evidence_path=relay,
    )

    assert report["ok"] is True
    assert report["runtime_non_crashing"] is True
    assert not report["blockers"]
    assert (tmp_path / "out" / "telemetry_conformance_report.json").exists()


def test_telemetry_conformance_blocks_drops_without_runtime_exception(tmp_path: Path) -> None:
    telemetry, scheduled, relay = _write_inputs(
        tmp_path,
        relay_payload={
            "ok": True,
            "classification_counts": {"enqueued": 1, "quarantined": 1},
            "dropped_field_counts": {"strategy_id": 2},
            "sample_events": [{"event_id": "evt-1", "bot_id": "ibkr"}],
        },
    )

    report = run_telemetry_conformance_check(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "out",
        telemetry_manifest_path=telemetry,
        scheduled_shadow_report_path=scheduled,
        relay_ingest_evidence_path=relay,
    )

    assert report["ok"] is False
    assert report["runtime_non_crashing"] is True
    assert any("quarantined" in blocker for blocker in report["blockers"])
    assert any("dropped_field_counts.strategy_id" in blocker for blocker in report["blockers"])
    assert any("missing required fields" in blocker for blocker in report["blockers"])


def test_telemetry_conformance_blocks_malformed_counts_without_exception(
    tmp_path: Path,
) -> None:
    telemetry, scheduled, relay = _write_inputs(
        tmp_path,
        relay_payload={
            "ok": True,
            "dropped_count": "many",
            "classification_counts": {"quarantined": "one"},
            "sample_events": [{"event_id": "evt-1", "bot_id": "ibkr", "event_type": "order"}],
        },
    )
    telemetry.write_text(
        json.dumps(
            {
                "manifest_version": "telemetry_manifest_v1",
                "total_events": -1,
                "missing_field_counts": {"event_id": "one"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = run_telemetry_conformance_check(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "out",
        telemetry_manifest_path=telemetry,
        scheduled_shadow_report_path=scheduled,
        relay_ingest_evidence_path=relay,
    )

    assert report["ok"] is False
    assert report["runtime_non_crashing"] is True
    assert any("total_events cannot be negative" in blocker for blocker in report["blockers"])
    assert any("must be an integer count" in blocker for blocker in report["blockers"])


def test_telemetry_conformance_omits_placeholder_paths_when_inputs_are_missing(
    tmp_path: Path,
) -> None:
    report = run_telemetry_conformance_check(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        artifact_root=tmp_path / "out",
    )

    assert report["ok"] is False
    assert report["telemetry_manifest_path"] == ""
    assert report["relay_ingest_evidence_path"] == ""
    assert report["checks"][0]["evidence_paths"] == []


def _write_inputs(
    root: Path,
    *,
    relay_payload: dict | None = None,
) -> tuple[Path, Path, Path]:
    telemetry = _write_json(
        root / "telemetry_manifest.json",
        {
            "manifest_version": "telemetry_manifest_v1",
            "total_events": 1,
            "missing_field_counts": {"event_id": 0, "bot_id": 0, "event_type": 0},
        },
    )
    relay = _write_json(
        root / "relay_ingest_evidence.json",
        relay_payload
        or {
            "ok": True,
            "classification_counts": {"enqueued": 1, "duplicate": 0},
            "sample_events": [
                {
                    "event_id": "evt-1",
                    "bot_id": "ibkr",
                    "event_type": "order",
                    "payload": {},
                }
            ],
        },
    )
    scheduled = _write_json(
        root / "scheduled_shadow_cycle_report.json",
        {
            "schema_version": "scheduled_shadow_cycle_report_v1",
            "ok": True,
            "relay_ingest_evidence_path": str(relay),
        },
    )
    return telemetry, scheduled, relay


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
