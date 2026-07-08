from __future__ import annotations

import json
from pathlib import Path

from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.validation import bridge_promotion


def test_bridge_promotion_dry_run_blocks_ineligible_bundle(tmp_path: Path) -> None:
    contract = _write_contract(tmp_path)
    bundle = _write_bundle(
        tmp_path,
        eligible=False,
        blockers=["production fixture breadth missing"],
        evidence_hashes={str(contract): sha256_file(contract)},
    )

    report = bridge_promotion.promote_bridge(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        approval_evidence_path=bundle,
        artifact_root=tmp_path / "promotion",
    )

    assert report["promoted"] is False
    assert report["promotion_decision"] == "blocked"
    assert "approval_evidence_eligible: production fixture breadth missing" in report["blockers"]
    assert _read_json(contract)["maturity"] == "shadow_validated"
    assert (tmp_path / "promotion" / "pre_promotion_report.json").exists()


def test_bridge_promotion_blocks_stale_evidence_hash(tmp_path: Path) -> None:
    contract = _write_contract(tmp_path)
    bundle = _write_bundle(
        tmp_path,
        eligible=True,
        blockers=[],
        evidence_hashes={str(contract): "0" * 64},
    )

    report = bridge_promotion.promote_bridge(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        approval_evidence_path=bundle,
        artifact_root=tmp_path / "promotion",
        write=True,
    )

    assert report["promoted"] is False
    assert any("evidence hash mismatch" in blocker for blocker in report["blockers"])
    assert _read_json(contract)["maturity"] == "shadow_validated"


def test_bridge_promotion_write_mutates_only_maturity_and_runs_post_validators(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract = _write_contract(tmp_path)
    before = _read_json(contract)
    source = _write_json(tmp_path / "evidence" / "fixture.json", {"status": "pass"})
    bundle = _write_bundle(
        tmp_path,
        eligible=True,
        blockers=[],
        evidence_hashes={
            str(contract): sha256_file(contract),
            str(source): sha256_file(source),
        },
    )
    calls: list[str] = []

    def bridge_stub(**kwargs):
        calls.append("bridge")
        return {"ok": True, "artifact_path": str(tmp_path / "bridge.json")}

    def matrix_stub(**kwargs):
        calls.append("matrix")
        return {"ok": True, "artifact_path": str(tmp_path / "matrix.json")}

    def audit_stub(**kwargs):
        calls.append("audit")
        return {"ok": True, "artifact_path": str(tmp_path / "audit.json")}

    monkeypatch.setattr(bridge_promotion, "run_bridge_readiness_audit", bridge_stub)
    monkeypatch.setattr(bridge_promotion, "run_validation_matrix_audit", matrix_stub)
    monkeypatch.setattr(bridge_promotion, "run_approval_grade_audit", audit_stub)

    report = bridge_promotion.promote_bridge(
        agent_root=tmp_path,
        scope_id="trading_stock_family",
        approval_evidence_path=bundle,
        artifact_root=tmp_path / "promotion",
        write=True,
    )

    after = _read_json(contract)
    before["maturity"] = "approval_ready"
    assert after == before
    assert report["promoted"] is True
    assert report["promotion_decision"] == "promoted"
    assert calls == ["bridge", "matrix", "audit"]
    post = _read_json(tmp_path / "promotion" / "post_promotion_report.json")
    assert post["mutation_scope_ok"] is True
    assert post["post_validation_ok"] is True


def test_bridge_promotion_cli_accepts_plan_gate_aliases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "approval_evidence_bundle.json"
    captured: dict = {}

    def promote_stub(**kwargs):
        captured.update(kwargs)
        return {"promoted": False}

    monkeypatch.setattr(bridge_promotion, "promote_bridge", promote_stub)

    result = bridge_promotion.main(
        [
            "--agent-root",
            str(tmp_path),
            "--bridge-id",
            "trading_stock_family",
            "--approval-evidence-bundle",
            str(bundle),
        ]
    )

    assert result == 0
    assert captured["scope_id"] == "trading_stock_family"
    assert captured["approval_evidence_path"] == bundle


def _write_contract(root: Path) -> Path:
    return _write_json(
        root
        / "contracts"
        / "strategy_plugins"
        / "trading_stock_family"
        / "strategy_plugin_contract.json",
        {
            "plugin_id": "trading-stock-family",
            "live_repo_path": "trading/ibkr_trader",
            "live_repo_commit_sha": "a" * 40,
            "backtest_adapter_path": "adapter.py",
            "backtest_adapter_commit_sha": "b" * 64,
            "config_schema_version": "config_v1",
            "decision_api_version": "decision_v1",
            "required_telemetry_schemas": ["trade_event_v1"],
            "supported_symbols": ["MSFT"],
            "supported_timeframes": ["5m"],
            "parity_fixture_set": ["fixture.json"],
            "maturity": "shadow_validated",
        },
    )


def _write_bundle(
    root: Path,
    *,
    eligible: bool,
    blockers: list[str],
    evidence_hashes: dict[str, str],
) -> Path:
    return _write_json(
        root
        / "artifacts"
        / "validation"
        / "approval_evidence"
        / "trading_stock_family"
        / "approval_evidence_bundle.json",
        {
            "schema_version": "approval_evidence_bundle_v1",
            "scope_id": "trading_stock_family",
            "eligible_for_promotion": eligible,
            "promotion_decision": "eligible" if eligible else "blocked",
            "blockers": blockers,
            "evidence_hashes": evidence_hashes,
            "artifact_path": str(
                root
                / "artifacts"
                / "validation"
                / "approval_evidence"
                / "trading_stock_family"
                / "approval_evidence_bundle.json"
            ),
        },
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
