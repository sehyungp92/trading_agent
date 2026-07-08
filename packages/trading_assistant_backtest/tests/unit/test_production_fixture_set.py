from __future__ import annotations

import json
from pathlib import Path

from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.validation.approval_evidence_spine import (
    _production_fixture_breadth_check,
)
from trading_assistant_backtest.validation.production_fixture_set import (
    REQUIRED_CASE_CLASSES,
    build_production_fixture_set_manifest,
)


def test_production_fixture_set_passes_with_hashed_live_shadow_source(tmp_path: Path) -> None:
    _write_contract(tmp_path)
    fixture = _write_fixture(tmp_path)
    parity_report = _write_parity_report(tmp_path, fixture)
    parity_summary = _write_json(
        tmp_path
        / "artifacts"
        / "validation"
        / "decision_parity_matrix"
        / "trading_stock_family"
        / "decision_parity"
        / "decision_parity_validation_summary.json",
        {"decision_parity_status": "pass"},
    )
    _write_context_artifacts(tmp_path, telemetry_identity=("ibkr", "trading_stock_family"))

    manifest = build_production_fixture_set_manifest(
        agent_root=tmp_path,
        bridge_id="trading_stock_family",
        parity_report_path=parity_report,
        parity_summary_path=parity_summary,
    )

    assert manifest["ok"] is True
    assert manifest["status"] == "pass"
    assert manifest["missing_case_classes"] == []
    observed = set(manifest["case_classes"])
    for aliases in REQUIRED_CASE_CLASSES.values():
        assert observed.intersection(aliases)
    fixture_record = next(
        record
        for record in manifest["source_records"]
        if record["path"].endswith("stock_family_collision.json")
    )
    assert fixture_record["sha256"] == sha256_file(fixture)
    check = _production_fixture_breadth_check(
        tmp_path
        / "artifacts"
        / "validation"
        / "decision_parity_matrix"
        / "trading_stock_family"
        / "production_fixture_set_manifest.json",
        tmp_path,
    )
    assert check["passed"] is True


def test_production_fixture_set_blocks_generic_runtime_identity(tmp_path: Path) -> None:
    _write_contract(tmp_path)
    fixture = _write_fixture(tmp_path)
    parity_report = _write_parity_report(tmp_path, fixture)
    parity_summary = _write_json(
        tmp_path
        / "artifacts"
        / "validation"
        / "decision_parity_matrix"
        / "trading_stock_family"
        / "decision_parity"
        / "decision_parity_validation_summary.json",
        {"decision_parity_status": "pass"},
    )
    _write_context_artifacts(tmp_path, telemetry_identity=("bot1", "strat1"))

    manifest = build_production_fixture_set_manifest(
        agent_root=tmp_path,
        bridge_id="trading_stock_family",
        parity_report_path=parity_report,
        parity_summary_path=parity_summary,
    )

    assert manifest["ok"] is False
    assert manifest["status"] == "blocked"
    assert "live_shadow_telemetry_source" in manifest["missing_case_classes"]
    assert any("does not match bridge identity" in blocker for blocker in manifest["blockers"])


def test_production_fixture_set_uses_latest_phase2_manifest_by_default(
    tmp_path: Path,
) -> None:
    _write_contract(tmp_path)
    fixture = _write_fixture(tmp_path)
    parity_report = _write_parity_report(tmp_path, fixture)
    parity_summary = _write_json(
        tmp_path
        / "artifacts"
        / "validation"
        / "decision_parity_matrix"
        / "trading_stock_family"
        / "decision_parity"
        / "decision_parity_validation_summary.json",
        {"decision_parity_status": "pass"},
    )
    _write_context_artifacts(
        tmp_path,
        telemetry_identity=("bot1", "strat1"),
        run_month="2026-06",
    )
    _write_context_artifacts(
        tmp_path,
        telemetry_identity=("ibkr", "trading_stock_family"),
        run_month="2026-07",
    )

    manifest = build_production_fixture_set_manifest(
        agent_root=tmp_path,
        bridge_id="trading_stock_family",
        parity_report_path=parity_report,
        parity_summary_path=parity_summary,
    )

    assert manifest["ok"] is True
    telemetry_records = [
        record
        for record in manifest["source_records"]
        if record["source_kind"] == "telemetry_manifest"
    ]
    assert telemetry_records
    assert "2026-07" in telemetry_records[0]["path"]


def test_production_fixture_set_blocks_malformed_telemetry_count(
    tmp_path: Path,
) -> None:
    _write_contract(tmp_path)
    fixture = _write_fixture(tmp_path)
    parity_report = _write_parity_report(tmp_path, fixture)
    parity_summary = _write_json(
        tmp_path
        / "artifacts"
        / "validation"
        / "decision_parity_matrix"
        / "trading_stock_family"
        / "decision_parity"
        / "decision_parity_validation_summary.json",
        {"decision_parity_status": "pass"},
    )
    _write_context_artifacts(tmp_path, telemetry_identity=("ibkr", "trading_stock_family"))
    telemetry = (
        tmp_path
        / "artifacts"
        / "learning_sufficiency"
        / "phase2_manifests"
        / "ibkr"
        / "2026-06"
        / "trading_stock_family"
        / "telemetry_manifest.json"
    )
    payload = json.loads(telemetry.read_text(encoding="utf-8"))
    payload["total_events"] = "many"
    telemetry.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest = build_production_fixture_set_manifest(
        agent_root=tmp_path,
        bridge_id="trading_stock_family",
        parity_report_path=parity_report,
        parity_summary_path=parity_summary,
    )

    assert manifest["ok"] is False
    assert any("must be an integer count" in blocker for blocker in manifest["blockers"])


def test_approval_spine_blocks_blocked_fixture_manifest(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "source.json", {"kind": "source"})
    manifest = _write_json(
        tmp_path
        / "artifacts"
        / "validation"
        / "decision_parity_matrix"
        / "trading_stock_family"
        / "production_fixture_set_manifest.json",
        {
            "schema_version": "production_fixture_set_manifest_v1",
            "status": "blocked",
            "ok": False,
            "source_kind": "production_derived_live_shadow",
            "case_classes": [
                "accepted_entry",
                "blocked_no_trade",
                "risk_portfolio_denial",
                "exit_close",
                "order_fill",
                "live_shadow_telemetry_source",
            ],
            "source_records": [{"path": str(source), "sha256": sha256_file(source)}],
        },
    )

    check = _production_fixture_breadth_check(manifest, tmp_path)

    assert check["passed"] is False
    assert "production fixture-set manifest is not ok" in check["errors"]
    assert "production fixture-set manifest status is blocked" in check["errors"]


def test_approval_spine_does_not_count_required_fixture_classes_as_observed(
    tmp_path: Path,
) -> None:
    source = _write_json(tmp_path / "source.json", {"kind": "source"})
    manifest = _write_json(
        tmp_path
        / "artifacts"
        / "validation"
        / "decision_parity_matrix"
        / "trading_stock_family"
        / "production_fixture_set_manifest.json",
        {
            "schema_version": "production_fixture_set_manifest_v1",
            "status": "pass",
            "ok": True,
            "source_kind": "production_derived_live_shadow",
            "required_case_classes": {
                "accepted_entry": ["accepted_entry"],
                "blocked_no_trade": ["blocked_no_trade"],
                "risk_portfolio_denial": ["risk_portfolio_denial"],
                "exit_close": ["exit_close"],
                "order_fill_or_explicit_non_fill": ["order_fill"],
                "live_shadow_telemetry_source": ["live_shadow_telemetry_source"],
            },
            "case_classes": [
                "accepted_entry",
                "blocked_no_trade",
                "risk_portfolio_denial",
                "exit_close",
                "order_fill",
            ],
            "source_records": [{"path": str(source), "sha256": sha256_file(source)}],
        },
    )

    check = _production_fixture_breadth_check(manifest, tmp_path)

    assert check["passed"] is False
    assert (
        "fixture-set manifest missing required case class: live_shadow_telemetry_source"
        in check["errors"]
    )


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


def _write_fixture(root: Path) -> Path:
    return _write_json(
        root / "fixtures" / "stock_family_collision.json",
        {
            "schema_version": 2,
            "artifacts": {"alcb": {"idle_market_input": {"reason": "fixture_child_idle"}}},
            "family_config": {
                "portfolio_rules": {"symbol_collision_action": "half_size"}
            },
            "broker_event_script": [
                {
                    "order_match": {
                        "strategy_id": "IARIC_v1",
                        "symbol": "MSFT",
                        "role": "ENTRY",
                        "side": "BUY",
                    },
                    "event": "fill",
                }
            ],
        },
    )


def _write_parity_report(root: Path, fixture: Path) -> Path:
    return _write_json(
        root
        / "artifacts"
        / "validation"
        / "decision_parity_matrix"
        / "trading_stock_family"
        / "decision_parity"
        / "decision_parity_report.json",
        {
            "status": "pass",
            "strategy_plugin_id": "trading-stock-family",
            "evidence_paths": [str(fixture)],
            "checks": [
                {"dimension": "entries", "status": "pass", "evidence_paths": [str(fixture)]},
                {"dimension": "exits", "status": "pass", "evidence_paths": [str(fixture)]},
            ],
        },
    )


def _write_context_artifacts(
    root: Path,
    *,
    telemetry_identity: tuple[str, str],
    run_month: str = "2026-06",
) -> None:
    _write_json(
        root
        / "artifacts"
        / "learning_sufficiency"
        / "ptg7_pilot"
        / "production_derived_fixture_window.json",
        {"schema_version": "ptg7_production_derived_fixture_window_v1", "status": "pass"},
    )
    manifest_dir = (
        root
        / "artifacts"
        / "learning_sufficiency"
        / "phase2_manifests"
        / "ibkr"
        / run_month
        / "trading_stock_family"
    )
    _write_json(
        manifest_dir / "learning_sufficiency_manifest.json",
        {
            "strategy_id": "trading_stock_family",
            "telemetry_authoritative_eligibility": "learning_authoritative",
        },
    )
    bot_id, strategy_id = telemetry_identity
    _write_json(
        manifest_dir / "telemetry_manifest.json",
        {
            "bot_id": bot_id,
            "strategy_id": strategy_id,
            "total_events": 1,
            "authoritative_eligibility": "learning_authoritative",
        },
    )
    _write_json(
        manifest_dir / "runtime_evidence_support.json",
        {"schema_version": "runtime_evidence_support_v1", "status": "pass"},
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
