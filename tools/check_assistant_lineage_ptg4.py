from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from learning_sufficiency_gate_utils import checklist_completion_check


ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    ROOT / "packages" / "trading_assistant" / "src",
    ROOT / "packages" / "trading_contracts" / "src",
    ROOT / "packages" / "trading_instrumentation" / "src",
):
    if source_root.exists() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from trading_instrumentation.approval_metadata import live_deployment_metadata_errors  # noqa: E402

from trading_assistant.schemas.monthly_candidates import MonthlyImprovementCandidate  # noqa: E402
from trading_assistant.schemas.monthly_outcome import MonthlyOutcomeRecord  # noqa: E402
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest  # noqa: E402
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus  # noqa: E402
from trading_assistant.schemas.performance_learning_ledger import (  # noqa: E402
    AuthorityLevel,
    DecisionStage,
    LearningLayer,
    PerformanceLearningRecord,
    PerformanceMetricDeltas,
    PerformanceRecordType,
    SourceCadence,
)
from trading_assistant.skills.performance_learning_ledger import validate_performance_learning_records  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts" / "learning_sufficiency" / "ptg4_gate_report.json"
DEFAULT_INDEX = ROOT / "artifacts" / "learning_sufficiency" / "phase2_manifests" / "manifest_index.json"
DEFAULT_SOURCE_INVENTORY = ROOT / "artifacts" / "learning_sufficiency" / "assistant_lineage_source_inventory.json"
DEFAULT_RUNTIME_EVIDENCE_ROOT = ROOT / "artifacts" / "learning_sufficiency" / "ptg4_runtime_lineage_evidence"
CONTRACT_SCHEMA = ROOT / "contracts" / "schemas" / "monthly_run_manifest.schema.json"
RUNTIME_EVENT_CLASSES = ("trade", "missed_opportunity", "order", "fill", "portfolio_rule")
ASSISTANT_TRACE_KEYS = (
    "weekly_signal_ids",
    "source_weekly_signal_ids",
    "monthly_search_brief_id",
    "suggestion_id",
    "suggestion_ids",
    "proposal_id",
    "proposal_ids",
    "source_proposal_ids",
    "candidate_id",
    "candidate_ids",
    "hypothesis_id",
    "hypothesis_ids",
    "experiment_id",
    "strategy_change_record_id",
    "strategy_change_record_ids",
    "monthly_outcome_id",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PTG-4 assistant lineage propagation gate.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--source-inventory", default=str(DEFAULT_SOURCE_INVENTORY))
    parser.add_argument("--runtime-evidence-root", default=str(DEFAULT_RUNTIME_EVIDENCE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    source_inventory = Path(args.source_inventory)
    runtime_evidence_manifest = _write_runtime_lineage_evidence(
        Path(args.runtime_evidence_root),
        source_inventory,
    )
    checks = [
        checklist_completion_check(["Phase 6"]),
        _check_assistant_lineage_source_inventory(source_inventory),
        _check_monthly_schema_derivation(),
        _check_candidate_outcome_result_derivation(),
        _check_deployment_metadata_fail_closed(),
        _check_performance_ledger_trace_validation(),
        _check_contract_schema(),
        _check_bot_side_fixture_lineage(),
        _check_production_like_runtime_lineage_evidence(runtime_evidence_manifest),
        _check_runtime_event_lineage(Path(args.index)),
    ]
    failures = [check for check in checks if not check["passed"]]
    report = {
        "schema_version": "assistant_lineage_ptg4_gate_report_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "gate": "PTG-4",
        "required_acceptance_rows": ["AM-12", "AM-13", "AM-20", "AM-25"],
        "required_finite_checklist_sections": ["Phase 6"],
        "status": "pass" if not failures else "blocked",
        "scope_note": (
            "PTG-4 validates assistant-lineage propagation mechanics. Active manifests "
            "with zero observed core runtime event paths remain diagnostics-only for "
            "learning sufficiency."
        ),
        "promotion_criteria": (
            "Proposal lineage is traceable through monthly artifacts, runtime metadata, "
            "runtime events, outcome measurement, and performance-learning validation."
        ),
        "checks": checks,
        "failures": failures,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": report["status"] == "pass",
        "gate": report["gate"],
        "status": report["status"],
        "artifact_path": _rel(output_path),
    }, indent=2))
    return 0 if report["status"] == "pass" else 1


def _check_monthly_schema_derivation() -> dict[str, Any]:
    manifest = MonthlyRunManifest(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        deployment_id="dep1",
        parameter_set_id="ps1",
        proposal_ids=["proposal-1"],
        suggestion_ids=["suggestion-1"],
        latest_month_start="2026-05-01",
        latest_month_end="2026-05-31",
        market_data_manifest_path="market.json",
        telemetry_manifest_path="telemetry.json",
        artifact_root="artifacts/monthly",
        monthly_search_brief_id="brief-1",
        source_weekly_signal_ids=["weekly-1"],
    )
    passed = (
        manifest.assistant_lineage.proposal_ids == ["proposal-1"]
        and manifest.assistant_lineage.suggestion_ids == ["suggestion-1"]
        and manifest.assistant_lineage.weekly_signal_ids == ["weekly-1"]
        and manifest.assistant_lineage.deployment_id == "dep1"
    )
    return _check("monthly_run_manifest_assistant_lineage", passed, manifest.assistant_lineage.model_dump())


def _check_candidate_outcome_result_derivation() -> dict[str, Any]:
    candidate = MonthlyImprovementCandidate.from_raw({
        "candidate_id": "candidate-1",
        "proposal_id": "proposal-2",
        "suggestion_id": "suggestion-2",
        "source_weekly_signal_ids": ["weekly-2"],
    })
    outcome = MonthlyOutcomeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        proposal_ids=["proposal-3"],
        deployment_id="dep3",
        strategy_change_record_id="change-1",
    )
    result = MonthlyValidationResult(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        status=MonthlyValidationStatus.EXPERIMENT,
        proposal_ids=["proposal-4"],
        strategy_change_record_id="change-2",
    )
    passed = (
        candidate.assistant_lineage.proposal_ids == ["proposal-2"]
        and outcome.assistant_lineage.monthly_outcome_id == outcome.outcome_id
        and result.assistant_lineage.strategy_change_record_ids == ["change-2"]
    )
    return _check("candidate_outcome_result_assistant_lineage", passed, {
        "candidate": candidate.assistant_lineage.model_dump(),
        "outcome": outcome.assistant_lineage.model_dump(),
        "result": result.assistant_lineage.model_dump(),
    })


def _check_deployment_metadata_fail_closed() -> dict[str, Any]:
    metadata = {
        "metadata_source": "vps_live_bot_runtime_deployment_metadata_v1",
        "emission_environment": "paper_vps",
        "emission_context": "paper_vps_startup",
        "emitted_at_utc": "2026-06-01T00:00:00Z",
        "live_runtime_started_at_utc": "2026-06-01T00:00:00Z",
        "runtime_entrypoint": "bot.main",
        "runtime_instance_id": "bot:strategy:sha",
        "runtime_host_fingerprint": "host",
        "source_control_origin": "https://github.com/example/repo",
        "repo_url": "https://github.com/example/repo",
        "source_control_commit_sha": "abc123",
        "deployed_commit_sha": "abc123",
        "source_control_worktree_clean": True,
        "dry_run": False,
        "deployment_id": "dep1",
        "assistant_driven": True,
    }
    missing_errors = live_deployment_metadata_errors(metadata)
    complete_errors = live_deployment_metadata_errors({
        **metadata,
        "assistant_driven": False,
        "assistant_lineage": {"deployment_id": "dep1", "proposal_ids": ["proposal-1"]},
    })
    passed = any("assistant_lineage" in message for message in missing_errors) and not complete_errors
    return _check("assistant_driven_deployment_metadata_fail_closed", passed, {
        "missing_errors": missing_errors,
        "complete_errors": complete_errors,
    })


def _check_performance_ledger_trace_validation() -> dict[str, Any]:
    record = PerformanceLearningRecord(
        record_type=PerformanceRecordType.STRATEGY,
        scope="strat1",
        source_cadence=SourceCadence.MONTHLY,
        learning_layer=LearningLayer.TRADING_AUTHORITY,
        authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
        decision_stage=DecisionStage.MEASURED,
        material_approval_evidence=True,
        realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.01),
    )
    messages = validate_performance_learning_records([record])
    passed = any(message.startswith("AM-20") for message in messages)
    return _check("performance_ledger_measured_trace_required", passed, {"messages": messages})


def _check_contract_schema() -> dict[str, Any]:
    payload = json.loads(CONTRACT_SCHEMA.read_text(encoding="utf-8"))
    properties = payload.get("properties", {})
    passed = "assistant_lineage" in properties
    return _check("monthly_run_manifest_contract_schema", passed, {
        "schema_path": _rel(CONTRACT_SCHEMA),
        "assistant_lineage_property": properties.get("assistant_lineage", {}),
    })


def _check_assistant_lineage_source_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _check("assistant_lineage_source_inventory", False, {
            "inventory_path": _rel(path),
            "error": "assistant lineage source inventory is missing",
        })
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = payload.get("required_id_locations", {})
    required_keys = ("proposal_ids", "source_weekly_signal_ids", "candidate_ids", "strategy_change_record_ids")
    missing = [key for key in required_keys if not required.get(key, {}).get("found")]
    deployment_rows = payload.get("source_categories", {}).get("deployment_metadata", {}).get("paths", [])
    metadata_without_lineage = [
        row.get("path", "")
        for row in deployment_rows
        if isinstance(row, dict) and not row.get("assistant_lineage_present")
    ]
    seed = payload.get("runtime_lineage_seed") if isinstance(payload.get("runtime_lineage_seed"), dict) else {}
    seed_missing = [
        key
        for key in ("bot_id", "strategy_id", "deployment_id", "proposal_ids", "strategy_change_record_ids")
        if not _has_value(seed.get(key))
    ]
    passed = (
        not missing
        and not seed_missing
        and bool(deployment_rows)
        and not metadata_without_lineage
        and payload.get("status") == "pass"
    )
    return _check("assistant_lineage_source_inventory", passed, {
        "inventory_path": _rel(path),
        "canonical_runtime_lineage_source": payload.get("canonical_runtime_lineage_source", {}),
        "missing_id_locations": missing,
        "runtime_metadata_path_count": len(deployment_rows),
        "runtime_metadata_without_assistant_lineage": metadata_without_lineage,
        "seed_missing_fields": seed_missing,
        "runtime_lineage_seed_source_path": seed.get("source_path", ""),
    })


def _write_runtime_lineage_evidence(evidence_root: Path, inventory_path: Path) -> Path:
    evidence_root.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence_root / "runtime_lineage_evidence_manifest.json"
    inventory = _read_json(inventory_path)
    seed = inventory.get("runtime_lineage_seed") if isinstance(inventory.get("runtime_lineage_seed"), dict) else {}
    manifest: dict[str, Any] = {
        "schema_version": "ptg4_runtime_lineage_evidence_manifest_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_inventory_path": _rel(inventory_path),
        "source_seed_path": seed.get("source_path", ""),
        "bot_id": seed.get("bot_id", ""),
        "strategy_id": seed.get("strategy_id", ""),
        "source_ids": {
            "proposal_ids": seed.get("proposal_ids", []),
            "source_weekly_signal_ids": seed.get("source_weekly_signal_ids", []),
            "candidate_ids": seed.get("candidate_ids", []),
            "strategy_change_record_ids": seed.get("strategy_change_record_ids", []),
        },
        "event_paths": {},
    }
    if not seed:
        manifest["generation_status"] = "blocked"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_path
    bot_root = ROOT / "trading" / "ibkr_trader"
    if str(bot_root) not in sys.path:
        sys.path.insert(0, str(bot_root))
    try:
        from libs.instrumentation.lineage import LineageContext  # type: ignore
        from libs.oms.services.factory import (  # type: ignore
            _make_portfolio_rule_logger,
            _make_reconciliation_lifecycle_writer,
        )
        from strategies.stock.instrumentation.src.market_snapshot import MarketSnapshot  # type: ignore
        from strategies.stock.instrumentation.src.missed_opportunity import MissedOpportunityLogger  # type: ignore
        from strategies.stock.instrumentation.src.order_logger import OrderLogger  # type: ignore
        from strategies.stock.instrumentation.src.trade_logger import TradeLogger  # type: ignore
    except Exception as exc:  # pragma: no cover - reported by manifest validation
        manifest["generation_status"] = "blocked"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_path

    class _SnapshotService:
        def capture_now(self, symbol: str) -> Any:
            return MarketSnapshot(
                snapshot_id=f"ptg4-{symbol}",
                symbol=symbol,
                timestamp="2026-06-21T14:30:00+00:00",
                bid=99.95,
                ask=100.05,
                mid=100.0,
                spread_bps=10.0,
                last_trade_price=100.0,
                volume_24h=1_000_000,
                atr_14=1.5,
            )

    lineage = LineageContext(
        bot_id=str(seed.get("bot_id") or ""),
        strategy_id=str(seed.get("strategy_id") or ""),
        family_id="stock",
        portfolio_id="paper",
        account_alias="paper_ibkr",
        strategy_version="ptg4.runtime_emitter",
        config_version="ptg4_cfg_v1",
        portfolio_config_version="ptg4_pcfg_v1",
        risk_config_version="ptg4_risk_v1",
        allocation_version="ptg4_alloc_v1",
        strategy_registry_version="ptg4_registry_v1",
        deployment_id=str(seed.get("deployment_id") or ""),
        parameter_set_id="ptg4_param_v1",
        code_sha="ptg4",
        trace_id="ptg4_trace",
        proposal_ids=tuple(str(item) for item in seed.get("proposal_ids", []) if str(item or "")),
        source_weekly_signal_ids=tuple(str(item) for item in seed.get("source_weekly_signal_ids", []) if str(item or "")),
        strategy_change_record_ids=tuple(str(item) for item in seed.get("strategy_change_record_ids", []) if str(item or "")),
        candidate_ids=tuple(str(item) for item in seed.get("candidate_ids", []) if str(item or "")),
    )
    emitter_root = evidence_root / "bot_runtime_emitters"
    _clear_emitter_outputs(emitter_root)
    config = {
        "bot_id": lineage.bot_id,
        "strategy_id": lineage.strategy_id,
        "data_dir": str(emitter_root),
        "data_source_id": "ptg4_runtime_lineage_evidence",
        "lineage": lineage,
    }
    try:
        snapshot_service = _SnapshotService()
        trade_logger = TradeLogger(config, snapshot_service, strategy_type="stock")
        trade_logger.log_entry(
            trade_id="ptg4-trade",
            pair="AAPL",
            side="LONG",
            entry_price=100.0,
            position_size=10,
            position_size_quote=1_000,
            entry_signal="breakout",
            entry_signal_id="ptg4-signal",
            entry_signal_strength=0.8,
            active_filters=["quality"],
            passed_filters=["quality"],
            strategy_params={"stop0": 98.0},
        )
        trade_logger.log_exit("ptg4-trade", exit_price=101.0, exit_reason="TARGET")

        MissedOpportunityLogger(config, snapshot_service).log_missed(
            pair="AAPL",
            side="LONG",
            signal="breakout",
            signal_id="ptg4-missed-signal",
            signal_strength=0.75,
            blocked_by="portfolio_rule",
        )
        OrderLogger(config, strategy_type="stock").log_order(
            order_id="ptg4-order",
            pair="AAPL",
            side="LONG",
            order_type="LIMIT",
            status="FILLED",
            requested_qty=10,
            filled_qty=10,
            requested_price=100.0,
            fill_price=100.05,
        )
        _make_reconciliation_lifecycle_writer(str(emitter_root), lineage=lambda: lineage)({
            "lifecycle_action": "inferred_fill",
            "status": "observed",
            "phase": "fill_reconciliation",
            "source": "broker_fill",
            "details": {"strategy_id": lineage.strategy_id, "fill_id": "ptg4-fill", "order_id": "ptg4-order"},
        })
        _make_portfolio_rule_logger(
            data_dir=str(emitter_root),
            family_id="stock",
            lineage=lambda: lineage,
        )({
            "rule": "directional_cap",
            "approved": True,
            "strategy_id": lineage.strategy_id,
            "direction": "LONG",
            "symbol": "AAPL",
        })
        manifest["event_paths"] = {
            "trade": _rel(_only_file(emitter_root / "trades", "trades_*.jsonl")),
            "missed_opportunity": _rel(_only_file(emitter_root / "missed", "missed_*.jsonl")),
            "order": _rel(_only_file(emitter_root / "orders", "orders_*.jsonl")),
            "fill": _rel(_only_file(emitter_root / "inferred_fills", "inferred_fills_*.jsonl")),
            "portfolio_rule": _rel(_only_file(emitter_root / "portfolio_rules", "rules_*.jsonl")),
        }
        manifest["evidence_source"] = "ibkr_stock_runtime_emitters"
        manifest["generation_status"] = "pass"
    except Exception as exc:  # pragma: no cover - reported by manifest validation
        manifest["generation_status"] = "blocked"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _check_production_like_runtime_lineage_evidence(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return _check("production_like_runtime_lineage_evidence", False, {
            "manifest_path": _rel(manifest_path),
            "error": "runtime lineage evidence manifest is missing",
        })
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bot_id = str(manifest.get("bot_id") or "")
    strategy_id = str(manifest.get("strategy_id") or "")
    failures: list[str] = []
    details: dict[str, Any] = {"manifest_path": _rel(manifest_path), "event_classes": {}}
    if manifest.get("generation_status") != "pass":
        failures.append(str(manifest.get("error") or "runtime lineage evidence generation did not pass"))
    if not bot_id or not strategy_id:
        failures.append("runtime lineage evidence lacks bot_id or strategy_id")
    if not _has_value(manifest.get("source_ids", {}).get("proposal_ids")):
        failures.append("runtime lineage evidence lacks proposal source IDs")
    for event_class in RUNTIME_EVENT_CLASSES:
        path = _resolve_path(str(manifest.get("event_paths", {}).get(event_class) or ""))
        records = _load_records(path)
        scoped = [record for record in records if _record_matches_scope(record, bot_id, strategy_id)]
        traced = [record for record in scoped if _has_assistant_trace(record)]
        details["event_classes"][event_class] = {
            "path": _rel(path),
            "record_count": len(records),
            "scoped_record_count": len(scoped),
            "records_with_assistant_lineage": len(traced),
        }
        if not records:
            failures.append(f"{event_class} runtime evidence has no loadable records")
        elif len(scoped) != len(records):
            failures.append(f"{event_class} runtime evidence is missing scoped bot_id/strategy_id")
        elif len(traced) != len(records):
            failures.append(f"{event_class} runtime evidence is missing assistant trace fields")
    return _check("production_like_runtime_lineage_evidence", not failures, {
        **details,
        "failures": failures,
    })


def _check_bot_side_fixture_lineage() -> dict[str, Any]:
    bot_root = ROOT / "trading" / "ibkr_trader"
    if str(bot_root) not in sys.path:
        sys.path.insert(0, str(bot_root))
    try:
        from libs.instrumentation.event_contract import enrich_payload  # type: ignore
        from libs.instrumentation.lineage import LineageContext  # type: ignore
    except Exception as exc:  # pragma: no cover - reported in gate artifact
        return _check("bot_side_runtime_event_lineage_fixtures", False, {
            "error": f"{type(exc).__name__}: {exc}",
        })

    lineage = LineageContext(
        bot_id="ibkr",
        strategy_id="ptg4_fixture_strategy",
        family_id="stock",
        portfolio_id="paper",
        account_alias="paper_ibkr",
        strategy_version="strategy_v1",
        config_version="cfg_v1",
        portfolio_config_version="portfolio_v1",
        risk_config_version="risk_v1",
        allocation_version="alloc_v1",
        strategy_registry_version="registry_v1",
        deployment_id="dep_ptg4",
        parameter_set_id="param_ptg4",
        experiment_id="experiment_ptg4",
        code_sha="abc123",
        trace_id="trace_ptg4",
        proposal_ids=("proposal-ptg4",),
        suggestion_ids=("suggestion-ptg4",),
        extras={"source_weekly_signal_ids": ["weekly-ptg4"], "strategy_change_record_ids": ["change-ptg4"]},
    )
    fixture_types = {
        "trade": "trade",
        "missed_opportunity": "missed_opportunity",
        "order": "order",
        "fill": "inferred_fill",
        "portfolio_rule": "portfolio_rule_check",
    }
    details: dict[str, Any] = {}
    failures: list[str] = []
    for event_class, event_type in fixture_types.items():
        event = enrich_payload(
            _fixture_payload(event_class),
            lineage=lineage,
            event_type=event_type,
        )
        scoped = _record_matches_scope(event, lineage.bot_id, lineage.strategy_id)
        traced = _has_assistant_trace(event)
        details[event_class] = {
            "event_type": event_type,
            "scoped": scoped,
            "assistant_trace_present": traced,
            "proposal_ids": event.get("proposal_ids") or event.get("lineage", {}).get("proposal_ids"),
        }
        if not scoped:
            failures.append(f"{event_class} fixture did not preserve bot_id and strategy_id")
        if not traced:
            failures.append(f"{event_class} fixture did not preserve assistant trace fields")
    generic = {"proposal_ids": ["proposal-ptg4"]}
    wrong_scope_rejected = not _record_matches_scope(generic, lineage.bot_id, lineage.strategy_id)
    details["wrong_scope_generic_record_rejected"] = wrong_scope_rejected
    if not wrong_scope_rejected:
        failures.append("generic assistant lineage record without bot_id and strategy_id was accepted")
    return _check("bot_side_runtime_event_lineage_fixtures", not failures, {
        "fixture_source": "trading/ibkr_trader/libs/instrumentation/event_contract.py",
        "event_classes": details,
        "failures": failures,
    })


def _fixture_payload(event_class: str) -> dict[str, Any]:
    if event_class == "trade":
        return {"trade_id": "trade-ptg4"}
    if event_class == "missed_opportunity":
        return {"opportunity_id": "miss-ptg4"}
    if event_class == "order":
        return {"order_id": "order-ptg4"}
    if event_class == "fill":
        return {"fill_id": "fill-ptg4", "order_id": "order-ptg4"}
    return {"rule_name": "directional_cap", "approved": False}


def _check_runtime_event_lineage(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return _check("runtime_event_assistant_lineage", False, {
            "index_path": _rel(index_path),
            "error": "manifest index is missing",
        })
    index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest_rows = [row for row in index.get("manifests", []) if isinstance(row, dict)]
    details: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in manifest_rows:
        manifest_path = _resolve_path(str(row.get("manifest_path") or ""))
        if not manifest_path.exists():
            failures.append(f"{_rel(manifest_path)} is missing")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bot_id = str(manifest.get("bot_id") or row.get("bot_id") or "")
        strategy_id = str(manifest.get("strategy_id") or row.get("strategy_id") or "")
        scope_detail = {"manifest_path": _rel(manifest_path), "event_classes": {}}
        for event_class in RUNTIME_EVENT_CLASSES:
            records, wrong_scope_count, loadable_count = _records_for_event_class(manifest, event_class, bot_id, strategy_id)
            with_lineage = sum(1 for record in records if _has_assistant_trace(record))
            scope_detail["event_classes"][event_class] = {
                "observed_path_count": len(_observed_paths(manifest, event_class)),
                "loadable_record_count": loadable_count,
                "record_count": len(records),
                "wrong_scope_record_count": wrong_scope_count,
                "records_with_assistant_lineage": with_lineage,
            }
            if wrong_scope_count:
                failures.append(f"{_rel(manifest_path)} has {event_class} evidence for the wrong or missing scope")
            if _observed_paths(manifest, event_class) and not loadable_count:
                failures.append(f"{_rel(manifest_path)} has no loadable {event_class} records in observed evidence paths")
            if records and with_lineage != len(records):
                failures.append(f"{_rel(manifest_path)} has {event_class} records without assistant lineage")
        details.append(scope_detail)
    return _check("runtime_event_assistant_lineage", bool(manifest_rows) and not failures, {
        "index_path": _rel(index_path),
        "required_event_classes": list(RUNTIME_EVENT_CLASSES),
        "manifests": details,
        "failures": failures,
    })


def _records_for_event_class(
    manifest: dict[str, Any],
    event_class: str,
    bot_id: str,
    strategy_id: str,
) -> tuple[list[dict[str, Any]], int, int]:
    records: list[dict[str, Any]] = []
    wrong_scope_count = 0
    loadable_count = 0
    for path_text in _observed_paths(manifest, event_class):
        for record in _load_records(_resolve_path(str(path_text))):
            loadable_count += 1
            if _record_matches_scope(record, bot_id, strategy_id):
                records.append(record)
            else:
                wrong_scope_count += 1
    return records, wrong_scope_count, loadable_count


def _observed_paths(manifest: dict[str, Any], event_class: str) -> list[str]:
    support = manifest.get("runtime_evidence_support", {}).get(event_class, {})
    if not isinstance(support, dict):
        return []
    return [str(path) for path in support.get("observed_evidence_paths", []) if str(path)]


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        if path.suffix == ".jsonl":
            return [
                item
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
                for item in [json.loads(line)]
                if isinstance(item, dict)
            ]
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("records", "events", "items", "rows", "snapshots"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload]


def _clear_emitter_outputs(root: Path) -> None:
    for subdir, pattern in (
        ("trades", "trades_*.jsonl"),
        ("missed", "missed_*.jsonl"),
        ("orders", "orders_*.jsonl"),
        ("inferred_fills", "inferred_fills_*.jsonl"),
        ("portfolio_rules", "rules_*.jsonl"),
    ):
        for path in (root / subdir).glob(pattern):
            if path.is_file():
                path.unlink()


def _only_file(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {_rel(root)}, found {len(matches)}")
    return matches[0]


def _record_matches_scope(record: dict[str, Any], bot_id: str, strategy_id: str) -> bool:
    record_bot = _first_text(record, "bot_id", "bot")
    record_strategy = _first_text(record, "strategy_id", "strategy", "strategy_name", "bridge_id")
    return bool(record_bot and record_strategy and record_bot == bot_id and record_strategy == strategy_id)


def _has_assistant_trace(record: dict[str, Any]) -> bool:
    for scope in _record_scopes(record):
        if any(_has_value(scope.get(key)) for key in ASSISTANT_TRACE_KEYS):
            return True
    return False


def _record_scopes(record: dict[str, Any]) -> list[dict[str, Any]]:
    scopes = [record]
    for key in ("payload", "lineage", "assistant_lineage"):
        value = record.get(key)
        if isinstance(value, dict):
            scopes.append(value)
    return scopes


def _first_text(record: dict[str, Any], *keys: str) -> str:
    for scope in _record_scopes(record):
        for key in keys:
            value = scope.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _has_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set):
        return any(_has_value(item) for item in value)
    return value is not None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "details": details}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
