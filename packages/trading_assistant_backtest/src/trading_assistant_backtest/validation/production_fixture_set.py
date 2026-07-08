"""Build production-derived fixture-set manifests for approval evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.paths import (
    monorepo_root,
    normalize_workspace_path,
    resolve_workspace_path,
)
from trading_assistant_backtest.validation.approval_grade_audit import CONTRACT_PATHS
from trading_assistant_backtest.validation.validation_matrix import SCOPES

SCHEMA_VERSION = "production_fixture_set_manifest_v1"
DEFAULT_BRIDGE_ID = "trading_stock_family"
REQUIRED_CASE_CLASSES = {
    "accepted_entry": {"accepted_entry", "entry_accept", "accepted_trade_entry"},
    "blocked_no_trade": {"blocked_no_trade", "blocked_trade", "no_trade_blocked"},
    "risk_portfolio_denial": {
        "portfolio_denial",
        "portfolio_collision",
        "risk_denial",
        "risk_portfolio_denial",
    },
    "exit_close": {"close", "exit_close", "position_exit"},
    "order_fill_or_explicit_non_fill": {
        "explicit_non_fill",
        "fill",
        "non_fill",
        "order_fill",
        "order_fill_or_explicit_non_fill",
    },
    "live_shadow_telemetry_source": {
        "live_shadow_telemetry_source",
        "live_telemetry",
        "runtime_telemetry",
        "shadow_telemetry",
    },
}

BRIDGE_BOT_ALIASES = {
    "trading_stock_family": {"ibkr", "trading", "paper_ibkr"},
    "trading_momentum_family": {"ibkr", "trading", "paper_ibkr"},
    "trading_swing_family": {"ibkr", "trading", "paper_ibkr"},
    "k_stock_olr_kalcb": {"k_stock", "olr_kalcb", "kis", "paper_kis"},
    "crypto_trend_v1": {"crypto", "crypto_trader", "paper_crypto"},
    "crypto_momentum_v1": {"crypto", "crypto_trader", "paper_crypto"},
    "crypto_breakout_v1": {"crypto", "crypto_trader", "paper_crypto"},
}
NON_FILL_EVENTS = {"cancel", "cancelled", "canceled", "reject", "rejected", "non_fill"}
LIVE_SOURCE_MARKERS = ("live", "paper", "shadow", "runtime", "vps", "broker")


@dataclass
class FixtureSource:
    name: str
    path: Path
    source_kind: str
    case_classes: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


@dataclass
class BuildState:
    agent_root: Path
    bridge_id: str
    expected_bot_ids: set[str]
    expected_strategy_ids: set[str]
    sources: dict[str, FixtureSource] = field(default_factory=dict)
    case_evidence: dict[str, set[str]] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hashed production fixture-set manifest for approval-ready "
            "bridge evidence."
        )
    )
    parser.add_argument("--agent-root", type=Path, default=monorepo_root())
    parser.add_argument("--bridge-id", default=DEFAULT_BRIDGE_ID, choices=sorted(CONTRACT_PATHS))
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--parity-report", type=Path, default=None)
    parser.add_argument("--parity-summary", type=Path, default=None)
    parser.add_argument("--fixture-window", type=Path, default=None)
    parser.add_argument("--learning-manifest", type=Path, default=None)
    parser.add_argument("--telemetry-manifest", type=Path, default=None)
    parser.add_argument("--runtime-evidence-support", type=Path, default=None)
    parser.add_argument(
        "--run-month",
        default=None,
        help=(
            "Optional YYYY-MM phase-2 manifest month. Defaults to the latest existing "
            "month for the selected bridge."
        ),
    )
    parser.add_argument(
        "--runtime-event-path",
        action="append",
        type=Path,
        default=[],
        help="Live/shadow telemetry JSON or JSONL source to scan for matching bridge events.",
    )
    parser.add_argument(
        "--source-record",
        action="append",
        type=Path,
        default=[],
        help="Additional source artifact to hash and include.",
    )
    parser.add_argument(
        "--case-class",
        action="append",
        default=[],
        help="Additional case class asserted by an operator-provided source record.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero if the emitted manifest is blocked.",
    )
    args = parser.parse_args(argv)

    report = build_production_fixture_set_manifest(
        agent_root=args.agent_root,
        bridge_id=args.bridge_id,
        artifact_root=args.artifact_root,
        parity_report_path=args.parity_report,
        parity_summary_path=args.parity_summary,
        fixture_window_path=args.fixture_window,
        learning_manifest_path=args.learning_manifest,
        telemetry_manifest_path=args.telemetry_manifest,
        runtime_evidence_support_path=args.runtime_evidence_support,
        run_month=args.run_month,
        runtime_event_paths=args.runtime_event_path,
        source_record_paths=args.source_record,
        case_classes=args.case_class,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] or not args.require_pass else 1


def build_production_fixture_set_manifest(
    *,
    agent_root: Path,
    bridge_id: str = DEFAULT_BRIDGE_ID,
    artifact_root: Path | None = None,
    parity_report_path: Path | None = None,
    parity_summary_path: Path | None = None,
    fixture_window_path: Path | None = None,
    learning_manifest_path: Path | None = None,
    telemetry_manifest_path: Path | None = None,
    runtime_evidence_support_path: Path | None = None,
    run_month: str | None = None,
    runtime_event_paths: list[Path] | None = None,
    source_record_paths: list[Path] | None = None,
    case_classes: list[str] | None = None,
) -> dict[str, Any]:
    agent_root = Path(agent_root).resolve()
    artifact_root = _default_artifact_root(agent_root, bridge_id, artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    state = BuildState(
        agent_root=agent_root,
        bridge_id=bridge_id,
        expected_bot_ids=_expected_bot_ids(bridge_id),
        expected_strategy_ids=_expected_strategy_ids(agent_root, bridge_id),
    )

    parity_report = _resolved_default(
        agent_root,
        parity_report_path,
        f"artifacts/validation/decision_parity_matrix/{bridge_id}/decision_parity/"
        "decision_parity_report.json",
    )
    parity_summary = _resolved_default(
        agent_root,
        parity_summary_path,
        f"artifacts/validation/decision_parity_matrix/{bridge_id}/decision_parity/"
        "decision_parity_validation_summary.json",
    )
    fixture_window = _resolved_default(
        agent_root,
        fixture_window_path,
        "artifacts/learning_sufficiency/ptg7_pilot/production_derived_fixture_window.json",
    )
    learning_manifest = _resolved_default(
        agent_root,
        learning_manifest_path,
        _default_learning_manifest_path(agent_root, bridge_id, run_month),
    )
    telemetry_manifest = _resolved_default(
        agent_root,
        telemetry_manifest_path,
        _default_telemetry_manifest_path(agent_root, bridge_id, run_month),
    )
    runtime_evidence_support = _resolved_default(
        agent_root,
        runtime_evidence_support_path,
        _default_runtime_evidence_support_path(agent_root, bridge_id, run_month),
    )

    _ingest_decision_parity(state, parity_report, parity_summary)
    _ingest_context_artifact(state, fixture_window, "ptg7_fixture_window", "production_context")
    _ingest_learning_manifest(state, learning_manifest)
    _ingest_telemetry_manifest(state, telemetry_manifest)
    _ingest_context_artifact(
        state,
        runtime_evidence_support,
        "runtime_evidence_support",
        "runtime_support_contract",
    )
    for path in runtime_event_paths or []:
        _ingest_runtime_event_source(state, _resolve(agent_root, path))
    for path in source_record_paths or []:
        source = _add_source(state, _resolve(agent_root, path), "operator_source_record")
        for case_class in case_classes or []:
            _add_case(state, str(case_class), source.path)
            source.case_classes.add(str(case_class))
    if case_classes and not source_record_paths:
        state.blockers.append("operator case classes require at least one --source-record")

    manifest = _build_manifest(state, artifact_root)
    manifest_path = artifact_root / "production_fixture_set_manifest.json"
    manifest["artifact_path"] = _workspace_display_path(agent_root, manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _ingest_decision_parity(state: BuildState, report_path: Path, summary_path: Path) -> None:
    report_source = _add_source(state, report_path, "decision_parity_report")
    _add_source(state, summary_path, "decision_parity_summary")
    report = _read_json(report_path)
    if not report:
        state.blockers.append("decision parity report missing or malformed")
        return
    if str(report.get("status") or "").lower() != "pass":
        state.blockers.append("decision parity report is not passing")
    if _strategy_plugin_id(report.get("strategy_plugin_id")) not in {
        _strategy_plugin_id(value) for value in state.expected_strategy_ids
    }:
        state.blockers.append("decision parity strategy_plugin_id does not match bridge")

    for evidence_path in _decision_parity_evidence_paths(report):
        fixture_path = _resolve(state.agent_root, evidence_path)
        fixture_source = _add_source(state, fixture_path, "parity_fixture")
        for case_class in _classify_parity_fixture(fixture_path):
            _add_case(state, case_class, fixture_path)
            fixture_source.case_classes.add(case_class)
    for check in report.get("checks") or []:
        if not isinstance(check, dict) or str(check.get("status") or "").lower() != "pass":
            continue
        dimension = str(check.get("dimension") or "").strip().lower()
        if dimension == "exits":
            _add_case(state, "exit_close", report_path)
            report_source.case_classes.add("exit_close")


def _ingest_context_artifact(
    state: BuildState,
    path: Path,
    name: str,
    source_kind: str,
) -> None:
    payload = _read_json(path)
    source = _add_source(state, path, source_kind, name=name)
    if not payload:
        source.notes.append("missing_or_malformed")
        state.blockers.append(f"{name} missing or malformed")
        return
    status = str(payload.get("status") or payload.get("generation_status") or "").lower()
    if status and status not in {"pass", "ok", "passed"}:
        source.notes.append(f"status={status}")


def _ingest_learning_manifest(state: BuildState, path: Path) -> None:
    payload = _read_json(path)
    source = _add_source(state, path, "learning_sufficiency_manifest")
    if not payload:
        source.notes.append("missing_or_malformed")
        state.blockers.append("learning sufficiency manifest missing or malformed")
        return
    if str(payload.get("strategy_id") or payload.get("family_id") or "") != state.bridge_id:
        source.notes.append("strategy_id_mismatch")
    eligibility = str(
        payload.get("telemetry_authoritative_eligibility")
        or payload.get("eligibility")
        or ""
    )
    if eligibility not in {"learning_authoritative", "authoritative"}:
        source.notes.append(f"eligibility={eligibility or 'missing'}")


def _ingest_telemetry_manifest(state: BuildState, path: Path) -> None:
    payload = _read_json(path)
    source = _add_source(state, path, "telemetry_manifest")
    if not payload:
        source.notes.append("missing_or_malformed")
        state.blockers.append("telemetry manifest missing or malformed")
        return
    total_events = _count_value(
        payload.get("total_events"),
        label="telemetry manifest total_events",
        blockers=state.blockers,
        notes=source.notes,
    )
    bot_ok = _identifier(payload, "bot_id") in state.expected_bot_ids
    strategy_ok = _identifier(payload, "strategy_id") in state.expected_strategy_ids
    source.notes.append(f"total_events={total_events}")
    if total_events <= 0:
        source.notes.append("no_runtime_events")
        return
    if bot_ok and strategy_ok:
        _add_case(state, "live_shadow_telemetry_source", path)
        source.case_classes.add("live_shadow_telemetry_source")
    else:
        source.notes.append("identity_mismatch")
        state.blockers.append("telemetry manifest has events but does not match bridge identity")


def _ingest_runtime_event_source(state: BuildState, path: Path) -> None:
    source = _add_source(state, path, "runtime_event_source")
    records = _read_event_records(path)
    if not records:
        source.notes.append("no_records")
        state.blockers.append(
            "runtime event source has no readable events: "
            f"{_workspace_display_path(state.agent_root, path)}"
        )
        return
    matched = False
    mismatched = 0
    for record in records[:100]:
        bot_id = _record_identifier(record, "bot_id")
        strategy_id = _record_identifier(record, "strategy_id")
        if bot_id in state.expected_bot_ids and strategy_id in state.expected_strategy_ids:
            matched = True
            event_type = _record_identifier(record, "event_type")
            if _is_live_runtime_record(record):
                _add_case(state, "live_shadow_telemetry_source", path)
                source.case_classes.add("live_shadow_telemetry_source")
            if event_type in {"fill", "inferred_fill"}:
                _add_case(state, "order_fill", path)
                source.case_classes.add("order_fill")
            elif event_type in {"order"} and _record_identifier(record, "status") in {
                "cancelled",
                "canceled",
                "rejected",
                "non_fill",
            }:
                _add_case(state, "explicit_non_fill", path)
                source.case_classes.add("explicit_non_fill")
        else:
            mismatched += 1
    if not matched:
        source.notes.append(f"identity_mismatched_records={mismatched}")
        state.blockers.append(
            "runtime event source contains no events matching bridge bot/strategy identity"
        )


def _classify_parity_fixture(path: Path) -> set[str]:
    payload = _read_json(path)
    if not payload:
        return set()
    classes: set[str] = set()
    broker_events = payload.get("broker_event_script") or []
    if isinstance(broker_events, list):
        for event in broker_events:
            if not isinstance(event, dict):
                continue
            event_name = str(event.get("event") or "").strip().lower()
            order_match = (
                event.get("order_match")
                if isinstance(event.get("order_match"), dict)
                else {}
            )
            role = str(order_match.get("role") or "").strip().upper()
            if event_name == "fill":
                classes.add("order_fill")
                if role == "ENTRY":
                    classes.add("accepted_entry")
                if role in {"EXIT", "CLOSE"}:
                    classes.add("exit_close")
            elif event_name in NON_FILL_EVENTS:
                classes.add("explicit_non_fill")
    if _contains_key(payload, "idle_market_input"):
        classes.add("blocked_no_trade")
    if _has_portfolio_collision(payload):
        classes.add("risk_portfolio_denial")
    if _contains_text(payload, {"exit", "close"}) and _contains_key(payload, "exit_order_model"):
        classes.add("exit_close")
    return classes


def _build_manifest(state: BuildState, artifact_root: Path) -> dict[str, Any]:
    observed_classes = {
        case_class
        for case_class, evidence in state.case_evidence.items()
        if case_class and evidence
    }
    missing: list[str] = []
    for required, aliases in REQUIRED_CASE_CLASSES.items():
        if not observed_classes.intersection(aliases):
            missing.append(required)
    blockers = list(
        dict.fromkeys(
            [*state.blockers, *[f"missing case class: {item}" for item in missing]]
        )
    )
    source_records = []
    for source in sorted(
        state.sources.values(),
        key=lambda item: _workspace_display_path(state.agent_root, item.path),
    ):
        record: dict[str, Any] = {
            "name": source.name,
            "path": _workspace_display_path(state.agent_root, source.path),
            "source_kind": source.source_kind,
            "case_classes": sorted(source.case_classes),
            "exists": source.path.exists() and source.path.is_file(),
        }
        if record["exists"]:
            record["sha256"] = sha256_file(source.path)
        else:
            record["sha256"] = ""
            blockers.append(f"source record missing: {record['path']}")
        if source.notes:
            record["notes"] = source.notes
        source_records.append(record)
    ok = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "bridge_id": state.bridge_id,
        "status": "pass" if ok else "blocked",
        "ok": ok,
        "source_kind": (
            "production_derived_live_shadow"
            if observed_classes.intersection(REQUIRED_CASE_CLASSES["live_shadow_telemetry_source"])
            else "production_derived_pending_live_shadow_telemetry"
        ),
        "required_case_classes": {
            key: sorted(value) for key, value in REQUIRED_CASE_CLASSES.items()
        },
        "case_classes": sorted(observed_classes),
        "missing_case_classes": missing,
        "cases": [
            {
                "case_class": case_class,
                "evidence_paths": sorted(state.case_evidence[case_class]),
            }
            for case_class in sorted(state.case_evidence)
        ],
        "source_records": source_records,
        "blockers": list(dict.fromkeys(blockers)),
        "artifact_root": _workspace_display_path(state.agent_root, artifact_root),
    }


def _add_source(
    state: BuildState,
    path: Path,
    source_kind: str,
    *,
    name: str | None = None,
) -> FixtureSource:
    resolved = path.resolve()
    key = str(resolved).lower()
    source = state.sources.get(key)
    if source is None:
        source = FixtureSource(name=name or resolved.stem, path=resolved, source_kind=source_kind)
        state.sources[key] = source
    return source


def _add_case(state: BuildState, case_class: str, path: Path) -> None:
    value = str(case_class).strip()
    if not value:
        return
    state.case_evidence.setdefault(value, set()).add(
        _workspace_display_path(state.agent_root, path)
    )


def _decision_parity_evidence_paths(report: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for raw in report.get("evidence_paths") or []:
        if raw:
            paths.append(str(raw))
    for check in report.get("checks") or []:
        if not isinstance(check, dict):
            continue
        for raw in check.get("evidence_paths") or []:
            if raw:
                paths.append(str(raw))
    return sorted(dict.fromkeys(paths))


def _read_event_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records
    payload = _read_json(path)
    if not payload:
        return []
    for key in ("events", "records", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value:
            return True
        return any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _contains_text(value: Any, needles: set[str]) -> bool:
    if isinstance(value, dict):
        return any(_contains_text(item, needles) for item in value.values())
    if isinstance(value, list):
        return any(_contains_text(item, needles) for item in value)
    text = str(value).lower()
    return any(needle in text for needle in needles)


def _has_portfolio_collision(payload: dict[str, Any]) -> bool:
    rules = (
        payload.get("family_config", {}).get("portfolio_rules", {})
        if isinstance(payload.get("family_config"), dict)
        else {}
    )
    collision_action = str(rules.get("symbol_collision_action") or "").strip().lower()
    if collision_action and collision_action != "none":
        return True
    decisions = (
        payload.get("expected_normalized_outputs", {}).get("family_decisions", [])
        if isinstance(payload.get("expected_normalized_outputs"), dict)
        else []
    )
    if not isinstance(decisions, list):
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("status") or "").strip().lower() in {"blocked", "denied", "reduced"}
        for item in decisions
    )


def _identifier(payload: dict[str, Any], key: str) -> str:
    return str(payload.get(key) or "").strip()


def _record_identifier(record: dict[str, Any], key: str) -> str:
    for container in (
        record,
        record.get("event_metadata") if isinstance(record.get("event_metadata"), dict) else {},
        record.get("lineage") if isinstance(record.get("lineage"), dict) else {},
    ):
        value = container.get(key) if isinstance(container, dict) else None
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _is_live_runtime_record(record: dict[str, Any]) -> bool:
    source_values = [
        _record_identifier(record, "data_source_id"),
        _record_identifier(record, "source"),
        _record_identifier(record, "scope"),
    ]
    joined = " ".join(value.lower() for value in source_values if value)
    if "fixture" in joined and "production" not in joined:
        return False
    return any(marker in joined for marker in LIVE_SOURCE_MARKERS) or bool(
        _record_identifier(record, "deployment_id")
    )


def _expected_bot_ids(bridge_id: str) -> set[str]:
    return {bridge_id, *BRIDGE_BOT_ALIASES.get(bridge_id, {bridge_id})}


def _expected_strategy_ids(agent_root: Path, bridge_id: str) -> set[str]:
    values = {bridge_id}
    contract_path = (
        resolve_workspace_path(agent_root, CONTRACT_PATHS[bridge_id])
        / "strategy_plugin_contract.json"
    )
    contract = _read_json(contract_path)
    if contract.get("plugin_id"):
        values.add(str(contract["plugin_id"]))
    for scope in SCOPES:
        bridge_ids = set(scope.decision_bridge_ids or (scope.decision_bridge_id,))
        if bridge_id in bridge_ids or scope.scope_id == bridge_id:
            values.update(scope.strategies)
            values.add(scope.scope_id)
            values.add(scope.decision_bridge_id)
    return values


def _strategy_plugin_id(value: Any) -> str:
    return str(value or "").strip().replace("-", "_")


def _default_artifact_root(agent_root: Path, bridge_id: str, artifact_root: Path | None) -> Path:
    if artifact_root is not None:
        return _resolve(agent_root, artifact_root)
    return agent_root / "artifacts" / "validation" / "decision_parity_matrix" / bridge_id


def _default_learning_manifest_path(
    agent_root: Path,
    bridge_id: str,
    run_month: str | None,
) -> str:
    return _default_phase2_manifest_path(
        agent_root,
        bridge_id,
        "learning_sufficiency_manifest.json",
        run_month,
    )


def _default_telemetry_manifest_path(
    agent_root: Path,
    bridge_id: str,
    run_month: str | None,
) -> str:
    return _default_phase2_manifest_path(
        agent_root,
        bridge_id,
        "telemetry_manifest.json",
        run_month,
    )


def _default_runtime_evidence_support_path(
    agent_root: Path,
    bridge_id: str,
    run_month: str | None,
) -> str:
    return _default_phase2_manifest_path(
        agent_root,
        bridge_id,
        "runtime_evidence_support.json",
        run_month,
    )


def _default_phase2_manifest_path(
    agent_root: Path,
    bridge_id: str,
    filename: str,
    run_month: str | None,
) -> str:
    bot = _bridge_bot_id(bridge_id)
    base = Path("artifacts") / "learning_sufficiency" / "phase2_manifests"
    if run_month:
        return (base / bot / run_month / bridge_id / filename).as_posix()
    search_root = agent_root / base / bot
    candidates = sorted(
        search_root.glob(f"????-??/{bridge_id}/{filename}"),
        key=lambda item: item.as_posix(),
        reverse=True,
    )
    if candidates:
        return _workspace_display_path(agent_root, candidates[0])
    return (base / bot / "2026-06" / bridge_id / filename).as_posix()


def _bridge_bot_id(bridge_id: str) -> str:
    if bridge_id.startswith("crypto_"):
        return "crypto"
    if bridge_id.startswith("trading_"):
        return "ibkr"
    return "k_stock"


def _count_value(
    value: Any,
    *,
    label: str,
    blockers: list[str],
    notes: list[str],
) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        notes.append(f"{label}=invalid")
        blockers.append(f"{label} must be an integer count")
        return 0
    try:
        count = int(value)
    except (TypeError, ValueError):
        notes.append(f"{label}=invalid")
        blockers.append(f"{label} must be an integer count")
        return 0
    if count < 0:
        notes.append(f"{label}={count}")
        blockers.append(f"{label} cannot be negative")
    return count


def _resolved_default(agent_root: Path, explicit: Path | None, default: str) -> Path:
    return _resolve(agent_root, explicit if explicit is not None else Path(default))


def _resolve(agent_root: Path, path: str | Path) -> Path:
    return normalize_workspace_path(agent_root, path)


def _workspace_display_path(agent_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(agent_root.resolve()).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
