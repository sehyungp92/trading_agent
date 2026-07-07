"""Inventory live-repo bridge readiness without promoting unfinished adapters."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.contract_models import (
    DecisionParityReport,
    DecisionParityStatus,
    StrategyPluginMaturity,
)
from trading_assistant_backtest.strategies.contracts import load_strategy_plugin_contract
from trading_assistant_backtest.strategies.live_clone import (
    LiveRepoCheckoutSpec,
    LiveRepoCloneManager,
)
from trading_assistant_backtest.paths import monorepo_root, package_root, resolve_workspace_path

CRYPTO_CONTRACT = Path(
    "contracts/strategy_plugins/crypto_trend_v1/strategy_plugin_contract.json"
)
CRYPTO_PARITY_SUMMARY = Path(
    "artifacts/validation/decision_parity_matrix/crypto_trend_v1/"
    "decision_parity/decision_parity_validation_summary.json"
)
CRYPTO_PARITY_REPORT = Path(
    "artifacts/validation/decision_parity_matrix/crypto_trend_v1/"
    "decision_parity/decision_parity_report.json"
)
CRYPTO_MOMENTUM_CONTRACT = Path(
    "contracts/strategy_plugins/crypto_momentum_v1/strategy_plugin_contract.json"
)
CRYPTO_MOMENTUM_PARITY_SUMMARY = Path(
    "artifacts/validation/decision_parity_matrix/crypto_momentum_v1/"
    "decision_parity/decision_parity_validation_summary.json"
)
CRYPTO_MOMENTUM_PARITY_REPORT = Path(
    "artifacts/validation/decision_parity_matrix/crypto_momentum_v1/"
    "decision_parity/decision_parity_report.json"
)
CRYPTO_BREAKOUT_CONTRACT = Path(
    "contracts/strategy_plugins/crypto_breakout_v1/strategy_plugin_contract.json"
)
CRYPTO_BREAKOUT_PARITY_SUMMARY = Path(
    "artifacts/validation/decision_parity_matrix/crypto_breakout_v1/"
    "decision_parity/decision_parity_validation_summary.json"
)
CRYPTO_BREAKOUT_PARITY_REPORT = Path(
    "artifacts/validation/decision_parity_matrix/crypto_breakout_v1/"
    "decision_parity/decision_parity_report.json"
)
K_STOCK_CONTRACT = Path(
    "contracts/strategy_plugins/k_stock_olr_kalcb/strategy_plugin_contract.json"
)
K_STOCK_PARITY_SUMMARY = Path(
    "artifacts/validation/decision_parity_matrix/k_stock_olr_kalcb/"
    "decision_parity/decision_parity_validation_summary.json"
)
K_STOCK_PARITY_REPORT = Path(
    "artifacts/validation/decision_parity_matrix/k_stock_olr_kalcb/"
    "decision_parity/decision_parity_report.json"
)
TRADING_STOCK_CONTRACT = Path(
    "contracts/strategy_plugins/trading_stock_family/strategy_plugin_contract.json"
)
TRADING_STOCK_PARITY_SUMMARY = Path(
    "artifacts/validation/decision_parity_matrix/trading_stock_family/"
    "decision_parity/decision_parity_validation_summary.json"
)
TRADING_STOCK_PARITY_REPORT = Path(
    "artifacts/validation/decision_parity_matrix/trading_stock_family/"
    "decision_parity/decision_parity_report.json"
)
TRADING_MOMENTUM_CONTRACT = Path(
    "contracts/strategy_plugins/trading_momentum_family/strategy_plugin_contract.json"
)
TRADING_MOMENTUM_PARITY_SUMMARY = Path(
    "artifacts/validation/decision_parity_matrix/trading_momentum_family/"
    "decision_parity/decision_parity_validation_summary.json"
)
TRADING_MOMENTUM_PARITY_REPORT = Path(
    "artifacts/validation/decision_parity_matrix/trading_momentum_family/"
    "decision_parity/decision_parity_report.json"
)
TRADING_SWING_CONTRACT = Path(
    "contracts/strategy_plugins/trading_swing_family/strategy_plugin_contract.json"
)
TRADING_SWING_PARITY_SUMMARY = Path(
    "artifacts/validation/decision_parity_matrix/trading_swing_family/"
    "decision_parity/decision_parity_validation_summary.json"
)
TRADING_SWING_PARITY_REPORT = Path(
    "artifacts/validation/decision_parity_matrix/trading_swing_family/"
    "decision_parity/decision_parity_report.json"
)

TRADING_SHARED_EVIDENCE_PATHS = (
    "tests/integration/parity/source_contract.py",
    "tests/integration/parity/live_shadow_contract.py",
    "tests/integration/parity/test_live_shadow_layer2.py",
    "tests/integration/parity/test_live_shadow_families.py",
)

TRADING_STOCK_EVIDENCE_PATHS = (
    *TRADING_SHARED_EVIDENCE_PATHS,
    "tests/fixtures/parity/layer2/iaric_entry_fill.json",
    "tests/fixtures/parity/layer3/stock_family_collision.json",
    "strategies/stock/iaric",
    "strategies/stock/alcb",
)

TRADING_MOMENTUM_EVIDENCE_PATHS = (
    *TRADING_SHARED_EVIDENCE_PATHS,
    "tests/fixtures/parity/layer2/nq_regime_entry_fill.json",
    "tests/fixtures/parity/layer3/momentum_family_shared_risk.json",
    "strategies/momentum/nqdtc",
    "strategies/momentum/nq_regime",
    "strategies/momentum/vdub",
    "strategies/momentum/downturn",
)

TRADING_SWING_EVIDENCE_PATHS = (
    *TRADING_SHARED_EVIDENCE_PATHS,
    "tests/fixtures/parity/layer2/tpc_entry_fill.json",
    "tests/fixtures/parity/layer3/swing_family_overlay_rebalance.json",
    "strategies/swing/atrss",
    "strategies/swing/akc_helix",
    "strategies/swing/tpc",
    "strategies/swing/overlay",
)

K_STOCK_EVIDENCE_PATHS = (
    "tests/fixtures/live_replay_parity/olr/manifest.json",
    "tests/fixtures/live_replay_parity/kalcb/manifest.json",
    "tests/backtests/strategies/test_olr_kalcb_live_replay_artifact_parity.py",
    "deployment/olr_kalcb/replay.py",
    "deployment/olr_kalcb/offline_replay.py",
    "strategy_olr/core/logic.py",
    "strategy_kalcb/core/logic.py",
)

FORMAL_BRIDGE_BLOCKERS = (
    "strategy_plugin_contract_missing_in_trading_assistant_backtest",
    "normalized_decision_trace_adapter_missing_in_trading_assistant_backtest",
    "vps_deployment_metadata_missing_in_bridge_format",
    "formal_decision_parity_report_missing",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit live repo bridge readiness and emit bridge_readiness_report.json."
    )
    parser.add_argument("--agent-root", type=Path, default=_default_agent_root())
    parser.add_argument("--artifact-root", type=Path, default=_default_artifact_root())
    parser.add_argument(
        "--crypto-parity-artifact-root",
        type=Path,
        default=None,
        help="Optional artifact root containing crypto decision_parity_report.json and summary.",
    )
    args = parser.parse_args(argv)

    result = run_bridge_readiness_audit(
        agent_root=args.agent_root,
        artifact_root=args.artifact_root,
        crypto_parity_artifact_root=args.crypto_parity_artifact_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["ok"] else 1


def run_bridge_readiness_audit(
    *,
    agent_root: Path,
    artifact_root: Path,
    crypto_parity_artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Emit an artifact describing which live repo bridges are formal enough to trust."""

    agent_root = Path(agent_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    bridges = [
        _crypto_bridge(agent_root, crypto_parity_artifact_root=crypto_parity_artifact_root),
        _formal_shadow_bridge(
            repo_id="crypto_momentum_v1",
            agent_root=agent_root,
            contract_path=CRYPTO_MOMENTUM_CONTRACT,
            parity_summary_path=CRYPTO_MOMENTUM_PARITY_SUMMARY,
            parity_report_path=CRYPTO_MOMENTUM_PARITY_REPORT,
            supported_scope=["portfolio:crypto_portfolio", "strategy:momentum"],
            weekly_focus_week=4,
            recommended_next_step=(
                "Keep crypto_momentum_v1 in shadow validation while expanding fixtures "
                "against live runtime telemetry."
            ),
        ),
        _formal_shadow_bridge(
            repo_id="crypto_breakout_v1",
            agent_root=agent_root,
            contract_path=CRYPTO_BREAKOUT_CONTRACT,
            parity_summary_path=CRYPTO_BREAKOUT_PARITY_SUMMARY,
            parity_report_path=CRYPTO_BREAKOUT_PARITY_REPORT,
            supported_scope=["portfolio:crypto_portfolio", "strategy:breakout"],
            weekly_focus_week=4,
            recommended_next_step=(
                "Keep crypto_breakout_v1 in shadow validation while expanding fixtures "
                "against live runtime telemetry."
            ),
        ),
        _formal_shadow_bridge(
            repo_id="trading_stock_family",
            agent_root=agent_root,
            contract_path=TRADING_STOCK_CONTRACT,
            parity_summary_path=TRADING_STOCK_PARITY_SUMMARY,
            parity_report_path=TRADING_STOCK_PARITY_REPORT,
            supported_scope=["portfolio:stock", "strategy:IARIC_v1", "strategy:ALCB_v1"],
            weekly_focus_week=1,
            recommended_next_step=(
                "Keep the stock-family bridge in shadow validation while expanding fixtures "
                "beyond IARIC entry/family collision cases."
            ),
        ),
        _formal_shadow_bridge(
            repo_id="k_stock_olr_kalcb",
            agent_root=agent_root,
            contract_path=K_STOCK_CONTRACT,
            parity_summary_path=K_STOCK_PARITY_SUMMARY,
            parity_report_path=K_STOCK_PARITY_REPORT,
            supported_scope=["strategy:OLR", "strategy:KALCB", "portfolio:OLR_KALCB"],
            weekly_focus_week=1,
            recommended_next_step=(
                "Keep OLR/KALCB in shadow validation while adding runtime-session fixtures "
                "covering fills, exits, non-candidate bars, and portfolio arbitration."
            ),
        ),
        _formal_shadow_bridge(
            repo_id="trading_momentum_family",
            agent_root=agent_root,
            contract_path=TRADING_MOMENTUM_CONTRACT,
            parity_summary_path=TRADING_MOMENTUM_PARITY_SUMMARY,
            parity_report_path=TRADING_MOMENTUM_PARITY_REPORT,
            supported_scope=[
                "portfolio:momentum",
                "asset:futures",
                "strategy:NQDTC_v2.1",
                "strategy:NQ_REGIME",
                "strategy:VdubusNQ_v4",
                "strategy:DownturnDominator_v1",
            ],
            weekly_focus_week=2,
            recommended_next_step=(
                "Keep trading momentum in shadow validation while broadening fixtures beyond "
                "NQ_REGIME entry and family shared-risk cases."
            ),
        ),
        _formal_shadow_bridge(
            repo_id="trading_swing_family",
            agent_root=agent_root,
            contract_path=TRADING_SWING_CONTRACT,
            parity_summary_path=TRADING_SWING_PARITY_SUMMARY,
            parity_report_path=TRADING_SWING_PARITY_REPORT,
            supported_scope=[
                "portfolio:swing",
                "strategy:ATRSS",
                "strategy:AKC_HELIX",
                "strategy:TPC",
                "strategy:OVERLAY",
            ],
            weekly_focus_week=3,
            recommended_next_step=(
                "Keep trading swing in shadow validation while adding blocked-trade, exit, "
                "and overlay rebalance edge fixtures."
            ),
        ),
    ]

    artifact_path = artifact_root / "bridge_readiness_report.json"
    formal_bridge_backlog = [
        bridge["repo_id"]
        for bridge in bridges
        if "needs_formal_bridge" in str(bridge["status"])
    ]
    report = {
        "ok": all(bridge["audit_passed"] for bridge in bridges),
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_path": str(artifact_path),
        "approval_ready_bridges": [
            bridge["repo_id"] for bridge in bridges if bridge["approval_ready"] is True
        ],
        "shadow_validated_bridges": [
            bridge["repo_id"]
            for bridge in bridges
            if bridge["maturity"] == "shadow_validated" and bridge["audit_passed"]
        ],
        "configured_shadow_validated_bridges": [
            bridge["repo_id"] for bridge in bridges if bridge["maturity"] == "shadow_validated"
        ],
        "formal_bridge_backlog": formal_bridge_backlog,
        "weekly_focus_rotation": [
            {
                "cycle_week": 1,
                "focus_id": "k_stock_and_trading_stock",
                "bridge_ids": ["k_stock_olr_kalcb", "trading_stock_family"],
            },
            {
                "cycle_week": 2,
                "focus_id": "trading_momentum",
                "bridge_ids": ["trading_momentum_family"],
            },
            {
                "cycle_week": 3,
                "focus_id": "trading_swing",
                "bridge_ids": ["trading_swing_family"],
            },
            {
                "cycle_week": 4,
                "focus_id": "crypto_trader",
                "bridge_ids": [
                    "crypto_trend_v1",
                    "crypto_momentum_v1",
                    "crypto_breakout_v1",
                ],
            },
        ],
        "recommended_next_bridge": (
            formal_bridge_backlog[0]
            if formal_bridge_backlog
            else "replay_backed_evaluator_and_fixture_coverage"
        ),
        "bridges": bridges,
    }
    artifact_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _crypto_bridge(
    agent_root: Path,
    *,
    crypto_parity_artifact_root: Path | None,
) -> dict[str, Any]:
    contract_path = resolve_workspace_path(agent_root, CRYPTO_CONTRACT)
    if crypto_parity_artifact_root is None:
        parity_summary_path = resolve_workspace_path(agent_root, CRYPTO_PARITY_SUMMARY)
        parity_report_path = resolve_workspace_path(agent_root, CRYPTO_PARITY_REPORT)
    else:
        root = Path(crypto_parity_artifact_root).resolve()
        parity_summary_path = root / "decision_parity_validation_summary.json"
        parity_report_path = root / "decision_parity_report.json"
    contract, contract_errors = load_strategy_plugin_contract(contract_path)
    parity_summary = _read_json(parity_summary_path)
    parity_report = _read_json(parity_report_path)
    approval_blockers: list[str] = []
    evidence_errors = list(contract_errors)

    parity_status = str(parity_summary.get("decision_parity_status") or "")
    parity_report_passed = parity_report.get("status") == DecisionParityStatus.PASS.value
    if parity_status != DecisionParityStatus.PASS.value or not parity_report_passed:
        evidence_errors.append("formal crypto decision parity summary is missing or not passing")
        evidence_errors.extend(_summary_check_errors(parity_summary))

    try:
        if parity_report:
            DecisionParityReport.model_validate(parity_report)
        else:
            evidence_errors.append("formal crypto decision parity report is missing")
    except Exception as exc:
        evidence_errors.append(f"formal crypto decision parity report is invalid: {exc}")

    if contract is None:
        maturity = "missing"
        approval_ready = False
        eligible_for_optimizer = False
    else:
        maturity = contract.maturity.value
        approval_ready = contract.eligible_for_approval
        eligible_for_optimizer = contract.eligible_for_optimizer
        if contract.maturity != StrategyPluginMaturity.SHADOW_VALIDATED:
            evidence_errors.append(f"crypto contract maturity is {contract.maturity.value}")
        if approval_ready:
            evidence_errors.append("crypto contract is unexpectedly approval-ready")
        else:
            approval_blockers.append("plugin_maturity_is_shadow_validated_not_approval_ready")

    repo_path = _source_repo_path(
        repo_id="crypto_trend_v1",
        parity_summary=parity_summary,
        contract=contract,
        contract_path=contract_path,
        agent_root=agent_root,
        fallback=agent_root / "bots" / "crypto_trader",
    )

    return {
        "repo_id": "crypto_trend_v1",
        "repo_path": str(repo_path),
        **_git_state(repo_path),
        "status": (
            "formal_decision_parity_passed"
            if not evidence_errors
            else "formal_decision_parity_missing_or_failed"
        ),
        "maturity": maturity,
        "eligible_for_optimizer": eligible_for_optimizer,
        "approval_ready": approval_ready,
        "audit_passed": not evidence_errors,
        "evidence": [
            _path_evidence(agent_root, CRYPTO_CONTRACT),
            _path_evidence(agent_root, CRYPTO_PARITY_SUMMARY),
            _path_evidence(agent_root, CRYPTO_PARITY_REPORT),
        ],
        "approval_blockers": approval_blockers,
        "errors": evidence_errors,
        "recommended_next_step": (
            "Keep crypto_trend_v1 in shadow validation until live deployment metadata and "
            "broader shadow cycles justify promotion."
        ),
    }


def _formal_shadow_bridge(
    *,
    repo_id: str,
    agent_root: Path,
    contract_path: Path,
    parity_summary_path: Path,
    parity_report_path: Path,
    supported_scope: list[str],
    weekly_focus_week: int,
    recommended_next_step: str,
) -> dict[str, Any]:
    resolved_contract_path = resolve_workspace_path(agent_root, contract_path)
    contract, contract_errors = load_strategy_plugin_contract(resolved_contract_path)
    parity_summary = _read_json(resolve_workspace_path(agent_root, parity_summary_path))
    repo_path = _source_repo_path(
        repo_id=repo_id,
        parity_summary=parity_summary,
        contract=contract,
        contract_path=resolved_contract_path,
        agent_root=agent_root,
        fallback=(
            Path(contract.live_repo_path)
            if contract is not None and contract.live_repo_path
            else _fallback_repo_path(agent_root, repo_id)
        ),
    )
    git_state = _git_state(repo_path)
    parity_report = _read_json(resolve_workspace_path(agent_root, parity_report_path))
    errors = [*contract_errors]
    approval_blockers: list[str] = []

    parity_status = str(parity_summary.get("decision_parity_status") or "")
    parity_report_passed = parity_report.get("status") == DecisionParityStatus.PASS.value
    if parity_status != DecisionParityStatus.PASS.value or not parity_report_passed:
        errors.append("formal decision parity summary is missing or not passing")
        errors.extend(_summary_check_errors(parity_summary))

    try:
        if parity_report:
            DecisionParityReport.model_validate(parity_report)
        else:
            errors.append("formal decision parity report is missing")
    except Exception as exc:
        errors.append(f"formal decision parity report is invalid: {exc}")

    if contract is None:
        maturity = "missing"
        eligible_for_optimizer = False
        approval_ready = False
    else:
        maturity = contract.maturity.value
        eligible_for_optimizer = contract.eligible_for_optimizer
        approval_ready = contract.eligible_for_approval
        if contract.maturity != StrategyPluginMaturity.SHADOW_VALIDATED:
            errors.append(f"contract maturity is {contract.maturity.value}")
        if approval_ready:
            errors.append("contract is unexpectedly approval-ready")
        else:
            approval_blockers.append("plugin_maturity_is_shadow_validated_not_approval_ready")
        errors.extend(_git_errors_for_maturity(git_state, contract.maturity))

    return {
        "repo_id": repo_id,
        "repo_path": str(repo_path),
        **git_state,
        "status": (
            "formal_decision_parity_passed"
            if not errors
            else "formal_decision_parity_missing_or_failed"
        ),
        "maturity": maturity,
        "eligible_for_optimizer": eligible_for_optimizer,
        "approval_ready": approval_ready,
        "audit_passed": not errors,
        "supported_scope": supported_scope,
        "weekly_focus_week": weekly_focus_week,
        "evidence": [
            _path_evidence(agent_root, contract_path),
            _path_evidence(agent_root, parity_summary_path),
            _path_evidence(agent_root, parity_report_path),
        ],
        "approval_blockers": approval_blockers,
        "errors": errors,
        "recommended_next_step": recommended_next_step,
    }


def _reference_repo_bridge(
    *,
    repo_id: str,
    repo_path: Path,
    evidence_paths: tuple[str, ...],
    status_if_evidence_present: str,
    supported_scope: list[str],
    recommended_next_step: str,
    weekly_focus_week: int,
) -> dict[str, Any]:
    git_state = _git_state(repo_path)
    evidence = [_repo_path_evidence(repo_path, relative_path) for relative_path in evidence_paths]
    missing = [item["relative_path"] for item in evidence if not item["exists"]]
    errors = [*git_state["errors"], *[f"missing evidence: {path}" for path in missing]]
    status = status_if_evidence_present if not missing else "parity_surface_incomplete"
    return {
        "repo_id": repo_id,
        "repo_path": str(repo_path),
        **git_state,
        "status": status,
        "maturity": "diagnostic",
        "eligible_for_optimizer": False,
        "approval_ready": False,
        "audit_passed": not errors,
        "supported_scope": supported_scope,
        "weekly_focus_week": weekly_focus_week,
        "evidence": evidence,
        "approval_blockers": list(FORMAL_BRIDGE_BLOCKERS),
        "errors": errors,
        "recommended_next_step": recommended_next_step,
    }


def _git_state(repo_path: Path) -> dict[str, Any]:
    if not repo_path.exists():
        return {
            "source_checkout_clean": False,
            "commit_sha": "",
            "errors": [f"repo path does not exist: {repo_path}"],
        }
    head = _git(repo_path, "rev-parse", "HEAD")
    status = _git(repo_path, "status", "--short")
    errors: list[str] = []
    if head["returncode"] != 0:
        errors.append(head["stderr"] or "git rev-parse HEAD failed")
    if status["returncode"] != 0:
        errors.append(status["stderr"] or "git status --short failed")
    dirty_lines = [line for line in status["stdout"].splitlines() if line.strip()]
    if dirty_lines:
        errors.append("source checkout has uncommitted changes")
    return {
        "source_checkout_clean": not errors,
        "commit_sha": head["stdout"].strip() if head["returncode"] == 0 else "",
        "errors": errors,
    }


def _git_errors_for_maturity(
    git_state: dict[str, Any],
    maturity: StrategyPluginMaturity,
) -> list[str]:
    errors = [str(error) for error in git_state.get("errors", []) if str(error)]
    if maturity == StrategyPluginMaturity.APPROVAL_READY:
        return errors
    return [
        error
        for error in errors
        if error != "source checkout has uncommitted changes"
    ]


def _source_repo_path(
    *,
    repo_id: str,
    parity_summary: dict[str, Any],
    contract: Any,
    contract_path: Path,
    agent_root: Path,
    fallback: Path,
) -> Path:
    if parity_summary.get("validated_live_repo_path"):
        validated_path = Path(parity_summary["validated_live_repo_path"])
        if validated_path.exists():
            return validated_path
    if contract is None or not getattr(contract, "live_repo_commit_sha", ""):
        return fallback
    metadata = _read_json(contract_path.with_name("deployment_metadata.json"))
    repo_url = str(metadata.get("repo_url") or "")
    if not repo_url:
        return fallback
    try:
        source = _clone_source(repo_url, agent_root=agent_root)
        checkout_root = Path(tempfile.gettempdir()) / "ta_live_checkouts"
        manager = LiveRepoCloneManager(checkout_root)
        return manager.prepare(
            LiveRepoCheckoutSpec(
                repo_url=source,
                commit_sha=contract.live_repo_commit_sha,
                checkout_root=manager.checkout_root,
            )
        )
    except Exception:
        return fallback


def _fallback_repo_path(agent_root: Path, repo_id: str) -> Path:
    if repo_id == "trading":
        return agent_root / "bots" / "ibkr_trading"
    return agent_root / "bots" / repo_id


def _clone_source(repo_url: str, *, agent_root: Path) -> str:
    if not repo_url.startswith("local://"):
        return repo_url
    raw = repo_url.removeprefix("local://")
    if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    path = Path(raw)
    if not path.is_absolute():
        path = resolve_workspace_path(agent_root, path)
    return str(path.resolve())


def _git(repo_path: Path, *args: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _repo_path_evidence(repo_path: Path, relative_path: str) -> dict[str, Any]:
    path = repo_path / relative_path
    return {
        "relative_path": relative_path,
        "path": str(path),
        "exists": path.exists(),
    }


def _path_evidence(agent_root: Path, relative_path: Path) -> dict[str, Any]:
    path = resolve_workspace_path(agent_root, relative_path)
    relative_text = relative_path.as_posix()
    return {
        "relative_path": relative_text,
        "path": str(path),
        "exists": path.exists(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _summary_check_errors(summary: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for check in summary.get("checks", []):
        if not isinstance(check, dict) or check.get("passed") is True:
            continue
        name = str(check.get("name") or "unnamed_check")
        errors = check.get("errors")
        if isinstance(errors, list) and errors:
            rows.extend(f"{name}: {error}" for error in errors)
        else:
            rows.append(f"{name}: failed")
    return rows


def _default_artifact_root() -> Path:
    return monorepo_root() / "artifacts" / "validation" / "bridge_readiness"


def _default_agent_root() -> Path:
    return monorepo_root()


def _repo_root() -> Path:
    return package_root()


if __name__ == "__main__":
    raise SystemExit(main())
