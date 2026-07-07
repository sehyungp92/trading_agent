"""Native monthly runner CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trading_assistant_backtest.artifact_writer import ArtifactWriter
from trading_assistant_backtest.contract_models import (
    DECISION_PARITY_DIMENSIONS,
    DataBundleManifest,
    DecisionParityCheck,
    DecisionParityReport,
    DecisionParityStatus,
    MonthlyRunManifest,
)
from trading_assistant_backtest.replay.parity import insufficient_decision_parity_report
from trading_assistant_backtest.strategies.contracts import (
    load_strategy_plugin_contract,
    strategy_plugin_errors,
)


def write_structural_placeholders(
    writer: ArtifactWriter,
    manifest: MonthlyRunManifest,
    bundle: DataBundleManifest | None,
) -> None:
    patch_artifacts = [
        "live_repo_patch.diff",
        "backtest_adapter_patch.diff",
        "config_patch.diff",
    ]
    writer.write_json(
        "structural_candidate_plan.json",
        {
            "schema_version": "structural_candidate_plan_v1",
            "run_id": manifest.run_id,
            "status": "blocked",
            "reason": "no structural candidate adopted by deterministic runner",
            "selection_gate_path": str(writer.path("structural_selection_gate.json")),
            "required_patch_artifacts": patch_artifacts,
        },
    )
    writer.write_text(
        "live_repo_patch.diff",
        "# No live repo patch generated for deterministic no-adoption run.\n",
    )
    writer.write_text(
        "backtest_adapter_patch.diff",
        "# No backtest adapter patch generated for deterministic no-adoption run.\n",
    )
    writer.write_text(
        "config_patch.diff",
        "# No config patch generated for deterministic no-adoption run.\n",
    )
    parity_report = _structural_decision_parity_report(writer, manifest, bundle)
    writer.write_json("decision_parity_report.json", parity_report)
    writer.write_json(
        "structural_selection_gate.json",
        _structural_selection_gate_payload(
            writer,
            manifest,
            parity_report=parity_report,
            patch_artifacts=patch_artifacts,
        ),
    )


def _structural_selection_gate_payload(
    writer: ArtifactWriter,
    manifest: MonthlyRunManifest,
    *,
    parity_report: DecisionParityReport,
    patch_artifacts: list[str],
) -> dict[str, Any]:
    patch_checks = [
        _structural_patch_check(writer.path(artifact_name), artifact_name=artifact_name)
        for artifact_name in patch_artifacts
    ]
    parity_passed = bool(parity_report.eligible_for_structural_approval)
    blocking_reasons = [
        f"{item['artifact_name']} is not a real patch artifact"
        for item in patch_checks
        if not item["usable_for_structural_selection"]
    ]
    if not parity_passed:
        blocking_reasons.append(
            f"decision parity report status is {parity_report.status.value}"
        )
    selection_allowed = not blocking_reasons
    return {
        "schema_version": "structural_selection_gate_v1",
        "run_id": manifest.run_id,
        "manifest_id": manifest.manifest_id,
        "candidate_id": (
            parity_report.candidate_id
            if parity_report.candidate_id not in {"", "no-adoption"}
            else ""
        ),
        "status": "selection_ready" if selection_allowed else "blocked",
        "selection_allowed": selection_allowed,
        "change_kind": "structural_change",
        "patch_checks": patch_checks,
        "decision_parity": {
            "report_path": str(writer.path("decision_parity_report.json")),
            "status": parity_report.status.value,
            "eligible_for_structural_approval": parity_passed,
            "strategy_plugin_id": parity_report.strategy_plugin_id,
            "live_repo_commit_sha": parity_report.live_repo_commit_sha,
            "backtest_adapter_commit_sha": parity_report.backtest_adapter_commit_sha,
            "evidence_paths": parity_report.evidence_paths,
        },
        "blocking_reasons": blocking_reasons,
        "required_before_selection": [
            "live_repo_patch.diff",
            "backtest_adapter_patch.diff",
            "config_patch.diff",
            "decision_parity_report.json:pass",
        ],
        "evidence_paths": [
            str(writer.path("structural_candidate_plan.json")),
            str(writer.path("decision_parity_report.json")),
            *[str(writer.path(name)) for name in patch_artifacts],
        ],
    }


def _structural_patch_check(path: Path, *, artifact_name: str) -> dict[str, Any]:
    exists = path.exists()
    text = path.read_text(encoding="utf-8") if exists else ""
    stripped = text.strip()
    has_diff_marker = (
        stripped.startswith(("diff --git", "--- ", "+++ ", "@@ "))
        or any(marker in stripped for marker in ("\ndiff --git", "\n--- ", "\n+++ ", "\n@@ "))
    )
    usable = bool(stripped) and not stripped.startswith("# No ") and has_diff_marker
    return {
        "artifact_name": artifact_name,
        "path": str(path),
        "exists": exists,
        "non_empty": bool(stripped),
        "placeholder": stripped.startswith("# No "),
        "has_diff_marker": has_diff_marker,
        "usable_for_structural_selection": usable,
    }


def _structural_decision_parity_report(
    writer: ArtifactWriter,
    manifest: MonthlyRunManifest,
    bundle: DataBundleManifest | None,
) -> DecisionParityReport:
    fallback = insufficient_decision_parity_report(
        manifest,
        candidate_id="no-adoption",
        evidence_path=writer.path("end_of_round_diagnostics.json"),
    )
    if not manifest.strategy_plugin_contract_path:
        return fallback
    contract, errors = load_strategy_plugin_contract(manifest.strategy_plugin_contract_path)
    if errors or contract is None:
        return _failed_decision_parity_report(
            manifest,
            candidate_id="strategy-plugin-contract",
            errors=errors or ["strategy plugin contract is unavailable"],
            evidence_paths=[manifest.strategy_plugin_contract_path],
        )
    validation_errors = strategy_plugin_errors(manifest, bundle)
    if validation_errors:
        return _failed_decision_parity_report(
            manifest,
            candidate_id="strategy-plugin-contract",
            errors=validation_errors,
            evidence_paths=[manifest.strategy_plugin_contract_path],
        )
    from trading_assistant_backtest.monthly_execution.structural_registry import (
        STRUCTURAL_PARITY_BUILDERS,
    )

    wired = STRUCTURAL_PARITY_BUILDERS.get(contract.plugin_id)
    if wired is None:
        return fallback
    expected_api_version, builder = wired
    if contract.decision_api_version != expected_api_version:
        return _failed_decision_parity_report(
            manifest,
            candidate_id="strategy-plugin-contract",
            errors=[
                "strategy plugin contract decision_api_version does not match "
                "the wired decision API"
            ],
            evidence_paths=[manifest.strategy_plugin_contract_path],
        )
    if not contract.parity_fixture_set:
        return _failed_decision_parity_report(
            manifest,
            candidate_id="strategy-plugin-contract",
            errors=["strategy plugin contract has no parity_fixture_set"],
            evidence_paths=[manifest.strategy_plugin_contract_path],
        )
    try:
        return builder(
            manifest,
            candidate_id="strategy-plugin-contract",
            fixture_paths=contract.parity_fixture_set,
            live_repo_path=contract.live_repo_path,
            live_repo_commit_sha=contract.live_repo_commit_sha,
            backtest_adapter_commit_sha=contract.backtest_adapter_commit_sha,
        )
    except Exception as exc:
        return _failed_decision_parity_report(
            manifest,
            candidate_id="strategy-plugin-contract",
            errors=[f"strategy plugin decision parity failed: {exc}"],
            evidence_paths=[manifest.strategy_plugin_contract_path, *contract.parity_fixture_set],
        )


def _failed_decision_parity_report(
    manifest: MonthlyRunManifest,
    *,
    candidate_id: str,
    errors: list[str],
    evidence_paths: list[str],
) -> DecisionParityReport:
    notes = "; ".join(errors)
    return DecisionParityReport(
        run_id=manifest.run_id,
        candidate_id=candidate_id,
        strategy_plugin_id=manifest.strategy_plugin_id,
        live_repo_commit_sha=manifest.trading_repo_commit_sha,
        backtest_adapter_commit_sha=manifest.backtest_repo_commit_sha,
        status=DecisionParityStatus.FAIL,
        evidence_paths=evidence_paths,
        checks=[
            DecisionParityCheck(
                dimension=dimension,
                status=DecisionParityStatus.FAIL,
                match_rate=0.0,
                mismatch_count=1,
                notes=notes,
                evidence_paths=evidence_paths,
            )
            for dimension in sorted(DECISION_PARITY_DIMENSIONS)
        ],
    )
