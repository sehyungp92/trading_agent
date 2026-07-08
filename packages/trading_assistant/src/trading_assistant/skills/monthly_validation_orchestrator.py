"""Shadow monthly validation orchestration."""
from __future__ import annotations

import json
from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_assistant.analysis.monthly_validation_report_builder import MonthlyValidationReportBuilder
from trading_assistant.orchestrator.backtest_invocation import backtest_repo_commit_sha, validate_backtest_repo
from trading_assistant.orchestrator.learning_sufficiency_audit import LearningSufficiencyAuditor
from trading_assistant.orchestrator.lineage_audit import LineageAuditor
from trading_assistant.paths import monorepo_root, package_root
from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex, REQUIRED_BACKTEST_ARTIFACTS
from trading_assistant.schemas.data_bundle_manifest import DataBundleManifest, DataBundleSlice, DataBundleStatus
from trading_assistant.schemas.market_data_manifest import MarketDataManifest
from trading_assistant.schemas.monthly_outcome import OutcomeSource
from trading_assistant.schemas.monthly_run_manifest import MonthlyApprovalMode, MonthlyRunManifest, MonthlyRunMode
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.schemas.monthly_optimizer import OptimizerSequenceStatus
from trading_assistant.schemas.replay_parity import ReplayParityReport, ReplayParityStatus
from trading_assistant.schemas.strategy_plugin_contract import StrategyPluginContract
from trading_assistant.schemas.strategy_change_ledger import StrategyChangeRecordType
from trading_assistant.schemas.telemetry_manifest import TelemetryEligibility, TelemetryManifest
from trading_assistant.skills.backtest_runner_client import BacktestRunnerClient
from trading_assistant.skills.monthly_candidate_pipeline import MonthlyCandidatePipeline
from trading_assistant.skills.monthly_gap_attribution import MonthlyGapAttributor
from trading_assistant.skills.monthly_artifact_contract import MonthlyArtifactContract
from trading_assistant.skills.monthly_model_review_runner import MonthlyModelReviewInvoker, MonthlyModelReviewRunner
from trading_assistant.skills.monthly_optimizer_runner import MonthlyOptimizerRunner
from trading_assistant.skills.monthly_outcome_measurer import MonthlyOutcomeMeasurer
from trading_assistant.skills.monthly_repair_planner import MonthlyRepairPlanner
from trading_assistant.skills.monthly_search_brief_builder import MonthlySearchBriefBuilder
from trading_assistant.skills.outcome_prior_store import OutcomePriorStore
from trading_assistant.skills.proposal_ledger import ProposalLedger
from trading_assistant.skills.replay_parity_checker import ReplayParityChecker
from trading_assistant.skills.scheduled_shadow_report import (
    write_scheduled_shadow_cycle_report,
)
from trading_assistant.skills.strategy_discovery_packet_builder import StrategyDiscoveryPacketBuilder
from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger


@dataclass(frozen=True)
class MonthlyValidationRequest:
    bot_id: str
    strategy_id: str
    run_month: str = ""
    strategy_version: str = ""
    config_version: str = ""
    config_hash: str = ""
    deployment_id: str = ""
    parameter_set_id: str = ""
    market_data_manifest_path: Path | None = None
    data_bundle_manifest_path: Path | None = None
    data_bundle_checksum: str = ""
    telemetry_manifest_path: Path | None = None
    backtest_command: list[str] | None = None
    optimizer_sequence_enabled: bool = True
    in_sample_start: date | None = None
    in_sample_end: date | None = None
    strategy_plugin_id: str = ""
    strategy_plugin_contract_path: Path | None = None
    round_id: str = ""
    prior_round_id: str = ""
    next_round_id: str = ""
    round_n_strategy_config_path: str = ""
    round_n_strategy_config_version: str = ""
    round_n_portfolio_config_path: str = ""
    round_n_portfolio_config_version: str = ""
    trading_repo_path: str = ""
    trading_repo_branch: str = ""
    trading_repo_commit_sha: str = ""
    deployment_metadata_path: Path | None = None
    deployment_metadata_install_report_paths: list[Path] | None = None
    operational_evidence_path: Path | None = None
    relay_ingest_evidence_path: Path | None = None
    vps_host_id: str = ""
    assistant_host_id: str = "local"
    expected_session_paths: list[Path] | None = None
    runtime_support_paths: list[Path] | None = None
    workflow_contract_path: str = ""
    workflow_contract_version: str = ""
    max_workers: int = 2
    shadow: bool = True
    approval_evidence_mode: bool = False


class MonthlyValidationOrchestrator:
    """Runs monthly validation and optional optimizer replay with fail-closed gates."""

    def __init__(
        self,
        *,
        curated_dir: Path,
        findings_dir: Path,
        market_data_root: Path,
        backtest_repo_path: Path,
        backtest_artifact_root: Path,
        required_market_coverage_ratio: float = 0.95,
        required_lineage_ratio: float = 0.95,
        timeout_seconds: int = 3600,
        proposal_ledger: object | None = None,
        approval_tracker: object | None = None,
        model_review_invoker: MonthlyModelReviewInvoker | None = None,
    ) -> None:
        self.curated_dir = Path(curated_dir)
        self.findings_dir = Path(findings_dir)
        self.market_data_root = Path(market_data_root)
        backtest_repo_raw = str(backtest_repo_path).strip()
        self.backtest_repo_path = Path(backtest_repo_raw) if backtest_repo_raw else None
        self.backtest_artifact_root = Path(backtest_artifact_root)
        self.required_market_coverage_ratio = required_market_coverage_ratio
        self.required_lineage_ratio = required_lineage_ratio
        self.runner = BacktestRunnerClient(timeout_seconds=timeout_seconds)
        self.optimizer_runner = MonthlyOptimizerRunner(self.runner)
        self.ledger = StrategyChangeLedger(self.findings_dir)
        self.outcome_prior_store = OutcomePriorStore(self.findings_dir)
        self.proposal_ledger = proposal_ledger or ProposalLedger(self.findings_dir)
        self.outcome_measurer = MonthlyOutcomeMeasurer(
            self.findings_dir,
            proposal_ledger=self.proposal_ledger,
            strategy_change_ledger=self.ledger,
            prior_store=self.outcome_prior_store,
        )
        self.auditor = LineageAuditor(
            self.curated_dir,
            self.findings_dir,
            required_lineage_ratio=required_lineage_ratio,
            proposal_ledger=self.proposal_ledger,
        )
        self.learning_sufficiency_auditor = LearningSufficiencyAuditor(
            self.curated_dir,
            self.findings_dir,
            required_lineage_ratio=required_lineage_ratio,
            lineage_auditor=self.auditor,
        )
        self.strategy_discovery_builder = StrategyDiscoveryPacketBuilder(self.curated_dir)
        self.attributor = MonthlyGapAttributor()
        self.report_builder = MonthlyValidationReportBuilder()
        self.repair_planner = MonthlyRepairPlanner()
        self.model_review_runner = MonthlyModelReviewRunner(invoker=model_review_invoker)
        self.candidate_pipeline = MonthlyCandidatePipeline(
            approval_tracker=approval_tracker,
            proposal_ledger=self.proposal_ledger,
            strategy_change_ledger=self.ledger,
            outcome_prior_store=self.outcome_prior_store,
        )

    def run_shadow(self, request: MonthlyValidationRequest) -> MonthlyValidationResult:
        return self.run(request if request.shadow else replace(request, shadow=True))

    def run(self, request: MonthlyValidationRequest) -> MonthlyValidationResult:
        run_month = request.run_month or latest_completed_month()
        window_start, window_end = month_window(run_month)
        artifact_base = _approval_evidence_artifact_base(
            self.backtest_artifact_root,
            request,
        )
        artifact_root = artifact_base / request.bot_id / run_month / request.strategy_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        run_id = f"monthly-{request.bot_id}-{request.strategy_id}-{run_month}"
        monthly_review_created_at = datetime.now(timezone.utc)

        telemetry_path = request.telemetry_manifest_path or artifact_root / "telemetry_manifest.json"
        telemetry = self._ensure_telemetry_manifest(
            request=request,
            run_month=run_month,
            window_start=window_start,
            window_end=window_end,
            output_path=telemetry_path,
        )
        learning_sufficiency_path = artifact_root / "learning_sufficiency_manifest.json"
        learning_sufficiency = self.learning_sufficiency_auditor.build_manifest(
            bot_id=request.bot_id,
            strategy_id=request.strategy_id,
            run_month=run_month,
            window_start=window_start,
            window_end=window_end,
            telemetry_manifest_path=telemetry_path,
            deployment_metadata_paths=(
                [request.deployment_metadata_path]
                if request.deployment_metadata_path is not None
                else []
            ),
            strategy_contract_path=request.strategy_plugin_contract_path,
            expected_session_paths=request.expected_session_paths or [],
            runtime_support_paths=request.runtime_support_paths or [],
        )
        learning_sufficiency.artifact_paths["learning_sufficiency_manifest"] = str(learning_sufficiency_path)
        strategy_discovery_packet_path = artifact_root / "strategy_discovery_packet.json"
        strategy_discovery_packet = self.strategy_discovery_builder.build(
            run_id=run_id,
            run_month=run_month,
            bot_id=request.bot_id,
            strategy_id=request.strategy_id,
            window_start=window_start,
            window_end=window_end,
            learning_sufficiency_manifest_path=learning_sufficiency_path,
        )
        if strategy_discovery_packet_path.exists():
            strategy_discovery_packet_path.unlink()
        strategy_discovery_packet_path_value = (
            str(strategy_discovery_packet_path)
            if self.strategy_discovery_builder.has_material_clusters(strategy_discovery_packet)
            else ""
        )
        market_data_path = request.market_data_manifest_path or self._default_market_manifest_path(
            request.bot_id,
            request.strategy_id,
            run_month,
        )
        requested_data_bundle_path = request.data_bundle_manifest_path
        data_bundle = (
            self._load_data_bundle_manifest(requested_data_bundle_path)
            if requested_data_bundle_path is not None
            else None
        )
        coverage = (
            self._load_market_manifest(market_data_path)
            if requested_data_bundle_path is None or request.market_data_manifest_path is not None
            else None
        )
        stage_status: dict[str, bool | str] = {
            "manifest_written": False,
            "runner_started": False,
            "artifact_index_validated": False,
            "optimizer_contract_validated": False,
            "parity_loaded": False,
            "model_review_requested": False,
            "candidate_gates_evaluated": False,
            "approval_created": False,
            "approval_suppressed": False,
            "repair_required": False,
        }

        blocking_reasons: list[str] = []
        data_blocking_reasons: list[str] = []
        if telemetry.authoritative_eligibility == TelemetryEligibility.INSUFFICIENT_LINEAGE:
            blocking_reasons.append("telemetry lineage below required threshold")
        elif telemetry.authoritative_eligibility == TelemetryEligibility.INSUFFICIENT_DATA:
            blocking_reasons.append("telemetry has insufficient events")
        if requested_data_bundle_path is not None:
            if data_bundle is None:
                reason = f"data bundle manifest missing or malformed: {requested_data_bundle_path}"
                data_blocking_reasons.append(reason)
                blocking_reasons.append(reason)
            else:
                if request.data_bundle_checksum and request.data_bundle_checksum != data_bundle.bundle_checksum:
                    reason = "data bundle checksum does not match request"
                    data_blocking_reasons.append(reason)
                    blocking_reasons.append(reason)
                if not data_bundle.usable_for_authoritative_validation:
                    reason = f"data bundle is not authoritative: {data_bundle.status.value}"
                    if data_bundle.diagnostics_only_reason:
                        reason += f" ({data_bundle.diagnostics_only_reason})"
                    data_blocking_reasons.append(reason)
                    blocking_reasons.append(reason)
        else:
            if coverage is None:
                reason = f"market data manifest missing or malformed: {market_data_path}"
                data_blocking_reasons.append(reason)
                blocking_reasons.append(reason)
            elif not coverage.usable_for_authoritative_validation:
                reasons = coverage.blocking_reasons or ["market data manifest is not authoritative"]
                data_blocking_reasons.extend(reasons)
                blocking_reasons.extend(reasons)
            elif coverage.coverage_ratio < self.required_market_coverage_ratio:
                reason = "market data coverage below required threshold"
                data_blocking_reasons.append(reason)
                blocking_reasons.append(reason)

        strategy_plugin_contract = self._load_strategy_plugin_contract(
            request.strategy_plugin_contract_path
        )
        if request.optimizer_sequence_enabled:
            if strategy_plugin_contract is None:
                blocking_reasons.append("strategy plugin contract missing or malformed for optimizer run")
            elif not strategy_plugin_contract.eligible_for_optimizer:
                blocking_reasons.append(
                    "strategy plugin contract is not mature enough for optimizer run: "
                    f"{strategy_plugin_contract.maturity.value}"
                )
            elif data_bundle is not None:
                blocking_reasons.extend(
                    self._strategy_plugin_bundle_errors(strategy_plugin_contract, data_bundle)
                )
            elif coverage is not None:
                if (
                    strategy_plugin_contract.supported_symbols
                    and coverage.symbol not in strategy_plugin_contract.supported_symbols
                ):
                    blocking_reasons.append("strategy plugin contract does not support market-data symbol")
                if (
                    strategy_plugin_contract.supported_timeframes
                    and coverage.timeframe not in strategy_plugin_contract.supported_timeframes
                ):
                    blocking_reasons.append("strategy plugin contract does not support market-data timeframe")

        backtest_repo_path = str(self.backtest_repo_path) if self.backtest_repo_path else ""
        backtest_repo_ok, _ = validate_backtest_repo(backtest_repo_path)
        backtest_commit_sha = backtest_repo_commit_sha(backtest_repo_path) if backtest_repo_ok else ""
        outcome_prior_snapshot_path = artifact_root / "outcome_priors_snapshot.json"
        outcome_prior_snapshot_path.write_text(
            json.dumps(
                self.outcome_prior_store.snapshot(
                    bot_id=request.bot_id,
                    strategy_id=request.strategy_id,
                ),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        monthly_search_brief_path: Path | None = artifact_root / "monthly_search_brief.json"
        monthly_search_brief_id = ""
        source_weekly_signal_ids: list[str] = []
        monthly_search_guidance: dict = {}
        try:
            search_brief_builder = MonthlySearchBriefBuilder(self.findings_dir)
            monthly_search_brief = search_brief_builder.build(
                run_month=run_month,
                bot_id=request.bot_id,
                strategy_id=request.strategy_id,
                report_only=True,
            )
            search_brief_builder.write(
                monthly_search_brief,
                monthly_search_brief_path,
            )
            monthly_search_brief_id = monthly_search_brief.monthly_search_brief_id
            source_weekly_signal_ids = monthly_search_brief.source_weekly_signal_ids
            monthly_search_guidance = monthly_search_brief.to_optimizer_guidance()
        except Exception:
            monthly_search_brief_path = None

        runner_market_data_path = requested_data_bundle_path or market_data_path
        data_bundle_manifest_path = str(requested_data_bundle_path) if requested_data_bundle_path else ""
        data_bundle_checksum = (
            data_bundle.bundle_checksum
            if data_bundle is not None
            else (request.data_bundle_checksum if requested_data_bundle_path else "")
        )
        if request.optimizer_sequence_enabled and requested_data_bundle_path is None and coverage is not None:
            data_bundle = self._write_data_bundle_manifest(
                coverage=coverage,
                coverage_path=market_data_path,
                output_path=artifact_root / "data_bundle_manifest.json",
            )
            runner_market_data_path = artifact_root / "data_bundle_manifest.json"
            data_bundle_manifest_path = str(runner_market_data_path)
            data_bundle_checksum = data_bundle.bundle_checksum
            if not data_bundle.usable_for_authoritative_validation:
                reason = f"data bundle is not authoritative: {data_bundle.status.value}"
                if data_bundle.diagnostics_only_reason:
                    reason += f" ({data_bundle.diagnostics_only_reason})"
                data_blocking_reasons.append(reason)
                blocking_reasons.append(reason)

        default_in_sample_start = request.in_sample_start
        default_in_sample_end = request.in_sample_end
        data_manifest_checksum = ""
        if data_bundle is not None:
            data_manifest_checksum = data_bundle.bundle_checksum
            if request.optimizer_sequence_enabled:
                (
                    default_in_sample_start,
                    default_in_sample_end,
                    in_sample_errors,
                ) = self._data_bundle_in_sample_window(
                    data_bundle,
                    window_start=window_start,
                    window_end=window_end,
                    requested_start=default_in_sample_start,
                    requested_end=default_in_sample_end,
                )
                data_blocking_reasons.extend(in_sample_errors)
                blocking_reasons.extend(in_sample_errors)
        elif coverage is not None:
            data_manifest_checksum = data_bundle_checksum or coverage.checksum
            if request.optimizer_sequence_enabled:
                candidate_in_sample_start = default_in_sample_start or coverage.start_ts.date()
                candidate_in_sample_end = default_in_sample_end or (window_start - timedelta(days=1))
                if coverage.start_ts.date() > window_start or coverage.end_ts.date() < window_end:
                    reason = "market data does not include latest-month selection coverage"
                    data_blocking_reasons.append(reason)
                    blocking_reasons.append(reason)
                if candidate_in_sample_end < candidate_in_sample_start:
                    reason = "market data does not include pre-latest-month in-sample coverage"
                    data_blocking_reasons.append(reason)
                    blocking_reasons.append(reason)
                else:
                    default_in_sample_start = candidate_in_sample_start
                    default_in_sample_end = candidate_in_sample_end

        manifest = MonthlyRunManifest(
            run_id=run_id,
            run_month=run_month,
            mode=(
                MonthlyRunMode.PHASED_AUTO
                if request.optimizer_sequence_enabled
                else MonthlyRunMode.INCUMBENT_VALIDATION
            ),
            bot_id=request.bot_id,
            strategy_id=request.strategy_id,
            strategy_version=request.strategy_version,
            config_version=request.config_version,
            config_hash=request.config_hash,
            deployment_id=request.deployment_id,
            parameter_set_id=request.parameter_set_id,
            latest_month_start=window_start,
            latest_month_end=window_end,
            market_data_manifest_path=str(runner_market_data_path),
            telemetry_manifest_path=str(telemetry_path),
            learning_sufficiency_manifest_path=str(learning_sufficiency_path),
            learning_sufficiency_status=learning_sufficiency.eligibility.value,
            supported_learning_capabilities=learning_sufficiency.supported_learning_capabilities,
            blocked_learning_capabilities=learning_sufficiency.blocked_learning_capabilities,
            strategy_discovery_packet_path=strategy_discovery_packet_path_value,
            backtest_repo_path=backtest_repo_path,
            backtest_repo_commit_sha=backtest_commit_sha,
            control_plane_commit_sha=backtest_repo_commit_sha(package_root()),
            trading_repo_path=request.trading_repo_path,
            trading_repo_branch=request.trading_repo_branch,
            trading_repo_commit_sha=request.trading_repo_commit_sha,
            deployment_metadata_path=(
                str(request.deployment_metadata_path)
                if request.deployment_metadata_path else ""
            ),
            expected_session_paths=[str(path) for path in (request.expected_session_paths or [])],
            runtime_support_paths=[str(path) for path in (request.runtime_support_paths or [])],
            backtest_command=request.backtest_command or [],
            artifact_root=str(artifact_root),
            strategy_plugin_id=(
                request.strategy_plugin_id
                or (strategy_plugin_contract.plugin_id if strategy_plugin_contract is not None else "")
            ),
            strategy_plugin_contract_path=(
                str(request.strategy_plugin_contract_path)
                if request.strategy_plugin_contract_path else ""
            ),
            strategy_plugin_contract_version=(
                strategy_plugin_contract.contract_version
                if strategy_plugin_contract is not None else ""
            ),
            round_id=request.round_id,
            prior_round_id=request.prior_round_id,
            next_round_id=request.next_round_id,
            round_n_strategy_config_path=request.round_n_strategy_config_path,
            round_n_strategy_config_version=request.round_n_strategy_config_version,
            round_n_portfolio_config_path=request.round_n_portfolio_config_path,
            round_n_portfolio_config_version=request.round_n_portfolio_config_version,
            data_manifest_checksum=data_manifest_checksum,
            data_bundle_manifest_path=data_bundle_manifest_path,
            data_bundle_checksum=data_bundle_checksum,
            in_sample_start=default_in_sample_start,
            in_sample_end=default_in_sample_end,
            outcome_prior_snapshot_path=str(outcome_prior_snapshot_path),
            monthly_search_brief_path=str(monthly_search_brief_path) if monthly_search_brief_path else "",
            monthly_search_brief_id=monthly_search_brief_id,
            source_weekly_signal_ids=source_weekly_signal_ids,
            monthly_search_guidance=monthly_search_guidance,
            workflow_contract_path=request.workflow_contract_path,
            workflow_contract_version=request.workflow_contract_version or "monthly_incumbent_validation_v1",
            max_workers=max(1, request.max_workers),
            approval_mode=MonthlyApprovalMode.NONE if request.shadow else MonthlyApprovalMode.MANUAL_REQUIRED,
            approval_evidence_mode=bool(request.approval_evidence_mode),
            expected_outputs=list(REQUIRED_BACKTEST_ARTIFACTS),
        )
        if request.optimizer_sequence_enabled:
            manifest = self.optimizer_runner.prepare_manifest(manifest)
        manifest_path = artifact_root / "run_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        learning_sufficiency_path.write_text(
            learning_sufficiency.model_dump_json(indent=2),
            encoding="utf-8",
        )
        stage_status["manifest_written"] = True

        artifact_index: BacktestArtifactIndex | None = None
        parity_report: ReplayParityReport | None = None
        runner_error = ""
        optimizer_sequence_status = ""
        optimizer_sequence_result_path = ""
        adopted_candidate_id = ""
        optimizer_no_adoption_reason = ""
        if not blocking_reasons:
            stage_status["runner_started"] = True
            runner_result = self.runner.run(manifest, manifest_path)
            if not runner_result.success:
                runner_error = runner_result.error or "unknown error"
                blocking_reasons.append(
                    f"backtest runner failed: {runner_error}"
                )
            artifact_index = runner_result.artifact_index
            stage_status["artifact_index_validated"] = bool(runner_result.success and artifact_index is not None)
            if artifact_index is not None:
                parity_report = self._load_parity_report(artifact_index)
                stage_status["parity_loaded"] = parity_report is not None
                if parity_report is None:
                    blocking_reasons.append("replay parity report missing or malformed")
                elif not parity_report.eligible_for_authoritative_validation:
                    blocking_reasons.append(f"replay parity status is {parity_report.status.value}")
                if request.optimizer_sequence_enabled:
                    optimizer_sequence_result = self.optimizer_runner.validate_artifacts(
                        manifest,
                        artifact_index,
                        manifest_path=manifest_path,
                    )
                    optimizer_sequence_status = optimizer_sequence_result.status.value
                    adopted_candidate_id = optimizer_sequence_result.adopted_candidate_id
                    optimizer_no_adoption_reason = optimizer_sequence_result.no_adoption_reason
                    optimizer_sequence_result_path = str(artifact_root / "optimizer_sequence_result.json")
                    Path(optimizer_sequence_result_path).write_text(
                        optimizer_sequence_result.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    if optimizer_sequence_result.status in {
                        OptimizerSequenceStatus.BLOCKED,
                        OptimizerSequenceStatus.FAILED,
                    }:
                        blocking_reasons.extend(optimizer_sequence_result.blocking_reasons)
                    else:
                        stage_status["optimizer_contract_validated"] = True
        else:
            artifact_index = BacktestArtifactIndex(
                run_id=run_id,
                manifest_id=manifest.manifest_id,
                artifact_root=str(artifact_root),
            )
            (artifact_root / "artifact_index.json").write_text(
                artifact_index.model_dump_json(indent=2),
                encoding="utf-8",
            )

        artifact_contract = MonthlyArtifactContract.from_index(artifact_index) if artifact_index is not None else None
        written_strategy_discovery_path = self.strategy_discovery_builder.write(strategy_discovery_packet, artifact_root)
        if written_strategy_discovery_path is not None:
            strategy_discovery_packet_path_value = str(written_strategy_discovery_path)

        gap = self.attributor.attribute(
            coverage_blocking_reasons=data_blocking_reasons,
            telemetry_known_gaps=telemetry.known_gaps if telemetry.authoritative_eligibility != TelemetryEligibility.AUTHORITATIVE else [],
            parity_report=parity_report,
        )
        status = self._status_from_evidence(blocking_reasons, telemetry, parity_report)
        if not blocking_reasons and artifact_index is not None:
            status = self._status_from_mode_decision(artifact_index) or status
            if _status_expects_candidates(status) and _selected_candidates_empty(artifact_index):
                blocking_reasons.append(
                    f"selected_candidates artifact contains no candidates for {status.value} monthly status"
                )
        evidence_paths = [
            str(manifest_path),
            str(telemetry_path),
            str(learning_sufficiency_path),
            str(runner_market_data_path),
            str(outcome_prior_snapshot_path),
        ]
        if strategy_discovery_packet_path_value:
            evidence_paths.append(strategy_discovery_packet_path_value)
        if data_bundle_manifest_path:
            evidence_paths.append(data_bundle_manifest_path)
        if request.strategy_plugin_contract_path:
            evidence_paths.append(str(request.strategy_plugin_contract_path))
        evidence_paths.extend(str(path) for path in (request.expected_session_paths or []))
        evidence_paths.extend(str(path) for path in (request.runtime_support_paths or []))
        if monthly_search_brief_path:
            evidence_paths.append(str(monthly_search_brief_path))
        if artifact_index is not None:
            evidence_paths.append(str(artifact_root / "artifact_index.json"))
        if parity_report is not None:
            evidence_paths.extend(parity_report.evidence_paths)
        if optimizer_sequence_result_path:
            evidence_paths.append(optimizer_sequence_result_path)

        result = MonthlyValidationResult(
            run_id=run_id,
            run_month=run_month,
            bot_id=request.bot_id,
            strategy_id=request.strategy_id,
            status=status,
            telemetry_manifest_path=str(telemetry_path),
            learning_sufficiency_manifest_path=str(learning_sufficiency_path),
            learning_sufficiency_status=learning_sufficiency.eligibility.value,
            supported_learning_capabilities=learning_sufficiency.supported_learning_capabilities,
            blocked_learning_capabilities=learning_sufficiency.blocked_learning_capabilities,
            learning_sufficiency_gate_paths=[str(learning_sufficiency_path)],
            learning_sufficiency_blocking_reasons=[
                gap.remediation for gap in learning_sufficiency.known_gaps[:10]
            ],
            strategy_discovery_packet_path=strategy_discovery_packet_path_value,
            market_data_manifest_path=str(runner_market_data_path),
            run_manifest_path=str(manifest_path),
            artifact_index_path=str(artifact_root / "artifact_index.json"),
            replay_parity_path=(
                artifact_contract.path_str("replay_parity_report.json", require_exists=False)
                if artifact_contract is not None else ""
            ),
            gap_attribution=gap,
            blocking_reasons=blocking_reasons,
            evidence_paths=dedupe(evidence_paths),
            optimizer_sequence_result_path=optimizer_sequence_result_path,
            optimizer_sequence_status=optimizer_sequence_status,
            adopted_candidate_id=adopted_candidate_id,
            optimizer_no_adoption_reason=optimizer_no_adoption_reason,
            shadow=request.shadow,
        )
        if blocking_reasons:
            repair = self.repair_planner.build(
                run_id=run_id,
                bot_id=request.bot_id,
                strategy_id=request.strategy_id,
                run_month=run_month,
                blocking_reasons=blocking_reasons,
                artifact_root=artifact_root,
                artifact_index=artifact_index,
                runner_error=runner_error,
                evidence_paths=result.evidence_paths,
            )
            repair_path = self.repair_planner.write(repair, artifact_root)
            result.repair_required = True
            result.repair_request_path = str(repair_path)
            result.evidence_paths = dedupe([*result.evidence_paths, str(repair_path)])
            stage_status["repair_required"] = True
        result_path = artifact_root / "monthly_validation_result.json"
        result.monthly_report_path = str(artifact_root / "monthly_report.md")
        planned_record = self.ledger.build_monthly_review_record(
            bot_id=request.bot_id,
            strategy_id=request.strategy_id,
            run_id=run_id,
            run_month=run_month,
            monthly_status=status.value,
            evidence_paths=[],
            decision_reason=_monthly_review_decision_reason(result),
            created_at=monthly_review_created_at,
        )
        result.strategy_change_record_id = planned_record.record_id
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        if artifact_index is not None and not blocking_reasons:
            model_review_path = artifact_root / "model_review.json"
            stage_status["model_review_requested"] = True
            model_review_result = self.model_review_runner.run(
                monthly_result=result,
                artifact_index=artifact_index,
                artifact_root=artifact_root,
                existing_review_path=model_review_path,
            )
            result.model_review_provider = model_review_result.provider
            result.model_review_model = model_review_result.model
            result.model_review_runtime = model_review_result.runtime
            result.model_review_cost_usd = model_review_result.cost_usd
            if model_review_result.error or (
                model_review_result.skipped_reason == "no monthly model-review invoker configured"
            ):
                repair = self.repair_planner.build(
                    run_id=run_id,
                    bot_id=request.bot_id,
                    strategy_id=request.strategy_id,
                    run_month=run_month,
                    blocking_reasons=result.blocking_reasons,
                    artifact_root=artifact_root,
                    artifact_index=artifact_index,
                    model_review_error=model_review_result.error,
                    model_review_skipped_reason=model_review_result.skipped_reason,
                    evidence_paths=[
                        *result.evidence_paths,
                        model_review_result.request_path,
                        model_review_result.prompt_path,
                    ],
                )
                repair_path = self.repair_planner.write(repair, artifact_root)
                result.repair_required = True
                result.repair_request_path = str(repair_path)
                result.evidence_paths = dedupe([*result.evidence_paths, str(repair_path)])
                stage_status["repair_required"] = True
            if model_review_result.request_path:
                result.evidence_paths = dedupe([
                    *result.evidence_paths,
                    model_review_result.request_path,
                    model_review_result.prompt_path,
                ])
            if model_review_result.model_review_path:
                model_review_path = Path(model_review_result.model_review_path)
            result.model_review_path = str(model_review_path) if model_review_path.exists() else ""
            if not result.repair_required:
                candidate_result = self.candidate_pipeline.process(
                    monthly_result=result,
                    artifact_index=artifact_index,
                    coverage=coverage,
                    telemetry=telemetry,
                    parity_report=parity_report,
                    artifact_root=artifact_root,
                    monthly_result_path=result_path,
                    shadow=request.shadow,
                    model_review_path=str(model_review_path) if model_review_path.exists() else "",
                )
                result.candidate_summary_path = candidate_result.candidate_summary_path
                result.candidate_gate_report_path = candidate_result.gate_report_path
                result.approval_packet_paths = candidate_result.approval_packet_paths
                result.proposal_ids = candidate_result.proposal_ids
                result.approval_request_ids = candidate_result.approval_request_ids
                result.selected_candidate_count = len(candidate_result.selected_candidates)
                result.rejected_candidate_count = len(candidate_result.rejected_candidates)
                result.gate_passed_candidate_count = candidate_result.gate_passed_candidate_count
                result.approval_ready_candidate_count = candidate_result.approval_ready_candidate_count
                stage_status["candidate_gates_evaluated"] = True
                stage_status["approval_created"] = bool(candidate_result.approval_request_ids)
                stage_status["approval_suppressed"] = bool(
                    candidate_result.approval_packets
                    and not candidate_result.approval_request_ids
                )
                result.model_review_validation_path = candidate_result.model_review_validation_path
                result.monthly_evidence_verification_paths = candidate_result.evidence_verification_paths
                result.model_review_valid = candidate_result.model_review_valid
                result.model_review_issues = candidate_result.model_review_issues
                self._record_model_review_validation_evidence(
                    result=result,
                    provider=model_review_result.provider,
                    model=model_review_result.model,
                    runtime=model_review_result.runtime,
                )
                result.proposed_strategy_change_record_ids = [
                    packet.strategy_change_record_id
                    for packet in candidate_result.approval_packets
                    if packet.strategy_change_record_id
                ]
                result.evidence_paths = dedupe([
                    *result.evidence_paths,
                    candidate_result.candidate_summary_path,
                    candidate_result.gate_report_path,
                    candidate_result.model_review_validation_path,
                    *candidate_result.evidence_verification_paths,
                    *candidate_result.approval_packet_paths,
                ])

        observability_path = self._write_runner_observability_status(
            artifact_root=artifact_root,
            run_id=run_id,
            stage_status=stage_status,
        )
        scheduled_shadow_report_path = ""
        if request.approval_evidence_mode:
            scheduled_shadow_report_path = str(
                write_scheduled_shadow_cycle_report(
                    result=result,
                    monthly_validation_result_path=result_path,
                    deployment_metadata_install_report_paths=(
                        request.deployment_metadata_install_report_paths or []
                    ),
                    operational_evidence_path=(
                        request.operational_evidence_path
                        or monorepo_root() / "deployments" / "operational_evidence.json"
                    ),
                    relay_ingest_evidence_path=request.relay_ingest_evidence_path,
                    learning_sufficiency_manifest_path=learning_sufficiency_path,
                    optimizer_run_manifest_path=artifact_root / "optimizer_run_manifest.json",
                    approval_evidence_mode=True,
                    adoption_disabled=request.shadow and not bool(result.adopted_candidate_id),
                    scope_id=request.strategy_id,
                    bot_id=request.bot_id,
                    vps_host_id=request.vps_host_id,
                    assistant_host_id=request.assistant_host_id,
                )
            )
        final_evidence_paths = dedupe([
            *result.evidence_paths,
            str(observability_path),
            scheduled_shadow_report_path,
            str(result_path),
            result.monthly_report_path,
        ])
        result.evidence_paths = final_evidence_paths
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (artifact_root / "monthly_report.md").write_text(
            self.report_builder.build(result),
            encoding="utf-8",
        )
        record = self.ledger.record_monthly_review(
            bot_id=request.bot_id,
            strategy_id=request.strategy_id,
            run_id=run_id,
            run_month=run_month,
            monthly_status=status.value,
            evidence_paths=final_evidence_paths,
            decision_reason=_monthly_review_decision_reason(result),
            created_at=monthly_review_created_at,
        )
        if result.strategy_change_record_id != record.record_id:
            result.strategy_change_record_id = record.record_id
            result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            (artifact_root / "monthly_report.md").write_text(
                self.report_builder.build(result),
                encoding="utf-8",
            )
        self._record_deployed_change_outcomes(
            result=result,
            window_start=window_start,
            artifact_index=artifact_index,
        )
        return result

    def _record_deployed_change_outcomes(
        self,
        *,
        result: MonthlyValidationResult,
        window_start: date,
        artifact_index: BacktestArtifactIndex | None = None,
    ) -> None:
        """Close deployed changes whose first completed month is this run.

        Shadow runs remain diagnostic only. Approval-gated runs may write an
        inconclusive monthly outcome when data/parity is insufficient, which is
        still useful provenance but does not strengthen positive priors.
        """
        if result.shadow:
            return
        for record in self.ledger.get_for_strategy(result.bot_id, result.strategy_id, days=730):
            if record.record_type != StrategyChangeRecordType.DEPLOYED_CHANGE:
                continue
            if record.deployed_at is None:
                continue
            if record.deployed_at.date() >= window_start:
                continue
            if record.monthly_verdict and record.follow_up_verdict:
                continue
            source = OutcomeSource.MONTHLY
            if record.monthly_verdict:
                if not _follow_up_due(record.monthly_verdict, result.run_month):
                    continue
                source = OutcomeSource.FOLLOW_UP
            mutation_family, category = _mutation_family_and_category(record.mutation_diff)
            self.outcome_measurer.record_from_monthly_validation(
                result,
                strategy_change_record_id=record.record_id,
                deployment_id=record.deployment_id or "",
                config_version=record.new_config_version,
                strategy_version=record.strategy_version,
                commit_sha=record.commit_sha or "",
                proposal_ids=record.source_proposal_ids,
                suggestion_ids=record.source_suggestion_ids,
                mutation_family=mutation_family,
                category=category,
                objective_deltas=_extract_objective_deltas(artifact_index),
                minimum_trade_count_met=_minimum_trade_count_met(artifact_index),
                source=source,
                source_provider=result.model_review_provider,
                source_model=result.model_review_model,
            )

    def _record_model_review_validation_evidence(
        self,
        *,
        result: MonthlyValidationResult,
        provider: str = "",
        model: str = "",
        runtime: str = "",
    ) -> None:
        if not result.model_review_validation_path or not provider:
            return
        record = {
            "workflow": "monthly_model_review",
            "run_id": result.run_id,
            "run_month": result.run_month,
            "bot_id": result.bot_id,
            "strategy_id": result.strategy_id,
            "provider": provider,
            "model": model,
            "runtime": runtime,
            "valid": result.model_review_valid,
            "issues": [{"message": issue} for issue in result.model_review_issues],
            "model_review_path": result.model_review_path,
            "model_review_validation_path": result.model_review_validation_path,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.findings_dir / "monthly_model_review_validations.jsonl"
        key = (
            result.run_id,
            result.model_review_validation_path,
            provider,
            model,
        )
        existing = _read_jsonl(path)
        if any(
            (
                row.get("run_id"),
                row.get("model_review_validation_path"),
                row.get("provider"),
                row.get("model"),
            ) == key
            for row in existing
        ):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _ensure_telemetry_manifest(
        self,
        *,
        request: MonthlyValidationRequest,
        run_month: str,
        window_start: date,
        window_end: date,
        output_path: Path,
    ) -> TelemetryManifest:
        if request.telemetry_manifest_path and request.telemetry_manifest_path.exists():
            return TelemetryManifest.model_validate(json.loads(request.telemetry_manifest_path.read_text(encoding="utf-8")))
        return self.auditor.build_telemetry_manifest(
            bot_id=request.bot_id,
            strategy_id=request.strategy_id,
            run_month=run_month,
            window_start=window_start,
            window_end=window_end,
            output_path=output_path,
        )

    def _default_market_manifest_path(self, bot_id: str, strategy_id: str, run_month: str) -> Path:
        return self.market_data_root / "manifests" / bot_id / strategy_id / f"{run_month}.coverage_manifest.json"

    @staticmethod
    def _load_market_manifest(path: Path) -> MarketDataManifest | None:
        try:
            return MarketDataManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
        except Exception:
            return None

    @staticmethod
    def _load_data_bundle_manifest(path: Path) -> DataBundleManifest | None:
        try:
            return DataBundleManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
        except Exception:
            return None

    @staticmethod
    def _load_strategy_plugin_contract(path: Path | None) -> StrategyPluginContract | None:
        if path is None:
            return None
        try:
            return StrategyPluginContract.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
        except Exception:
            return None

    @staticmethod
    def _strategy_plugin_bundle_errors(
        contract: StrategyPluginContract,
        bundle: DataBundleManifest,
    ) -> list[str]:
        errors: list[str] = []
        if contract.supported_symbols:
            supported_symbols = {symbol.upper() for symbol in contract.supported_symbols}
            unsupported_symbols = sorted({
                item.symbol.upper()
                for item in bundle.slice_manifests
                if item.symbol.upper() not in supported_symbols
            })
            if unsupported_symbols:
                errors.append(
                    "strategy plugin contract does not support data-bundle symbol(s): "
                    + ", ".join(unsupported_symbols)
                )
        if contract.supported_timeframes:
            supported_timeframes = set(contract.supported_timeframes)
            unsupported_timeframes = sorted({
                item.timeframe
                for item in bundle.slice_manifests
                if item.timeframe not in supported_timeframes
            })
            if unsupported_timeframes:
                errors.append(
                    "strategy plugin contract does not support data-bundle timeframe(s): "
                    + ", ".join(unsupported_timeframes)
                )
        return errors

    @staticmethod
    def _data_bundle_in_sample_window(
        bundle: DataBundleManifest,
        *,
        window_start: date,
        window_end: date,
        requested_start: date | None,
        requested_end: date | None,
    ) -> tuple[date | None, date | None, list[str]]:
        slice_starts = [
            item.start_ts.date()
            for item in bundle.slice_manifests
            if item.start_ts is not None
        ]
        slice_ends = [
            item.end_ts.date()
            for item in bundle.slice_manifests
            if item.end_ts is not None
        ]
        if (
            len(slice_starts) != len(bundle.slice_manifests)
            or len(slice_ends) != len(bundle.slice_manifests)
        ):
            if requested_start is None or requested_end is None:
                return (
                    None,
                    None,
                    ["data bundle slice timestamps missing for optimizer in-sample window"],
                )
            return (
                requested_start,
                requested_end,
                ["data bundle slice timestamps missing for optimizer in-sample window"],
            )

        common_start = max(slice_starts)
        common_end = min(slice_ends)
        target_start = requested_start or common_start
        target_end = requested_end or (window_start - timedelta(days=1))
        errors: list[str] = []
        if target_start < common_start or target_end > common_end:
            errors.append("data bundle does not cover optimizer in-sample window")
        if common_start > window_start or common_end < window_end:
            errors.append("data bundle does not cover latest-month selection window")
        if target_end < target_start:
            errors.append("data bundle does not include pre-latest-month in-sample coverage")
        return target_start, target_end, errors

    def _write_data_bundle_manifest(
        self,
        *,
        coverage: MarketDataManifest,
        coverage_path: Path,
        output_path: Path,
    ) -> DataBundleManifest:
        data_repo_commit_sha = backtest_repo_commit_sha(self.market_data_root) or coverage.source_version
        bundle_blockers = [
            label for label, value in (
                ("data repo commit SHA missing", data_repo_commit_sha),
                ("market-data checksum missing", coverage.checksum),
                ("session calendar missing", coverage.session_calendar),
                ("fee model version missing", coverage.fee_model_version),
                ("slippage model version missing", coverage.slippage_model_version),
                ("adjustment policy missing", coverage.adjustment_policy),
            )
            if not str(value or "").strip()
        ]
        status = (
            DataBundleStatus.AUTHORITATIVE
            if coverage.usable_for_authoritative_validation
            and coverage.coverage_ratio >= self.required_market_coverage_ratio
            and not bundle_blockers
            else DataBundleStatus.DIAGNOSTICS_ONLY
        )
        diagnostics_only_reason = "; ".join([
            *(coverage.blocking_reasons or []),
            *bundle_blockers,
        ])
        bundle = DataBundleManifest(
            data_repo_path=str(self.market_data_root),
            data_repo_commit_sha=data_repo_commit_sha,
            slice_manifests=[
                DataBundleSlice(
                    manifest_path=str(coverage_path),
                    manifest_id=coverage.manifest_id,
                    source=coverage.source,
                    market=coverage.market,
                    symbol=coverage.symbol,
                    timeframe=coverage.timeframe,
                    start_ts=coverage.start_ts,
                    end_ts=coverage.end_ts,
                    checksum=coverage.checksum,
                    calendar=coverage.session_calendar,
                    authoritative=coverage.usable_for_authoritative_validation,
                )
            ],
            calendars=[coverage.session_calendar] if coverage.session_calendar else [],
            fee_model_version=coverage.fee_model_version,
            slippage_model_version=coverage.slippage_model_version,
            adjustment_policy=coverage.adjustment_policy,
            status=status,
            diagnostics_only_reason=diagnostics_only_reason,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        return bundle

    @staticmethod
    def _write_runner_observability_status(
        *,
        artifact_root: Path,
        run_id: str,
        stage_status: dict[str, bool | str],
    ) -> Path:
        path = artifact_root / "runner_observability.json"
        entry = {
            "run_id": run_id,
            "phase": "control_plane",
            "attempt_state": "control_plane",
            "monthly_stage_status": stage_status,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        entries: list[Any] = []
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                entries = current if isinstance(current, list) else [current]
            except Exception:
                entries = []
        entries.append(entry)
        path.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
        return path

    @staticmethod
    def _load_parity_report(index: BacktestArtifactIndex) -> ReplayParityReport | None:
        contract = MonthlyArtifactContract.from_index(index)
        path = contract.path("replay_parity_report.json")
        if path is None:
            return None
        report = contract.load_model("replay_parity_report.json", ReplayParityReport)
        if report is None:
            return None
        computed_status = ReplayParityChecker().classify(report)
        if report.status == ReplayParityStatus.INSUFFICIENT_DATA and (
            report.trade_count_live > 0 or report.trade_count_replay > 0
        ):
            report.status = computed_status
        elif report.status in {ReplayParityStatus.PASS, ReplayParityStatus.PASS_WITH_KNOWN_GAPS}:
            report.status = computed_status
        raw_path = str(path)
        if raw_path not in report.evidence_paths:
            report.evidence_paths.append(raw_path)
        return report

    @staticmethod
    def _status_from_mode_decision(index: BacktestArtifactIndex) -> MonthlyValidationStatus | None:
        data = MonthlyArtifactContract.from_index(index).load_json_object("mode_decision.json")
        if not data:
            return None
        raw = (
            data.get("monthly_status")
            or data.get("status")
            or data.get("decision")
            or data.get("routing")
            or data.get("mode")
        )
        if not raw:
            return None
        normalized = str(raw).strip().lower().replace("-", "_")
        aliases = {
            "smoke_repair": MonthlyValidationStatus.REPAIR,
            "phased_auto": MonthlyValidationStatus.EXPERIMENT,
            "manual_review": MonthlyValidationStatus.EXPERIMENT,
            "manual_required": MonthlyValidationStatus.EXPERIMENT,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return MonthlyValidationStatus(normalized)
        except ValueError:
            return None

    @staticmethod
    def _status_from_evidence(
        blocking_reasons: list[str],
        telemetry: TelemetryManifest,
        parity_report: ReplayParityReport | None,
    ) -> MonthlyValidationStatus:
        if telemetry.authoritative_eligibility == TelemetryEligibility.INSUFFICIENT_LINEAGE:
            return MonthlyValidationStatus.INSUFFICIENT_LINEAGE
        if telemetry.authoritative_eligibility == TelemetryEligibility.INSUFFICIENT_DATA:
            return MonthlyValidationStatus.INSUFFICIENT_DATA
        if parity_report is not None:
            if parity_report.status == ReplayParityStatus.FAIL:
                return MonthlyValidationStatus.UNSUPPORTED_NO_REPLAY_PLUGIN
            if parity_report.status == ReplayParityStatus.INSUFFICIENT_DATA:
                return MonthlyValidationStatus.INSUFFICIENT_DATA
        if any("backtest" in reason.lower() or "backtest_repo_path" in reason.lower() for reason in blocking_reasons):
            return MonthlyValidationStatus.UNSUPPORTED_NO_REPLAY_PLUGIN
        if blocking_reasons:
            return MonthlyValidationStatus.INSUFFICIENT_DATA
        return MonthlyValidationStatus.NO_CHANGE


def latest_completed_month(now: datetime | None = None) -> str:
    current = (now or datetime.now(timezone.utc)).date()
    year = current.year
    month = current.month - 1
    if month == 0:
        year -= 1
        month = 12
    return f"{year:04d}-{month:02d}"


def month_window(run_month: str) -> tuple[date, date]:
    year_s, month_s = run_month.split("-", 1)
    year = int(year_s)
    month = int(month_s)
    last_day = monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)


def _approval_evidence_artifact_base(
    artifact_root: Path,
    request: MonthlyValidationRequest,
) -> Path:
    root = Path(artifact_root)
    if not request.approval_evidence_mode:
        return root
    if any(part.lower() == "approval_grade" for part in root.parts):
        return root
    return root / "approval_grade"


def _mutation_family_and_category(mutation_diff: dict) -> tuple[str, str]:
    if not isinstance(mutation_diff, dict):
        return "unknown", "unknown"
    family = str(
        mutation_diff.get("family")
        or mutation_diff.get("candidate_family")
        or mutation_diff.get("source")
        or mutation_diff.get("mode")
        or "unknown"
    )
    category = str(
        mutation_diff.get("change_kind")
        or mutation_diff.get("category")
        or mutation_diff.get("proposal_type")
        or family
        or "unknown"
    )
    return family, category


def _extract_objective_deltas(index: BacktestArtifactIndex | None) -> dict[str, float]:
    if index is None:
        return {}
    contract = MonthlyArtifactContract.from_index(index)
    payloads = [
        contract.load_json_object("objective_breakdown.json"),
        contract.load_json_object("incumbent_validation.json"),
    ]
    aliases = {
        "live_vs_expected": (
            "live_vs_expected",
            "live_vs_expected_objective_delta",
            "objective_delta",
            "composite_delta",
            "delta",
        ),
        "trade_frequency": (
            "trade_frequency_delta",
            "trade_count_delta",
            "frequency_delta",
        ),
        "drawdown": (
            "drawdown_delta",
            "max_drawdown_delta",
            "drawdown_delta_pct",
        ),
        "execution_slippage": (
            "execution_slippage_delta",
            "slippage_delta",
            "fee_slippage_delta_bps",
        ),
    }
    deltas: dict[str, float] = {}
    for target, keys in aliases.items():
        for payload in payloads:
            value = _find_number(payload, keys)
            if value is not None:
                deltas[target] = value
                break
    return deltas


def _minimum_trade_count_met(index: BacktestArtifactIndex | None) -> bool:
    if index is None:
        return False
    contract = MonthlyArtifactContract.from_index(index)
    for payload in (
        contract.load_json_object("objective_breakdown.json"),
        contract.load_json_object("incumbent_validation.json"),
    ):
        value = _find_value(payload, (
            "minimum_trade_count_met",
            "sufficient_trade_count",
            "trade_count_gate_passed",
        ))
        if isinstance(value, bool):
            return value
    return False


def _follow_up_due(monthly_verdict: dict | None, run_month: str) -> bool:
    if not isinstance(monthly_verdict, dict) or not run_month:
        return False
    verdict_month = str(monthly_verdict.get("run_month") or "")
    if not verdict_month:
        return False
    return run_month >= _add_months(verdict_month, 3)


def _add_months(run_month: str, months: int) -> str:
    year_s, month_s = run_month.split("-", 1)
    year = int(year_s)
    month = int(month_s) + months
    while month > 12:
        year += 1
        month -= 12
    return f"{year:04d}-{month:02d}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _status_expects_candidates(status: MonthlyValidationStatus) -> bool:
    return status in {
        MonthlyValidationStatus.REPAIR,
        MonthlyValidationStatus.ROLLBACK,
        MonthlyValidationStatus.EXPERIMENT,
        MonthlyValidationStatus.QUARANTINE,
    }


def _selected_candidates_empty(index: BacktestArtifactIndex) -> bool:
    return not MonthlyArtifactContract.from_index(index).load_candidate_rows("selected_candidates.json")


def _monthly_review_decision_reason(result: MonthlyValidationResult) -> str:
    if result.blocking_reasons:
        return "; ".join(result.blocking_reasons)
    if result.repair_required:
        if result.repair_request_path:
            return f"Monthly validation requires repair before approval: {result.repair_request_path}"
        return "Monthly validation requires repair before approval."
    if result.approval_packet_paths:
        return (
            f"Monthly validation completed with status {result.status.value}; "
            f"{len(result.approval_packet_paths)} approval packet(s) generated."
        )
    return f"Monthly validation completed with status {result.status.value}."


def _find_number(payload: Any, keys: tuple[str, ...]) -> float | None:
    value = _find_value(payload, keys)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_value(payload: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        for value in payload.values():
            found = _find_value(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_value(value, keys)
            if found is not None:
                return found
    return None


def dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
