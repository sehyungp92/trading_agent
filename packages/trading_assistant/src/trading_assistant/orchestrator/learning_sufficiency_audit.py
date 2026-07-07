"""Learning-sufficiency manifest builder for monthly evidence windows."""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

from trading_assistant.orchestrator.action_handlers.daily_data import (
    DAILY_CURATED_EVENT_FILES,
    DAILY_RAW_EVENT_TAXONOMY,
    iter_daily_event_input_paths,
)
from trading_assistant.orchestrator.lineage_audit import LineageAuditor
from trading_assistant.schemas.learning_sufficiency import (
    CoverageCheck,
    CoverageStatus,
    ExpectedLearningValue,
    LearningCapabilityAuthority,
    LearningCapabilityStatus,
    LearningEligibility,
    LearningGap,
    LearningSufficiencyManifest,
    RuntimeEvidenceSupport,
    RuntimeEvidenceSupportState,
)
from trading_assistant.schemas.telemetry_manifest import TelemetryEligibility, TelemetryManifest
from trading_assistant.skills.lineage_utils import event_strategy_id, event_value


__all__ = [
    "LearningSufficiencyAuditor",
    "DAILY_CURATED_EVENT_FILES",
    "DAILY_RAW_EVENT_TAXONOMY",
    "runtime_source_authority_for_checks",
    "manifest_runtime_support_source_authoritative",
]


CANONICAL_DECISION_KEYS = (
    "entry_decision_id",
    "exit_decision_id",
    "decision_id",
    "decision_ref",
    "signal_id",
    "source_signal_id",
    "entry_signal_id",
    "action_ref",
    "bar_id",
)
CANONICAL_ORDER_KEYS = (
    "intent_id",
    "idempotency_key",
    "client_order_ids",
    "client_order_id",
    "entry_order_ids",
    "entry_order_id",
    "exit_order_ids",
    "exit_order_id",
    "order_ids",
    "order_id",
    "exchange_order_ids",
    "broker_order_id",
    "oms_order_id",
    "kis_order_id",
)
PLURAL_CANONICAL_ORDER_KEYS = (
    "client_order_ids",
    "entry_order_ids",
    "exit_order_ids",
    "order_ids",
    "exchange_order_ids",
)
SINGULAR_CANONICAL_ORDER_KEYS = tuple(
    key for key in CANONICAL_ORDER_KEYS
    if key not in PLURAL_CANONICAL_ORDER_KEYS
)
PLACEHOLDER_KEY_VALUES = {"unknown", "none", "null", "n/a", "na", "missing", "undefined"}
CANONICAL_FILL_KEYS = (
    "fill_ids",
    "entry_fill_ids",
    "exit_fill_ids",
    "fill_id",
    "kis_exec_id",
)
CANONICAL_PORTFOLIO_KEYS = (
    "portfolio_rule_event_id",
    "risk_decision_id",
    "risk_decision_ref",
    "portfolio_decision_ref",
    "portfolio_rule_id",
    "portfolio_rule",
    "rule_id",
)
CANONICAL_ASSISTANT_KEYS = (
    "weekly_signal_ids",
    "source_weekly_signal_ids",
    "monthly_search_brief_id",
    "suggestion_id",
    "suggestion_ids",
    "proposal_id",
    "proposal_ids",
    "hypothesis_id",
    "hypothesis_ids",
    "experiment_id",
    "deployment_id",
    "strategy_change_record_id",
    "strategy_change_record_ids",
    "monthly_outcome_id",
)

CAPABILITY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "filter_threshold_learning": (
        "trade_outcome_lineage",
        "decision_to_trade_join",
        "filter_decision_coverage",
        "denominator_coverage",
        "counterfactual_coverage",
        "after_cost_coverage",
    ),
    "execution_learning": (
        "decision_to_order_join",
        "order_to_fill_join",
        "orderbook_context_coverage",
        "after_cost_coverage",
    ),
    "sizing_learning": (
        "trade_outcome_lineage",
        "risk_portfolio_join",
        "after_cost_coverage",
    ),
    "portfolio_interaction_learning": (
        "risk_portfolio_join",
        "portfolio_rule_coverage",
        "denominator_coverage",
    ),
    "new_strategy_discovery": (
        "missed_opportunity_lineage",
        "denominator_coverage",
        "counterfactual_coverage",
        "after_cost_coverage",
    ),
    "approval_grade_strategy_change": (
        "trade_outcome_lineage",
        "missed_opportunity_lineage",
        "decision_to_trade_join",
        "decision_to_order_join",
        "order_to_fill_join",
        "risk_portfolio_join",
        "denominator_coverage",
        "after_cost_coverage",
        "proposal_trace_coverage",
        "deployment_metadata_coverage",
    ),
}

CHECK_EVENT_TYPE = {
    "trade_outcome_lineage": "trade",
    "missed_opportunity_lineage": "missed_opportunity",
    "filter_decision_coverage": "filter_decision",
    "orderbook_context_coverage": "orderbook_context",
    "portfolio_rule_coverage": "portfolio_rule",
    "decision_to_trade_join": "trade",
    "decision_to_order_join": "order",
    "order_to_fill_join": "fill",
    "risk_portfolio_join": "portfolio_rule",
    "denominator_coverage": "pipeline_funnel",
    "after_cost_coverage": "trade",
    "counterfactual_coverage": "missed_opportunity",
    "proposal_trace_coverage": "deployment_metadata",
    "deployment_metadata_coverage": "deployment_metadata",
}

CHECK_RUNTIME_EVENT_CLASSES = {
    **{check_id: (event_class,) for check_id, event_class in CHECK_EVENT_TYPE.items()},
    "decision_to_trade_join": ("filter_decision", "order", "trade"),
    "decision_to_order_join": ("filter_decision", "order"),
    "order_to_fill_join": ("order", "fill"),
    "risk_portfolio_join": ("portfolio_rule", "trade", "missed_opportunity"),
}

RUNTIME_EVIDENCE_CLASSES = (
    "trade",
    "missed_opportunity",
    "filter_decision",
    "orderbook_context",
    "portfolio_rule",
    "order",
    "fill",
    "pipeline_funnel",
    "deployment_metadata",
)
EXPECTED_SESSION_FILENAMES = (
    "expected_active_sessions.json",
    "session_manifest.json",
    "trading_sessions.json",
    "paper_session_manifest.json",
)
RUNTIME_SUPPORT_FILENAMES = (
    "runtime_evidence_support.json",
    "runtime_exporter_support.json",
    "sidecar_support.json",
)
RUNTIME_BLOCKER_PREFIX = "runtime_evidence_support:"
LEARNING_AUTHORITY_EVENT_VALUE_CLASS = "learning_authority"


class LearningSufficiencyAuditor:
    """Build learning authority from existing telemetry and daily evidence."""

    def __init__(
        self,
        curated_dir: Path,
        findings_dir: Path,
        *,
        raw_data_dir: Path | None = None,
        required_lineage_ratio: float = 0.95,
        lineage_auditor: LineageAuditor | None = None,
    ) -> None:
        self._curated_dir = Path(curated_dir)
        self._raw_data_dir = Path(raw_data_dir) if raw_data_dir is not None else self._infer_raw_dir(self._curated_dir)
        self._findings_dir = Path(findings_dir)
        self._lineage_auditor = lineage_auditor or LineageAuditor(
            self._curated_dir,
            self._findings_dir,
            required_lineage_ratio=required_lineage_ratio,
        )

    def build_manifest(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        run_month: str,
        window_start: date,
        window_end: date,
        output_path: Path | None = None,
        telemetry_manifest_path: Path | None = None,
        deployment_metadata_paths: list[Path] | None = None,
        strategy_contract_path: Path | None = None,
        expected_session_paths: list[Path] | None = None,
        runtime_support_paths: list[Path] | None = None,
        family_id: str = "",
        portfolio_id: str = "",
    ) -> LearningSufficiencyManifest:
        telemetry = self._load_or_build_telemetry_manifest(
            bot_id=bot_id,
            strategy_id=strategy_id,
            run_month=run_month,
            window_start=window_start,
            window_end=window_end,
            output_path=telemetry_manifest_path,
        )
        evidence = self._load_window_evidence(
            bot_id=bot_id,
            strategy_id=strategy_id,
            window_start=window_start,
            window_end=window_end,
            expected_session_paths=expected_session_paths or [],
        )
        deployment_metadata = self._load_deployment_metadata(deployment_metadata_paths or [])
        event_counts = Counter(evidence.counts)
        for event_type, count in telemetry.event_counts_by_type.items():
            event_counts[event_type] = max(event_counts.get(event_type, 0), count)
        if deployment_metadata:
            event_counts["deployment_metadata"] += len(deployment_metadata)
        not_applicable_checks = self._declared_not_applicable_checks(strategy_contract_path)
        runtime_support = self._build_runtime_evidence_support(
            bot_id=bot_id,
            strategy_id=strategy_id,
            window_start=window_start,
            window_end=window_end,
            evidence=evidence,
            deployment_metadata_count=len(deployment_metadata),
            deployment_metadata_paths=deployment_metadata_paths or [],
            strategy_contract_path=strategy_contract_path,
            runtime_support_paths=runtime_support_paths or [],
        )

        checks = self._build_checks(
            telemetry=telemetry,
            evidence=evidence,
            deployment_metadata=deployment_metadata,
            deployment_metadata_paths=deployment_metadata_paths or [],
            not_applicable_checks=not_applicable_checks,
        )
        capability_status = self._build_capability_status(checks, runtime_support)
        gaps = self._build_gaps(
            bot_id=bot_id,
            strategy_id=strategy_id,
            family_id=family_id,
            portfolio_id=portfolio_id,
            checks=checks,
            capability_status=capability_status,
            runtime_support=runtime_support,
        )
        eligibility = self._eligibility(telemetry.authoritative_eligibility, checks, capability_status)
        evidence_paths = _dedupe([
            *(_path_text(path) for path in evidence.paths),
            *(_path_text(path) for path in (deployment_metadata_paths or [])),
            *(_path_text(telemetry_manifest_path) for _ in [0] if telemetry_manifest_path is not None),
        ])

        manifest = LearningSufficiencyManifest(
            bot_id=bot_id,
            strategy_id=strategy_id,
            family_id=family_id,
            portfolio_id=portfolio_id,
            run_month=run_month,
            window_start=window_start,
            window_end=window_end,
            telemetry_manifest_path=str(telemetry_manifest_path or ""),
            telemetry_authoritative_eligibility=telemetry.authoritative_eligibility.value,
            eligibility=eligibility,
            event_counts_by_type=dict(sorted(event_counts.items())),
            required_event_coverage={
                key: checks[key]
                for key in (
                    "trade_outcome_lineage",
                    "missed_opportunity_lineage",
                    "filter_decision_coverage",
                    "orderbook_context_coverage",
                    "portfolio_rule_coverage",
                )
            },
            lineage_coverage=checks["lineage_coverage"],
            join_coverage={
                key: checks[key]
                for key in (
                    "decision_to_trade_join",
                    "decision_to_order_join",
                    "order_to_fill_join",
                    "risk_portfolio_join",
                )
            },
            denominator_coverage={"denominator_coverage": checks["denominator_coverage"]},
            after_cost_coverage=checks["after_cost_coverage"],
            counterfactual_coverage=checks["counterfactual_coverage"],
            proposal_trace_coverage=checks["proposal_trace_coverage"],
            deployment_metadata_coverage=checks["deployment_metadata_coverage"],
            runtime_evidence_support=runtime_support,
            capability_status=capability_status,
            artifact_paths={
                "telemetry_manifest": str(telemetry_manifest_path or ""),
                "learning_sufficiency_manifest": str(output_path or ""),
            },
            known_gaps=gaps,
            evidence_paths=evidence_paths,
        )
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest

    def _build_checks(
        self,
        *,
        telemetry: Any,
        evidence: "_WindowEvidence",
        deployment_metadata: list[dict[str, Any]],
        deployment_metadata_paths: list[Path],
        not_applicable_checks: set[str],
    ) -> dict[str, CoverageCheck]:
        required_event_checks = {
            "trade_outcome_lineage": self._event_check(
                "trade_outcome_lineage",
                evidence.records.get("trade", []),
                required_fields=["strategy_version", "config_version", "deployment_id"],
            ),
            "missed_opportunity_lineage": self._event_check(
                "missed_opportunity_lineage",
                evidence.records.get("missed_opportunity", []),
                required_fields=["strategy_version", "config_version", "deployment_id"],
            ),
            "filter_decision_coverage": self._event_check(
                "filter_decision_coverage",
                evidence.records.get("filter_decision", []),
                required_fields=["filter_name", "threshold", "actual_value", "passed"],
            ),
            "orderbook_context_coverage": self._event_count_check(
                "orderbook_context_coverage",
                evidence.counts.get("orderbook_context", 0),
                evidence.paths_by_type.get("orderbook_context", []),
            ),
            "portfolio_rule_coverage": self._event_count_check(
                "portfolio_rule_coverage",
                evidence.counts.get("portfolio_rule", 0),
                evidence.paths_by_type.get("portfolio_rule", []),
            ),
        }
        required_total = max(1, telemetry.total_events)
        observed_lineage = round(float(telemetry.lineage_coverage_ratio) * required_total)
        checks = {
            **required_event_checks,
            "lineage_coverage": CoverageCheck(
                check_id="lineage_coverage",
                observed_count=observed_lineage if telemetry.total_events else 0,
                required_count=required_total,
                coverage_ratio=telemetry.lineage_coverage_ratio,
                required_fields=["strategy_version", "config_version", "deployment_id"],
                missing_fields=[
                    field
                    for field in ("strategy_version", "config_version", "deployment_id")
                    if telemetry.missing_field_counts.get(field, 0)
                ],
                reason="composed_from_telemetry_manifest",
            ),
            "decision_to_trade_join": self._cross_record_join_check(
                "decision_to_trade_join",
                left_records=[
                    *evidence.records.get("filter_decision", []),
                    *evidence.records.get("order", []),
                ],
                right_records=evidence.records.get("trade", []),
                left_keys=CANONICAL_DECISION_KEYS,
                right_keys=CANONICAL_DECISION_KEYS,
                evidence_paths=[
                    *evidence.paths_by_type.get("filter_decision", []),
                    *evidence.paths_by_type.get("order", []),
                    *evidence.paths_by_type.get("trade", []),
                ],
                consumed_runtime_event_classes=_consumed_event_classes(
                    ("filter_decision", evidence.records.get("filter_decision", [])),
                    ("order", evidence.records.get("order", [])),
                    ("trade", evidence.records.get("trade", [])),
                ),
                not_applicable="decision_to_trade_join" in not_applicable_checks,
            ),
            "decision_to_order_join": self._cross_record_join_check(
                "decision_to_order_join",
                left_records=evidence.records.get("filter_decision", []),
                right_records=[
                    *evidence.records.get("order", []),
                    *_explicit_no_order_decisions(evidence.records.get("filter_decision", [])),
                ],
                left_keys=CANONICAL_DECISION_KEYS,
                right_keys=CANONICAL_DECISION_KEYS,
                evidence_paths=[
                    *evidence.paths_by_type.get("filter_decision", []),
                    *evidence.paths_by_type.get("order", []),
                ],
                consumed_runtime_event_classes=_consumed_event_classes(
                    ("filter_decision", evidence.records.get("filter_decision", [])),
                    ("order", evidence.records.get("order", [])),
                ),
                not_applicable="decision_to_order_join" in not_applicable_checks,
            ),
            "order_to_fill_join": self._order_to_fill_check(
                evidence,
                not_applicable="order_to_fill_join" in not_applicable_checks,
            ),
            "risk_portfolio_join": self._cross_record_join_check(
                "risk_portfolio_join",
                left_records=evidence.records.get("portfolio_rule", []),
                right_records=[
                    *evidence.records.get("trade", []),
                    *evidence.records.get("missed_opportunity", []),
                    *_explicit_risk_denials(evidence.records.get("portfolio_rule", [])),
                ],
                left_keys=CANONICAL_PORTFOLIO_KEYS,
                right_keys=CANONICAL_PORTFOLIO_KEYS,
                evidence_paths=[
                    *evidence.paths_by_type.get("portfolio_rule", []),
                    *evidence.paths_by_type.get("trade", []),
                    *evidence.paths_by_type.get("missed_opportunity", []),
                ],
                consumed_runtime_event_classes=_consumed_event_classes(
                    ("portfolio_rule", evidence.records.get("portfolio_rule", [])),
                    ("trade", evidence.records.get("trade", [])),
                    ("missed_opportunity", evidence.records.get("missed_opportunity", [])),
                ),
                not_applicable="risk_portfolio_join" in not_applicable_checks,
            ),
            "denominator_coverage": self._denominator_session_check(evidence),
            "after_cost_coverage": self._after_cost_check(evidence.records.get("trade", [])),
            "counterfactual_coverage": self._field_presence_check(
                "counterfactual_coverage",
                evidence.records.get("missed_opportunity", []),
                (
                    "would_have_pnl",
                    "would_have_won",
                    "estimated_pnl",
                    "counterfactual_outcome",
                    "post_exit_backfill_status",
                ),
            ),
            "proposal_trace_coverage": self._field_presence_check(
                "proposal_trace_coverage",
                [
                    *evidence.records.get("trade", []),
                    *evidence.records.get("missed_opportunity", []),
                    *deployment_metadata,
                ],
                CANONICAL_ASSISTANT_KEYS,
                not_applicable="proposal_trace_coverage" in not_applicable_checks,
            ),
            "deployment_metadata_coverage": self._event_count_check(
                "deployment_metadata_coverage",
                len(deployment_metadata),
                [_path_text(path) for path in deployment_metadata_paths],
            ),
        }
        return checks

    def _load_or_build_telemetry_manifest(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        run_month: str,
        window_start: date,
        window_end: date,
        output_path: Path | None,
    ) -> TelemetryManifest:
        if output_path is not None and Path(output_path).exists():
            payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
            telemetry = TelemetryManifest.model_validate(payload)
            if (
                telemetry.bot_id != bot_id
                or telemetry.strategy_id != strategy_id
                or telemetry.run_month != run_month
                or telemetry.window_start != window_start
                or telemetry.window_end != window_end
            ):
                raise ValueError("supplied telemetry manifest scope does not match sufficiency request")
            return telemetry
        return self._lineage_auditor.build_telemetry_manifest(
            bot_id=bot_id,
            strategy_id=strategy_id,
            run_month=run_month,
            window_start=window_start,
            window_end=window_end,
            output_path=output_path,
        )

    def _event_check(
        self,
        check_id: str,
        records: list[dict[str, Any]],
        *,
        required_fields: list[str],
    ) -> CoverageCheck:
        observed = 0
        missing_counter: Counter[str] = Counter()
        if not records:
            return CoverageCheck(
                check_id=check_id,
                observed_count=0,
                required_count=1,
                required_fields=required_fields,
                missing_fields=required_fields,
            )
        for record in records:
            missing = [field for field in required_fields if not _has_any_value(record, (field,))]
            if not missing:
                observed += 1
            for field in missing:
                missing_counter[field] += 1
        return CoverageCheck(
            check_id=check_id,
            observed_count=observed,
            required_count=max(1, len(records)),
            required_fields=required_fields,
            observed_fields=[field for field in required_fields if not missing_counter.get(field)],
            missing_fields=sorted(missing_counter),
        )

    def _event_count_check(self, check_id: str, count: int, evidence_paths: list[str]) -> CoverageCheck:
        return CoverageCheck(
            check_id=check_id,
            observed_count=count,
            required_count=1,
            evidence_paths=evidence_paths,
        )

    def _denominator_session_check(self, evidence: "_WindowEvidence") -> CoverageCheck:
        expected_days = set(evidence.expected_session_days)
        observed_days = set(evidence.denominator_session_days) & expected_days
        required = len(expected_days)
        observed = len(observed_days)
        if not required:
            return CoverageCheck(
                check_id="denominator_coverage",
                status=CoverageStatus.MISSING,
                observed_count=0,
                required_count=1,
                min_required_ratio=0.90,
                evidence_paths=evidence.expected_session_paths,
                reason="missing_expected_active_session_source",
                details={
                    "active_session_count": len(evidence.active_session_days),
                    "denominator_session_count": len(evidence.denominator_session_days),
                    "expected_session_source_paths": evidence.expected_session_paths,
                    "active_session_days": sorted(evidence.active_session_days)[:40],
                    "denominator_session_days": sorted(evidence.denominator_session_days)[:40],
                },
            )
        return CoverageCheck(
            check_id="denominator_coverage",
            observed_count=observed,
            required_count=max(1, required),
            coverage_ratio=(observed / required) if required else 0.0,
            min_required_ratio=0.90,
            evidence_paths=_dedupe([
                *evidence.expected_session_paths,
                *evidence.paths_by_type.get("pipeline_funnel", []),
            ]),
            details={
                "expected_session_count": required,
                "active_session_count": len(evidence.active_session_days),
                "denominator_session_count": observed,
                "required_session_coverage_ratio": 0.90,
                "expected_session_source_paths": evidence.expected_session_paths,
                "missing_expected_session_days": sorted(expected_days - observed_days)[:40],
                "expected_session_days": sorted(expected_days)[:40],
                "active_session_days": sorted(evidence.active_session_days)[:40],
                "denominator_session_days": sorted(observed_days)[:40],
            },
        )

    def _field_presence_check(
        self,
        check_id: str,
        records: list[dict[str, Any]],
        field_names: tuple[str, ...],
        *,
        not_applicable: bool = False,
    ) -> CoverageCheck:
        if not_applicable:
            return CoverageCheck(
                check_id=check_id,
                observed_count=0,
                required_count=0,
                declared_not_applicable=True,
                reason="declared_not_applicable_by_strategy_contract",
            )
        observed = sum(1 for record in records if _has_any_value(record, field_names))
        missing_fields = [] if observed == len(records) and records else list(field_names)
        return CoverageCheck(
            check_id=check_id,
            observed_count=observed,
            required_count=max(1, len(records)),
            required_fields=list(field_names),
            observed_fields=list(field_names) if observed else [],
            missing_fields=missing_fields,
        )

    def _cross_record_join_check(
        self,
        check_id: str,
        *,
        left_records: list[dict[str, Any]],
        right_records: list[dict[str, Any]],
        left_keys: tuple[str, ...],
        right_keys: tuple[str, ...],
        evidence_paths: list[str],
        consumed_runtime_event_classes: list[str],
        not_applicable: bool = False,
    ) -> CoverageCheck:
        if not_applicable:
            return CoverageCheck(
                check_id=check_id,
                observed_count=0,
                required_count=0,
                declared_not_applicable=True,
                reason="declared_not_applicable_by_strategy_contract",
            )
        left_ids = (
            set().union(*(_record_key_values(record, left_keys) for record in left_records))
            if left_records else set()
        )
        joined = 0
        missing_refs = 0
        orphan_refs: set[str] = set()
        for record in right_records:
            refs = _record_key_values(record, right_keys)
            if not refs:
                missing_refs += 1
            elif refs & left_ids:
                joined += 1
            else:
                orphan_refs.update(refs)
        required = max(1, len(right_records))
        passing = (
            bool(right_records)
            and (joined / required) >= 0.95
            and not missing_refs
            and not orphan_refs
        )
        authority_observed = joined if passing else 0
        return CoverageCheck(
            check_id=check_id,
            observed_count=authority_observed,
            required_count=required,
            min_required_ratio=0.95,
            required_fields=list(right_keys),
            observed_fields=list(right_keys) if passing else [],
            missing_fields=[] if passing else list(right_keys),
            evidence_paths=_dedupe(evidence_paths),
            reason="" if passing else "missing_or_mismatched_join_ids",
            details={
                "source_record_count": len(left_records),
                "target_record_count": len(right_records),
                "source_join_id_count": len(left_ids),
                "joined_target_count": joined,
                "missing_join_ref_count": missing_refs,
                "orphan_join_ref_count": len(orphan_refs),
                "orphan_join_refs": sorted(orphan_refs)[:20],
                "consumed_runtime_event_classes": consumed_runtime_event_classes,
            },
        )

    def _order_to_fill_check(self, evidence: "_WindowEvidence", *, not_applicable: bool = False) -> CoverageCheck:
        if not_applicable:
            return CoverageCheck(
                check_id="order_to_fill_join",
                observed_count=0,
                required_count=0,
                declared_not_applicable=True,
                reason="declared_not_applicable_by_strategy_contract",
            )
        orders = evidence.records.get("order", [])
        fills = evidence.records.get("fill", [])
        (
            order_groups,
            missing_fill_required_order_id_count,
            terminal_order_missing_id_count,
        ) = _canonical_order_groups(orders)
        fill_required_order_ids = [ids for ids, is_terminal in order_groups if not is_terminal]
        terminal_order_ids = [ids for ids, is_terminal in order_groups if is_terminal]
        order_ids = set().union(*(ids for ids, _ in order_groups)) if order_groups else set()
        fill_ids = set().union(*(
            _record_key_values(fill, CANONICAL_FILL_KEYS) for fill in fills
        )) if fills else set()
        valid_fill_order_ids: set[str] = set()
        missing_fill_order_refs = 0
        missing_fill_id_count = 0
        orphan_fill_count = 0
        orphan_fill_refs: set[str] = set()
        for fill in fills:
            refs = _record_key_values(fill, CANONICAL_ORDER_KEYS)
            fill_key_values = _record_key_values(fill, CANONICAL_FILL_KEYS)
            if not fill_key_values:
                missing_fill_id_count += 1
            if not refs:
                missing_fill_order_refs += 1
            elif not (refs & order_ids):
                orphan_fill_count += 1
                orphan_fill_refs.update(refs)
            elif fill_key_values:
                valid_fill_order_ids.update(refs & order_ids)
        matched_fill_orders = sum(1 for ids, _ in order_groups if ids & valid_fill_order_ids)
        unfilled_required_order_count = sum(
            1 for ids in fill_required_order_ids
            if not (ids & valid_fill_order_ids)
        )
        terminal_no_fill_order_count = sum(
            1 for ids in terminal_order_ids
            if not (ids & valid_fill_order_ids)
        )
        observed = matched_fill_orders + terminal_no_fill_order_count
        required = max(
            1,
            len(fill_required_order_ids)
            + len(terminal_order_ids)
            + orphan_fill_count
            + missing_fill_order_refs
            + missing_fill_id_count
            + missing_fill_required_order_id_count
            + terminal_order_missing_id_count,
        )
        hard_failure = bool(
            orphan_fill_count
            or missing_fill_order_refs
            or missing_fill_id_count
            or missing_fill_required_order_id_count
            or terminal_order_missing_id_count
            or unfilled_required_order_count
        )
        authority_observed = 0 if hard_failure else observed
        passing = authority_observed >= required
        return CoverageCheck(
            check_id="order_to_fill_join",
            observed_count=authority_observed,
            required_count=required,
            required_fields=["order_id", "fill_id"],
            observed_fields=["order_id", "fill_id"] if passing else [],
            missing_fields=[] if passing else ["order_id", "fill_id"],
            reason="" if passing else "missing_or_mismatched_order_fill_ids",
            evidence_paths=[
                *evidence.paths_by_type.get("order", []),
                *evidence.paths_by_type.get("fill", []),
            ],
            details={
                "matched_fill_order_count": matched_fill_orders,
                "order_lifecycle_row_count": len(orders),
                "order_identity_group_count": len(order_groups),
                "terminal_order_count": len(terminal_order_ids),
                "terminal_no_fill_order_count": terminal_no_fill_order_count,
                "terminal_no_fill_order_with_id_count": terminal_no_fill_order_count,
                "fill_required_order_count": len(fill_required_order_ids),
                "fill_record_count": len(fills),
                "fill_id_count": len(fill_ids),
                "orphan_fill_count": orphan_fill_count,
                "orphan_fill_refs": sorted(orphan_fill_refs)[:20],
                "missing_fill_order_ref_count": missing_fill_order_refs,
                "missing_fill_id_count": missing_fill_id_count,
                "missing_fill_required_order_id_count": missing_fill_required_order_id_count,
                "terminal_order_missing_id_count": terminal_order_missing_id_count,
                "unfilled_required_order_count": unfilled_required_order_count,
                "consumed_runtime_event_classes": _consumed_event_classes(
                    ("order", orders),
                    ("fill", fills),
                ),
            },
        )

    def _after_cost_check(self, trades: list[dict[str, Any]]) -> CoverageCheck:
        observed = 0
        inferred = 0
        unavailable = 0
        status_only = 0
        for trade in trades:
            status = str(event_value(trade, "after_cost_status") or "").lower()
            source = str(event_value(trade, "net_pnl_source") or "").lower()
            has_net = _has_numeric_value(trade, ("net_pnl", "realized_pnl_net", "after_cost_pnl"))
            observed_flag = status == "observed" and source.startswith("observed")
            has_cost = _has_any_value(trade, (
                "fees_paid",
                "total_fees",
                "commission",
                "tax",
                "slippage_pct",
                "entry_slippage_bps",
                "exit_slippage_bps",
                "funding_paid",
                "borrow_cost",
                "spread_cost",
            ))
            if observed_flag and has_net:
                observed += 1
            elif status == "inferred" or source.startswith("inferred") or (has_net and has_cost):
                inferred += 1
            elif status == "observed" or source.startswith("observed"):
                status_only += 1
                unavailable += 1
            else:
                unavailable += 1
        required = max(1, len(trades))
        check = CoverageCheck(
            check_id="after_cost_coverage",
            observed_count=observed,
            required_count=required,
            required_fields=["net_pnl", "net_pnl_source", "after_cost_status"],
            observed_fields=["net_pnl", "net_pnl_source", "after_cost_status"] if observed else [],
            missing_fields=[] if observed == len(trades) and trades else ["net_pnl_source", "after_cost_status"],
            details={
                "observed_after_cost_count": observed,
                "inferred_after_cost_count": inferred,
                "status_only_without_net_count": status_only,
                "unavailable_after_cost_count": unavailable,
            },
        )
        if inferred and not observed == len(trades):
            check.status = CoverageStatus.PARTIAL
            check.reason = "inferred_after_costs_are_diagnostics_only"
        if status_only:
            check.reason = "observed_status_without_numeric_net_pnl"
        if unavailable:
            check.reason = check.reason or "gross_as_net_or_missing_cost_fields"
        return check

    def _build_capability_status(
        self,
        checks: dict[str, CoverageCheck],
        runtime_support: dict[str, RuntimeEvidenceSupport],
    ) -> dict[str, LearningCapabilityStatus]:
        statuses: dict[str, LearningCapabilityStatus] = {}
        for capability_id, required_checks in sorted(CAPABILITY_REQUIREMENTS.items()):
            satisfied = [
                check_id
                for check_id in required_checks
                if checks[check_id].satisfies_learning_authority
            ]
            blocking = [check_id for check_id in required_checks if check_id not in satisfied]
            runtime_blockers, runtime_evidence_paths = runtime_source_authority_for_checks(
                satisfied,
                runtime_support,
                checks,
            )
            blocking = _dedupe([*blocking, *runtime_blockers])
            statuses[capability_id] = LearningCapabilityStatus(
                capability_id=capability_id,
                status=(
                    LearningCapabilityAuthority.LEARNING_AUTHORITATIVE
                    if not blocking
                    else LearningCapabilityAuthority.BLOCKED
                ),
                required_checks=list(required_checks),
                satisfied_checks=satisfied,
                blocking_checks=blocking,
                evidence_paths=_dedupe([
                    path
                    for check_id in required_checks
                    for path in checks[check_id].evidence_paths
                ] + runtime_evidence_paths),
                blocking_reasons=[
                    _blocking_reason(check_id, checks, runtime_support)
                    for check_id in blocking
                ],
            )
        return statuses

    def _build_gaps(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        family_id: str,
        portfolio_id: str,
        checks: dict[str, CoverageCheck],
        capability_status: dict[str, LearningCapabilityStatus],
        runtime_support: dict[str, RuntimeEvidenceSupport],
    ) -> list[LearningGap]:
        gaps: list[LearningGap] = []
        for capability_id, status in capability_status.items():
            if status.status != LearningCapabilityAuthority.BLOCKED:
                continue
            for check_id in status.blocking_checks:
                check = checks.get(check_id)
                if check is None and check_id.startswith(RUNTIME_BLOCKER_PREFIX):
                    parts = check_id[len(RUNTIME_BLOCKER_PREFIX):].split(":")
                    source_check_id = parts[0] if parts else ""
                    event_class = parts[1] if len(parts) > 1 else source_check_id
                    support = runtime_support.get(event_class)
                    evidence_paths = (
                        [*support.support_source_paths, *support.observed_evidence_paths]
                        if support is not None else []
                    )
                    gaps.append(LearningGap(
                        bot_id=bot_id,
                        strategy_id=strategy_id,
                        family_id=family_id,
                        portfolio_id=portfolio_id,
                        event_type=event_class,
                        missing_field="runtime_evidence_support",
                        blocked_learning_capability=capability_id,
                        expected_learning_value=_expected_value(capability_id),
                        frequency=1,
                        evidence_paths=evidence_paths,
                        remediation=f"Declare and observe runtime support for {event_class}.",
                        details={
                            "check_id": check_id,
                            "source_check_id": source_check_id,
                            "status": support.support_state.value if support is not None else "missing",
                        },
                    ))
                    continue
                if check is None:
                    continue
                missing_field = ",".join(check.missing_fields[:3]) or ",".join(check.missing_event_types[:3])
                if not missing_field:
                    missing_field = check_id
                gaps.append(LearningGap(
                    bot_id=bot_id,
                    strategy_id=strategy_id,
                    family_id=family_id,
                    portfolio_id=portfolio_id,
                    event_type=CHECK_EVENT_TYPE.get(check_id, ""),
                    missing_field=missing_field,
                    blocked_learning_capability=capability_id,
                    expected_learning_value=_expected_value(capability_id),
                    frequency=max(0, check.required_count - check.observed_count),
                    evidence_paths=check.evidence_paths,
                    remediation=f"Provide {check_id} evidence for {capability_id}.",
                    details={
                        "check_id": check_id,
                        "status": check.status.value,
                        "observed_count": check.observed_count,
                        "required_count": check.required_count,
                    },
                ))
        priority = {
            ExpectedLearningValue.CRITICAL: 0,
            ExpectedLearningValue.HIGH: 1,
            ExpectedLearningValue.MEDIUM: 2,
            ExpectedLearningValue.LOW: 3,
        }
        return sorted(
            gaps,
            key=lambda gap: (
                priority[gap.expected_learning_value],
                -gap.frequency,
                gap.blocked_learning_capability,
                gap.event_type,
                gap.missing_field,
            ),
        )

    def _eligibility(
        self,
        telemetry_eligibility: TelemetryEligibility,
        checks: dict[str, CoverageCheck],
        capability_status: dict[str, LearningCapabilityStatus],
    ) -> LearningEligibility:
        if telemetry_eligibility == TelemetryEligibility.INSUFFICIENT_LINEAGE:
            return LearningEligibility.INSUFFICIENT_LINEAGE
        if telemetry_eligibility == TelemetryEligibility.INSUFFICIENT_DATA:
            return LearningEligibility.DIAGNOSTICS_ONLY
        if any(status.status == LearningCapabilityAuthority.LEARNING_AUTHORITATIVE for status in capability_status.values()):
            if any(
                not checks[check_id].satisfies_learning_authority
                for check_id in ("decision_to_order_join", "order_to_fill_join", "risk_portfolio_join")
            ):
                return LearningEligibility.INSUFFICIENT_JOINS
            if checks["denominator_coverage"].status != CoverageStatus.PASS:
                return LearningEligibility.INSUFFICIENT_DENOMINATORS
            if checks["after_cost_coverage"].status != CoverageStatus.PASS:
                return LearningEligibility.INSUFFICIENT_AFTER_COSTS
            return LearningEligibility.LEARNING_AUTHORITATIVE
        if checks["denominator_coverage"].status != CoverageStatus.PASS:
            return LearningEligibility.INSUFFICIENT_DENOMINATORS
        if checks["after_cost_coverage"].status != CoverageStatus.PASS:
            return LearningEligibility.INSUFFICIENT_AFTER_COSTS
        return LearningEligibility.DIAGNOSTICS_ONLY

    def _load_window_evidence(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        window_start: date,
        window_end: date,
        expected_session_paths: list[Path],
    ) -> "_WindowEvidence":
        records: dict[str, list[dict[str, Any]]] = {key: [] for key in DAILY_RAW_EVENT_TAXONOMY}
        paths_by_type: dict[str, list[str]] = {key: [] for key in DAILY_RAW_EVENT_TAXONOMY}
        expected_session_days, loaded_expected_session_paths = self._load_expected_session_days(
            bot_id=bot_id,
            strategy_id=strategy_id,
            window_start=window_start,
            window_end=window_end,
            explicit_paths=expected_session_paths,
        )
        active_session_days: set[str] = set()
        denominator_session_days: set[str] = set()
        days = sorted({
            *self._iter_days(self._curated_dir, window_start, window_end),
            *self._iter_days(self._raw_data_dir, window_start, window_end),
            *[
                datetime.strptime(day, "%Y-%m-%d").date()
                for day in expected_session_days
            ],
        })
        for day in days:
            day_key = day.isoformat()
            for event_type, path in iter_daily_event_input_paths(
                curated_dir=self._curated_dir,
                raw_data_dir=self._raw_data_dir,
                bot_id=bot_id,
                day=day,
            ):
                loaded = self._load_records(path, strategy_id=strategy_id)
                if loaded.records:
                    records[event_type].extend(loaded.records)
                    paths_by_type[event_type].append(_path_text(path))
                    active_session_days.add(day_key)
                    if event_type == "pipeline_funnel":
                        denominator_session_days.add(day_key)
                elif loaded.exists:
                    paths_by_type[event_type].append(_path_text(path))

        counts = {event_type: len(rows) for event_type, rows in records.items()}
        return _WindowEvidence(
            records=records,
            counts=counts,
            paths=_dedupe([path for paths in paths_by_type.values() for path in paths]),
            paths_by_type={key: _dedupe(paths) for key, paths in paths_by_type.items()},
            active_session_days=active_session_days,
            expected_session_days=expected_session_days,
            expected_session_paths=loaded_expected_session_paths,
            denominator_session_days=denominator_session_days,
        )

    def _load_expected_session_days(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        window_start: date,
        window_end: date,
        explicit_paths: list[Path],
    ) -> tuple[set[str], list[str]]:
        days: set[str] = set()
        loaded_paths: list[str] = []
        for path in self._candidate_sidecar_paths(
            bot_id=bot_id,
            window_start=window_start,
            window_end=window_end,
            explicit_paths=explicit_paths,
            filenames=EXPECTED_SESSION_FILENAMES,
        ):
            payload = _read_json_payload(path)
            if payload is None:
                continue
            path_days = _expected_days_from_payload(
                payload,
                bot_id=bot_id,
                strategy_id=strategy_id,
                window_start=window_start,
                window_end=window_end,
            )
            if path_days:
                days.update(path_days)
                loaded_paths.append(_path_text(path))
        return days, _dedupe(loaded_paths)

    def _build_runtime_evidence_support(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        window_start: date,
        window_end: date,
        evidence: "_WindowEvidence",
        deployment_metadata_count: int,
        deployment_metadata_paths: list[Path],
        strategy_contract_path: Path | None,
        runtime_support_paths: list[Path],
    ) -> dict[str, RuntimeEvidenceSupport]:
        declarations = {
            event_class: {
                "configured_event_types": [],
                "support_source_paths": [],
                "event_value_classifications": {},
                "declared_unavailable": False,
                "reason": "",
            }
            for event_class in RUNTIME_EVIDENCE_CLASSES
        }
        support_paths = self._candidate_sidecar_paths(
            bot_id=bot_id,
            window_start=window_start,
            window_end=window_end,
            explicit_paths=runtime_support_paths,
            filenames=RUNTIME_SUPPORT_FILENAMES,
        )
        for path in support_paths:
            payload = _read_json_payload(path)
            if isinstance(payload, dict):
                _merge_runtime_support_payload(
                    declarations,
                    payload,
                    declaration_path=_path_text(path),
                    bot_id=bot_id,
                    strategy_id=strategy_id,
                )
        _merge_unavailable_contract_support(declarations, strategy_contract_path)

        observed_counts = {**evidence.counts, "deployment_metadata": deployment_metadata_count}
        observed_paths = {
            **evidence.paths_by_type,
            "deployment_metadata": [_path_text(path) for path in deployment_metadata_paths],
        }
        result: dict[str, RuntimeEvidenceSupport] = {}
        for event_class in RUNTIME_EVIDENCE_CLASSES:
            declaration = declarations[event_class]
            observed_count = int(observed_counts.get(event_class, 0))
            configured = bool(declaration["configured_event_types"])
            authority_configured = bool(_learning_authority_configured_event_types(declaration))
            declared_unavailable = bool(declaration["declared_unavailable"])
            if declared_unavailable or not configured:
                state = RuntimeEvidenceSupportState.UNSUPPORTED
            elif authority_configured and observed_count > 0:
                state = RuntimeEvidenceSupportState.OBSERVED
            else:
                state = RuntimeEvidenceSupportState.SUPPORTED_BUT_UNOBSERVED
            reason = str(declaration["reason"] or "")
            if not reason:
                if declared_unavailable:
                    reason = "declared_unavailable_by_strategy_contract"
                elif not configured:
                    reason = "runtime_support_not_declared"
                elif not authority_configured:
                    reason = "configured_runtime_support_not_learning_authority"
                elif state == RuntimeEvidenceSupportState.SUPPORTED_BUT_UNOBSERVED:
                    reason = "configured_runtime_support_has_no_observed_window_evidence"
            result[event_class] = RuntimeEvidenceSupport(
                evidence_class=event_class,
                support_state=state,
                configured_event_types=declaration["configured_event_types"],
                observed_event_count=observed_count,
                support_source_paths=declaration["support_source_paths"],
                observed_evidence_paths=observed_paths.get(event_class, []),
                event_value_classifications=declaration["event_value_classifications"],
                declared_unavailable=declared_unavailable,
                reason=reason,
            )
        return result

    def _candidate_sidecar_paths(
        self,
        *,
        bot_id: str,
        window_start: date,
        window_end: date,
        explicit_paths: list[Path],
        filenames: tuple[str, ...],
    ) -> list[Path]:
        paths = [Path(path) for path in explicit_paths if path]
        for root in (self._curated_dir, self._raw_data_dir):
            paths.extend(root / filename for filename in filenames)
            paths.extend(root / bot_id / filename for filename in filenames)
            for day in self._iter_days(root, window_start, window_end):
                paths.extend(root / day.isoformat() / bot_id / filename for filename in filenames)
        return [Path(path) for path in _dedupe([_path_text(path) for path in paths])]

    def _load_records(self, path: Path, *, strategy_id: str) -> "_LoadedRecords":
        if not path.exists():
            return _LoadedRecords(exists=False, records=[])
        rows: list[dict[str, Any]] = []
        try:
            if path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    rows.extend(item for item in payload if isinstance(item, dict))
                elif isinstance(payload, dict):
                    extracted = _extract_records_from_summary(payload)
                    rows.extend(extracted or [payload])
        except (json.JSONDecodeError, OSError):
            return _LoadedRecords(exists=True, records=[])
        if strategy_id:
            filtered = [row for row in rows if not event_strategy_id(row) or event_strategy_id(row) == strategy_id]
            rows = filtered
        return _LoadedRecords(exists=True, records=rows)

    def _load_deployment_metadata(self, paths: list[Path]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in paths:
            if not path or not Path(path).exists():
                continue
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def _declared_not_applicable_checks(self, strategy_contract_path: Path | None) -> set[str]:
        if strategy_contract_path is None or not Path(strategy_contract_path).exists():
            return set()
        try:
            payload = json.loads(Path(strategy_contract_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return set()
        declared: set[str] = set()
        for key in (
            "not_applicable_learning_checks",
            "declared_not_applicable_checks",
            "unavailable_learning_surfaces",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                declared.update(str(item) for item in value if str(item))
        return declared

    @staticmethod
    def _iter_days(root: Path, window_start: date, window_end: date) -> list[date]:
        if not root.exists():
            return []
        days: list[date] = []
        for date_dir in sorted(root.iterdir()):
            if not date_dir.is_dir():
                continue
            try:
                day = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if window_start <= day <= window_end:
                days.append(day)
        return days

    @staticmethod
    def _infer_raw_dir(curated_dir: Path) -> Path:
        if curated_dir.name == "curated":
            return curated_dir.parent / "raw"
        return curated_dir.parent / "raw"


class _LoadedRecords:
    def __init__(self, *, exists: bool, records: list[dict[str, Any]]) -> None:
        self.exists = exists
        self.records = records


class _WindowEvidence:
    def __init__(
        self,
        *,
        records: dict[str, list[dict[str, Any]]],
        counts: dict[str, int],
        paths: list[str],
        paths_by_type: dict[str, list[str]],
        active_session_days: set[str],
        expected_session_days: set[str],
        expected_session_paths: list[str],
        denominator_session_days: set[str],
    ) -> None:
        self.records = records
        self.counts = counts
        self.paths = paths
        self.paths_by_type = paths_by_type
        self.active_session_days = active_session_days
        self.expected_session_days = expected_session_days
        self.expected_session_paths = expected_session_paths
        self.denominator_session_days = denominator_session_days


def _extract_records_from_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("records", "events", "snapshots", "items", "details", "rules", "orders", "fills"):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return list(value)
    if any(key.endswith("_count") or key in {"total", "count", "coverage"} for key in payload):
        return [payload]
    return []


def _has_any_value(record: dict[str, Any], field_names: tuple[str, ...]) -> bool:
    for field in field_names:
        value = event_value(record, field)
        if value not in (None, "", [], {}):
            return True
        lineage = event_value(record, "assistant_lineage")
        if isinstance(lineage, dict):
            lineage_value = lineage.get(field)
            if field == "source_weekly_signal_ids":
                lineage_value = lineage_value or lineage.get("weekly_signal_ids")
            if lineage_value not in (None, "", [], {}):
                return True
    return False


def _has_numeric_value(record: dict[str, Any], field_names: tuple[str, ...]) -> bool:
    for field in field_names:
        value = event_value(record, field)
        if isinstance(value, bool) or value in (None, "", [], {}):
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            continue
        return True
    return False


def _record_key_values(record: dict[str, Any], field_names: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for field in field_names:
        values.update(_key_values(event_value(record, field)))
    return values


def _key_values(value: Any) -> set[str]:
    if value in (None, "", [], {}):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {
            text for item in value
            if (text := str(item).strip()) and text.lower() not in PLACEHOLDER_KEY_VALUES
        }
    text = str(value).strip()
    return {text} if text and text.lower() not in PLACEHOLDER_KEY_VALUES else set()


def _order_record_identity_sets(record: dict[str, Any]) -> list[set[str]]:
    singular_ids = _record_key_values(record, SINGULAR_CANONICAL_ORDER_KEYS)
    plural_ids = {
        value
        for field in PLURAL_CANONICAL_ORDER_KEYS
        for value in _key_values(event_value(record, field))
    }
    if len(plural_ids) <= 1 or (singular_ids and _record_declares_order_aliases(record)):
        ids = singular_ids | plural_ids
        return [ids] if ids else []
    return [{value} for value in sorted(plural_ids)]


def _record_declares_order_aliases(record: dict[str, Any]) -> bool:
    if event_value(record, "order_ids_are_aliases") is True:
        return True
    relationship = str(event_value(record, "order_id_relationship") or "").strip().lower()
    return relationship in {"alias", "aliases", "equivalent", "equivalents"}


def _canonical_order_groups(records: list[dict[str, Any]]) -> tuple[list[tuple[set[str], bool]], int, int]:
    groups: list[tuple[set[str], bool]] = []
    missing_fill_required = 0
    missing_terminal = 0
    for record in records:
        identity_sets = _order_record_identity_sets(record)
        is_terminal = _explicit_non_fill_order(record)
        if not identity_sets:
            if is_terminal:
                missing_terminal += 1
            else:
                missing_fill_required += 1
            continue
        for ids in identity_sets:
            matches = [index for index, (group_ids, _) in enumerate(groups) if group_ids & ids]
            if not matches:
                groups.append((set(ids), is_terminal))
                continue
            first = matches[0]
            merged_ids, merged_terminal = groups[first]
            merged_ids.update(ids)
            merged_terminal = merged_terminal or is_terminal
            for index in reversed(matches[1:]):
                merged_ids.update(groups[index][0])
                merged_terminal = merged_terminal or groups[index][1]
                del groups[index]
            groups[first] = (merged_ids, merged_terminal)
    return groups, missing_fill_required, missing_terminal


def _explicit_non_fill_order(record: dict[str, Any]) -> bool:
    status = str(
        event_value(record, "status")
        or event_value(record, "order_status")
        or event_value(record, "execution_status")
        or ""
    ).lower()
    if status in {"cancelled", "canceled", "rejected", "expired", "no_order", "not_submitted"}:
        return True
    return _has_any_value(record, ("cancel_reason", "reject_reason", "no_order_reason"))


def _explicit_no_order_decisions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if _explicit_non_fill_order(record)]


def _explicit_risk_denials(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    denial_values = {"block", "blocked", "deny", "denied", "reject", "rejected"}
    result_keys = ("result", "decision", "status", "action", "risk_status")
    reason_keys = ("deny_reason", "denial_reason", "block_reason", "risk_denial_reason", "reject_reason")
    return [
        record for record in records
        if any(str(event_value(record, key) or "").lower() in denial_values for key in result_keys)
        or _has_any_value(record, reason_keys)
    ]


def _expected_value(capability_id: str) -> ExpectedLearningValue:
    if capability_id == "approval_grade_strategy_change":
        return ExpectedLearningValue.CRITICAL
    if capability_id in {"filter_threshold_learning", "execution_learning", "new_strategy_discovery"}:
        return ExpectedLearningValue.HIGH
    return ExpectedLearningValue.MEDIUM


def runtime_source_authority_for_checks(
    required_checks: Any,
    runtime_support: Any,
    checks: Any | None = None,
) -> tuple[list[str], list[str]]:
    if not isinstance(runtime_support, dict) or not runtime_support:
        return [f"{RUNTIME_BLOCKER_PREFIX}missing"], []
    check_map = checks if isinstance(checks, dict) else {}
    blockers: list[str] = []
    evidence_paths: list[str] = []
    for check_id in _string_list(required_checks):
        for event_class in _runtime_evidence_classes_for_check(check_id, check_map.get(check_id)):
            support = runtime_support.get(event_class)
            if support is None:
                blockers.append(f"{RUNTIME_BLOCKER_PREFIX}{check_id}:{event_class}:missing")
                continue
            evidence_paths.extend(_runtime_support_paths(support))
            state = _runtime_support_state_value(support)
            has_authority_source = _support_entry_has_learning_authority_source(support)
            if not has_authority_source and _runtime_support_reason(support) == "configured_runtime_support_not_learning_authority":
                blockers.append(f"{RUNTIME_BLOCKER_PREFIX}{check_id}:{event_class}:runtime_support_not_learning_authority")
            elif state != RuntimeEvidenceSupportState.OBSERVED.value:
                reason = _runtime_support_reason(support)
                suffix = f" ({reason})" if reason else ""
                blockers.append(f"{RUNTIME_BLOCKER_PREFIX}{check_id}:{event_class}:{state or 'missing'}{suffix}")
            elif not has_authority_source:
                blockers.append(f"{RUNTIME_BLOCKER_PREFIX}{check_id}:{event_class}:runtime_support_not_learning_authority")
    return _dedupe(blockers), _dedupe(evidence_paths)


def _runtime_evidence_classes_for_check(check_id: str, check: Any | None = None) -> list[str]:
    details = _check_details(check)
    classes = _string_list(details.get("consumed_runtime_event_classes"))
    if not classes:
        classes = list(CHECK_RUNTIME_EVENT_CLASSES.get(check_id, ()))
    return [
        event_class for event_class in _dedupe(_normalize_event_class(item) for item in classes)
        if event_class in RUNTIME_EVIDENCE_CLASSES
    ]


def _check_details(check: Any | None) -> dict[str, Any]:
    if check is None:
        return {}
    if isinstance(check, dict):
        details = check.get("details")
    else:
        details = getattr(check, "details", None)
    return details if isinstance(details, dict) else {}


def _runtime_support_paths(support: Any) -> list[str]:
    return _dedupe([
        *_string_list(_runtime_support_value(support, "support_source_paths")),
        *_string_list(_runtime_support_value(support, "observed_evidence_paths")),
    ])


def _runtime_support_state_value(support: Any) -> str:
    state = _runtime_support_value(support, "support_state")
    return str(getattr(state, "value", state) or "").strip().lower()


def _runtime_support_reason(support: Any) -> str:
    return str(_runtime_support_value(support, "reason") or "").strip()


def _runtime_support_value(support: Any, key: str) -> Any:
    return support.get(key) if isinstance(support, dict) else getattr(support, key, None)


def _consumed_event_classes(*items: tuple[str, list[dict[str, Any]]]) -> list[str]:
    return [event_class for event_class, records in items if records]


def _blocking_reason(
    check_id: str,
    checks: dict[str, CoverageCheck],
    runtime_support: dict[str, RuntimeEvidenceSupport],
) -> str:
    if check_id.startswith(RUNTIME_BLOCKER_PREFIX):
        parts = check_id[len(RUNTIME_BLOCKER_PREFIX):].split(":")
        event_class = parts[1] if len(parts) > 1 else (parts[0] if parts else "")
        support = runtime_support.get(event_class)
        state = support.support_state.value if support is not None else "missing"
        return f"{check_id}:{state}"
    return f"{check_id}:{checks[check_id].status.value}"


def _read_json_payload(path: Path) -> Any | None:
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _expected_days_from_payload(
    payload: Any,
    *,
    bot_id: str,
    strategy_id: str,
    window_start: date,
    window_end: date,
) -> set[str]:
    days: set[str] = set()
    for item in _session_items(payload):
        if isinstance(item, str):
            parsed = _parse_day(item)
            if parsed and window_start <= parsed <= window_end:
                days.add(parsed.isoformat())
            continue
        if not isinstance(item, dict) or not _record_matches_scope(item, bot_id, strategy_id):
            continue
        if not _session_is_expected_active(item):
            continue
        parsed = _parse_day(_first_text(item, "date", "session_date", "active_session_date", "trading_day", "day"))
        if parsed and window_start <= parsed <= window_end:
            days.add(parsed.isoformat())
    return days


def _session_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in (
        "expected_session_days",
        "active_session_days",
        "trading_sessions",
        "sessions",
        "session_days",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    if all(_parse_day(str(key)) is not None for key in payload):
        return [
            {"date": str(key), "active": bool(value)}
            for key, value in payload.items()
        ]
    return [payload]


def _session_is_expected_active(record: dict[str, Any]) -> bool:
    for key in ("active", "expected", "is_active", "is_trading_day"):
        if key in record and record.get(key) is False:
            return False
    status = str(record.get("status") or "").strip().lower()
    return status not in {"closed", "inactive", "disabled", "cancelled", "canceled", "not_trading"}


def _merge_runtime_support_payload(
    declarations: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    *,
    declaration_path: str,
    bot_id: str,
    strategy_id: str,
) -> None:
    if not _record_matches_scope(payload, bot_id, strategy_id):
        return
    source_paths = _dedupe(_string_list(
        payload.get("support_source_paths")
        or payload.get("source_paths")
        or payload.get("runtime_support_source_paths")
        or [declaration_path]
    ))
    _merge_event_value_classifications(
        declarations,
        payload.get("event_value_classifications") or payload.get("event_value_classes"),
    )
    capabilities = payload.get("capabilities")
    if isinstance(capabilities, dict):
        for details in capabilities.values():
            if isinstance(details, dict):
                _merge_capability_support(declarations, details, source_paths)
    for key in ("runtime_evidence_support", "evidence_classes", "events", "support"):
        value = payload.get(key)
        if isinstance(value, dict):
            for event_class, details in value.items():
                _merge_event_support(declarations, str(event_class), details, source_paths)
        elif isinstance(value, list):
            for details in value:
                if isinstance(details, dict):
                    event_class = str(details.get("evidence_class") or details.get("event_type") or "")
                    _merge_event_support(declarations, event_class, details, source_paths)
    if payload.get("declares_complete_runtime_support"):
        for target in declarations.values():
            if not target["support_source_paths"]:
                target["support_source_paths"] = source_paths
                target["reason"] = target["reason"] or "not_configured_by_runtime_support_source"


def _merge_capability_support(
    declarations: dict[str, dict[str, Any]],
    details: dict[str, Any],
    source_paths: list[str],
) -> None:
    _merge_event_value_classifications(
        declarations,
        details.get("event_value_classifications") or details.get("event_value_classes"),
    )
    required = _string_list(details.get("required_event_types"))
    missing = {_normalize_event_class(item) for item in _string_list(details.get("missing_configured_event_types"))}
    detail_sources = _dedupe(_string_list(details.get("support_source_paths") or details.get("source_paths")) or source_paths)
    for event_type in required:
        event_class = _normalize_event_class(event_type)
        if event_class not in declarations:
            continue
        target = declarations[event_class]
        authority_aliases = _learning_authority_source_event_types_for_class(target, event_class)
        if event_class in missing:
            if not authority_aliases:
                target["support_source_paths"] = _dedupe([*target["support_source_paths"], *detail_sources])
                target["reason"] = target["reason"] or "not_configured_by_runtime_support_source"
                continue
            target["reason"] = "" if target["reason"] == "not_configured_by_runtime_support_source" else target["reason"]
        configured = authority_aliases or [str(event_type)]
        target["configured_event_types"] = _dedupe([*target["configured_event_types"], *configured])
        target["support_source_paths"] = _dedupe([*target["support_source_paths"], *detail_sources])


def _merge_event_support(
    declarations: dict[str, dict[str, Any]],
    event_class: str,
    details: Any,
    source_paths: list[str],
) -> None:
    normalized = _normalize_event_class(event_class)
    if normalized not in declarations:
        return
    detail = details if isinstance(details, dict) else {"support_state": details}
    target = declarations[normalized]
    detail_sources = _dedupe(_string_list(detail.get("support_source_paths") or detail.get("source_paths")) or source_paths)
    if _entry_declares_unavailable(detail):
        target["declared_unavailable"] = True
        target["reason"] = str(detail.get("reason") or "declared_unavailable_by_runtime_support")
    elif _entry_declares_support(detail):
        configured = _string_list(detail.get("configured_event_types")) or [str(event_class)]
        target["configured_event_types"] = _dedupe([*target["configured_event_types"], *configured])
        value_class = str(detail.get("event_value_class") or detail.get("value_class") or "").strip()
        if value_class:
            for event_type in configured:
                target["event_value_classifications"][str(event_type)] = value_class
    target["support_source_paths"] = _dedupe([*target["support_source_paths"], *detail_sources])


def _merge_event_value_classifications(
    declarations: dict[str, dict[str, Any]],
    classifications: Any,
) -> None:
    if not isinstance(classifications, dict):
        return
    for event_type, value_class in classifications.items():
        event_text = str(event_type or "").strip()
        class_text = str(value_class or "").strip()
        normalized = _normalize_event_class(event_text)
        if event_text and class_text and normalized in declarations:
            declarations[normalized]["event_value_classifications"][event_text] = class_text


def _learning_authority_configured_event_types(declaration: dict[str, Any]) -> list[str]:
    classifications = declaration.get("event_value_classifications", {})
    return [
        event_type for event_type in _string_list(declaration.get("configured_event_types"))
        if str(classifications.get(event_type) or "").strip().lower() == LEARNING_AUTHORITY_EVENT_VALUE_CLASS
    ]


def _manifest_payload_required_checks(payload: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    capability_status = payload.get("capability_status")
    if isinstance(capability_status, dict):
        for capability_id, details in capability_status.items():
            if not isinstance(details, dict):
                continue
            if str(details.get("status") or "").strip().lower() == LearningCapabilityAuthority.LEARNING_AUTHORITATIVE.value:
                checks.extend(_string_list(details.get("required_checks")) or CAPABILITY_REQUIREMENTS.get(capability_id, ()))
    for capability_id in _string_list(payload.get("supported_learning_capabilities")):
        checks.extend(CAPABILITY_REQUIREMENTS.get(capability_id, ()))
    return _dedupe(checks)


def _manifest_payload_checks(payload: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for key in ("required_event_coverage", "join_coverage", "denominator_coverage"):
        value = payload.get(key)
        if isinstance(value, dict):
            checks.update({str(check_id): details for check_id, details in value.items() if isinstance(details, dict)})
    for check_id in (
        "lineage_coverage",
        "after_cost_coverage",
        "counterfactual_coverage",
        "proposal_trace_coverage",
        "deployment_metadata_coverage",
    ):
        details = payload.get(check_id)
        if isinstance(details, dict):
            checks[check_id] = details
    return checks


def manifest_runtime_support_source_authoritative(payload: dict[str, Any]) -> bool:
    runtime_support = payload.get("runtime_evidence_support")
    if not isinstance(runtime_support, dict):
        return False
    required_checks = _manifest_payload_required_checks(payload)
    if required_checks:
        blockers, _ = runtime_source_authority_for_checks(
            required_checks,
            runtime_support,
            _manifest_payload_checks(payload),
        )
        return not blockers
    return all(
        _runtime_support_state_value(details) != RuntimeEvidenceSupportState.OBSERVED.value
        or _support_entry_has_learning_authority_source(details)
        for details in runtime_support.values()
    )


def _support_entry_has_learning_authority_source(details: Any) -> bool:
    classifications = _runtime_support_value(details, "event_value_classifications")
    if not isinstance(classifications, dict):
        classifications = {}
    return any(
        str(classifications.get(event_type) or "").strip().lower() == LEARNING_AUTHORITY_EVENT_VALUE_CLASS
        for event_type in _string_list(_runtime_support_value(details, "configured_event_types"))
    )


def _learning_authority_source_event_types_for_class(declaration: dict[str, Any], event_class: str) -> list[str]:
    return [
        str(event_type)
        for event_type, value_class in sorted(declaration.get("event_value_classifications", {}).items())
        if _normalize_event_class(event_type) == event_class
        and str(value_class or "").strip().lower() == LEARNING_AUTHORITY_EVENT_VALUE_CLASS
    ]


def _merge_unavailable_contract_support(
    declarations: dict[str, dict[str, Any]],
    strategy_contract_path: Path | None,
) -> None:
    payload = _read_json_payload(Path(strategy_contract_path)) if strategy_contract_path else None
    if not isinstance(payload, dict):
        return
    for key in ("unavailable_learning_surfaces", "unsupported_runtime_evidence", "unsupported_evidence_classes"):
        for item in _string_list(payload.get(key)):
            event_class = _normalize_event_class(CHECK_EVENT_TYPE.get(item, item))
            if event_class in declarations:
                declarations[event_class]["declared_unavailable"] = True
                declarations[event_class]["support_source_paths"] = _dedupe([
                    *declarations[event_class]["support_source_paths"],
                    str(strategy_contract_path),
                ])


def _entry_declares_unavailable(detail: dict[str, Any]) -> bool:
    state = str(detail.get("support_state") or detail.get("status") or "").strip().lower()
    return bool(detail.get("declared_unavailable")) or state in {"unsupported", "unavailable"}


def _entry_declares_support(detail: dict[str, Any]) -> bool:
    if detail.get("supported") is False or detail.get("configured") is False:
        return False
    if detail.get("supported") is True or detail.get("configured") is True:
        return True
    state = str(detail.get("support_state") or detail.get("status") or "supported").strip().lower()
    return state in {"supported", "supported_but_unobserved", "observed", "configured", "configured_unobserved"}


def _normalize_event_class(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "deployment": "deployment_metadata",
        "portfolio_rule_check": "portfolio_rule",
        "risk_decision": "portfolio_rule",
        "risk_decisions": "portfolio_rule",
        "pipeline_funnels": "pipeline_funnel",
        "inferred_fill": "fill",
    }
    return aliases.get(text, text)


def canonical_runtime_event_class(value: Any) -> str:
    return _normalize_event_class(value)


def _record_matches_scope(record: dict[str, Any], bot_id: str, strategy_id: str) -> bool:
    record_bot = str(record.get("bot_id") or "")
    record_strategy = str(record.get("strategy_id") or "")
    return (not record_bot or record_bot == bot_id) and (not record_strategy or record_strategy == strategy_id)


def _first_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _parse_day(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _append_unique(values: list[str], value: str) -> list[str]:
    text = str(value or "")
    if text and text not in values:
        values.append(text)
    return values


def _path_text(path: Path | str | None) -> str:
    return "" if path is None else str(path).replace("\\", "/")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
