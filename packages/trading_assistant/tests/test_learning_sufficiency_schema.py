from __future__ import annotations

from datetime import date

from trading_contracts.models import MonthlyRunManifest as ContractMonthlyRunManifest

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
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus


def test_complete_learning_sufficiency_manifest_validates_and_derives_ids() -> None:
    manifest = LearningSufficiencyManifest(
        bot_id="crypto_trader",
        strategy_id="MomentumPullback_M15",
        family_id="crypto",
        portfolio_id="main",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path="telemetry_manifest.json",
        telemetry_authoritative_eligibility="authoritative",
        eligibility=LearningEligibility.LEARNING_AUTHORITATIVE,
        event_counts_by_type={"trade": 10, "missed_opportunity": 5, "pipeline_funnel": 31},
        required_event_coverage={
            "trade_outcome_lineage": CoverageCheck(
                check_id="trade_outcome_lineage",
                observed_count=10,
                required_count=10,
                required_fields=["strategy_id", "deployment_id", "config_version"],
                observed_fields=["strategy_id", "deployment_id", "config_version"],
                evidence_paths=["telemetry_manifest.json"],
            )
        },
        lineage_coverage=CoverageCheck(
            check_id="lineage_coverage",
            observed_count=15,
            required_count=15,
        ),
        join_coverage={
            "decision_to_order_join": CoverageCheck(
                check_id="decision_to_order_join",
                observed_count=10,
                required_count=10,
            )
        },
        denominator_coverage={
            "denominator_coverage": CoverageCheck(
                check_id="denominator_coverage",
                observed_count=31,
                required_count=31,
            )
        },
        after_cost_coverage=CoverageCheck(
            check_id="after_cost_coverage",
            observed_count=10,
            required_count=10,
        ),
        runtime_evidence_support={
            "trade": RuntimeEvidenceSupport(
                evidence_class="trade",
                support_state=RuntimeEvidenceSupportState.OBSERVED,
                configured_event_types=["trade"],
                observed_event_count=10,
                support_source_paths=["sidecar.py"],
                observed_evidence_paths=["trades.jsonl"],
            )
        },
        capability_status={
            "filter_threshold_learning": LearningCapabilityStatus(
                capability_id="filter_threshold_learning",
                status=LearningCapabilityAuthority.LEARNING_AUTHORITATIVE,
                required_checks=["trade_outcome_lineage", "denominator_coverage"],
                satisfied_checks=["trade_outcome_lineage", "denominator_coverage"],
                evidence_paths=["telemetry_manifest.json"],
            )
        },
        artifact_paths={
            "telemetry_manifest": "telemetry_manifest.json",
            "learning_sufficiency_manifest": "learning_sufficiency_manifest.json",
        },
        evidence_paths=["learning_sufficiency_manifest.json"],
    )

    payload = manifest.model_dump(mode="json")

    assert manifest.manifest_id
    assert manifest.total_events == 46
    assert manifest.is_learning_authoritative is True
    assert manifest.required_event_coverage["trade_outcome_lineage"].status == CoverageStatus.PASS
    assert manifest.supported_learning_capabilities == ["filter_threshold_learning"]
    assert "telemetry_manifest.json" in manifest.evidence_paths
    assert "sidecar.py" in manifest.evidence_paths
    assert payload["eligibility"] == "learning_authoritative"


def test_diagnostics_only_learning_sufficiency_manifest_accepts_empty_evidence() -> None:
    manifest = LearningSufficiencyManifest(
        bot_id="k_stock_trader",
        strategy_id="olr",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
    )

    assert manifest.eligibility == LearningEligibility.DIAGNOSTICS_ONLY
    assert manifest.total_events == 0
    assert manifest.lineage_coverage.status == CoverageStatus.UNKNOWN
    assert manifest.supported_learning_capabilities == []
    assert manifest.blocked_learning_capabilities == []


def test_not_applicable_requires_declared_contract_authority() -> None:
    undeclared = CoverageCheck(
        check_id="order_to_fill_join",
        status=CoverageStatus.NOT_APPLICABLE,
    )
    declared = CoverageCheck(
        check_id="order_to_fill_join",
        declared_not_applicable=True,
    )

    assert undeclared.satisfies_learning_authority is False
    assert declared.status == CoverageStatus.NOT_APPLICABLE
    assert declared.satisfies_learning_authority is True


def test_insufficient_lineage_manifest_records_actionable_gap() -> None:
    manifest = LearningSufficiencyManifest(
        bot_id="ibkr_trading",
        strategy_id="AKC_HELIX",
        family_id="stock",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        eligibility=LearningEligibility.INSUFFICIENT_LINEAGE,
        event_counts_by_type={"trade": 5},
        lineage_coverage=CoverageCheck(
            check_id="trade_outcome_lineage",
            observed_count=2,
            required_count=5,
            required_fields=["strategy_id", "deployment_id"],
            observed_fields=["strategy_id"],
        ),
        capability_status={
            "approval_grade_strategy_change": LearningCapabilityStatus(
                capability_id="approval_grade_strategy_change",
                status=LearningCapabilityAuthority.BLOCKED,
                required_checks=["trade_outcome_lineage"],
                blocking_checks=["trade_outcome_lineage"],
                blocking_reasons=["deployment_id missing from trade lineage"],
            )
        },
        known_gaps=[
            LearningGap(
                bot_id="ibkr_trading",
                strategy_id="AKC_HELIX",
                family_id="stock",
                event_type="trade",
                missing_field="deployment_id",
                blocked_learning_capability="approval_grade_strategy_change",
                expected_learning_value=ExpectedLearningValue.HIGH,
                frequency=3,
                remediation="Propagate deployment_id into trade events.",
            )
        ],
    )

    assert manifest.lineage_coverage.status == CoverageStatus.PARTIAL
    assert manifest.lineage_coverage.missing_fields == ["deployment_id"]
    assert manifest.blocked_learning_capabilities == ["approval_grade_strategy_change"]
    assert manifest.known_gaps[0].gap_id
    assert manifest.known_gaps[0].expected_learning_value == ExpectedLearningValue.HIGH


def test_old_monthly_manifest_payloads_validate_without_sufficiency_fields(tmp_path) -> None:
    payload = {
        "run_id": "monthly-bot1-strat1-2026-05",
        "run_month": "2026-05",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "latest_month_start": "2026-05-01",
        "latest_month_end": "2026-05-31",
        "market_data_manifest_path": str(tmp_path / "market_data_manifest.json"),
        "telemetry_manifest_path": str(tmp_path / "telemetry_manifest.json"),
        "artifact_root": str(tmp_path),
    }

    assistant_manifest = MonthlyRunManifest.model_validate(payload)
    contract_manifest = ContractMonthlyRunManifest.model_validate(payload)
    monthly_result = MonthlyValidationResult(
        run_id=payload["run_id"],
        run_month=payload["run_month"],
        bot_id=payload["bot_id"],
        strategy_id=payload["strategy_id"],
        status=MonthlyValidationStatus.INSUFFICIENT_LINEAGE,
    )

    assert assistant_manifest.learning_sufficiency_manifest_path == ""
    assert assistant_manifest.supported_learning_capabilities == []
    assert assistant_manifest.expected_session_paths == []
    assert assistant_manifest.runtime_support_paths == []
    assert contract_manifest.learning_sufficiency_status == ""
    assert contract_manifest.expected_session_paths == []
    assert contract_manifest.runtime_support_paths == []
    assert monthly_result.learning_sufficiency_manifest_path == ""
    assert monthly_result.blocked_learning_capabilities == []
