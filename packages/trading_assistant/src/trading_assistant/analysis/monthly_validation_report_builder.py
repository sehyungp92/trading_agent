"""Markdown report builder for monthly validation."""
from __future__ import annotations

from trading_assistant.schemas.monthly_validation import MonthlyValidationResult


class MonthlyValidationReportBuilder:
    def build(self, result: MonthlyValidationResult) -> str:
        lines = [
            f"# Monthly Validation: {result.bot_id} / {result.strategy_id} / {result.run_month}",
            "",
            f"Status: {result.status.value}",
            f"Shadow mode: {str(result.shadow).lower()}",
            f"Objective version: {result.objective_version}",
            "",
            "## Evidence",
        ]
        for path in result.evidence_paths:
            lines.append(f"- {path}")
        if result.blocking_reasons:
            lines.extend(["", "## Blocking Reasons"])
            for reason in result.blocking_reasons:
                lines.append(f"- {reason}")
        if result.repair_required or result.repair_request_path:
            lines.extend(["", "## Repair Request"])
            if result.repair_request_path:
                lines.append(f"Request: {result.repair_request_path}")
            lines.append("Approval packets remain blocked until the repair request is resolved.")
        if result.selected_candidate_count or result.rejected_candidate_count or result.candidate_summary_path:
            lines.extend([
                "",
                "## Candidate Generation",
                f"Selected: {result.selected_candidate_count}",
                f"Rejected: {result.rejected_candidate_count}",
                f"Gate-passed: {result.gate_passed_candidate_count}",
                f"Approval-ready: {result.approval_ready_candidate_count}",
            ])
            if result.candidate_summary_path:
                lines.append(f"Summary: {result.candidate_summary_path}")
            if result.model_review_validation_path:
                model_status = "valid" if result.model_review_valid else "invalid"
                lines.append(f"Model review: {model_status} ({result.model_review_validation_path})")
            if result.approval_request_ids:
                lines.append(f"Approval requests: {', '.join(result.approval_request_ids)}")
        lines.extend([
            "",
            "## Gap Attribution",
            f"Primary: {result.gap_attribution.primary_category.value}",
            result.gap_attribution.summary or "No attribution summary.",
        ])
        return "\n".join(lines).strip() + "\n"
