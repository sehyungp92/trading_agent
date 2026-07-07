"""First-class monthly model-review request and invocation support."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any

from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex
from trading_assistant.schemas.monthly_candidates import MonthlyImprovementCandidate
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult
from trading_assistant.schemas.prompt_package import PromptPackage
from trading_assistant.skills.monthly_artifact_contract import MonthlyArtifactContract

MONTHLY_MODEL_REVIEW_PROMPT_VERSION = "monthly_model_review_v1"


@dataclass(frozen=True)
class MonthlyModelReviewInvocationResult:
    response: str
    provider: str = ""
    model: str = ""
    runtime: str = ""
    cost_usd: float = 0.0


@dataclass(frozen=True)
class MonthlyModelReviewRunResult:
    request_path: str = ""
    prompt_path: str = ""
    model_review_path: str = ""
    invoked: bool = False
    skipped_reason: str = ""
    error: str = ""
    provider: str = ""
    model: str = ""
    runtime: str = ""
    cost_usd: float = 0.0


MonthlyModelReviewInvoker = Callable[[PromptPackage, str], str | MonthlyModelReviewInvocationResult]


class MonthlyModelReviewRunner:
    """Prepare and optionally invoke the model layer after deterministic evidence.

    The runner never creates strategy changes directly. It writes a frozen request
    artifact, invokes the configured runtime when available, and stores the raw
    model output for the candidate pipeline to validate fail-closed.
    """

    def __init__(
        self,
        *,
        invoker: MonthlyModelReviewInvoker | None = None,
    ) -> None:
        self._invoker = invoker

    def run(
        self,
        *,
        monthly_result: MonthlyValidationResult,
        artifact_index: BacktestArtifactIndex,
        artifact_root: Path,
        existing_review_path: Path,
    ) -> MonthlyModelReviewRunResult:
        artifact_root = Path(artifact_root)
        if existing_review_path.exists():
            attribution = _load_invocation_metadata(existing_review_path)
            return MonthlyModelReviewRunResult(
                model_review_path=str(existing_review_path),
                skipped_reason="existing model review artifact",
                provider=str(attribution.get("provider") or ""),
                model=str(attribution.get("model") or ""),
                runtime=str(attribution.get("runtime") or ""),
                cost_usd=float(attribution.get("cost_usd") or 0.0),
            )

        artifact_contract = MonthlyArtifactContract.from_index(artifact_index)
        selected = [
            artifact_contract.normalize_candidate_paths(candidate)
            for candidate in artifact_contract.load_selected_candidates(
                bot_id=monthly_result.bot_id,
                strategy_id=monthly_result.strategy_id,
            )
        ]
        rejected = artifact_contract.load_rejected_candidates()
        if not selected:
            return MonthlyModelReviewRunResult(skipped_reason="no selected candidates")

        request = self._build_request(
            monthly_result=monthly_result,
            artifact_contract=artifact_contract,
            selected=selected,
            rejected=rejected,
        )
        request_path = artifact_root / "model_review_request.json"
        prompt_path = artifact_root / "model_review_prompt.md"
        request_path.write_text(json.dumps(request, indent=2, default=str), encoding="utf-8")
        prompt = self._build_prompt(request)
        prompt_path.write_text(prompt, encoding="utf-8")

        if self._invoker is None:
            return MonthlyModelReviewRunResult(
                request_path=str(request_path),
                prompt_path=str(prompt_path),
                skipped_reason="no monthly model-review invoker configured",
            )

        package = PromptPackage(
            system_prompt=_SYSTEM_PROMPT,
            task_prompt=prompt,
            instructions=_OUTPUT_INSTRUCTIONS,
            data=request,
            context_files=list(monthly_result.evidence_paths),
            metadata={
                "workflow": "monthly_model_review",
                "prompt_version": MONTHLY_MODEL_REVIEW_PROMPT_VERSION,
                "run_id": monthly_result.run_id,
            },
        )
        try:
            invocation = self._invoker(package, f"{monthly_result.run_id}-model-review")
        except Exception as exc:
            error_path = artifact_root / "model_review_error.json"
            error_path.write_text(
                json.dumps({"error": str(exc), "recorded_at": _now()}, indent=2),
                encoding="utf-8",
            )
            return MonthlyModelReviewRunResult(
                request_path=str(request_path),
                prompt_path=str(prompt_path),
                error=str(exc),
            )

        if isinstance(invocation, MonthlyModelReviewInvocationResult):
            response = invocation.response
            provider = invocation.provider
            model = invocation.model
            runtime = invocation.runtime
            cost_usd = invocation.cost_usd
        else:
            response = str(invocation)
            provider = ""
            model = ""
            runtime = ""
            cost_usd = 0.0
        existing_review_path.write_text(response, encoding="utf-8")
        _write_invocation_metadata(
            existing_review_path,
            monthly_result=monthly_result,
            provider=provider,
            model=model,
            runtime=runtime,
            cost_usd=cost_usd,
        )
        return MonthlyModelReviewRunResult(
            request_path=str(request_path),
            prompt_path=str(prompt_path),
            model_review_path=str(existing_review_path),
            invoked=True,
            provider=provider,
            model=model,
            runtime=runtime,
            cost_usd=cost_usd,
        )

    @staticmethod
    def _build_request(
        *,
        monthly_result: MonthlyValidationResult,
        artifact_contract: MonthlyArtifactContract,
        selected: list[MonthlyImprovementCandidate],
        rejected: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidate_evidence = [
            path
            for candidate in selected
            for path in [*candidate.evidence_paths, *candidate.artifact_paths]
            if path
        ]
        allowed_evidence_paths = _dedupe([
            *monthly_result.evidence_paths,
            monthly_result.monthly_report_path,
            monthly_result.candidate_summary_path,
            monthly_result.candidate_gate_report_path,
            *candidate_evidence,
        ])
        return {
            "prompt_version": MONTHLY_MODEL_REVIEW_PROMPT_VERSION,
            "generated_at": _now(),
            "run_id": monthly_result.run_id,
            "run_month": monthly_result.run_month,
            "bot_id": monthly_result.bot_id,
            "strategy_id": monthly_result.strategy_id,
            "monthly_status": monthly_result.status.value,
            "gap_attribution": monthly_result.gap_attribution.model_dump(mode="json"),
            "allowed_evidence_paths": allowed_evidence_paths,
            "artifact_index_path": artifact_contract.path_str(
                "artifact_index.json",
                require_exists=False,
            ),
            "selected_candidates": [
                candidate.model_dump(mode="json")
                for candidate in selected
            ],
            "rejected_candidates": rejected[:25],
            "governance": {
                "trading_authority": "monthly_full_fidelity_validation",
                "requires_human_approval": True,
                "model_review_can_only_route_or_hypothesize": True,
                "allowed_actionable_routes": [
                    "smoke_repair",
                    "phased_auto",
                    "experiment",
                    "manual_design_review",
                ],
            },
        }

    @staticmethod
    def _build_prompt(request: dict[str, Any]) -> str:
        return (
            "Review the deterministic monthly validation evidence and selected "
            "smoke-repair/phased-auto candidates. Use only the listed evidence "
            "paths and candidate payloads. Do not propose live trading actions, "
            "do not bypass human approval, and route unsupported ideas as "
            "hypothesis_only.\n\n"
            "Frozen request:\n"
            f"```json\n{json.dumps(request, indent=2, default=str)}\n```\n"
        )


_SYSTEM_PROMPT = (
    "You are the monthly model-review layer for a trading assistant. You inspect "
    "deterministic full-fidelity validation evidence after the replay layer has "
    "finished. You may recommend routing, structural hypotheses, or manual review, "
    "but you have no authority to deploy or alter trading logic."
)

_OUTPUT_INSTRUCTIONS = """Return only a JSON object wrapped in:
<!-- MONTHLY_MODEL_REVIEW
{...}
-->

Required top-level fields:
- run_id, bot_id, strategy_id
- candidate_reviews: one object per selected candidate
- structural_proposals: optional list
- rejected_actions: list

Every actionable candidate review must include candidate_id, recommendation,
evidence_paths, expected_objective_impact, replay_or_experiment_plan,
acceptance_criteria, rollback_plan, routing, risk_classification, and confidence.
"""


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _invocation_metadata_path(model_review_path: Path) -> Path:
    return Path(model_review_path).with_name("model_review_invocation.json")


def _load_invocation_metadata(model_review_path: Path) -> dict[str, Any]:
    path = _invocation_metadata_path(model_review_path)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write_invocation_metadata(
    model_review_path: Path,
    *,
    monthly_result: MonthlyValidationResult,
    provider: str,
    model: str,
    runtime: str,
    cost_usd: float = 0.0,
) -> None:
    path = _invocation_metadata_path(model_review_path)
    path.write_text(
        json.dumps(
            {
                "run_id": monthly_result.run_id,
                "run_month": monthly_result.run_month,
                "bot_id": monthly_result.bot_id,
                "strategy_id": monthly_result.strategy_id,
                "provider": provider,
                "model": model,
                "runtime": runtime,
                "cost_usd": cost_usd,
                "model_review_path": str(model_review_path),
                "recorded_at": _now(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
