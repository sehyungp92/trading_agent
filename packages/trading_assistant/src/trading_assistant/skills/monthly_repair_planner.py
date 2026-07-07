"""Deterministic monthly repair request builder."""
from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex
from trading_assistant.schemas.monthly_repair_request import MonthlyRepairClassification, MonthlyRepairRequest


class MonthlyRepairPlanner:
    """Classify incomplete monthly evidence into operator repair actions."""

    def build(
        self,
        *,
        run_id: str,
        bot_id: str,
        strategy_id: str,
        run_month: str,
        blocking_reasons: list[str],
        artifact_root: Path,
        artifact_index: BacktestArtifactIndex | None = None,
        runner_error: str = "",
        model_review_error: str = "",
        model_review_skipped_reason: str = "",
        evidence_paths: list[str] | None = None,
    ) -> MonthlyRepairRequest:
        missing = artifact_index.missing_required() if artifact_index is not None else []
        malformed = artifact_index.malformed_required() if artifact_index is not None else []
        if artifact_index is None and _mentions([*blocking_reasons, runner_error], "artifact_index"):
            missing.append("artifact_index.json")

        classification = self._classify(
            blocking_reasons=blocking_reasons,
            missing_artifacts=missing,
            malformed_artifacts=malformed,
            runner_error=runner_error,
            model_review_error=model_review_error,
            model_review_skipped_reason=model_review_skipped_reason,
        )
        owner, hints, retry = self._owner_hints_retry(classification, artifact_root)
        summary = self._summary(
            classification=classification,
            blocking_reasons=blocking_reasons,
            missing=missing,
            malformed=malformed,
            model_review_error=model_review_error,
            model_review_skipped_reason=model_review_skipped_reason,
            runner_error=runner_error,
        )
        return MonthlyRepairRequest(
            run_id=run_id,
            bot_id=bot_id,
            strategy_id=strategy_id,
            run_month=run_month,
            classification=classification,
            missing_artifact_keys=sorted(set(missing)),
            malformed_artifacts=sorted(set(malformed)),
            blocking_gates=list(dict.fromkeys([reason for reason in blocking_reasons if reason])),
            owner_component=owner,
            repair_command_hints=hints,
            retry_eligible=retry,
            evidence_paths=_dedupe(evidence_paths or []),
            human_summary=summary,
        )

    def write(self, request: MonthlyRepairRequest, artifact_root: Path) -> Path:
        path = Path(artifact_root) / "monthly_repair_request.json"
        path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _classify(
        *,
        blocking_reasons: list[str],
        missing_artifacts: list[str],
        malformed_artifacts: list[str],
        runner_error: str,
        model_review_error: str,
        model_review_skipped_reason: str,
    ) -> MonthlyRepairClassification:
        reasons = [*blocking_reasons, runner_error, model_review_error, model_review_skipped_reason]
        if model_review_error or "model-review invoker" in model_review_skipped_reason:
            return MonthlyRepairClassification.MODEL_REVIEW
        if _mentions(reasons, "strategy plugin contract", "strategy_plugin"):
            return MonthlyRepairClassification.STRATEGY_PLUGIN
        if _mentions(reasons, "data bundle checksum", "data checksum", "checksum does not match"):
            return MonthlyRepairClassification.DATA_CHECKSUM
        if _mentions(reasons, "schema_version", "schema version", "contract_version", "contract version"):
            return MonthlyRepairClassification.SCHEMA_VERSION
        if _mentions(reasons, "timed out", "timeout"):
            return MonthlyRepairClassification.RUNNER_TIMEOUT
        if _mentions(reasons, "stale artifacts", "older than run manifest"):
            return MonthlyRepairClassification.STALE_ARTIFACTS
        if _mentions(reasons, "outside artifact_root", "outside configured root", "path containment"):
            return MonthlyRepairClassification.PATH_CONTAINMENT
        if _mentions(reasons, "decision parity", "decision_parity"):
            return MonthlyRepairClassification.DECISION_PARITY
        if missing_artifacts or malformed_artifacts or _mentions(reasons, "artifact_index", "artifact index"):
            return MonthlyRepairClassification.ARTIFACT_CONTRACT
        if _mentions(reasons, "replay parity", "parity"):
            return MonthlyRepairClassification.REPLAY_PARITY
        if _mentions(reasons, "market data", "coverage", "data bundle is not authoritative"):
            return MonthlyRepairClassification.DATA
        if _mentions(reasons, "telemetry", "lineage"):
            return MonthlyRepairClassification.TELEMETRY
        if _mentions(reasons, "candidate", "selected_candidates"):
            return MonthlyRepairClassification.CANDIDATE_GENERATION
        if _mentions(reasons, "backtest", "runner", "command"):
            return MonthlyRepairClassification.BACKTEST_RUNNER
        return MonthlyRepairClassification.UNKNOWN

    @staticmethod
    def _owner_hints_retry(
        classification: MonthlyRepairClassification,
        artifact_root: Path,
    ) -> tuple[str, list[str], bool]:
        manifest_hint = f"Re-run the monthly backtest command with the manifest under {artifact_root}."
        mapping = {
            MonthlyRepairClassification.DATA: (
                "market_data_sync",
                ["Refresh market-data coverage and data-bundle manifests for the blocked bot/strategy/month."],
                True,
            ),
            MonthlyRepairClassification.TELEMETRY: (
                "lineage_audit",
                ["Repair missing strategy/config/deployment lineage in curated telemetry, then rebuild telemetry_manifest.json."],
                True,
            ),
            MonthlyRepairClassification.ARTIFACT_CONTRACT: (
                "backtest_runner",
                [manifest_hint, "Ensure artifact_index.json lists every required artifact and all JSON/JSONL artifacts parse."],
                True,
            ),
            MonthlyRepairClassification.STRATEGY_PLUGIN: (
                "strategy_plugin_owner",
                ["Publish a strategy_plugin_contract.json with shadow_validated or approval_ready maturity before optimizer runs."],
                True,
            ),
            MonthlyRepairClassification.DATA_CHECKSUM: (
                "market_data_sync",
                ["Regenerate the data bundle manifest and ensure coverage_manifest.json echoes the same bundle checksum."],
                True,
            ),
            MonthlyRepairClassification.SCHEMA_VERSION: (
                "contract_owner",
                ["Update the emitting repo to the current monthly contract schema versions before retrying."],
                True,
            ),
            MonthlyRepairClassification.RUNNER_TIMEOUT: (
                "backtest_runner",
                [manifest_hint, "Inspect runner_observability.json and increase or fix the timed-out runner stage."],
                True,
            ),
            MonthlyRepairClassification.STALE_ARTIFACTS: (
                "backtest_runner",
                [manifest_hint, "Delete stale artifacts or force the runner to rewrite every artifact after run_manifest.json."],
                True,
            ),
            MonthlyRepairClassification.PATH_CONTAINMENT: (
                "backtest_runner",
                ["Rewrite artifact_index.json and candidate paths so every artifact stays under artifact_root or candidate_workspace_root."],
                True,
            ),
            MonthlyRepairClassification.DECISION_PARITY: (
                "strategy_plugin_owner",
                ["Regenerate decision_parity_report.json against live strategy code and the backtest adapter before structural review."],
                True,
            ),
            MonthlyRepairClassification.REPLAY_PARITY: (
                "replay_adapter",
                ["Fix replay/live parity gaps or record deterministic known-gap explanations before candidate review."],
                True,
            ),
            MonthlyRepairClassification.CANDIDATE_GENERATION: (
                "monthly_candidate_pipeline",
                ["Regenerate selected/rejected candidate artifacts or record a deterministic no-candidate reason."],
                True,
            ),
            MonthlyRepairClassification.MODEL_REVIEW: (
                "monthly_model_review_invoker",
                ["Configure or retry the monthly model-review invoker; do not create approval packets from missing review evidence."],
                True,
            ),
            MonthlyRepairClassification.BACKTEST_RUNNER: (
                "backtest_runner",
                [manifest_hint, "Inspect stdout.log, stderr.log, and exit_status.json before retrying."],
                True,
            ),
        }
        return mapping.get(
            classification,
            ("operator", ["Inspect blocking reasons and monthly artifacts before retrying."], False),
        )

    @staticmethod
    def _summary(
        *,
        classification: MonthlyRepairClassification,
        blocking_reasons: list[str],
        missing: list[str],
        malformed: list[str],
        model_review_error: str,
        model_review_skipped_reason: str,
        runner_error: str,
    ) -> str:
        details: list[str] = []
        if blocking_reasons:
            details.append("; ".join(blocking_reasons))
        if missing:
            details.append("missing artifacts: " + ", ".join(sorted(set(missing))))
        if malformed:
            details.append("malformed artifacts: " + ", ".join(sorted(set(malformed))))
        if model_review_error:
            details.append(f"model review error: {model_review_error}")
        if model_review_skipped_reason:
            details.append(f"model review skipped: {model_review_skipped_reason}")
        if runner_error:
            details.append(f"runner error: {runner_error}")
        suffix = " ".join(details) if details else "No specific blocking detail was recorded."
        return f"Monthly validation requires {classification.value} repair. {suffix}"


def load_repair_request(path: Path) -> MonthlyRepairRequest | None:
    try:
        return MonthlyRepairRequest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:
        return None


def _mentions(values: list[str], *needles: str) -> bool:
    text = " ".join(str(value or "") for value in values).lower()
    return any(needle.lower() in text for needle in needles)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
