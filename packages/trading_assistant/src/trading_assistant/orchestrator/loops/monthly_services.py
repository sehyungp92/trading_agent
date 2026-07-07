"""Monthly loop support services."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from trading_assistant.orchestrator.orchestrator_brain import Action
from trading_assistant.skills.performance_learning_ledger import (
    PerformanceLearningRefreshMarkerError,
)

logger = logging.getLogger(__name__)


def bool_detail(details: dict, key: str, default: bool) -> bool:
    if key not in details:
        return default
    value = details.get(key)
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    return []


def optional_path(value: object) -> Path | None:
    text = str(value or "").strip()
    return Path(text) if text else None


def optional_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def monthly_stage_status_for_result(result: Any) -> dict:
    artifact_index_path = str(getattr(result, "artifact_index_path", "") or "")
    if not artifact_index_path:
        return {}
    path = Path(artifact_index_path).parent / "runner_observability.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    entries = payload if isinstance(payload, list) else [payload]
    for entry in reversed(entries):
        if isinstance(entry, dict) and isinstance(entry.get("monthly_stage_status"), dict):
            return entry["monthly_stage_status"]
    return {}


class MonthlyRunRecorder:
    def __init__(self, *, run_history_path: Path, runs_dir: Path) -> None:
        self._run_history_path = run_history_path
        self._runs_dir = runs_dir

    def record_run(
        self,
        run_id: str,
        agent_type: str,
        status: str,
        started_at: str = "",
        finished_at: str = "",
        error: str = "",
        duration_ms: int = 0,
        metadata: dict | None = None,
    ) -> None:
        try:
            self._run_history_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "run_id": run_id,
                "agent_type": agent_type,
                "handler": agent_type,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_ms": duration_ms,
                "error": error,
            }
            if metadata:
                entry["metadata"] = metadata
            with open(self._run_history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.warning("Failed to write run history for %s", run_id)

    def write_artifact_index(
        self,
        *,
        run_id: str,
        started_at: str,
        finished_at: str,
        results: list,
    ) -> Path:
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result_payloads = [result.model_dump(mode="json") for result in results]
        artifact_roots: list[str] = []
        seen_roots: set[str] = set()
        for result in results:
            for path in (
                getattr(result, "monthly_report_path", ""),
                getattr(result, "run_manifest_path", ""),
                getattr(result, "artifact_index_path", ""),
            ):
                if not str(path):
                    continue
                root = str(Path(path).parent)
                if root not in seen_roots:
                    seen_roots.add(root)
                    artifact_roots.append(root)
        stage_statuses: list[dict] = []
        for root in artifact_roots:
            observability_path = Path(root) / "runner_observability.json"
            if not observability_path.exists():
                continue
            try:
                payload = json.loads(observability_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            entries = payload if isinstance(payload, list) else [payload]
            for entry in entries:
                if isinstance(entry, dict) and isinstance(entry.get("monthly_stage_status"), dict):
                    stage_statuses.append({
                        "artifact_root": root,
                        "monthly_stage_status": entry["monthly_stage_status"],
                    })
        metadata = {
            "agent_type": "monthly_validation",
            "workflow": "monthly_validation",
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "bot_id": getattr(results[0], "bot_id", "") if len(results) == 1 else "",
            "strategy_id": getattr(results[0], "strategy_id", "") if len(results) == 1 else "",
            "monthly_artifact_roots": artifact_roots,
            "monthly_stage_status": stage_statuses,
            "results": result_payloads,
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )
        (run_dir / "monthly_validation_results.json").write_text(
            json.dumps(result_payloads, indent=2, default=str),
            encoding="utf-8",
        )
        return run_dir


class ScheduledMonthlyProjection:
    def __init__(self, *, scheduled_run_store: Any, memory_dir: Path) -> None:
        self._scheduled_run_store = scheduled_run_store
        self._memory_dir = memory_dir

    async def signal_result(self, action: Action, *, success: bool, error: str = "") -> None:
        if self._scheduled_run_store is None:
            return
        details = action.details or {}
        marker = details.get("__scheduled_run__")
        if not isinstance(marker, dict):
            return
        try:
            scheduled_for = datetime.fromisoformat(marker["scheduled_for"])
        except (KeyError, ValueError, TypeError):
            return
        try:
            if success:
                await self._scheduled_run_store.mark_completed(
                    marker["job_key"],
                    marker["scope_key"],
                    scheduled_for,
                )
            else:
                await self._scheduled_run_store.mark_failed(
                    marker["job_key"],
                    marker["scope_key"],
                    scheduled_for,
                    error=error or "Scheduled handler did not complete successfully",
                )
        except Exception:
            logger.warning("Failed to mark scheduled run result: %s", marker)

    async def project_results(
        self,
        action: Action,
        *,
        results: list,
        run_id: str,
        run_dir: Path | None,
        duration_ms: int,
        error: str = "",
    ) -> None:
        if self._scheduled_run_store is None or not results:
            return
        details = action.details or {}
        marker = details.get("__scheduled_run__")
        if not isinstance(marker, dict) or marker.get("job_key") != "monthly_validation":
            return
        try:
            scheduled_for = datetime.fromisoformat(marker["scheduled_for"])
            records = await self._scheduled_run_store.get_records(
                marker["job_key"],
                marker["scope_key"],
                since=scheduled_for,
                until=scheduled_for,
            )
        except Exception:
            logger.warning("Failed to load scheduled monthly run for projection: %s", marker)
            return
        record = next((item for item in records if item.scheduled_for == scheduled_for), None)
        if record is None:
            return

        try:
            from trading_assistant.skills.loop_run_ledger import RuntimeLoopProjectionWriter

            RuntimeLoopProjectionWriter(self._memory_dir).project_record(
                record,
                run_metadata=_monthly_projection_run_metadata(
                    action,
                    results,
                    details=details,
                    marker=marker,
                    run_id=run_id,
                    run_dir=run_dir,
                    duration_ms=duration_ms,
                ),
                monthly_artifacts=_monthly_projection_artifacts(results, run_dir),
                blocking_reasons=_monthly_projection_blockers(results, error),
                approval_packet_paths=_monthly_projection_approval_packets(results),
                strategy_change_record_ids=_monthly_projection_strategy_change_ids(results),
                proposal_ids=_monthly_projection_proposal_ids(results),
                summary=_monthly_projection_summary(results),
            )
            try:
                from trading_assistant.skills.performance_learning_ledger import (
                    refresh_performance_learning_projection,
                )

                refresh_performance_learning_projection(self._memory_dir / "findings")
            except PerformanceLearningRefreshMarkerError:
                raise
            except Exception:
                logger.warning("Failed to refresh performance-learning projection for %s", run_id)
        except PerformanceLearningRefreshMarkerError:
            raise
        except Exception:
            logger.warning("Failed to project scheduled monthly artifacts for %s", run_id)


def _monthly_projection_artifacts(results: list, run_dir: Path | None) -> dict[str, str]:
    artifacts: dict[str, str] = {}

    def add(key: str, value: object) -> None:
        text = str(value or "").strip()
        if text:
            artifacts[key] = text

    if run_dir is not None:
        add("monthly_run_metadata", run_dir / "metadata.json")
        add("monthly_validation_results", run_dir / "monthly_validation_results.json")

    single = len(results) == 1
    for index, result in enumerate(results, start=1):
        scope = "" if single else _projection_scope(getattr(result, "strategy_id", "") or str(index))

        def key(name: str, scope: str = scope) -> str:
            return name if single else f"{scope}_{name}"

        add(key("artifact_index"), getattr(result, "artifact_index_path", ""))
        add(key("monthly_report"), getattr(result, "monthly_report_path", ""))
        add(key("run_manifest"), getattr(result, "run_manifest_path", ""))
        add(key("model_review"), getattr(result, "model_review_path", ""))
        add(key("model_review_validation"), getattr(result, "model_review_validation_path", ""))
        add(key("candidate_summary"), getattr(result, "candidate_summary_path", ""))
        add(key("candidate_gate_report"), getattr(result, "candidate_gate_report_path", ""))

        root = _monthly_artifact_root(result)
        if root is not None:
            add(key("runner_observability"), _existing_path(root / "runner_observability.json"))
            add(key("candidate_attempts"), _existing_path(root / "candidate_attempts.jsonl"))

        for item_index, path in enumerate(
            getattr(result, "monthly_evidence_verification_paths", []) or [],
            start=1,
        ):
            name = (
                "monthly_evidence_verification"
                if single and item_index == 1
                else f"monthly_evidence_verification_{item_index}"
            )
            add(key(name), path)
        for item_index, path in enumerate(getattr(result, "approval_packet_paths", []) or [], start=1):
            name = "approval_packet" if single and item_index == 1 else f"approval_packet_{item_index}"
            add(key(name), path)

    return artifacts


def _monthly_projection_blockers(results: list, error: str) -> list[str]:
    blockers: list[str] = []
    for result in results:
        blockers.extend(str(item) for item in (getattr(result, "blocking_reasons", []) or []))
    if error:
        blockers.append(error)
    return _dedupe_strings(blockers)


def _monthly_projection_approval_packets(results: list) -> list[str]:
    packets: list[str] = []
    for result in results:
        packets.extend(str(path) for path in (getattr(result, "approval_packet_paths", []) or []))
    return _dedupe_strings(packets)


def _monthly_projection_strategy_change_ids(results: list) -> list[str]:
    ids: list[str] = []
    for result in results:
        ids.append(str(getattr(result, "strategy_change_record_id", "") or ""))
        ids.extend(str(item) for item in (getattr(result, "proposed_strategy_change_record_ids", []) or []))
    return _dedupe_strings(ids)


def _monthly_projection_proposal_ids(results: list) -> list[str]:
    ids: list[str] = []
    for result in results:
        ids.extend(str(item) for item in (getattr(result, "proposal_ids", []) or []))
    return _dedupe_strings(ids)


def _monthly_projection_run_metadata(
    action: Action,
    results: list,
    *,
    details: dict,
    marker: dict,
    run_id: str,
    run_dir: Path | None,
    duration_ms: int,
) -> dict[str, object]:
    return {
        "bot_id": _projection_join_attr(results, "bot_id") or action.bot_id,
        "strategy_id": _projection_join_attr(results, "strategy_id"),
        "task_id": str(details.get("task_id") or marker.get("task_id") or ""),
        "provider": _projection_join_attr(results, "model_review_provider"),
        "model": _projection_join_attr(results, "model_review_model"),
        "cost_usd": _projection_sum_attr(results, "model_review_cost_usd"),
        "duration_ms": duration_ms,
        "run_id": run_id,
        "run_index_id": run_id,
        "run_dir": str(run_dir or ""),
    }


def _monthly_projection_summary(results: list) -> str:
    if not results:
        return "monthly_validation completed with no result artifacts"
    if len(results) == 1:
        result = results[0]
        status = _status_value(getattr(result, "status", "unknown"))
        bot_id = getattr(result, "bot_id", "")
        strategy_id = getattr(result, "strategy_id", "") or "all"
        run_month = getattr(result, "run_month", "")
        packets = len(getattr(result, "approval_packet_paths", []) or [])
        return (
            f"monthly_validation {status} for {bot_id}/{strategy_id}"
            f" {run_month}; {packets} approval packet(s)"
        )
    statuses = ", ".join(
        f"{getattr(result, 'strategy_id', '') or index}:{_status_value(getattr(result, 'status', 'unknown'))}"
        for index, result in enumerate(results, start=1)
    )
    return f"monthly_validation completed for {len(results)} strategy result(s): {statuses}"


def _monthly_artifact_root(result: Any) -> Path | None:
    for field in ("artifact_index_path", "monthly_report_path", "run_manifest_path"):
        value = str(getattr(result, field, "") or "")
        if value:
            return Path(value).parent
    return None


def _existing_path(path: Path) -> str:
    return str(path) if path.exists() else ""


def _projection_scope(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:48] or "result"


def _projection_join_attr(results: list, attr: str) -> str:
    return ", ".join(_dedupe_strings([
        str(getattr(result, attr, "") or "")
        for result in results
    ]))


def _projection_sum_attr(results: list, attr: str) -> float:
    total = 0.0
    for result in results:
        try:
            total += float(getattr(result, attr, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return round(total, 12)


def _status_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
