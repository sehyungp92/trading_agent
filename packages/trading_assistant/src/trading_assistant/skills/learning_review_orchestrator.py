"""Evidence-bounded post-run learning review for harness and memory curation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from trading_assistant.schemas.learning_card import CardType, LearningCard
from trading_assistant.skills.evidence_ref_utils import evidence_ref_exists_within_roots, is_virtual_evidence_ref
from trading_assistant.skills.learning_card_store import LearningCardStore
from trading_assistant.skills.learning_write_coordinator import LearningWriteCoordinator

_ALLOWED_WRITE_TARGETS = {
    "learning_reviews.jsonl",
    "focused_recall_cards.jsonl",
}

_ALLOWED_REVIEW_ACTION_TYPES = {
    "focused_recall",
    "provider_route_watch",
    "discarded_harness_warning",
    "run_parse_missing",
    "run_parse_warning",
    "validator_block_recall",
    "report_checklist_failure",
    "retrieval_context_warning",
    "monthly_candidate_recall",
    "monthly_model_review_issue",
    "monthly_approval_packet_issue",
    "approval_lifecycle_recall",
    "deployment_lifecycle_recall",
}

StructuredReviewInvoker = Callable[[dict[str, Any]], dict[str, Any]]


class LearningReviewOrchestrator:
    """Create focused recall cards and review records from measured evidence.

    This reviewer is deliberately non-authoritative for trading. It only curates
    harness/memory/provider evidence for future prompts and offline benchmarks.
    """

    def __init__(
        self,
        findings_dir: Path,
        *,
        review_mode: str = "deterministic",
        runs_dir: Path | None = None,
        structured_reviewer: StructuredReviewInvoker | None = None,
        disabled_workflows: list[str] | None = None,
    ) -> None:
        self._findings_dir = Path(findings_dir)
        self._review_mode = review_mode
        self._root_dir = _infer_root_dir(self._findings_dir)
        self._runs_dir = Path(runs_dir) if runs_dir is not None else self._root_dir / "runs"
        self._structured_reviewer = structured_reviewer
        self._disabled_workflows = {str(item) for item in (disabled_workflows or []) if str(item)}

    def run(self, *, week_start: str = "", week_end: str = "") -> dict[str, Any]:
        if self._review_mode == "disabled":
            return {
                "review_id": self._review_id(week_start, week_end, []),
                "week_start": week_start,
                "week_end": week_end,
                "reviewed_at": _now(),
                "scope": "analysis_harness_only",
                "trading_authority": "none",
                "review_mode": "disabled",
                "actions": [],
                "card_count": 0,
            }
        if self._review_mode not in {"deterministic", "llm_review"}:
            raise ValueError("review_mode must be disabled, deterministic, or llm_review")
        review_mode = "deterministic"
        review_notes: list[str] = []

        harness = self._window_jsonl(
            "harness_eval_results.jsonl",
            limit=12,
            week_start=week_start,
            week_end=week_end,
        )
        provider_scores = self._window_jsonl(
            "provider_route_scores.jsonl",
            limit=20,
            week_start=week_start,
            week_end=week_end,
        )
        discarded = self._window_jsonl(
            "discarded_harness_experiments.jsonl",
            limit=10,
            week_start=week_start,
            week_end=week_end,
        )

        actions: list[dict[str, Any]] = []
        cards: list[LearningCard] = []

        for result in harness:
            if result.get("variant_name") == "baseline":
                continue
            weak_metrics = [
                name for name, score in (result.get("per_metric") or {}).items()
                if _float(score) < 0.55
            ]
            failures = result.get("governance_failures") or []
            if result.get("kept") is False or weak_metrics or failures:
                action = {
                    "type": "focused_recall",
                    "source": "harness_eval",
                    "variant_name": result.get("variant_name", ""),
                    "weak_metrics": weak_metrics,
                    "governance_failures": failures,
                    "evidence_paths": ["memory/findings/harness_eval_results.jsonl"],
                }
                if self._action_has_evidence(action):
                    actions.append(action)
                    cards.append(self._card_from_action(action))

        for score in provider_scores:
            benchmark_quality = score.get("benchmark_quality")
            if benchmark_quality is not None and _float(benchmark_quality) < 0.6:
                action = {
                    "type": "provider_route_watch",
                    "source": "provider_route_scores",
                    "workflow": score.get("workflow", ""),
                    "provider": score.get("provider", ""),
                    "model": score.get("model", ""),
                    "benchmark_quality": benchmark_quality,
                    "evidence_paths": ["memory/findings/provider_route_scores.jsonl"],
                }
                if self._action_has_evidence(action):
                    actions.append(action)
                    cards.append(self._card_from_action(action))

        for entry in discarded:
            action = {
                "type": "discarded_harness_warning",
                "source": "discarded_harness_experiment",
                "variant_name": entry.get("variant_name", ""),
                "discard_reason": entry.get("discard_reason", ""),
                "future_warning_tags": entry.get("future_warning_tags", []),
                "evidence_paths": ["memory/findings/discarded_harness_experiments.jsonl"],
            }
            if self._action_has_evidence(action):
                actions.append(action)
                cards.append(self._card_from_action(action))

        for action in self._artifact_review_actions(
            limit=30,
            week_start=week_start,
            week_end=week_end,
        ):
            if self._action_has_evidence(action):
                actions.append(action)
                cards.append(self._card_from_action(action))

        for action in self._approval_deployment_actions(
            week_start=week_start,
            week_end=week_end,
        ):
            if self._action_has_evidence(action):
                actions.append(action)
                cards.append(self._card_from_action(action))

        if self._review_mode == "llm_review":
            if self._structured_reviewer is None:
                review_notes.append(
                    "llm_review requested without structured_reviewer; deterministic artifact review was used."
                )
            else:
                review_mode = "llm_review"
                structured_actions, structured_errors = self._run_structured_review(
                    week_start=week_start,
                    week_end=week_end,
                    deterministic_actions=actions,
                )
                review_notes.extend(structured_errors)
                for action in structured_actions:
                    actions.append(action)
                    cards.append(self._card_from_action(action))
        actions = [action for action in actions if self._action_workflow_enabled(action)]
        actions = _dedupe_actions(actions)
        cards = [self._card_from_action(action) for action in actions]

        review = {
            "review_id": self._review_id(week_start, week_end, actions),
            "week_start": week_start,
            "week_end": week_end,
            "reviewed_at": _now(),
            "scope": "analysis_harness_and_run_artifacts",
            "trading_authority": "none",
            "review_mode": review_mode,
            "requested_review_mode": self._review_mode,
            "actions": actions,
            "card_count": len(cards),
        }
        if review_notes:
            review["review_notes"] = review_notes
        if not self._review_already_written(review["review_id"]):
            self._write_grouped(review, cards)
        else:
            review["duplicate_skipped"] = True
        return review

    def _card_from_action(self, action: dict[str, Any]) -> LearningCard:
        title = _title(action)
        workflow = str(action.get("workflow", "") or "harness_meta_learning")
        tags = [
            "workflow:harness_meta_learning",
            f"source:{action.get('source', '')}",
            f"type:{action.get('type', '')}",
            *[str(tag) for tag in action.get("future_warning_tags", []) if str(tag)],
        ]
        return LearningCard(
            card_type=CardType.SYNTHESIS,
            source_id=(
                f"{action.get('type', '')}:"
                f"{action.get('variant_name') or action.get('provider') or action.get('run_id') or title}"
            ),
            bot_id=str(action.get("bot_id") or ""),
            title=title,
            content=json.dumps(action, sort_keys=True),
            evidence_summary=", ".join(action.get("evidence_paths", [])),
            tags=_dedupe(tags),
            source_workflow=workflow,
            confidence=0.75,
            impact_score=-0.2 if "warning" in action.get("type", "") else 0.1,
        )

    def _artifact_review_actions(
        self,
        *,
        limit: int,
        week_start: str,
        week_end: str,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for run_dir in self._recent_run_dirs(
            limit=limit,
            week_start=week_start,
            week_end=week_end,
        ):
            run_id = run_dir.name
            metadata = _read_json(run_dir / "metadata.json")
            workflow = str(metadata.get("agent_type") or metadata.get("workflow") or "")
            bot_id = str(metadata.get("bot_id") or "")
            strategy_id = str(metadata.get("strategy_id") or "")
            base = {
                "run_id": run_id,
                "workflow": workflow,
                "bot_id": bot_id,
                "strategy_id": strategy_id,
            }
            parsed_path = run_dir / "parsed_analysis.json"
            response_path = run_dir / "response.md"
            validator_path = run_dir / "validator_notes.md"
            checklist_path = _first_existing(run_dir, [
                "report_checklist.json",
                "report_checklist_result.json",
                "checklist.json",
            ])
            if response_path.exists() and not parsed_path.exists():
                actions.append({
                    **base,
                    "type": "run_parse_missing",
                    "source": "run_artifact_review",
                    "summary": "Run has response.md but no parsed_analysis.json for downstream learning.",
                    "evidence_paths": [_rel_to_root(response_path, self._root_dir)],
                })
            if parsed_path.exists():
                parsed = _read_json(parsed_path)
                if parsed.get("fallback_used") or parsed.get("parse_success") is False:
                    actions.append({
                        **base,
                        "type": "run_parse_warning",
                        "source": "run_artifact_review",
                        "summary": "Parser fallback or parse failure should be recalled before similar future runs.",
                        "evidence_paths": [_rel_to_root(parsed_path, self._root_dir)],
                    })
            if validator_path.exists():
                notes = validator_path.read_text(encoding="utf-8", errors="ignore")[:20_000]
                if "blocked" in notes.lower():
                    actions.append({
                        **base,
                        "type": "validator_block_recall",
                        "source": "run_artifact_review",
                        "summary": "Validator blocked one or more proposals in this run.",
                        "evidence_paths": [_rel_to_root(validator_path, self._root_dir)],
                    })
            if checklist_path is not None:
                checklist = _read_json(checklist_path)
                if _checklist_failed(checklist):
                    actions.append({
                        **base,
                        "type": "report_checklist_failure",
                        "source": "run_artifact_review",
                        "summary": "Report checklist failed or was incomplete.",
                        "evidence_paths": [_rel_to_root(checklist_path, self._root_dir)],
                    })
            metadata_path = run_dir / "metadata.json"
            retrieval_ids = _retrieval_ids(metadata)
            if retrieval_ids and self._run_has_review_signal(run_dir):
                actions.append({
                    **base,
                    "type": "retrieval_context_warning",
                    "source": "run_artifact_review",
                    "summary": "Run had parser/validator/checklist signals; preserve retrieved context IDs for future recall.",
                    "retrieved_card_ids": retrieval_ids.get("learning_card_ids", []),
                    "retrieved_playbook_ids": retrieval_ids.get("generated_playbook_ids", []),
                    "evidence_paths": [_rel_to_root(metadata_path, self._root_dir)],
                })
            for artifact_root in self._artifact_roots_for_run(run_dir, metadata):
                actions.extend(self._monthly_artifact_actions(artifact_root, base=base))
        return actions

    def _artifact_roots_for_run(self, run_dir: Path, metadata: dict[str, Any]) -> list[Path]:
        roots = [run_dir]

        def add_root(value: Any) -> None:
            path_value = str(value or "")
            if not path_value:
                return
            path = Path(path_value)
            if not path.is_absolute():
                path = self._root_dir / path
            try:
                resolved = path.resolve()
                resolved.relative_to(self._root_dir.resolve())
            except (OSError, ValueError):
                return
            if resolved.exists() and resolved.is_dir():
                roots.append(resolved)

        for key in (
            "artifact_root",
            "backtest_artifact_root",
            "monthly_artifact_root",
            "output_dir",
            "output_root",
        ):
            add_root(metadata.get(key))
        for key in (
            "artifact_roots",
            "backtest_artifact_roots",
            "monthly_artifact_roots",
        ):
            values = metadata.get(key) or []
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list | tuple):
                for value in values:
                    add_root(value)
        for result in metadata.get("results", []) or []:
            if not isinstance(result, dict):
                continue
            for key in (
                "monthly_report_path",
                "run_manifest_path",
                "artifact_index_path",
                "candidate_summary_path",
                "candidate_gate_report_path",
            ):
                value = str(result.get(key) or "")
                if value:
                    add_root(str(Path(value).parent))
        return _dedupe_paths(roots)

    def _monthly_artifact_actions(
        self,
        artifact_root: Path,
        *,
        base: dict[str, Any],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        selected_path = artifact_root / "selected_candidates.json"
        candidate_results_path = artifact_root / "candidate_results.jsonl"
        selected = _read_json_list(selected_path)
        if selected:
            actions.append({
                **base,
                "type": "monthly_candidate_recall",
                "source": "monthly_candidate_artifacts",
                "candidate_ids": [
                    str(item.get("candidate_id") or item.get("id") or "")
                    for item in selected if isinstance(item, dict)
                ],
                "candidate_attempt_ids": [
                    str(item.get("candidate_attempt_id") or "")
                    for item in selected if isinstance(item, dict) and item.get("candidate_attempt_id")
                ],
                "source_weekly_signal_ids": _dedupe([
                    str(signal)
                    for item in selected if isinstance(item, dict)
                    for signal in (item.get("source_weekly_signal_ids") or [])
                    if str(signal)
                ]),
                "summary": "Monthly selected candidates were emitted; keep candidate lineage visible in future review.",
                "evidence_paths": [_rel_to_root(selected_path, self._root_dir)],
            })
            if not (artifact_root / "model_review.json").exists():
                actions.append({
                    **base,
                    "type": "monthly_model_review_issue",
                    "source": "monthly_candidate_artifacts",
                    "summary": "Selected monthly candidates exist but model_review.json is missing.",
                    "evidence_paths": [_rel_to_root(selected_path, self._root_dir)],
                })
        elif candidate_results_path.exists() and candidate_results_path.stat().st_size > 0:
            actions.append({
                **base,
                "type": "monthly_candidate_recall",
                "source": "monthly_candidate_artifacts",
                "summary": "Monthly candidate_results.jsonl exists without selected_candidates.json selection.",
                "evidence_paths": [_rel_to_root(candidate_results_path, self._root_dir)],
            })

        model_validation_path = artifact_root / "model_review_validation.json"
        if model_validation_path.exists():
            validation = _read_json(model_validation_path)
            if validation.get("valid") is False or validation.get("issues"):
                actions.append({
                    **base,
                    "type": "monthly_model_review_issue",
                    "source": "monthly_model_review",
                    "summary": "Monthly model review validation reported issues.",
                    "issues": validation.get("issues", []),
                    "evidence_paths": [_rel_to_root(model_validation_path, self._root_dir)],
                })

        for packet_path in sorted(artifact_root.glob("*approval*packet*.json"))[:10]:
            packet = _read_json(packet_path)
            issue = _approval_packet_issue(packet)
            if issue:
                actions.append({
                    **base,
                    "type": "monthly_approval_packet_issue",
                    "source": "monthly_approval_packet",
                    "summary": issue,
                    "candidate_id": str(packet.get("candidate_id") or ""),
                    "evidence_paths": [_rel_to_root(packet_path, self._root_dir)],
                })
        return actions

    def _approval_deployment_actions(
        self,
        *,
        week_start: str,
        week_end: str,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        approvals_path = self._findings_dir / "approvals.jsonl"
        for row in self._window_jsonl(
            "approvals.jsonl",
            limit=30,
            week_start=week_start,
            week_end=week_end,
        ):
            monthly_run_id = str(row.get("monthly_run_id") or "")
            if not monthly_run_id:
                continue
            status = str(row.get("status") or "")
            evidence = _string_list(row.get("evidence_paths"))
            packet = str(row.get("approval_packet_path") or "")
            if packet:
                evidence.insert(0, packet)
            if not evidence:
                evidence = [_rel_to_root(approvals_path, self._root_dir)]
            actions.append({
                "type": "approval_lifecycle_recall",
                "source": "approval_tracker",
                "run_id": monthly_run_id,
                "workflow": "monthly_validation",
                "bot_id": str(row.get("bot_id") or ""),
                "strategy_id": str(row.get("strategy_id") or ""),
                "approval_request_id": str(row.get("request_id") or ""),
                "approval_status": status,
                "summary": f"Monthly approval request is {status or 'unknown'}.",
                "evidence_paths": evidence,
            })

        deployments_path = self._findings_dir / "deployments.jsonl"
        for row in self._window_jsonl(
            "deployments.jsonl",
            limit=30,
            week_start=week_start,
            week_end=week_end,
        ):
            status = str(row.get("status") or "")
            if status.lower() not in {"rolled_back", "rollback", "regression_detected", "failed"}:
                continue
            actions.append({
                "type": "deployment_lifecycle_recall",
                "source": "deployment_monitor",
                "run_id": str(row.get("monthly_run_id") or row.get("approval_request_id") or ""),
                "workflow": "deployment_monitoring",
                "bot_id": str(row.get("bot_id") or ""),
                "strategy_id": str(row.get("strategy_id") or ""),
                "deployment_id": str(row.get("deployment_id") or ""),
                "deployment_status": status,
                "summary": f"Deployment lifecycle warning: {status}.",
                "evidence_paths": [_rel_to_root(deployments_path, self._root_dir)],
            })
        return actions

    def _run_has_review_signal(self, run_dir: Path) -> bool:
        parsed = _read_json(run_dir / "parsed_analysis.json")
        if parsed.get("fallback_used") or parsed.get("parse_success") is False:
            return True
        validator_path = run_dir / "validator_notes.md"
        if validator_path.exists() and "blocked" in validator_path.read_text(encoding="utf-8", errors="ignore").lower():
            return True
        checklist_path = _first_existing(run_dir, [
            "report_checklist.json",
            "report_checklist_result.json",
            "checklist.json",
        ])
        return bool(checklist_path and _checklist_failed(_read_json(checklist_path)))

    def _run_structured_review(
        self,
        *,
        week_start: str,
        week_end: str,
        deterministic_actions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        assert self._structured_reviewer is not None
        payload = {
            "week_start": week_start,
            "week_end": week_end,
            "root_dir": str(self._root_dir),
            "allowed_write_targets": sorted(_ALLOWED_WRITE_TARGETS),
            "allowed_action_types": sorted(_ALLOWED_REVIEW_ACTION_TYPES),
            "deterministic_actions": deterministic_actions,
        }
        try:
            response = self._structured_reviewer(payload)
        except Exception as exc:
            return [], [f"structured reviewer failed: {exc}"]
        actions = response.get("actions") if isinstance(response, dict) else None
        if not isinstance(actions, list):
            return [], ["structured reviewer returned no actions list"]
        accepted: list[dict[str, Any]] = []
        errors: list[str] = []
        for item in actions[:20]:
            if not isinstance(item, dict):
                errors.append("structured reviewer emitted non-object action")
                continue
            action = dict(item)
            action.setdefault("source", "structured_learning_review")
            action_type = str(action.get("type") or "")
            if action_type not in _ALLOWED_REVIEW_ACTION_TYPES:
                errors.append(f"structured reviewer emitted disallowed action type: {action_type}")
                continue
            if not self._action_has_evidence(action):
                errors.append(f"structured reviewer emitted action without safe evidence: {action_type}")
                continue
            accepted.append(action)
        return accepted, errors

    def _recent_run_dirs(
        self,
        *,
        limit: int,
        week_start: str,
        week_end: str,
    ) -> list[Path]:
        roots = [self._runs_dir, self._root_dir / "data" / "runs"]
        dirs: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            dirs.extend(
                path for path in root.iterdir()
                if path.is_dir() and _path_in_window(path, week_start, week_end)
            )
        dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return dirs[:limit]

    def _write_grouped(self, review: dict[str, Any], cards: list[LearningCard]) -> None:
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        coordinator = LearningWriteCoordinator(self._findings_dir)
        group = coordinator.begin(
            source_workflow="learning_review",
            source_run_id=review.get("review_id", ""),
        )
        coordinator.add_jsonl_append(
            group,
            "record_learning_review",
            self._safe_target("learning_reviews.jsonl"),
            [review],
            dedup_key=f"learning_review:{review.get('review_id', '')}",
        )
        if cards:
            coordinator.add_jsonl_append(
                group,
                "record_focused_recall_cards",
                self._safe_target("focused_recall_cards.jsonl"),
                [card.model_dump(mode="json") for card in cards],
                dedup_key=f"focused_recall:{review.get('review_id', '')}",
            )
            coordinator.add_callback(
                group,
                "upsert_learning_cards",
                self._upsert_learning_cards,
                args=(cards,),
                dedup_key=f"learning_cards:{review.get('review_id', '')}",
            )
        coordinator.execute(group)

    def _upsert_learning_cards(self, cards: list[LearningCard]) -> None:
        store = LearningCardStore(self._findings_dir)
        index = store.load()
        for card in cards:
            index.add(card)
        store.save(index)

    def _latest_jsonl(self, filename: str, *, limit: int) -> list[dict[str, Any]]:
        return self._read_jsonl(filename)[-limit:]

    def _read_jsonl(self, filename: str) -> list[dict[str, Any]]:
        path = self._findings_dir / filename
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows

    def _window_jsonl(
        self,
        filename: str,
        *,
        limit: int,
        week_start: str,
        week_end: str,
    ) -> list[dict[str, Any]]:
        rows = self._read_jsonl(filename)
        if not week_start and not week_end:
            return rows[-limit:]
        return [
            row for row in rows
            if _date_in_window(_row_date(row), week_start, week_end)
        ][-limit:]

    def _action_has_evidence(self, action: dict[str, Any]) -> bool:
        paths = action.get("evidence_paths") or []
        if not isinstance(paths, list) or not paths:
            return False
        return all(self._safe_evidence_ref(str(path)) for path in paths)

    def _safe_evidence_ref(self, value: str) -> bool:
        if not value or ".." in value.replace("\\", "/"):
            return False
        normalized = value.replace("\\", "/")
        if normalized.startswith("memory/policies"):
            return False
        if is_virtual_evidence_ref(normalized):
            return True
        return evidence_ref_exists_within_roots(value, [self._root_dir, self._findings_dir])

    def _safe_target(self, target: str) -> str:
        if target not in _ALLOWED_WRITE_TARGETS:
            raise ValueError(f"learning review target not allowed: {target}")
        return target

    def _action_workflow_enabled(self, action: dict[str, Any]) -> bool:
        workflow = str(action.get("workflow") or "harness_meta_learning")
        return workflow not in self._disabled_workflows

    def _review_already_written(self, review_id: str) -> bool:
        if not review_id:
            return False
        for row in self._latest_jsonl("learning_reviews.jsonl", limit=1000):
            if row.get("review_id") == review_id:
                return True
        return False

    @staticmethod
    def _review_id(week_start: str, week_end: str, actions: list[dict[str, Any]]) -> str:
        import hashlib

        raw = json.dumps(
            {"week_start": week_start, "week_end": week_end, "actions": actions},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _title(action: dict[str, Any]) -> str:
    action_type = str(action.get("type", "") or "learning_review")
    if action_type == "provider_route_watch":
        return (
            f"Provider route watch: {action.get('workflow', '')} "
            f"{action.get('provider', '')}/{action.get('model', '')}"
        ).strip()
    if action.get("variant_name"):
        return f"Harness review: {action.get('variant_name')}"
    if action.get("run_id"):
        return f"Run artifact review: {action.get('run_id')}"
    return action_type.replace("_", " ").title()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in actions:
        key = json.dumps({
            "type": action.get("type", ""),
            "source": action.get("source", ""),
            "run_id": action.get("run_id", ""),
            "variant_name": action.get("variant_name", ""),
            "candidate_id": action.get("candidate_id", ""),
            "approval_request_id": action.get("approval_request_id", ""),
            "deployment_id": action.get("deployment_id", ""),
            "evidence_paths": action.get("evidence_paths", []),
        }, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    return []


def _retrieval_ids(metadata: dict[str, Any]) -> dict[str, list[str]]:
    learning = _string_list(
        metadata.get("_learning_card_ids")
        or metadata.get("learning_card_ids")
        or metadata.get("retrieved_card_ids")
    )
    playbooks = _string_list(
        metadata.get("_generated_playbook_ids")
        or metadata.get("generated_playbook_ids")
        or metadata.get("retrieved_playbook_ids")
    )
    result: dict[str, list[str]] = {}
    if learning:
        result["learning_card_ids"] = learning
    if playbooks:
        result["generated_playbook_ids"] = playbooks
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(value, dict):
        items = (
            value.get("candidates")
            or value.get("selected_candidates")
            or value.get("selected")
            or []
        )
    else:
        items = value
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _row_date(row: dict[str, Any]) -> str:
    for key in (
        "recorded_at",
        "created_at",
        "updated_at",
        "resolved_at",
        "measured_at",
        "deployed_at",
        "started_at",
        "completed_at",
        "date",
        "week_start",
    ):
        value = str(row.get(key) or "")
        if len(value) >= 10:
            return value[:10]
    return ""


def _date_in_window(value: str, week_start: str, week_end: str) -> bool:
    if not week_start and not week_end:
        return True
    if not value:
        return False
    if week_start and value < week_start:
        return False
    if week_end and value > week_end:
        return False
    return True


def _path_in_window(path: Path, week_start: str, week_end: str) -> bool:
    metadata_date = _row_date(_read_json(path / "metadata.json"))
    if metadata_date:
        return _date_in_window(metadata_date, week_start, week_end)
    if not week_start and not week_end:
        return True
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
    except OSError:
        return False
    return _date_in_window(mtime, week_start, week_end)


def _approval_packet_issue(packet: dict[str, Any]) -> str:
    if not packet:
        return ""
    suppressed = packet.get("approval_suppressed_reasons") or []
    if suppressed:
        return "Approval packet records suppressed approval reasons."
    payload = packet.get("machine_readable_payload") or {}
    if isinstance(payload, dict):
        model_validation = payload.get("model_review_validation") or {}
        if isinstance(model_validation, dict) and model_validation.get("valid") is False:
            return "Approval packet includes invalid monthly model review validation."
    payload_rollback_plan = payload.get("rollback_plan", "") if isinstance(payload, dict) else ""
    rollback_plan = str(packet.get("rollback_plan") or payload_rollback_plan)
    if not rollback_plan:
        return "Approval packet is missing rollback plan."
    evidence = _string_list(packet.get("artifact_paths") or packet.get("evidence_paths"))
    if not evidence and isinstance(payload, dict):
        evidence = _string_list(payload.get("evidence_paths"))
    if not evidence:
        return "Approval packet is missing evidence paths."
    return ""


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _checklist_failed(checklist: dict[str, Any]) -> bool:
    if not checklist:
        return False
    for key in ("passed", "complete", "is_complete", "all_passed"):
        if key in checklist:
            return checklist.get(key) is False
    results = checklist.get("results") or checklist.get("checks") or []
    if isinstance(results, list):
        return any(
            isinstance(item, dict)
            and str(item.get("status") or item.get("result") or "").lower() in {"fail", "failed", "missing"}
            for item in results
        )
    return False


def _rel_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _infer_root_dir(findings_dir: Path) -> Path:
    if findings_dir.name == "findings" and findings_dir.parent.name == "memory":
        return findings_dir.parent.parent
    return findings_dir.parent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
