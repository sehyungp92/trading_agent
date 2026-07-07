"""Build bounded weekly-to-monthly search-prior briefs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trading_assistant.schemas.monthly_search_brief import MonthlySearchBrief
from trading_assistant.schemas.weekly_focus_rotation import (
    weekly_focus_payload,
    weekly_focus_rotation_payload,
)
from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger


class MonthlySearchBriefBuilder:
    """Create report-only monthly search guidance from existing evidence."""

    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = Path(findings_dir)

    def build(
        self,
        *,
        run_month: str,
        bot_id: str = "",
        strategy_id: str = "",
        report_only: bool = True,
        week_start: str = "",
    ) -> MonthlySearchBrief:
        evidence_paths: list[str] = []
        source_ids: list[str] = []
        focus: list[dict[str, Any]] = []
        phase_families: list[dict[str, Any]] = []
        seed_candidates: list[dict[str, Any]] = []
        ablations: list[dict[str, Any]] = []
        rollback_candidates: list[dict[str, Any]] = []
        negative_priors: list[dict[str, Any]] = []
        confidence_caps: list[dict[str, Any]] = []
        attribution: dict[str, list[str]] = {}

        retrospective_path = self._findings_dir / "retrospective_synthesis.jsonl"
        focus_week_start = week_start or _latest_week_start(retrospective_path) or f"{run_month}-01"
        weekly_focus = weekly_focus_payload(focus_week_start)
        weekly_focus_signal_id = f"weekly_focus:{weekly_focus['focus_id']}:{focus_week_start}"
        focus.append({
            "source_weekly_signal_id": weekly_focus_signal_id,
            "category": "weekly_focus_rotation",
            "family": weekly_focus["focus_id"],
            "summary": weekly_focus["monthly_handoff"],
            "authority": "search_order_only",
            "portfolio_families": weekly_focus["portfolio_families"],
            "strategy_ids": weekly_focus["strategy_ids"],
        })
        source_ids.append(weekly_focus_signal_id)
        for family in weekly_focus["portfolio_families"]:
            phase_families.append({
                "family": family,
                "priority": "weekly_focus_rotation",
                "reason": weekly_focus["monthly_handoff"],
                "source_weekly_signal_id": weekly_focus_signal_id,
                "authority": "search_order_only",
            })

        for row in _tail_jsonl(retrospective_path, limit=8):
            evidence_paths.append(_rel(retrospective_path))
            signal_id = str(row.get("week_start") or row.get("review_id") or row.get("recorded_at") or "")
            for keep in _list(row.get("keep")):
                hint = _hint_from_signal(keep, signal_id=signal_id, bot_id=bot_id)
                if hint:
                    focus.append(hint)
                    seed_candidates.append({**hint, "seed_type": "weekly_keep"})
                    source_ids.append(hint["source_weekly_signal_id"])
            for discard in _list(row.get("discard")):
                hint = _hint_from_signal(discard, signal_id=signal_id, bot_id=bot_id)
                if hint:
                    negative_priors.append(hint)
                    confidence_caps.append({
                        "source_weekly_signal_id": hint["source_weekly_signal_id"],
                        "cap": 0.4,
                        "reason": "one-week or discarded weekly signal is advisory only",
                        "authority": "search_order_only",
                    })
                    source_ids.append(hint["source_weekly_signal_id"])

        priors_path = self._findings_dir / "outcome_priors.jsonl"
        for prior in _tail_jsonl(priors_path, limit=50):
            if bot_id and prior.get("bot_id") not in ("", bot_id):
                continue
            if strategy_id and prior.get("strategy_id") not in ("", strategy_id):
                continue
            evidence_paths.append(_rel(priors_path))
            family = str(prior.get("mutation_family") or prior.get("category") or "unknown")
            if int(prior.get("negative_count") or 0) > 0:
                negative_priors.append({
                    "family": family,
                    "category": prior.get("category", ""),
                    "reason": "negative monthly outcome prior requires stronger evidence",
                    "source": "outcome_priors",
                    "authority": "search_order_only",
                })
                ablations.append({
                    "family": family,
                    "priority": "high_if_oos_repair_triggers",
                    "reason": "prior negative monthly outcome",
                    "authority": "search_order_only",
                })
            if int(prior.get("positive_count") or 0) > 0:
                phase_families.append({
                    "family": family,
                    "priority": "measured_positive",
                    "reason": "positive monthly outcome prior",
                    "authority": "search_order_only",
                })

        hypotheses_path = self._findings_dir / "hypotheses.jsonl"
        for hypothesis in _tail_jsonl(hypotheses_path, limit=50):
            status = str(hypothesis.get("status") or "active").lower()
            if status == "retired":
                continue
            evidence_paths.append(_rel(hypotheses_path))
            signal_id = str(hypothesis.get("id") or hypothesis.get("title") or "")
            if not signal_id:
                continue
            source_ids.append(signal_id)
            hint = {
                "source_weekly_signal_id": signal_id,
                "category": hypothesis.get("category", ""),
                "family": "structural_hypothesis",
                "summary": hypothesis.get("title", ""),
                "authority": "search_order_only",
            }
            focus.append(hint)
            if status == "active":
                seed_candidates.append({**hint, "seed_type": "active_hypothesis"})

        experiments_path = self._findings_dir / "structural_experiments.jsonl"
        for experiment in _tail_jsonl(experiments_path, limit=50):
            if bot_id and experiment.get("bot_id") not in ("", bot_id):
                continue
            evidence_paths.append(_rel(experiments_path))
            signal_id = str(experiment.get("experiment_id") or experiment.get("suggestion_id") or "")
            if not signal_id:
                continue
            status = str(experiment.get("status") or "").lower()
            source_ids.append(signal_id)
            hint = {
                "source_weekly_signal_id": signal_id,
                "category": experiment.get("hypothesis_id", "") or "structural",
                "family": "structural_experiment",
                "summary": experiment.get("title", ""),
                "authority": "search_order_only",
            }
            if status in {"active", "proposed"}:
                seed_candidates.append({**hint, "seed_type": f"{status}_structural_experiment"})
            elif status in {"failed", "abandoned"}:
                negative_priors.append({**hint, "reason": f"structural experiment {status}"})

        overrides_path = self._findings_dir / "category_overrides.jsonl"
        for override in _tail_jsonl(overrides_path, limit=50):
            if bot_id and override.get("bot_id") not in ("", bot_id):
                continue
            multiplier = _float_or_none(override.get("confidence_multiplier"))
            if multiplier is None or multiplier >= 1.0:
                continue
            evidence_paths.append(_rel(overrides_path))
            category = str(override.get("category") or "")
            confidence_caps.append({
                "source_weekly_signal_id": str(override.get("source_id") or f"category_override:{category}"),
                "category": category,
                "cap": multiplier,
                "reason": override.get("reason", "category recalibration reduced confidence"),
                "authority": "search_order_only",
            })
            negative_priors.append({
                "category": category,
                "family": category,
                "reason": "category recalibration reduced confidence",
                "source": "category_overrides",
                "authority": "search_order_only",
            })

        suggestions_path = self._findings_dir / "suggestions.jsonl"
        for suggestion in _tail_jsonl(suggestions_path, limit=50):
            if bot_id and suggestion.get("bot_id") not in ("", bot_id):
                continue
            status = str(suggestion.get("status") or "").lower()
            if status in {"rolled_back", "rollback", "reverted", "failed"}:
                evidence_paths.append(_rel(suggestions_path))
                signal_id = str(suggestion.get("suggestion_id") or "")
                source_ids.append(signal_id)
                rollback_candidates.append({
                    "source_weekly_signal_id": signal_id,
                    "category": suggestion.get("category", ""),
                    "family": suggestion.get("tier", "") or suggestion.get("category", ""),
                    "summary": suggestion.get("title", ""),
                    "reason": f"suggestion lifecycle status={status}",
                    "authority": "search_order_only",
                })
                continue
            if status not in {"accepted", "implemented", "deployed"}:
                continue
            evidence_paths.append(_rel(suggestions_path))
            signal_id = str(suggestion.get("suggestion_id") or "")
            source_ids.append(signal_id)
            seed_candidates.append({
                "source_weekly_signal_id": signal_id,
                "category": suggestion.get("category", ""),
                "family": suggestion.get("tier", ""),
                "summary": suggestion.get("title", ""),
                "seed_type": "accepted_suggestion",
                "authority": "search_order_only",
            })

        monthly_outcomes_path = self._findings_dir / "monthly_outcomes.jsonl"
        for outcome in _tail_jsonl(monthly_outcomes_path, limit=50):
            if bot_id and outcome.get("bot_id") not in ("", bot_id):
                continue
            if strategy_id and outcome.get("strategy_id") not in ("", strategy_id):
                continue
            verdict = str(outcome.get("verdict") or "").lower()
            if verdict not in {"repair", "rollback", "quarantine"}:
                continue
            evidence_paths.append(_rel(monthly_outcomes_path))
            signal_id = str(outcome.get("outcome_id") or outcome.get("run_id") or outcome.get("run_month") or "")
            source_ids.append(signal_id)
            family = str(outcome.get("mutation_family") or outcome.get("category") or "unknown")
            rollback_candidates.append({
                "source_weekly_signal_id": signal_id,
                "category": outcome.get("category", ""),
                "family": family,
                "summary": outcome.get("recommended_next_action", "") or f"monthly verdict={verdict}",
                "reason": f"authoritative monthly outcome verdict={verdict}",
                "run_month": outcome.get("run_month", ""),
                "authority": "search_order_only",
            })
            ablations.append({
                "family": family,
                "priority": "highest_if_oos_repair_triggers",
                "reason": f"authoritative monthly outcome verdict={verdict}",
                "authority": "search_order_only",
            })

        strategy_changes_path = self._findings_dir / "strategy_change_ledger.jsonl"
        for change in StrategyChangeLedger(self._findings_dir).projected_records(
            bot_id=bot_id,
            strategy_id=strategy_id,
            days=730,
            limit=50,
        ):
            record_type = str(change.get("record_type") or "").lower()
            rollback_status = str(change.get("rollback_status") or "").lower()
            monthly_status = str(change.get("monthly_status") or "").lower()
            if (
                record_type not in {"rollback", "quarantine", "repair"}
                and rollback_status not in {"recommended", "executed"}
                and monthly_status not in {"repair", "rollback", "quarantine"}
            ):
                continue
            evidence_paths.append(_rel(strategy_changes_path))
            signal_id = str(change.get("record_id") or change.get("run_id") or "")
            source_ids.append(signal_id)
            status_label = rollback_status or monthly_status or record_type
            rollback_candidates.append({
                "source_weekly_signal_id": signal_id,
                "category": change.get("category", ""),
                "family": change.get("mutation_family", "") or monthly_status or record_type,
                "summary": change.get("decision_reason", "") or f"strategy change {record_type}",
                "reason": f"strategy ledger status={status_label}",
                "run_month": change.get("run_month", ""),
                "authority": "search_order_only",
            })

        for item in [*focus, *seed_candidates, *negative_priors, *rollback_candidates]:
            signal_id = str(item.get("source_weekly_signal_id") or item.get("source") or "")
            if signal_id:
                attribution.setdefault(signal_id, []).append(str(item.get("category") or item.get("family") or "search_hint"))

        phase_order_hints = _phase_order(phase_families, negative_priors)
        return MonthlySearchBrief(
            run_month=run_month,
            bot_id=bot_id,
            strategy_id=strategy_id,
            report_only=report_only,
            experiment_focus_hints=_dedupe_dicts(focus)[:12],
            phased_auto_priority_families=_dedupe_dicts(phase_families)[:12],
            phase_order_hints=phase_order_hints,
            seed_candidates=_dedupe_dicts(seed_candidates)[:20],
            conditional_oos_repair_ablation_priorities=_dedupe_dicts(ablations)[:20],
            rollback_candidates=_dedupe_dicts(rollback_candidates)[:12],
            negative_priors=_dedupe_dicts(negative_priors)[:20],
            confidence_caps=_dedupe_dicts(confidence_caps)[:20],
            weekly_focus=weekly_focus,
            weekly_focus_rotation=weekly_focus_rotation_payload(),
            evidence_paths=_dedupe(evidence_paths),
            source_weekly_signal_ids=_dedupe(source_ids),
            attribution=attribution,
        )

    def write(self, brief: MonthlySearchBrief, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(brief.model_dump_json(indent=2), encoding="utf-8")
        return Path(path)


def _hint_from_signal(signal: dict[str, Any], *, signal_id: str, bot_id: str) -> dict[str, Any]:
    if bot_id and signal.get("bot_id") not in ("", bot_id, None):
        return {}
    category = str(signal.get("category") or signal.get("family") or signal.get("mutation_family") or "")
    summary = str(signal.get("summary") or signal.get("title") or signal.get("lesson") or "")
    if not category and not summary:
        return {}
    source_id = str(signal.get("signal_id") or signal.get("suggestion_id") or signal_id or summary)[:80]
    return {
        "source_weekly_signal_id": source_id,
        "category": category,
        "family": str(signal.get("family") or signal.get("mutation_family") or category),
        "summary": summary,
        "authority": "search_order_only",
    }


def _phase_order(phase_families: list[dict[str, Any]], negative_priors: list[dict[str, Any]]) -> list[str]:
    hints: list[str] = []
    if phase_families:
        hints.append("prioritize measured-positive families inside phased-auto ordering")
    if negative_priors:
        hints.append("deprioritize negative-prior families unless stronger-evidence gates pass")
        hints.append("if OOS repair triggers, inspect negative-prior families early in the ablation queue")
    return hints


def _tail_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
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
    return rows[-limit:]


def _latest_week_start(path: Path) -> str:
    for row in reversed(_tail_jsonl(path, limit=20)):
        week_start = str(row.get("week_start") or "").strip()
        if week_start:
            return week_start
    return ""


def _list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, dict)]
    return []


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rel(path: Path) -> str:
    parts = path.parts
    if "memory" in parts:
        idx = parts.index("memory")
        return str(Path(*parts[idx:]))
    return str(path)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _dedupe_dicts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
