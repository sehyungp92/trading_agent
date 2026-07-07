"""Focused, provenance-rich recall over prior run and review artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_assistant.schemas.run_recall import FocusedRecallCard
from trading_assistant.skills.evidence_ref_utils import evidence_ref_exists_within_roots
from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger


class RunRecallSummarizer:
    """Normalize RunIndex hits and focused-recall records for prompt context."""

    def __init__(self, memory_dir: Path, run_index: object | None = None) -> None:
        self._memory_dir = Path(memory_dir)
        self._findings_dir = self._memory_dir / "findings"
        self._run_index = run_index
        self._root_dir = _infer_root_dir(self._memory_dir)

    def summarize(
        self,
        *,
        workflow: str = "",
        bot_id: str = "",
        strategy_id: str = "",
        tags: list[str] | None = None,
        limit: int = 5,
        days: int = 90,
    ) -> list[FocusedRecallCard]:
        tags = tags or []
        cards = [
            *self._load_existing_cards(workflow=workflow, bot_id=bot_id, strategy_id=strategy_id, tags=tags),
            *self._cards_from_run_index(workflow=workflow, bot_id=bot_id, days=days, limit=limit),
        ]
        cards = [card for card in cards if self._has_valid_provenance(card)]
        cards = self._apply_outcome_status(cards, bot_id=bot_id, strategy_id=strategy_id)
        cards = self._suppress_stale_contradicted(cards)
        cards.sort(key=lambda card: (_parse_dt(card.date), card.created_at), reverse=True)
        return cards[:limit]

    def _load_existing_cards(
        self,
        *,
        workflow: str,
        bot_id: str,
        strategy_id: str,
        tags: list[str],
    ) -> list[FocusedRecallCard]:
        path = self._findings_dir / "focused_recall_cards.jsonl"
        result: list[FocusedRecallCard] = []
        for row in _load_jsonl(path):
            card = self._normalize_existing(row)
            if card is None:
                continue
            if workflow and card.workflow and card.workflow != workflow:
                continue
            if bot_id and card.bot_id and card.bot_id != bot_id:
                continue
            if strategy_id and card.strategy_id and card.strategy_id != strategy_id:
                continue
            if tags and not card.workflow and not _tag_overlap(row, tags) and not card.reason_for_retrieval:
                continue
            result.append(card)
        return result

    def _normalize_existing(self, row: dict[str, Any]) -> FocusedRecallCard | None:
        if not isinstance(row, dict):
            return None
        if "proposal_or_finding_summary" in row or "how_this_matters_now" in row:
            try:
                return FocusedRecallCard.model_validate(row)
            except Exception:
                return None

        content = row.get("content", "")
        parsed_content: dict[str, Any] = {}
        if isinstance(content, str) and content.strip().startswith("{"):
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                parsed_content = {}
        evidence_paths = _string_list(row.get("evidence_paths"))
        if not evidence_paths:
            evidence_summary = str(row.get("evidence_summary") or "")
            evidence_paths = [
                item.strip()
                for item in evidence_summary.split(",")
                if item.strip()
            ]
        workflow = str(
            row.get("source_workflow")
            or parsed_content.get("workflow")
            or "harness_meta_learning"
        )
        return FocusedRecallCard(
            run_id=str(row.get("source_run_id") or parsed_content.get("run_id") or parsed_content.get("variant_name") or ""),
            date=str(row.get("created_at") or row.get("date") or ""),
            workflow=workflow,
            bot_id=str(row.get("bot_id") or parsed_content.get("bot_id") or ""),
            strategy_id=str(row.get("strategy_id") or parsed_content.get("strategy_id") or ""),
            reason_for_retrieval="existing focused recall card",
            proposal_or_finding_summary=str(row.get("title") or parsed_content.get("discard_reason") or content)[:500],
            validator_gate_status=_gate_status(parsed_content),
            evidence_paths=evidence_paths,
            supersession_notes=str(row.get("superseded_by") or ""),
            how_this_matters_now=_matters_now(parsed_content, row.get("title", "")),
            source="learning_review",
        )

    def _cards_from_run_index(
        self,
        *,
        workflow: str,
        bot_id: str,
        days: int,
        limit: int,
    ) -> list[FocusedRecallCard]:
        if self._run_index is None or not workflow:
            return []
        try:
            min_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            runs = []
            if hasattr(self._run_index, "search"):
                runs = self._run_index.search(
                    query=workflow.replace("_", " "),
                    limit=limit,
                    agent_type=workflow,
                    bot_id=bot_id,
                    min_date=min_date,
                )
            if not runs and hasattr(self._run_index, "get_recent_runs"):
                runs = self._run_index.get_recent_runs(
                    agent_type=workflow,
                    bot_id=bot_id,
                    limit=limit,
                    days=days,
                )
        except Exception:
            return []

        result: list[FocusedRecallCard] = []
        for run in runs:
            evidence = self._run_evidence_paths(run)
            if not evidence:
                continue
            result.append(FocusedRecallCard(
                run_id=str(run.get("run_id") or ""),
                date=str(run.get("date") or run.get("created_at") or ""),
                workflow=str(run.get("agent_type") or workflow),
                bot_id=bot_id,
                reason_for_retrieval="RunIndex match for current workflow context",
                proposal_or_finding_summary=str(run.get("snippet") or run.get("response_preview") or "")[:500],
                validator_gate_status="unknown",
                evidence_paths=evidence,
                how_this_matters_now="Use as prior evidence only; verify against current monthly artifacts before acting.",
                source="run_index",
            ))
        return result

    def _run_evidence_paths(self, run: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        run_dir = str(run.get("run_dir") or "")
        if run_dir:
            base = Path(run_dir)
            for name in ("response.md", "parsed_analysis.json", "validator_notes.md", "metadata.json"):
                candidates.append(str(base / name))
        run_id = str(run.get("run_id") or "")
        if run_id:
            for base in (self._root_dir / "runs", self._root_dir / "data" / "runs"):
                path = base / run_id
                if path.exists():
                    candidates.extend(str(path / name) for name in ("response.md", "parsed_analysis.json", "validator_notes.md"))
        return [path for path in _dedupe(candidates) if Path(path).exists()]

    def _apply_outcome_status(
        self,
        cards: list[FocusedRecallCard],
        *,
        bot_id: str,
        strategy_id: str,
    ) -> list[FocusedRecallCard]:
        outcomes = _load_jsonl(self._findings_dir / "monthly_outcomes.jsonl")
        changes = StrategyChangeLedger(self._findings_dir).projected_records(
            bot_id=bot_id,
            strategy_id=strategy_id,
            days=730,
            limit=200,
        )
        for card in cards:
            latest_outcome = _matching_record_for_card(
                card,
                outcomes,
                bot_id=bot_id,
                strategy_id=strategy_id,
            )
            latest_change = _matching_record_for_card(
                card,
                changes,
                bot_id=bot_id,
                strategy_id=strategy_id,
            )
            if not card.outcome_status and latest_outcome:
                verdict = latest_outcome.get("verdict") or latest_outcome.get("monthly_verdict")
                month = latest_outcome.get("run_month", "")
                if verdict:
                    card.outcome_status = f"{verdict} ({month})".strip()
            if latest_change:
                card.approval_status = card.approval_status or str(latest_change.get("approval_status") or "")
                card.deployment_status = card.deployment_status or str(latest_change.get("deployment_id") or "")
        return cards

    def _suppress_stale_contradicted(self, cards: list[FocusedRecallCard]) -> list[FocusedRecallCard]:
        active: list[FocusedRecallCard] = []
        for card in cards:
            text = " ".join([
                card.supersession_notes,
                card.contradiction_notes,
                card.outcome_status,
            ]).lower()
            if any(term in text for term in ("superseded", "contradicted", "rolled back", "rollback")):
                if "how this matters" not in card.how_this_matters_now.lower():
                    card.how_this_matters_now = (
                        card.how_this_matters_now
                        or "Treat as cautionary or superseded context, not as positive guidance."
                    )
                if "rollback" in text or "contradicted" in text:
                    continue
            active.append(card)
        return active

    def _has_valid_provenance(self, card: FocusedRecallCard) -> bool:
        if not card.evidence_paths:
            return False
        for raw in card.evidence_paths:
            if evidence_ref_exists_within_roots(raw, [self._root_dir, self._findings_dir]):
                return True
        return False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def _tag_overlap(row: dict[str, Any], tags: list[str]) -> bool:
    row_tags = set(_string_list(row.get("tags")) + _string_list(row.get("case_tags")))
    return bool(row_tags & set(tags))


def _gate_status(content: dict[str, Any]) -> str:
    failures = content.get("governance_failures") or content.get("weak_metrics") or []
    if failures:
        return "blocked_or_warning"
    if content.get("kept") is True:
        return "passed"
    if content.get("kept") is False:
        return "discarded"
    return "unknown"


def _matters_now(content: dict[str, Any], title: str) -> str:
    if content.get("discard_reason"):
        return "Avoid repeating this discarded harness or routing pattern without stronger evidence."
    if content.get("governance_failures"):
        return "Preserve approval and monthly-authority gates when this pattern recurs."
    if title:
        return "Use as advisory context; verify against current artifacts before proposing action."
    return ""


def _matching_record_for_card(
    card: FocusedRecallCard,
    rows: list[dict[str, Any]],
    *,
    bot_id: str,
    strategy_id: str,
) -> dict[str, Any]:
    tokens = _card_identity_tokens(card)
    filtered = [
        row for row in rows
        if (not bot_id or row.get("bot_id") in ("", bot_id))
        and (not strategy_id or row.get("strategy_id") in ("", strategy_id))
        and _row_matches_card(row, card, tokens)
    ]
    filtered.sort(key=lambda row: str(row.get("recorded_at") or row.get("created_at") or row.get("run_month") or ""))
    return filtered[-1] if filtered else {}


def _card_identity_tokens(card: FocusedRecallCard) -> set[str]:
    text = " ".join([
        card.run_id,
        card.proposal_or_finding_summary,
        card.reason_for_retrieval,
        *card.evidence_paths,
    ])
    return {
        token.strip(" ,.;:()[]{}\"'")
        for token in text.replace("\\", "/").replace("/", " ").split()
        if len(token.strip(" ,.;:()[]{}\"'")) >= 4
    }


def _row_matches_card(
    row: dict[str, Any],
    card: FocusedRecallCard,
    tokens: set[str],
) -> bool:
    if card.run_id and str(row.get("run_id") or "") == card.run_id:
        return True
    row_tokens = {
        str(row.get(key) or "")
        for key in (
            "outcome_id",
            "record_id",
            "strategy_change_record_id",
            "deployment_id",
            "approval_request_id",
            "commit_sha",
        )
    }
    row_tokens.update(str(item) for item in row.get("proposal_ids", []) or [])
    row_tokens.update(str(item) for item in row.get("suggestion_ids", []) or [])
    row_tokens = {token for token in row_tokens if token}
    return bool(row_tokens & tokens)


def _parse_dt(value: str) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _infer_root_dir(memory_dir: Path) -> Path:
    if memory_dir.name == "memory":
        return memory_dir.parent
    if memory_dir.name == "findings" and memory_dir.parent.name == "memory":
        return memory_dir.parent.parent
    return memory_dir.parent
