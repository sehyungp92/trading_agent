"""Lifecycle curator for generated advisory playbooks."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_assistant.schemas.generated_playbook import GeneratedPlaybook, PlaybookStatus, PlaybookTracking
from trading_assistant.skills.generated_playbook_guard import GeneratedPlaybookGuard


class GeneratedPlaybookCurator:
    """Consolidate, pin, archive, restore, patch, and quarantine playbooks."""

    def __init__(self, memory_dir: Path) -> None:
        self._memory_dir = Path(memory_dir)
        self._playbooks_dir = self._memory_dir / "playbooks" / "generated"
        self._manifest_path = self._playbooks_dir / "playbooks.jsonl"
        self._tracking_path = self._playbooks_dir / "playbook_tracking.jsonl"
        self._action_log_path = self._playbooks_dir / "generated_playbook_curator_actions.jsonl"
        self._guard = GeneratedPlaybookGuard(self._memory_dir)

    def curate(self, *, report_only: bool = True, stale_days: int = 120) -> list[dict[str, Any]]:
        playbooks = self._load_playbooks()
        tracking = self._load_tracking()
        actions: list[dict[str, Any]] = []
        actions.extend(self._safety_actions(playbooks))
        actions.extend(self._consolidation_actions(playbooks))
        actions.extend(self._stale_archive_actions(playbooks, stale_days=stale_days))
        actions.extend(self._harmful_quarantine_actions(playbooks, tracking))
        actions.extend(self._restore_actions(playbooks, tracking))
        actions = _dedupe_actions(actions)

        if not report_only and actions:
            by_id = {playbook.playbook_id: playbook for playbook in playbooks}
            for action in actions:
                self._apply_action(by_id, action)
            self._write_playbooks(list(by_id.values()))
        if actions:
            self._log_actions(actions, report_only=report_only)
        return actions

    def pin(self, playbook_id: str, *, pinned_by: str) -> bool:
        playbooks = self._load_playbooks()
        changed = False
        for playbook in playbooks:
            if playbook.playbook_id == playbook_id:
                playbook.pinned_by = pinned_by
                changed = True
        if changed:
            self._write_playbooks(playbooks)
            self._log_actions([self._action("pin", playbook_id, reason=f"pinned by {pinned_by}")], report_only=False)
        return changed

    def _safety_actions(self, playbooks: list[GeneratedPlaybook]) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for playbook in playbooks:
            issues = self._guard.validate(playbook)
            if issues and playbook.status == PlaybookStatus.ACTIVE:
                if playbook.pinned_by:
                    actions.append(self._action(
                        "manual_review",
                        playbook.playbook_id,
                        reason="pinned playbook has guard issues: " + "; ".join(issues),
                        evidence_refs=playbook.evidence_refs,
                    ))
                    continue
                actions.append(self._action(
                    "quarantine",
                    playbook.playbook_id,
                    reason="; ".join(issues),
                    evidence_refs=playbook.evidence_refs,
                ))
        return actions

    def _consolidation_actions(self, playbooks: list[GeneratedPlaybook]) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[GeneratedPlaybook]] = {}
        for playbook in playbooks:
            if playbook.status != PlaybookStatus.ACTIVE:
                continue
            groups.setdefault((playbook.workflow, _primary_tag(playbook)), []).append(playbook)

        actions: list[dict[str, Any]] = []
        for (_, _), group in groups.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda item: (bool(item.pinned_by), len(item.evidence_refs), item.updated_at), reverse=True)
            keeper = group[0]
            for duplicate in group[1:]:
                if duplicate.pinned_by:
                    continue
                actions.append(self._action(
                    "archive",
                    duplicate.playbook_id,
                    reason=f"consolidated into {keeper.playbook_id}",
                    superseded_by=keeper.playbook_id,
                    evidence_refs=duplicate.evidence_refs,
                ))
                actions.append(self._action(
                    "patch",
                    keeper.playbook_id,
                    reason=f"merge duplicate evidence from {duplicate.playbook_id}",
                    evidence_refs=duplicate.evidence_refs,
                    patch={
                        "add_evidence_refs": duplicate.evidence_refs,
                        "add_supersedes": [duplicate.playbook_id],
                        "provenance_note": f"Consolidated duplicate {duplicate.playbook_id}.",
                    },
                ))
        return actions

    def _stale_archive_actions(
        self,
        playbooks: list[GeneratedPlaybook],
        *,
        stale_days: int,
    ) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=stale_days)
        actions: list[dict[str, Any]] = []
        for playbook in playbooks:
            if playbook.status != PlaybookStatus.ACTIVE or playbook.pinned_by:
                continue
            updated = playbook.updated_at
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            if updated < cutoff:
                actions.append(self._action(
                    "archive",
                    playbook.playbook_id,
                    reason=f"stale for more than {stale_days} days",
                    evidence_refs=playbook.evidence_refs,
                ))
        return actions

    def _harmful_quarantine_actions(
        self,
        playbooks: list[GeneratedPlaybook],
        tracking: dict[str, PlaybookTracking],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for playbook in playbooks:
            if playbook.status != PlaybookStatus.ACTIVE or playbook.pinned_by:
                continue
            entry = tracking.get(playbook.playbook_id)
            if entry is None:
                continue
            total = entry.positive_outcomes + entry.negative_outcomes
            if total >= 3 and entry.effectiveness_rate < 0.35:
                actions.append(self._action(
                    "quarantine",
                    playbook.playbook_id,
                    reason="negative downstream outcomes exceeded curator threshold",
                    evidence_refs=playbook.evidence_refs,
                ))
        return actions

    def _restore_actions(
        self,
        playbooks: list[GeneratedPlaybook],
        tracking: dict[str, PlaybookTracking],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for playbook in playbooks:
            if playbook.status != PlaybookStatus.ARCHIVED:
                continue
            entry = tracking.get(playbook.playbook_id)
            if entry and entry.positive_outcomes >= 2 and entry.effectiveness_rate >= 0.6:
                actions.append(self._action(
                    "restore",
                    playbook.playbook_id,
                    reason="matching positive outcome pattern recurred",
                    evidence_refs=playbook.evidence_refs,
                ))
        return actions

    def _apply_action(self, by_id: dict[str, GeneratedPlaybook], action: dict[str, Any]) -> None:
        playbook = by_id.get(action["playbook_id"])
        if playbook is None:
            return
        kind = action["action"]
        if kind == "archive":
            if playbook.pinned_by:
                return
            playbook.curator_action_ids.append(action["action_id"])
            playbook.status = PlaybookStatus.ARCHIVED
            playbook.archived_at = datetime.now(timezone.utc)
            playbook.archive_reason = action.get("reason", "")
            playbook.superseded_by = action.get("superseded_by", "")
        elif kind == "restore":
            playbook.curator_action_ids.append(action["action_id"])
            playbook.status = PlaybookStatus.ACTIVE
            playbook.archived_at = None
            playbook.archive_reason = ""
        elif kind == "quarantine":
            if playbook.pinned_by:
                return
            playbook.curator_action_ids.append(action["action_id"])
            playbook.status = PlaybookStatus.QUARANTINED
        elif kind == "patch":
            playbook.curator_action_ids.append(action["action_id"])
            patch = action.get("patch") or {}
            for ref in patch.get("add_evidence_refs", []) or []:
                value = str(ref or "").strip()
                if value and value not in playbook.evidence_refs:
                    playbook.evidence_refs.append(value)
            for superseded in patch.get("add_supersedes", []) or []:
                value = str(superseded or "").strip()
                if value and value not in playbook.supersedes:
                    playbook.supersedes.append(value)
            note = str(patch.get("provenance_note") or "").strip()
            if note and note not in playbook.provenance:
                playbook.provenance = (playbook.provenance.rstrip() + "\n" + note).strip()
        elif kind == "supersede":
            playbook.curator_action_ids.append(action["action_id"])
            supersedes = action.get("supersedes", "")
            if supersedes and supersedes not in playbook.supersedes:
                playbook.supersedes.append(supersedes)
        elif kind == "manual_review":
            return
        playbook.updated_at = datetime.now(timezone.utc)

    def _load_playbooks(self) -> list[GeneratedPlaybook]:
        if not self._manifest_path.exists():
            return []
        playbooks: list[GeneratedPlaybook] = []
        for line in self._manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                playbooks.append(GeneratedPlaybook.model_validate_json(line))
            except Exception:
                continue
        return playbooks

    def _write_playbooks(self, playbooks: list[GeneratedPlaybook]) -> None:
        self._playbooks_dir.mkdir(parents=True, exist_ok=True)
        with self._manifest_path.open("w", encoding="utf-8") as handle:
            for playbook in playbooks:
                handle.write(playbook.model_dump_json() + "\n")
                md_path = self._playbooks_dir / f"{playbook.playbook_id}.md"
                if playbook.status == PlaybookStatus.ACTIVE:
                    md_path.write_text(playbook.to_prompt_text() + "\n", encoding="utf-8")
                else:
                    md_path.unlink(missing_ok=True)

    def _load_tracking(self) -> dict[str, PlaybookTracking]:
        if not self._tracking_path.exists():
            return {}
        result: dict[str, PlaybookTracking] = {}
        for line in self._tracking_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = PlaybookTracking.model_validate_json(line)
            except Exception:
                continue
            result[record.playbook_id] = record
        return result

    def _log_actions(self, actions: list[dict[str, Any]], *, report_only: bool) -> None:
        self._playbooks_dir.mkdir(parents=True, exist_ok=True)
        with self._action_log_path.open("a", encoding="utf-8") as handle:
            for action in actions:
                handle.write(json.dumps({**action, "report_only": report_only}, default=str) + "\n")

    @staticmethod
    def _action(
        action: str,
        playbook_id: str,
        *,
        reason: str,
        evidence_refs: list[str] | None = None,
        superseded_by: str = "",
        supersedes: str = "",
        patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        patch_payload = patch or {}
        raw = f"{action}:{playbook_id}:{reason}:{superseded_by}:{supersedes}:{json.dumps(patch_payload, sort_keys=True)}"
        return {
            "action_id": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
            "action": action,
            "playbook_id": playbook_id,
            "reason": reason,
            "evidence_refs": evidence_refs or [],
            "superseded_by": superseded_by,
            "supersedes": supersedes,
            "patch": patch_payload,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }


def _primary_tag(playbook: GeneratedPlaybook) -> str:
    return next(
        (tag for tag in playbook.trigger_tags if tag.startswith(("category:", "reason:"))),
        playbook.title.lower().strip(),
    )


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in actions:
        key = str(action.get("action_id") or "")
        if not key:
            key = ":".join(
                str(action.get(part, ""))
                for part in ("action", "playbook_id", "superseded_by", "supersedes")
            )
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result
