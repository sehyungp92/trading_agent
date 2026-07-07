"""Generate guarded advisory playbooks from repeated learning evidence."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.schemas.generated_playbook import GeneratedPlaybook, PlaybookStatus, PlaybookTracking
from trading_assistant.skills.generated_playbook_guard import GeneratedPlaybookGuard

_MAX_ACTIVE_PER_WORKFLOW = 10
_MAX_ACTIVE_PER_PRIMARY_TAG = 3


class PlaybookGenerator:
    """Build safe generated playbooks from repeated benchmark and retrospective signals."""

    def __init__(self, memory_dir: Path) -> None:
        self._memory_dir = Path(memory_dir)
        self._findings_dir = self._memory_dir / "findings"
        self._playbooks_dir = self._memory_dir / "playbooks" / "generated"
        self._manifest_path = self._playbooks_dir / "playbooks.jsonl"

    def generate(self, lookback_days: int = 90, min_evidence: int = 3) -> list[GeneratedPlaybook]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        clusters: dict[tuple[str, str], dict] = defaultdict(lambda: {
            "tags": set(),
            "evidence_refs": [],
            "sources": set(),
        })

        for case in self._load_jsonl(self._findings_dir / "benchmark_cases.jsonl"):
            ts = self._parse_ts(case.get("created_at", "") or case.get("date", ""))
            if ts and ts < cutoff:
                continue
            workflow = str(case.get("agent_type", "") or self._infer_workflow(case.get("source_run_id", ""))).strip()
            tags = [tag for tag in case.get("case_tags", []) if str(tag).startswith(("category:", "reason:", "regime:", "bot:"))]
            primary = next((tag for tag in tags if tag.startswith(("category:", "reason:"))), "")
            if not workflow or not primary:
                continue
            cluster = clusters[(workflow, primary)]
            cluster["tags"].update(tags)
            cluster["evidence_refs"].append(f"benchmark:{case.get('case_id', '')}")
            cluster["sources"].add("benchmark")

        for entry in self._load_jsonl(self._findings_dir / "retrospective_synthesis.jsonl"):
            ts = self._parse_ts(entry.get("recorded_at", "") or entry.get("created_at", "") or entry.get("week_end", ""))
            if ts and ts < cutoff:
                continue
            for discard in entry.get("discard", []):
                category = discard.get("category", "")
                if not category:
                    continue
                primary = self._structured_tag("category", category)
                cluster = clusters[("weekly_analysis", primary)]
                cluster["tags"].update({"workflow:weekly_analysis", primary})
                if discard.get("bot_id"):
                    cluster["tags"].add(self._structured_tag("bot", discard["bot_id"]))
                cluster["evidence_refs"].append(
                    f"retrospective:{entry.get('week_start', '')}:{category}"
                )
                cluster["sources"].add("retrospective")

        playbooks: list[GeneratedPlaybook] = []
        for (workflow, primary), cluster in sorted(clusters.items()):
            evidence_refs = list(dict.fromkeys(cluster["evidence_refs"]))
            if len(evidence_refs) < min_evidence:
                continue
            trigger_tags = sorted(tag for tag in cluster["tags"] if tag)
            title = f"Investigate recurring {self._humanize_tag(primary)} issues"
            playbook = GeneratedPlaybook(
                workflow=workflow,
                title=title,
                trigger_tags=trigger_tags,
                evidence_refs=evidence_refs[:10],
                trigger_conditions=[
                    f"Current task matches {self._humanize_tag(primary)} signals.",
                    f"At least {min_evidence} corroborating evidence items exist in the last {lookback_days} days.",
                ],
                required_evidence=[
                    "Confirm the current run or report shows the same trigger pattern.",
                    "Open linked artifacts or findings and verify the pattern is recent, not one-off noise.",
                    "Keep approval gates intact before proposing any follow-up action.",
                ],
                steps=[
                    "Review the linked evidence and compare the current case against the repeated failure pattern.",
                    "Check whether the issue clusters by category, regime, or bot before proposing changes.",
                    "Prefer reversible investigation or experiment steps over direct live-trading changes.",
                    "Only escalate a proposal when provenance is clear and approval gates still apply.",
                ],
                expected_outputs=[
                    "Evidence-linked investigation note or no-action rationale.",
                    "Any follow-up proposal remains advisory until deterministic gates and human approval pass.",
                ],
                failure_modes=[
                    "Do not bypass approval gates.",
                    "Do not change live trading logic directly from this playbook.",
                    "Do not use this playbook when provenance or current evidence is weak.",
                ],
                provenance=(
                    f"Generated from {len(evidence_refs)} corroborating evidence items "
                    f"across {', '.join(sorted(cluster['sources']))} in the last {lookback_days} days."
                ),
                status=PlaybookStatus.ACTIVE,
            )
            if self._is_safe(playbook):
                playbooks.append(playbook)

        # Exclude playbooks that were previously retired via tracking
        tracking = self._load_tracking()
        quarantined_ids = {
            pid for pid, t in tracking.items()
            if (t.positive_outcomes + t.negative_outcomes) >= 5
            and t.effectiveness_rate < 0.3
        }
        if quarantined_ids:
            playbooks = [p for p in playbooks if p.playbook_id not in quarantined_ids]

        playbooks = self._cap_active_playbooks(playbooks)
        self._write_playbooks(playbooks)
        return playbooks

    def _write_playbooks(self, playbooks: list[GeneratedPlaybook]) -> None:
        self._playbooks_dir.mkdir(parents=True, exist_ok=True)
        active_ids = {playbook.playbook_id for playbook in playbooks}
        with self._manifest_path.open("w", encoding="utf-8") as manifest:
            for playbook in playbooks:
                manifest.write(playbook.model_dump_json() + "\n")
                (self._playbooks_dir / f"{playbook.playbook_id}.md").write_text(
                    playbook.to_prompt_text() + "\n",
                    encoding="utf-8",
                )
        for path in self._playbooks_dir.glob("*.md"):
            if path.stem not in active_ids:
                path.unlink(missing_ok=True)

    @staticmethod
    def _is_safe(playbook: GeneratedPlaybook) -> bool:
        return GeneratedPlaybookGuard().is_safe(playbook)

    @staticmethod
    def _cap_active_playbooks(playbooks: list[GeneratedPlaybook]) -> list[GeneratedPlaybook]:
        """Bound generated prompt surface by workflow and primary trigger tag."""
        per_workflow: dict[str, int] = defaultdict(int)
        per_primary: dict[tuple[str, str], int] = defaultdict(int)
        selected: list[GeneratedPlaybook] = []

        def _rank(playbook: GeneratedPlaybook) -> tuple[int, str]:
            return (-len(playbook.evidence_refs), playbook.title)

        for playbook in sorted(playbooks, key=_rank):
            workflow = playbook.workflow or "default"
            primary = next(
                (
                    tag for tag in playbook.trigger_tags
                    if tag.startswith(("category:", "reason:"))
                ),
                "",
            )
            if per_workflow[workflow] >= _MAX_ACTIVE_PER_WORKFLOW:
                continue
            if primary and per_primary[(workflow, primary)] >= _MAX_ACTIVE_PER_PRIMARY_TAG:
                continue
            selected.append(playbook)
            per_workflow[workflow] += 1
            if primary:
                per_primary[(workflow, primary)] += 1
        return selected

    @staticmethod
    def _humanize_tag(tag: str) -> str:
        _, _, value = str(tag).partition(":")
        return value.replace("_", " ") if value else str(tag)

    @staticmethod
    def _structured_tag(prefix: str, value: str) -> str:
        text = str(value or "").strip().lower()
        chars: list[str] = []
        prev_sep = False
        for char in text:
            if char.isalnum():
                chars.append(char)
                prev_sep = False
            elif not prev_sep:
                chars.append("_")
                prev_sep = True
        slug = "".join(chars).strip("_")
        return f"{prefix}:{slug}" if slug else ""

    @staticmethod
    def _infer_workflow(run_id: str) -> str:
        lower = str(run_id or "").lower()
        mapping = {
            "daily": "daily_analysis",
            "weekly": "weekly_analysis",
            "monthly": "monthly_validation",
            "triage": "triage",
            "discovery": "discovery_analysis",
            "reasoning": "outcome_reasoning",
        }
        for prefix, workflow in mapping.items():
            if lower.startswith(prefix):
                return workflow
        return ""

    @staticmethod
    def _parse_ts(value: str) -> datetime | None:
        if not value:
            return None
        try:
            if "T" in value:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None

    # --- Playbook outcome tracking (separate file from manifest) ---

    @property
    def _tracking_path(self) -> Path:
        return self._playbooks_dir / "playbook_tracking.jsonl"

    def _load_tracking(self) -> dict[str, PlaybookTracking]:
        """Load tracking records keyed by playbook_id."""
        if not self._tracking_path.exists():
            return {}
        result: dict[str, PlaybookTracking] = {}
        for line in self._tracking_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = PlaybookTracking.model_validate_json(line)
                result[record.playbook_id] = record
            except Exception:
                continue
        return result

    def _save_tracking(self, tracking: dict[str, PlaybookTracking]) -> None:
        """Rewrite tracking JSONL."""
        self._playbooks_dir.mkdir(parents=True, exist_ok=True)
        with self._tracking_path.open("w", encoding="utf-8") as f:
            for record in tracking.values():
                f.write(record.model_dump_json() + "\n")

    def record_usage(self, playbook_id: str) -> None:
        """Record that a playbook was used."""
        tracking = self._load_tracking()
        entry = tracking.get(playbook_id, PlaybookTracking(playbook_id=playbook_id))
        entry.usage_count += 1
        entry.last_used_at = datetime.now(timezone.utc)
        tracking[playbook_id] = entry
        self._save_tracking(tracking)

    def record_outcome(self, playbook_id: str, positive: bool) -> None:
        """Record a positive or negative outcome for a playbook."""
        tracking = self._load_tracking()
        entry = tracking.get(playbook_id, PlaybookTracking(playbook_id=playbook_id))
        if positive:
            entry.positive_outcomes += 1
        else:
            entry.negative_outcomes += 1
        tracking[playbook_id] = entry
        self._save_tracking(tracking)

    def retire_ineffective(self, min_uses: int = 5, min_effectiveness: float = 0.3) -> int:
        """Quarantine playbooks with enough usage data and poor effectiveness.

        Returns the number of playbooks retired.
        """
        tracking = self._load_tracking()
        if not tracking:
            return 0

        # Load current manifest
        playbooks: list[GeneratedPlaybook] = []
        if self._manifest_path.exists():
            for line in self._manifest_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    playbooks.append(GeneratedPlaybook.model_validate_json(line))
                except Exception:
                    continue

        retired = 0
        for playbook in playbooks:
            entry = tracking.get(playbook.playbook_id)
            if entry is None:
                continue
            total_outcomes = entry.positive_outcomes + entry.negative_outcomes
            if total_outcomes >= min_uses and entry.effectiveness_rate < min_effectiveness:
                playbook.status = PlaybookStatus.QUARANTINED
                retired += 1

        if retired:
            # Rewrite manifest with updated statuses
            with self._manifest_path.open("w", encoding="utf-8") as f:
                for playbook in playbooks:
                    f.write(playbook.model_dump_json() + "\n")
            # Clean up .md files for quarantined playbooks
            for playbook in playbooks:
                if playbook.status == PlaybookStatus.QUARANTINED:
                    md_path = self._playbooks_dir / f"{playbook.playbook_id}.md"
                    md_path.unlink(missing_ok=True)

        return retired

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        result: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result
