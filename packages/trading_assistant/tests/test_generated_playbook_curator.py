from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.schemas.generated_playbook import GeneratedPlaybook, PlaybookStatus, PlaybookTracking
from trading_assistant.skills.generated_playbook_curator import GeneratedPlaybookCurator


def _safe_playbook(title: str, *, pinned_by: str = "") -> GeneratedPlaybook:
    return GeneratedPlaybook(
        workflow="daily_analysis",
        title=title,
        trigger_tags=["category:parameter"],
        evidence_refs=["benchmark:a", "benchmark:b", "benchmark:c"],
        trigger_conditions=["Current task matches parameter signals."],
        required_evidence=["Verify current run evidence."],
        steps=["Review evidence and preserve approval gates."],
        expected_outputs=["Evidence-linked recommendation or no-action note."],
        failure_modes=["Do not bypass approval gates."],
        provenance="Generated from repeated benchmark evidence.",
        pinned_by=pinned_by,
    )


def _write_manifest(memory: Path, playbooks: list[GeneratedPlaybook]) -> Path:
    generated = memory / "playbooks" / "generated"
    generated.mkdir(parents=True)
    manifest = generated / "playbooks.jsonl"
    manifest.write_text("\n".join(playbook.model_dump_json() for playbook in playbooks), encoding="utf-8")
    return manifest


def test_curator_consolidates_duplicate_playbooks(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    first = _safe_playbook("Investigate recurring parameter issues")
    second = _safe_playbook("Investigate recurring parameter issues duplicate")
    second.evidence_refs.append("benchmark:extra")
    _write_manifest(memory, [first, second])

    actions = GeneratedPlaybookCurator(memory).curate(report_only=False)

    assert any(action["action"] == "archive" for action in actions)
    assert any(action["action"] == "patch" for action in actions)
    rows = [
        GeneratedPlaybook.model_validate_json(line)
        for line in (memory / "playbooks" / "generated" / "playbooks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(playbook.status == PlaybookStatus.ARCHIVED for playbook in rows)
    keeper = next(playbook for playbook in rows if playbook.status == PlaybookStatus.ACTIVE)
    assert "benchmark:extra" in keeper.evidence_refs
    archived_ids = {playbook.playbook_id for playbook in rows if playbook.status == PlaybookStatus.ARCHIVED}
    assert archived_ids & set(keeper.supersedes)


def test_curator_does_not_archive_pinned_stale_playbook(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    pinned = _safe_playbook("Pinned playbook", pinned_by="human")
    pinned.updated_at = datetime.now(timezone.utc) - timedelta(days=365)
    _write_manifest(memory, [pinned])

    actions = GeneratedPlaybookCurator(memory).curate(report_only=False, stale_days=30)

    assert actions == []
    row = GeneratedPlaybook.model_validate_json(
        (memory / "playbooks" / "generated" / "playbooks.jsonl").read_text(encoding="utf-8")
    )
    assert row.status == PlaybookStatus.ACTIVE


def test_curator_quarantines_harmful_playbook(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    playbook = _safe_playbook("Harmful playbook")
    _write_manifest(memory, [playbook])
    tracking = PlaybookTracking(playbook_id=playbook.playbook_id, positive_outcomes=0, negative_outcomes=3)
    (memory / "playbooks" / "generated" / "playbook_tracking.jsonl").write_text(
        tracking.model_dump_json() + "\n",
        encoding="utf-8",
    )

    actions = GeneratedPlaybookCurator(memory).curate(report_only=False)

    assert any(action["action"] == "quarantine" for action in actions)
    row = GeneratedPlaybook.model_validate_json(
        (memory / "playbooks" / "generated" / "playbooks.jsonl").read_text(encoding="utf-8")
    )
    assert row.status == PlaybookStatus.QUARANTINED
