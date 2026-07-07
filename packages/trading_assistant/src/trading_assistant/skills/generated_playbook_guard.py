"""Safety guard for generated advisory playbooks."""
from __future__ import annotations

from pathlib import Path

from trading_assistant.schemas.generated_playbook import GeneratedPlaybook, PlaybookStatus

_MAX_PLAYBOOK_PROMPT_CHARS = 4000
_BANNED_DIRECTIVE_TERMS = (
    "bypass approval",
    "disable approval",
    "skip approval",
    "without approval",
    "auto-deploy",
    "autodeploy",
    "direct bot command",
    "live trading command",
    "send order",
    "place order",
    "change live trading logic directly",
    "ignore deterministic gate",
    "disable gate",
    "always approve",
    "always deploy",
)


class GeneratedPlaybookGuard:
    """Validate generated playbooks before injection or curation."""

    def __init__(self, memory_dir: Path | None = None) -> None:
        self._memory_dir = Path(memory_dir) if memory_dir else None
        self._root_dir = self._memory_dir.parent if self._memory_dir and self._memory_dir.name == "memory" else self._memory_dir

    def validate(self, playbook: GeneratedPlaybook) -> list[str]:
        issues: list[str] = []
        if playbook.status not in {PlaybookStatus.ACTIVE, PlaybookStatus.ARCHIVED, PlaybookStatus.QUARANTINED}:
            issues.append("unsupported playbook status")
        if not playbook.provenance:
            issues.append("missing provenance")
        if len(playbook.evidence_refs) < 3:
            issues.append("insufficient evidence refs")
        if not playbook.trigger_conditions:
            issues.append("missing trigger conditions")
        if not playbook.required_evidence:
            issues.append("missing required evidence")
        if not playbook.steps:
            issues.append("missing steps")
        if not playbook.expected_outputs:
            issues.append("missing expected outputs")
        if not playbook.failure_modes:
            issues.append("missing failure modes")
        if len(playbook.to_prompt_text()) > _MAX_PLAYBOOK_PROMPT_CHARS:
            issues.append("playbook prompt text too large")
        if self._has_bad_evidence_ref(playbook):
            issues.append("unsafe or missing evidence ref")
        directive_text = " ".join(
            playbook.trigger_conditions
            + playbook.required_evidence
            + playbook.steps
            + [playbook.provenance]
        ).lower()
        for term in _BANNED_DIRECTIVE_TERMS:
            if term in directive_text:
                issues.append(f"unsafe directive: {term}")
                break
        if "approval" not in " ".join(playbook.failure_modes).lower():
            issues.append("failure modes must preserve approval gates")
        return issues

    def is_safe(self, playbook: GeneratedPlaybook) -> bool:
        return not self.validate(playbook)

    def _has_bad_evidence_ref(self, playbook: GeneratedPlaybook) -> bool:
        for ref in playbook.evidence_refs:
            value = str(ref or "").strip()
            if not value or ".." in value.replace("\\", "/"):
                return True
            if value.startswith(("benchmark:", "retrospective:", "outcome:", "transfer:")):
                continue
            if self._root_dir is None:
                continue
            path = Path(value)
            if path.is_absolute():
                try:
                    path.resolve().relative_to(self._root_dir.resolve())
                except ValueError:
                    return True
                if not path.exists():
                    return True
            elif not (self._root_dir / path).exists():
                return True
        return False
