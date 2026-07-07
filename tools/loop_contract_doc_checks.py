from __future__ import annotations

from pathlib import Path


def check_auto_outcome_authority(src: Path) -> list[str]:
    messages: list[str] = []
    auto_path = src / "trading_assistant" / "skills" / "auto_outcome_measurer.py"
    app_path = src / "trading_assistant" / "orchestrator" / "app.py"
    callbacks_path = src / "trading_assistant" / "orchestrator" / "runtime_scheduled_callbacks.py"
    auto_text = auto_path.read_text(encoding="utf-8") if auto_path.exists() else ""
    app_text = app_path.read_text(encoding="utf-8") if app_path.exists() else ""
    callbacks_text = callbacks_path.read_text(encoding="utf-8") if callbacks_path.exists() else ""
    if "Monthly full-fidelity validation is authoritative" not in auto_text or "early-warning/context" not in auto_text:
        messages.append(_issue(
            "AM-12",
            str(auto_path),
            "module_docstring",
            "AutoOutcomeMeasurer does not state monthly validation authority and early-warning scope",
            "Document AutoOutcomeMeasurer as early-warning/context only for material changes.",
        ))
    if 'outcome_source="early_warning"' not in auto_text:
        messages.append(_issue(
            "AM-12",
            str(auto_path),
            "ProposalOutcome.outcome_source",
            "AutoOutcomeMeasurer proposal outcomes are not explicitly tagged early_warning",
            "Record AutoOutcomeMeasurer proposal outcomes with outcome_source='early_warning'.",
        ))
    if "final=not requires_monthly_outcome" not in callbacks_text:
        messages.append(_issue(
            "AM-12",
            str(callbacks_path),
            "suggestion_tracker.mark_measured",
            "early outcome measurement can finalize material strategy suggestions",
            "Pass final=False for suggestions that require monthly outcome authority.",
        ))
    if "requires_monthly_outcome as _requires_monthly_outcome" not in app_text:
        messages.append(_issue(
            "AM-12",
            str(app_path),
            "_requires_monthly_outcome",
            "app compatibility import no longer points at runtime authority helper",
            "Keep app.py importing the runtime-owned requires_monthly_outcome helper.",
        ))
    return messages


def check_retired_wfo_docs(root: Path, memory_dir: Path) -> list[str]:
    messages: list[str] = []
    paths = [
        root / "README.md",
        root / "packages" / "trading_assistant" / "README.md",
        root / "packages" / "trading_assistant" / "AGENTS.md",
        root / "packages" / "trading_assistant" / "CLAUDE.md",
        *sorted((memory_dir / "skills").glob("*.md")),
        *sorted((memory_dir / "policies").glob("**/*.md")),
        *sorted((root / "docs" / "architecture").glob("*.md")),
        *sorted((root / "docs" / "adr").glob("*.md")),
        *sorted((root / "docs").glob("loop-engineering*.md")),
    ]
    supersession_markers = (
        "monthly validation supersedes",
        "monthly validation is the authoritative",
        "replaced by monthly",
        "replaces legacy wfo",
        "replace current wfo",
        "not by legacy wfo",
        "stay as sensors",
        "legacy wfo runtime",
        "historical reference only",
        "monthly full-fidelity validation",
        "stay screening",
    )
    authority_markers = (
        "authoritative",
        "scheduler",
        "scheduled",
        "runtime",
        "approval",
        "deploy",
        "strategy-improvement path",
    )
    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        lower = path.read_text(encoding="utf-8", errors="ignore").lower()
        if "wfo" not in lower and "walk-forward" not in lower and "walk forward" not in lower:
            continue
        if any(marker in lower for marker in supersession_markers):
            continue
        if any(marker in lower for marker in authority_markers):
            messages.append(_issue(
                "AM-13",
                str(path),
                "retired_wfo_authority",
                "active documentation mentions WFO authority without monthly-validation supersession wording",
                "Add historical/superseded wording or remove active WFO authority claims.",
            ))
    return messages


def _issue(am_row: str, path: str, field: str, message: str, remediation: str) -> str:
    return f"{am_row} {path}:{field} - {message}\n  remediation: {remediation}"
