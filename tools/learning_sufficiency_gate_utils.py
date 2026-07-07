from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs" / "trading-assistant-learning-instrumentation-implementation-plan.md"


def checklist_completion_check(section_names: list[str], *, plan_path: Path = PLAN_PATH) -> dict[str, Any]:
    if not plan_path.exists():
        return {
            "name": "required_finite_checklist_sections_complete",
            "passed": False,
            "details": {"plan_path": _rel(plan_path), "error": "plan file is missing"},
        }
    text = plan_path.read_text(encoding="utf-8").splitlines()
    details: list[dict[str, Any]] = []
    for section_name in section_names:
        items = _section_checklist_items(text, section_name)
        unchecked = [item for item in items if not item["checked"]]
        details.append({
            "section": section_name,
            "item_count": len(items),
            "unchecked_count": len(unchecked),
            "unchecked_items": [item["text"] for item in unchecked],
        })
    return {
        "name": "required_finite_checklist_sections_complete",
        "passed": bool(details) and all(not row["unchecked_count"] for row in details),
        "details": details,
    }


def _section_checklist_items(lines: list[str], section_name: str) -> list[dict[str, Any]]:
    marker = f"### {section_name}:"
    start = next((index for index, line in enumerate(lines) if line.startswith(marker)), -1)
    if start < 0:
        return [{"checked": False, "text": f"{section_name} checklist section is missing"}]
    items: list[dict[str, Any]] = []
    for line in lines[start + 1:]:
        if line.startswith("### "):
            break
        stripped = line.strip()
        if stripped.startswith("- [ ] ") or stripped.startswith("- [x] "):
            items.append({"checked": stripped.startswith("- [x] "), "text": stripped[6:]})
    return items


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")
