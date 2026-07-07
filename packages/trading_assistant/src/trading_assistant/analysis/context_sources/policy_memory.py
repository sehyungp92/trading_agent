"""Policy prompt source."""

from __future__ import annotations

from pathlib import Path

POLICY_FILES = ["agent.md", "trading_rules.md", "soul.md"]


class PolicyMemorySource:
    def __init__(self, memory_dir: Path) -> None:
        self._policy_dir = Path(memory_dir) / "policies" / "v1"

    def build_system_prompt(self) -> str:
        parts: list[str] = []
        for name in POLICY_FILES:
            path = self._policy_dir / name
            if path.exists():
                parts.append(f"--- {name} ---\n{path.read_text(encoding='utf-8')}")
        return "\n\n".join(parts)

    def context_files(self) -> list[str]:
        return [
            str(path)
            for name in POLICY_FILES
            if (path := self._policy_dir / name).exists()
        ]
