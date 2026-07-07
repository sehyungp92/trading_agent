from __future__ import annotations

from pathlib import Path


def test_no_strategy_core_state_file_imports_retired_lifecycle_helper() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    state_files = list((repo_root / "strategies").glob("**/core/state.py"))
    offenders: list[str] = []
    for path in state_files:
        text = path.read_text(encoding="utf-8")
        if "strategies.core.lifecycle" in text or "LifecycleBarInput" in text:
            offenders.append(str(path.relative_to(repo_root)))

    assert offenders == []


def test_retired_lifecycle_helper_module_is_absent() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "strategies" / "core" / "lifecycle.py").exists()
