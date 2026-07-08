from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_IMPORT_PATTERN = re.compile(
    r"from backtest(?:\.|\s+import)"
    r"|import backtest(?:\.|\s|$)"
    r"|from strategy(?:_2|_3)?(?:\.|\s+import)"
    r"|import strategy(?:_2|_3)?(?:\.|\s|$)"
    r"|_aliases\.install\("
)


def test_repo_has_no_alias_backed_backtest_imports() -> None:
    offenders: list[str] = []
    for path in REPO_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts or any(part.startswith("_ref") for part in path.parts):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if not (
            rel.startswith("backtests/")
            or rel.startswith("tests/")
            or rel.startswith("strategies/")
        ):
            continue
        if LEGACY_IMPORT_PATTERN.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)

    assert offenders == []
