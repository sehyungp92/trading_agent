"""Archive retired-strategy records from memory/findings/.

Scheduled weekly. Walks each *.jsonl file under memory/findings/, partitions
records into:
  - active (rewritten in place via atomic rewrite) — records with no
    strategy_id OR with a strategy_id that is still in the StrategyRegistry
  - retired (appended to memory/findings/archive/<YYYY-MM-DD>.jsonl) — records
    whose strategy_id is no longer registered

Pure move; never deletes. Forensic history is preserved in archive/.

Drift filtering at context-assembly time (analysis/context_builder.py) already
prevents retired records from polluting prompts; this skill is the cleanup
companion that physically partitions the on-disk store so it doesn't grow
without bound.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.paths import memory_root

from trading_assistant.schemas.strategy_profile import StrategyRegistry

logger = logging.getLogger(__name__)


_SKIP_FILES = frozenset({
    # Files that don't carry strategy_id and have their own retention policies.
    "category_overrides.jsonl",
    "strategy_registry_drift.jsonl",
    "forecast_history.jsonl",
    "patterns_consolidated.jsonl",
})


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1,
    ):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(
                "archive_retired_strategies: skipping malformed JSON at %s:%d",
                path.name, line_no,
            )
    return records


def _atomic_rewrite(path: Path, records: list[dict]) -> None:
    """Atomically rewrite a JSONL file with the given records."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    tmp.replace(path)


def _append_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def archive_retired(
    findings_dir: Path,
    registry: StrategyRegistry | None = None,
) -> dict[str, int]:
    """Partition findings JSONLs into active vs retired-strategy archives.

    Returns a summary mapping filename → number of records archived.
    """
    if registry is None:
        from trading_assistant.orchestrator.strategy_registry_loader import load_strategy_registry
        registry = load_strategy_registry()

    if not findings_dir.exists():
        return {}

    archive_dir = findings_dir / "archive"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    archive_path = archive_dir / f"{today}.jsonl"

    summary: dict[str, int] = {}
    for entry in sorted(findings_dir.iterdir()):
        if not entry.is_file() or entry.suffix != ".jsonl":
            continue
        if entry.name in _SKIP_FILES:
            continue
        records = _read_jsonl(entry)
        if not records:
            continue

        active: list[dict] = []
        retired: list[dict] = []
        for rec in records:
            sid = rec.get("strategy_id")
            if not sid or registry.is_active(sid):
                active.append(rec)
            else:
                retired.append({"_archived_from": entry.name, **rec})

        if not retired:
            continue

        _append_jsonl(archive_path, retired)
        _atomic_rewrite(entry, active)
        summary[entry.name] = len(retired)
        logger.info(
            "Archived %d retired-strategy records from %s",
            len(retired), entry.name,
        )

    return summary


def _cli() -> int:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    findings_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else memory_root() / "findings"
    )
    summary = archive_retired(findings_dir)
    if summary:
        print(f"Archived {sum(summary.values())} records across {len(summary)} files:")
        for name, count in summary.items():
            print(f"  {name}: {count} records")
    else:
        print("No retired-strategy records found.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
