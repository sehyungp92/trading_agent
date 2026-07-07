#!/usr/bin/env python3
"""One-time JSONL schema migration - normalize field names to canonical forms.

Eliminates dual-schema technical debt by renaming legacy field names to their
canonical equivalents across all JSONL files in memory/findings/.

Canonical field mappings (legacy -> canonical):
  - timestamp, created_at -> proposed_at  (suggestions.jsonl)
  - measured_at, timestamp -> measurement_date  (outcomes.jsonl, portfolio_outcomes.jsonl)
  - timestamp -> reasoned_at  (outcome_reasonings.jsonl)
  - concluded_at -> resolved_at  (experiments.jsonl, structural_experiments.jsonl)

Usage:
  python scripts/migrate_jsonl_schema.py               # dry run
  python scripts/migrate_jsonl_schema.py --apply        # apply changes
  python scripts/migrate_jsonl_schema.py --apply --dir /path/to/memory/findings

After migration, the fallback chains in readers can optionally be simplified
(but keeping them is safe - they become no-ops on normalized data).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Per-file migration rules: {filename: {legacy_field: canonical_field}}
# Only renames when the canonical field is NOT already present.
_MIGRATION_RULES: dict[str, dict[str, str]] = {
    "suggestions.jsonl": {
        "timestamp": "proposed_at",
        "created_at": "proposed_at",
    },
    "outcomes.jsonl": {
        "measured_at": "measurement_date",
        "timestamp": "measurement_date",
    },
    "portfolio_outcomes.jsonl": {
        "measured_at": "measurement_date",
        "timestamp": "measurement_date",
    },
    "outcome_reasonings.jsonl": {
        "timestamp": "reasoned_at",
    },
    "experiments.jsonl": {
        "concluded_at": "resolved_at",
    },
    "structural_experiments.jsonl": {
        "concluded_at": "resolved_at",
    },
}


def _migrate_record(record: dict, rules: dict[str, str]) -> tuple[dict, list[str]]:
    """Apply field renames to a single record. Returns (updated_record, changes)."""
    changes: list[str] = []
    for legacy, canonical in rules.items():
        if legacy in record and canonical not in record:
            record[canonical] = record.pop(legacy)
            changes.append(f"{legacy} -> {canonical}")
    return record, changes


def migrate_file(path: Path, rules: dict[str, str], apply: bool) -> dict:
    """Migrate a single JSONL file. Returns stats."""
    stats = {"file": str(path), "total_records": 0, "migrated_records": 0, "changes": []}

    if not path.exists():
        stats["skipped"] = "file not found"
        return stats

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    migrated_lines: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        stats["total_records"] += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            migrated_lines.append(line)
            continue

        record, changes = _migrate_record(record, rules)
        if changes:
            stats["migrated_records"] += 1
            stats["changes"].extend(changes)

        migrated_lines.append(json.dumps(record, ensure_ascii=False))

    if apply and stats["migrated_records"] > 0:
        # Backup original
        backup = path.with_suffix(f".pre_migration_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.bak")
        shutil.copy2(path, backup)
        stats["backup"] = str(backup)
        # Write migrated
        path.write_text("\n".join(migrated_lines) + "\n", encoding="utf-8")
        stats["applied"] = True
    else:
        stats["applied"] = False

    return stats


def main():
    parser = argparse.ArgumentParser(description="Normalize JSONL field names")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--dir", type=Path, default=None, help="Path to findings directory")
    args = parser.parse_args()

    # Find the findings directory
    if args.dir:
        findings_dir = args.dir
    else:
        # Default: memory/findings/ relative to project root
        project_root = Path(__file__).resolve().parent.parent
        findings_dir = project_root / "memory" / "findings"

    if not findings_dir.is_dir():
        print(f"Findings directory not found: {findings_dir}")
        sys.exit(1)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Schema migration [{mode}] - scanning {findings_dir}\n")

    total_migrated = 0
    for filename, rules in _MIGRATION_RULES.items():
        path = findings_dir / filename
        stats = migrate_file(path, rules, apply=args.apply)

        status = ""
        if stats.get("skipped"):
            status = f"  SKIP ({stats['skipped']})"
        elif stats["migrated_records"] == 0:
            status = f"  OK - {stats['total_records']} records, no changes needed"
        else:
            status = (
                f"  {'MIGRATED' if stats.get('applied') else 'WOULD MIGRATE'}"
                f" - {stats['migrated_records']}/{stats['total_records']} records"
            )
            total_migrated += stats["migrated_records"]

        print(f"{filename}:{status}")
        if stats["changes"]:
            # Show unique change types
            unique_changes = sorted(set(stats["changes"]))
            for c in unique_changes:
                count = stats["changes"].count(c)
                print(f"    {c} (×{count})")
        if stats.get("backup"):
            print(f"    backup: {stats['backup']}")

    print(f"\nTotal records requiring migration: {total_migrated}")
    if not args.apply and total_migrated > 0:
        print("Run with --apply to execute migration.")


if __name__ == "__main__":
    main()
