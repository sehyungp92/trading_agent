from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "learning_sufficiency" / "assistant_lineage_source_inventory.json"

SOURCE_GLOBS = {
    "monthly_briefs": ("packages/trading_assistant/memory/findings/**/*monthly_search_brief*.json",),
    "monthly_candidates": (
        "packages/trading_assistant/memory/findings/**/*candidate*.json",
        "packages/trading_assistant/memory/findings/**/*approval*.json",
    ),
    "proposal_ledgers": ("packages/trading_assistant/memory/findings/proposal_ledger.jsonl",),
    "strategy_change_ledgers": ("packages/trading_assistant/memory/findings/strategy_change_ledger.jsonl",),
    "monthly_manifests": ("packages/trading_assistant/memory/findings/**/run_manifest.json",),
    "deployment_metadata": (
        "deployments/*/generated/runtime_deployment_metadata/*/deployment_metadata.json",
    ),
}

ID_ALIASES = {
    "proposal_ids": ("proposal_id", "proposal_ids", "source_proposal_ids"),
    "source_weekly_signal_ids": ("weekly_signal_ids", "source_weekly_signal_ids"),
    "candidate_ids": ("candidate_id", "candidate_ids", "adopted_candidate_id"),
    "strategy_change_record_ids": (
        "strategy_change_record_id",
        "strategy_change_record_ids",
        "proposed_strategy_change_record_ids",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory assistant lineage source IDs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    inventory = build_inventory()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": inventory["status"] == "pass", "artifact_path": _rel(output)}, indent=2))
    return 0 if inventory["status"] == "pass" else 1


def build_inventory() -> dict[str, Any]:
    categories = {name: _scan_category(name, globs) for name, globs in SOURCE_GLOBS.items()}
    required = {
        key: _locations_for_key(key, categories)
        for key in ID_ALIASES
    }
    seed = _runtime_lineage_seed(categories)
    missing = [key for key, value in required.items() if not value["found"]]
    if not seed:
        missing.append("runtime_lineage_seed")
    return {
        "schema_version": "assistant_lineage_source_inventory_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "canonical_runtime_lineage_source": {
            "kind": "runtime_deployment_metadata",
            "source": "deployment metadata generated from the bot LineageContext or active config lineage block",
            "path_patterns": list(SOURCE_GLOBS["deployment_metadata"]),
            "assistant_lineage_block": "assistant_lineage",
        },
        "source_categories": categories,
        "required_id_locations": required,
        "runtime_lineage_seed": seed,
        "status": "pass" if not missing else "blocked",
        "missing_requirements": missing,
    }


def _scan_category(name: str, globs: tuple[str, ...]) -> dict[str, Any]:
    paths: list[dict[str, Any]] = []
    for pattern in globs:
        for path in sorted(ROOT.glob(pattern)):
            ids = _ids_in_path(path, category=name)
            if any(ids.values()) or name == "deployment_metadata":
                row: dict[str, Any] = {"path": _rel(path), "ids": ids}
                if name == "deployment_metadata":
                    payloads = _load_payloads(path)
                    row["assistant_lineage_present"] = any(
                        isinstance(payload.get("assistant_lineage"), dict)
                        for payload in payloads
                    )
                paths.append(row)
    return {"path_count": len(paths), "paths": paths}


def _ids_in_path(path: Path, *, category: str) -> dict[str, list[str]]:
    ids = {key: [] for key in ID_ALIASES}
    for payload in _load_payloads(path):
        if category == "strategy_change_ledgers":
            record_id = _text(payload.get("record_id"))
            if record_id:
                ids["strategy_change_record_ids"].append(record_id)
        for key, aliases in ID_ALIASES.items():
            ids[key].extend(_values_for_aliases(payload, aliases))
    return {key: _unique(values)[:20] for key, values in ids.items()}


def _load_payloads(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        if path.suffix == ".jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item.get("payload") if isinstance(item.get("payload"), dict) else item)
            return rows
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return [payload] if isinstance(payload, dict) else []


def _values_for_aliases(value: Any, aliases: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in aliases:
                values.extend(_list_text(item))
            values.extend(_values_for_aliases(item, aliases))
    elif isinstance(value, list):
        for item in value:
            values.extend(_values_for_aliases(item, aliases))
    return values


def _locations_for_key(key: str, categories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    locations = []
    values: list[str] = []
    for category, payload in categories.items():
        for row in payload["paths"]:
            row_values = row["ids"].get(key, [])
            if row_values:
                locations.append({"category": category, "path": row["path"], "sample_values": row_values[:5]})
                values.extend(row_values)
    return {"found": bool(locations), "locations": locations, "values": _unique(values)[:20]}


def _runtime_lineage_seed(categories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ledger = ROOT / "packages" / "trading_assistant" / "memory" / "findings" / "strategy_change_ledger.jsonl"
    candidate_ids = _locations_for_key("candidate_ids", categories)["values"]
    for payload in _load_payloads(ledger):
        proposal_ids = _list_text(payload.get("source_proposal_ids")) or _list_text(payload.get("proposal_ids"))
        weekly_ids = _values_for_aliases(payload.get("mutation_diff") or {}, ("source_weekly_signal_ids", "weekly_signal_ids"))
        record_id = _text(payload.get("record_id"))
        bot_id = _text(payload.get("bot_id"))
        strategy_id = _text(payload.get("strategy_id"))
        deployment_id = _text(payload.get("deployment_id"))
        if bot_id and strategy_id and deployment_id and proposal_ids and record_id:
            return {
                "bot_id": bot_id,
                "strategy_id": strategy_id,
                "deployment_id": deployment_id,
                "proposal_ids": proposal_ids,
                "source_weekly_signal_ids": _unique(weekly_ids),
                "strategy_change_record_ids": [record_id],
                "candidate_ids": candidate_ids[:5],
                "source_path": _rel(ledger),
            }
    return {}


def _list_text(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "")]
    return [str(value)]


def _text(value: Any) -> str:
    return str(value) if value not in (None, "") else ""


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
