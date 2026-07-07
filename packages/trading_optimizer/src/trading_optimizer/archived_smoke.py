from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "cumulative_mutations",
    "gate_decisions",
    "selected_candidates",
    "canonical_round_outputs",
)


def dimension_payloads(records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    """Canonical smoke fixture view over frozen optimizer outputs."""

    payloads = {name: [] for name in DIMENSIONS}
    for record in records:
        manifest = _role_payload(record, root, "rounds_manifest")
        optimized = _role_payload(record, root, "optimized_config")
        latest_round = _latest_round(manifest, record.get("latest_round"))
        header = {
            "baseline_id": record.get("baseline_id", ""),
            "bot": record.get("bot", ""),
            "family": record.get("family", ""),
            "strategy": record.get("strategy", ""),
            "latest_round": record.get("latest_round"),
        }
        payloads["cumulative_mutations"].append({
            **header,
            "mutations": _first_mapping(
                latest_round,
                optimized,
                "cumulative_mutations",
                "selected_mutations",
                "changed_mutations",
                "mutations",
                "repair_mutations",
                "new_mutations",
            ),
        })
        payloads["gate_decisions"].append({
            **header,
            "gate_status": latest_round.get("gate_status") or latest_round.get("status") or "",
            "gate_passed": latest_round.get("gate_passed", latest_round.get("passed", "")),
            "reject_reason": latest_round.get("reject_reason", ""),
            "gate_failure_reasons": latest_round.get("gate_failure_reasons", []),
        })
        payloads["selected_candidates"].append({
            **header,
            "candidate": _first_mapping(
                latest_round,
                optimized,
                "selected_candidate",
                "best_candidate",
                "source_candidate",
                "candidate",
                "source_candidate_payload",
            ),
        })
        payloads["canonical_round_outputs"].append({
            **header,
            "files": [
                {
                    "role": item.get("role", ""),
                    "baseline_path": item.get("baseline_path", ""),
                    "sha256": item.get("sha256", ""),
                }
                for item in record.get("files", [])
            ],
        })
    return payloads


def stable_payload_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _role_payload(record: dict[str, Any], root: Path, role: str) -> dict[str, Any]:
    for item in record.get("files", []):
        if item.get("role") == role:
            path = root / str(item.get("baseline_path", ""))
            if path.exists() and path.suffix == ".json":
                return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _latest_round(manifest: dict[str, Any], latest_round: Any) -> dict[str, Any]:
    rounds = manifest.get("rounds", [])
    for row in rounds if isinstance(rounds, list) else []:
        if isinstance(row, dict) and str(row.get("round")) == str(latest_round):
            return row
    if rounds and isinstance(rounds[-1], dict):
        return rounds[-1]
    return {}


def _first_mapping(*items: Any) -> Any:
    payloads = [item for item in items if isinstance(item, dict)]
    keys = [item for item in items if isinstance(item, str)]
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value not in (None, "", {}, []):
                return value
    return {}
