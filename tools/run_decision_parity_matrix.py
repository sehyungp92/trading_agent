from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "artifacts" / "validation" / "decision_parity_matrix"
DIMENSIONS = {
    "entries",
    "exits",
    "filters",
    "order_intent",
    "risk_caps",
    "signals",
    "sizing",
    "stops",
}
SCOPES = (
    "crypto_trend_v1",
    "crypto_momentum_v1",
    "crypto_breakout_v1",
    "k_stock_olr_kalcb",
    "trading_stock_family",
    "trading_momentum_family",
    "trading_swing_family",
)
ACCEPTED_EXPLICIT_STATUSES = {
    "blocked_non_promotion",
    "non_promotion",
    "research_only",
    "superseded_by_portfolio_bundle",
}


def main() -> int:
    _parser().parse_args()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for scope in SCOPES:
        record, scope_errors = _record(scope)
        records.append(record)
        errors.extend(scope_errors)
    print(json.dumps({"valid": not errors, "records": records, "errors": errors}, indent=2))
    return 0 if not errors else 1


def _record(scope: str) -> tuple[dict[str, Any], list[str]]:
    path = REPORT_ROOT / scope / "decision_parity" / "decision_parity_report.json"
    record: dict[str, Any] = {
        "scope": scope,
        "decision_parity_report": _relative(path),
        "status": "missing",
        "dimensions": [],
    }
    if not path.exists():
        return record, [f"{scope}: archived decision parity report missing at {_relative(path)}"]
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        record["status"] = "invalid"
        return record, [f"{scope}: invalid decision parity JSON: {exc}"]
    statuses = _dimension_statuses(report)
    status = str(report.get("status") or "")
    record.update({
        "status": status,
        "dimensions": sorted(statuses),
        "dimension_statuses": statuses,
        "archived_provenance": report.get("archived_provenance", {}),
    })
    errors: list[str] = []
    missing = sorted(DIMENSIONS - set(statuses))
    if missing:
        errors.append(f"{scope}: missing decision parity dimensions {missing}")
    if status == "pass":
        for dimension, dimension_status in statuses.items():
            if dimension_status != "pass":
                errors.append(f"{scope}: {dimension} status is {dimension_status}")
    elif status in ACCEPTED_EXPLICIT_STATUSES:
        record["explicit_non_promotion_or_blocked_status"] = True
    else:
        errors.append(f"{scope}: decision parity status is {status or 'missing'}")
    return record, errors


def _dimension_statuses(report: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("dimension")): str(item.get("status"))
        for item in report.get("checks", [])
        if isinstance(item, dict) and item.get("dimension")
    }


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate archived decision parity evidence matrix.")
    parser.add_argument("--promoted-only", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
