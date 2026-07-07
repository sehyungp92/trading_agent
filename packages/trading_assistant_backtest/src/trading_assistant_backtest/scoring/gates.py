"""Deterministic gate payloads."""

from __future__ import annotations


def pass_gate_report(run_id: str, name: str) -> dict:
    return {
        "run_id": run_id,
        "status": "pass",
        "gate": name,
        "checks": [{"name": name, "passed": True, "reason": ""}],
    }
