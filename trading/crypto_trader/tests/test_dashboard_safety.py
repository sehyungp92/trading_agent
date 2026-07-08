"""Dashboard safety-query and health semantics regressions."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_safety_events_query_reads_canonical_envelope_and_nested_payload_fields():
    route = (ROOT / "infra/dashboard/src/app/api/live/route.ts").read_text(encoding="utf-8")

    assert "COALESCE(payload->'payload', '{}'::jsonb) AS body" in route
    assert "body->>'severity'" in route
    assert "body->>'status'" in route
    assert "body->>'approved'" in route
    assert "body->>'reject_reason'" in route
    assert "body_metadata->>'discrepancy_kind'" in route


def test_system_health_blocks_only_unresolved_or_error_safety_events():
    component = (ROOT / "infra/dashboard/src/components/SystemHealth.tsx").read_text(encoding="utf-8")

    assert 'event.event_type === "reconciliation_event"' not in component
    assert 'status === "open"' in component
    assert 'status === "failed"' in component
    assert 'status === "blocked"' in component
    assert 'status === "rejected"' in component
