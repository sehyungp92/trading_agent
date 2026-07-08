from __future__ import annotations

import json

from click.testing import CliRunner

from crypto_trader.cli import cli
from crypto_trader.live.oms_store import OmsStore


def test_admin_resolve_discrepancy_cli_emits_correction_and_resolves_oms(tmp_path) -> None:
    state_dir = tmp_path / "state"
    config_path = tmp_path / "live.json"
    config_path.write_text(
        json.dumps({
            "state_dir": str(state_dir),
            "data_dir": str(tmp_path / "data"),
            "bot_id": "synthetic_bot",
            "portfolio_id": "paper_portfolio",
            "account_alias": "paper",
            "symbols": ["BTC"],
        }),
        encoding="utf-8",
    )
    store = OmsStore(state_dir)
    discrepancy_id = store.record_discrepancy(
        kind="missing_position",
        description="Synthetic missing BTC position.",
        symbol="BTC",
        strategy_id="UNKNOWN",
        metadata={"fill_id": "unknown_fill"},
    )
    store.close()

    result = CliRunner().invoke(cli, [
        "admin",
        "resolve-discrepancy",
        "--config",
        str(config_path),
        "--id",
        str(discrepancy_id),
        "--resolution",
        "Assigned to manual correction ledger.",
        "--resolved-by",
        "operator_1",
        "--metadata",
        '{"ticket":"INC-1"}',
    ])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["resolved"] is True
    assert payload["status"] == "RESOLVED"

    reopened = OmsStore(state_dir)
    try:
        discrepancy = reopened.get_discrepancy(discrepancy_id)
        assert discrepancy["status"] == "RESOLVED"
        assert discrepancy["metadata"]["resolved_by"] == "operator_1"
        assert discrepancy["metadata"]["ticket"] == "INC-1"
    finally:
        reopened.close()

    event_files = list((state_dir / "instrumentation" / "events" / "reconciliation_event").glob("*.jsonl"))
    assert event_files
    events = [
        json.loads(line)
        for path in event_files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    correction = events[-1]["payload"]
    assert correction["lifecycle_event_kind"] == "admin_correction"
    assert correction["status"] == "resolved"
    assert correction["correction_applied"] is True
    assert correction["metadata"]["resolved_by"] == "operator_1"
