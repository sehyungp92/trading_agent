from __future__ import annotations

import json
import re

from libs.instrumentation.lineage import LineageContext
from libs.instrumentation.event_contract import write_startup_events
from libs.oms.risk.portfolio_rules import PortfolioRulesConfig


_RAW_ACCOUNT = re.compile(r"\b(?:U|DU)\d{5,}\b")


def _lineage() -> LineageContext:
    return LineageContext(
        bot_id="stock_trader",
        strategy_id="IARIC_v1",
        family_id="stock",
        portfolio_id="paper_default",
        account_alias="paper_ibkr_1",
        strategy_version="IARIC.1",
        config_version="cfg_1",
        portfolio_config_version="pcfg_1",
        risk_config_version="risk_1",
        allocation_version="alloc_1",
        strategy_registry_version="registry_1",
        deployment_id="dep_1",
        parameter_set_id="param_1",
        code_sha="abc123",
        trace_id="trace_1",
    )


def _walk(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def test_snapshot_and_lifecycle_contract_redacts_raw_accounts_and_preserves_alias(tmp_path) -> None:
    write_startup_events(
        tmp_path,
        _lineage(),
        effective_config={"broker_account_id": "U1234567"},
        allocation_state={"account_id": "U1234567"},
        portfolio_state={"account_id": "U1234567", "net_liquidation": 100_000.0},
        positions=[{"symbol": "AAPL", "qty": 2, "account_id": "U1234567"}],
        portfolio_rules_config=PortfolioRulesConfig(family_strategy_ids=("IARIC_v1",)),
    )

    for path in tmp_path.rglob("*"):
        if path.suffix != ".jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            assert payload.get("account_alias", "paper_ibkr_1") == "paper_ibkr_1"
            assert payload["lineage"].get("account_alias") == "paper_ibkr_1"
            assert not any(isinstance(item, str) and _RAW_ACCOUNT.search(item) for item in _walk(payload))
