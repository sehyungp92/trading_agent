from __future__ import annotations

import json
from datetime import datetime as real_datetime
from datetime import timezone

from libs.instrumentation.event_contract import enrich_payload, write_error_event
from libs.instrumentation.lineage import (
    LineageContext,
    canonical_json,
    compute_code_sha,
    compute_portfolio_config_version,
    compute_risk_config_version,
    lineage_from_config,
    lineage_from_runtime,
    redact_config,
    resolve_deployment_id,
    stable_hash,
)


def _complete_lineage(**overrides) -> LineageContext:
    values = {
        "bot_id": "bot",
        "strategy_id": "TPC",
        "family_id": "swing",
        "portfolio_id": "paper_default",
        "account_alias": "paper_ibkr_1",
        "strategy_version": "TPC.1",
        "config_version": "cfg_1",
        "portfolio_config_version": "pcfg_1",
        "risk_config_version": "risk_1",
        "allocation_version": "alloc_1",
        "strategy_registry_version": "registry_1",
        "deployment_id": "dep_1",
        "parameter_set_id": "param_1",
        "code_sha": "abc",
        "trace_id": "trace_1",
    }
    values.update(overrides)
    return LineageContext(**values)


def test_canonical_hashes_are_stable_and_key_order_independent() -> None:
    left = {"risk": {"max": 3, "symbols": ["AAPL", "MSFT"]}, "enabled": True}
    right = {"enabled": True, "risk": {"symbols": ["AAPL", "MSFT"], "max": 3}}

    assert canonical_json(left) == canonical_json(right)
    assert stable_hash("cfg_", left) == stable_hash("cfg_", right)
    assert compute_portfolio_config_version(left) == compute_portfolio_config_version(right)


def test_redaction_removes_secrets_and_raw_accounts() -> None:
    redacted = redact_config(
        {
            "api_key": "secret",
            "broker_account": "U1234567",
            "account_alias": "paper_ibkr_1",
            "nested": {"password": "secret"},
        }
    )

    assert redacted["api_key"] == "<redacted>"
    assert redacted["broker_account"] == "<redacted>"
    assert redacted["account_alias"] == "paper_ibkr_1"
    assert redacted["nested"]["password"] == "<redacted>"


def test_versions_change_when_relevant_config_changes() -> None:
    base = {"risk": {"heat_cap_R": 3.0}, "drawdown_tiers": [0.08, 0.12]}
    changed = {"risk": {"heat_cap_R": 4.0}, "drawdown_tiers": [0.08, 0.12]}

    assert compute_risk_config_version(base) != compute_risk_config_version(changed)


def test_lineage_from_config_risk_version_includes_dynamic_portfolio_rules() -> None:
    config = {"bot_id": "stock_trader", "strategy_id": "IARIC_v1", "family_id": "stock"}

    base = lineage_from_config(
        config,
        family_id="stock",
        strategy_id="IARIC_v1",
        portfolio_rules_config={"same_sector_heat_cap_R": 2.0},
    )
    changed = lineage_from_config(
        config,
        family_id="stock",
        strategy_id="IARIC_v1",
        portfolio_rules_config={"same_sector_heat_cap_R": 3.0},
    )

    assert base.risk_config_version != changed.risk_config_version


def test_risk_version_ignores_private_dataclass_validator_fields() -> None:
    left = {"limit": 1, "_VALID_VALUES": ["a"]}
    right = {"limit": 1, "_VALID_VALUES": ["b"]}

    assert compute_risk_config_version({}, left, {}) == compute_risk_config_version({}, right, {})


def test_lineage_from_runtime_uses_env_overrides_and_fail_open_git_sha(tmp_path) -> None:
    lineage = lineage_from_runtime(
        bot_id="stock_trader",
        strategy_id="IARIC_v1",
        family_id="stock",
        portfolio_config={"capital": {"nav": 100_000}, "risk": {"heat_cap_R": 3}},
        strategy_manifest={"family": "stock", "artifact_config": {"version": "iaric.test"}},
        effective_strategy_config={"risk_pct": 0.01},
        portfolio_rules_config={"sector_heat_cap": 3.8},
        repo_root=tmp_path,
        env={
            "DEPLOYMENT_ID": "dep_test",
            "TRACE_ID": "trace_test",
            "PORTFOLIO_ID": "paper_default",
            "ACCOUNT_ALIAS": "paper_ibkr_1",
        },
    )

    assert lineage.deployment_id == "dep_test"
    assert lineage.trace_id == "trace_test"
    assert lineage.portfolio_id == "paper_default"
    assert lineage.account_alias == "paper_ibkr_1"
    assert lineage.config_version.startswith("cfg_")
    assert lineage.portfolio_config_version.startswith("pcfg_")
    assert lineage.risk_config_version.startswith("risk_")
    assert lineage.allocation_version.startswith("alloc_")
    assert lineage.strategy_registry_version.startswith("registry_")
    assert compute_code_sha(tmp_path) == "unknown"


def test_generated_deployment_id_includes_config_version(monkeypatch) -> None:
    class FrozenDateTime:
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("libs.instrumentation.lineage.datetime", FrozenDateTime)

    first = resolve_deployment_id(
        LineageContext(
            bot_id="bot",
            strategy_id="IARIC_v1",
            family_id="stock",
            config_version="cfg_a",
            code_sha="abc123",
        ),
        env={},
    )
    second = resolve_deployment_id(
        LineageContext(
            bot_id="bot",
            strategy_id="IARIC_v1",
            family_id="stock",
            config_version="cfg_b",
            code_sha="abc123",
        ),
        env={},
    )

    assert first.startswith("dep_2026_06_03_")
    assert first != second


def test_lineage_from_config_accepts_existing_mapping_and_aliases() -> None:
    lineage = lineage_from_config(
        {
            "bot_id": "bot",
            "lineage": {
                "strategy_id": "TPC",
                "proposal_ids": ["proposal-1"],
                "suggestion_ids": ["suggestion-1"],
                "source_weekly_signal_ids": ["weekly-1"],
                "strategy_change_record_ids": ["change-1"],
                "candidate_ids": ["candidate-1"],
                "monthly_search_brief_id": "brief-1",
            },
        },
        family_id="swing",
    )

    assert isinstance(lineage, LineageContext)
    assert lineage.bot_id == "bot"
    assert lineage.family_id == "swing"
    assert lineage.proposal_ids == ("proposal-1",)
    assert lineage.suggestion_ids == ("suggestion-1",)
    assert lineage.source_weekly_signal_ids == ("weekly-1",)
    assert lineage.strategy_change_record_ids == ("change-1",)
    assert lineage.candidate_ids == ("candidate-1",)
    assert lineage.monthly_search_brief_id == "brief-1"


def test_enrich_payload_adds_top_level_lineage_and_param_alias() -> None:
    lineage = _complete_lineage(
        proposal_ids=("proposal-1",),
        source_weekly_signal_ids=("weekly-1",),
        strategy_change_record_ids=("change-1",),
        candidate_ids=("candidate-1",),
        monthly_search_brief_id="brief-1",
    )

    event = enrich_payload({"trade_id": "t1"}, lineage=lineage, event_type="trade")

    assert event["bot_id"] == "bot"
    assert event["strategy_id"] == "TPC"
    assert event["schema_version"] == "trade_event_v2"
    assert event["parameter_set_id"] == "param_1"
    assert event["param_set_id"] == "param_1"
    assert event["proposal_ids"] == ["proposal-1"]
    assert event["candidate_ids"] == ["candidate-1"]
    assert event["lineage"]["deployment_id"] == "dep_1"
    assert event["lineage"]["source_weekly_signal_ids"] == ["weekly-1"]
    assert event["lineage"]["strategy_change_record_ids"] == ["change-1"]
    assert event["lineage"]["monthly_search_brief_id"] == "brief-1"
    assert "lineage_gaps" not in event


def test_enrich_payload_keeps_param_aliases_identical_when_payload_conflicts() -> None:
    lineage = _complete_lineage(
        strategy_id="IARIC_v1",
        family_id="stock",
        strategy_version="IARIC_v1.0.0",
        parameter_set_id="param_lineage",
    )

    event = enrich_payload(
        {"trade_id": "t1", "param_set_id": "bare_payload_hash"},
        lineage=lineage,
        event_type="trade",
    )

    assert event["parameter_set_id"] == "param_lineage"
    assert event["param_set_id"] == "param_lineage"
    assert event["lineage"]["parameter_set_id"] == "param_lineage"


def test_enrich_payload_promotes_legacy_bare_param_set_id() -> None:
    event = enrich_payload(
        {"trade_id": "t1", "param_set_id": "bare_payload_hash"},
        lineage=None,
        event_type="trade",
    )

    assert event["parameter_set_id"] == "param_bare_payload_hash"
    assert event["param_set_id"] == "param_bare_payload_hash"


def test_enrich_payload_exposes_guide_lineage_gap_aliases() -> None:
    event = enrich_payload({"bot_id": "bot"}, lineage=None, event_type="trade")

    assert event["lineage_gap"] is True
    assert "lineage_missing_fields" in event
    assert event["lineage_gaps"] == event["lineage_missing_fields"]
    assert "deployment_id" in event["lineage_missing_fields"]


def test_write_error_event_uses_enriched_error_contract(tmp_path) -> None:
    lineage = _complete_lineage()

    path = write_error_event(
        tmp_path,
        lineage,
        component="trade_logger",
        method="log_entry",
        message="boom",
        error_type="ValueError",
        context={"trade_id": "trade_1"},
    )

    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert event["event_type"] == "error"
    assert event["schema_version"] == "error_event_v2"
    assert event["component"] == "trade_logger"
    assert event["context"]["trade_id"] == "trade_1"
    assert event["deployment_id"] == "dep_1"
    assert event["parameter_set_id"] == "param_1"
    assert event["param_set_id"] == "param_1"
    assert "lineage_gaps" not in event


def test_oms_contract_schema_versions_match_guide() -> None:
    lineage = _complete_lineage(
        bot_id="oms",
        strategy_id="",
        family_id="stock",
        strategy_version="oms.1",
    )

    rule_event = enrich_payload({}, lineage=lineage, event_type="portfolio_rule_check", scope="portfolio")
    denial_event = enrich_payload({}, lineage=lineage, event_type="risk_denial", scope="oms")

    assert rule_event["schema_version"] == "portfolio_rule_check_v2"
    assert denial_event["schema_version"] == "risk_denial_v2"
