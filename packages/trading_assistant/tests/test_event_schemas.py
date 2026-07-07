"""Tests for event-schema lineage fields — backward-compat round-trip."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from trading_assistant.schemas.events import MissedOpportunityEvent, TradeEvent


def _trade_kwargs(**overrides) -> dict:
    payload = {
        "trade_id": "t1",
        "bot_id": "bot_a",
        "pair": "BTC/USDT",
        "side": "LONG",
        "entry_time": datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
        "exit_time": datetime(2026, 5, 1, 11, 0, tzinfo=timezone.utc),
        "entry_price": 50000.0,
        "exit_price": 50500.0,
        "position_size": 0.1,
        "pnl": 50.0,
        "pnl_pct": 1.0,
    }
    payload.update(overrides)
    return payload


def _missed_kwargs() -> dict:
    return {
        "bot_id": "bot_a",
        "pair": "BTC/USDT",
        "signal": "RSI_OVERSOLD",
    }


def test_trade_event_round_trips_without_lineage() -> None:
    trade = TradeEvent(**_trade_kwargs())
    payload = trade.model_dump(mode="json")
    redo = TradeEvent(**payload)
    assert redo.trade_id == "t1"
    assert redo.deployment_id is None
    assert redo.experiment_id is None
    assert redo.net_pnl_source == "gross_as_net_fallback"
    assert redo.after_cost_status == "unavailable"


def test_trade_event_round_trips_with_lineage() -> None:
    kwargs = _trade_kwargs()
    kwargs.update(
        deployment_id="dep_001",
        experiment_id="exp_001",
        variant_id="A",
        parameter_set_id="ps_42",
        strategy_version="v3",
        config_version="2026-05",
        signal_generation_version="sg_7",
        code_sha="abc1234",
    )
    trade = TradeEvent(**kwargs)
    payload = json.loads(json.dumps(trade.model_dump(mode="json"), default=str))
    redo = TradeEvent(**payload)
    assert redo.deployment_id == "dep_001"
    assert redo.experiment_id == "exp_001"
    assert redo.variant_id == "A"
    assert redo.parameter_set_id == "ps_42"
    assert redo.strategy_version == "v3"
    assert redo.config_version == "2026-05"
    assert redo.signal_generation_version == "sg_7"
    assert redo.code_sha == "abc1234"


def test_trade_event_ignores_unknown_fields() -> None:
    """Forward-compat: an old reader should accept newer payloads with
    extra fields — model_config="ignore" was added for migration windows."""
    kwargs = _trade_kwargs()
    kwargs["future_field_we_dont_know"] = {"any": "shape"}
    trade = TradeEvent(**kwargs)
    assert not hasattr(trade, "future_field_we_dont_know")


def test_trade_event_derives_observed_after_cost_outcome() -> None:
    kwargs = _trade_kwargs()
    kwargs.update(
        gross_pnl=50.0,
        net_pnl=48.5,
        net_pnl_source="observed_broker_statement",
        after_cost_status="observed",
        fees_paid=1.0,
        entry_slippage_bps=2.0,
        exit_slippage_bps=1.0,
        cost_model_version="costs-v1",
        execution_model_version="exec-v1",
    )

    trade = TradeEvent(**kwargs)

    assert trade.after_cost_outcome is not None
    assert trade.after_cost_outcome.status.value == "observed"
    assert trade.after_cost_outcome.gross_pnl == 50.0
    assert trade.after_cost_outcome.net_pnl == 48.5
    assert trade.after_cost_outcome.fees == 1.0
    assert trade.after_cost_outcome.slippage_bps == 3.0
    assert trade.after_cost_outcome.cost_model_version == "costs-v1"


def test_trade_event_labels_inferred_after_costs() -> None:
    trade = TradeEvent(**_trade_kwargs(fees_paid=1.25, entry_slippage_bps=1.0))

    assert trade.net_pnl == 48.75
    assert trade.net_pnl_source == "inferred_from_cost_fields"
    assert trade.after_cost_status == "inferred"


def test_trade_event_preserves_kis_tax_and_commission_after_costs() -> None:
    trade = TradeEvent(**_trade_kwargs(
        bot_id="k_stock_trader",
        strategy_id="KALCB",
        pair="005930",
        fees_paid=0.5,
        commission=0.3,
        tax=0.2,
        borrow_cost=0.1,
        cost_model_version="kis-fees-v1",
    ))

    assert trade.net_pnl == 48.9
    assert trade.after_cost_status == "inferred"
    assert trade.after_cost_outcome is not None
    assert trade.after_cost_outcome.fees == 0.5
    assert trade.after_cost_outcome.commission == 0.3
    assert trade.after_cost_outcome.tax == 0.2
    assert trade.after_cost_outcome.funding_or_borrow == 0.1
    assert trade.after_cost_outcome.cost_model_version == "kis-fees-v1"


def test_trade_event_preserves_crypto_funding_and_slippage_after_costs() -> None:
    trade = TradeEvent(**_trade_kwargs(
        bot_id="crypto_trader",
        pair="BTC-PERP",
        total_fees=2.0,
        funding_paid=1.0,
        slippage_pct=0.0005,
        execution_model_version="hyperliquid-exec-v1",
    ))

    assert trade.net_pnl == 47.0
    assert trade.after_cost_status == "inferred"
    assert trade.after_cost_outcome is not None
    assert trade.after_cost_outcome.fees == 2.0
    assert trade.after_cost_outcome.funding_or_borrow == 1.0
    assert trade.after_cost_outcome.slippage_bps == 5.0
    assert trade.after_cost_outcome.execution_model_version == "hyperliquid-exec-v1"


def test_missed_event_round_trips_with_lineage() -> None:
    kwargs = _missed_kwargs()
    kwargs.update(
        deployment_id="dep_002",
        experiment_id="exp_002",
        variant_id="B",
    )
    evt = MissedOpportunityEvent(**kwargs)
    redo = MissedOpportunityEvent(**evt.model_dump(mode="json"))
    assert redo.deployment_id == "dep_002"
    assert redo.experiment_id == "exp_002"
    assert redo.variant_id == "B"


def test_missed_event_round_trips_without_lineage() -> None:
    evt = MissedOpportunityEvent(**_missed_kwargs())
    redo = MissedOpportunityEvent(**evt.model_dump(mode="json"))
    assert redo.deployment_id is None
    assert redo.experiment_id is None


def test_crypto_canonical_envelope_unwraps_payload_and_preserves_join_keys() -> None:
    envelope = {
        "event_id": "evt-crypto-trade-1",
        "bot_id": "crypto_trader",
        "event_type": "trade",
        "schema_version": "assistant_event_v1",
        "family_id": "crypto_perps",
        "portfolio_id": "default",
        "account_alias": "default",
        "strategy_id": "momentum",
        "assistant_strategy_id": "MomentumPullback_M15",
        "exchange_timestamp": "2026-05-01T12:00:00+00:00",
        "logical_event_id": "crypto:momentum:BTC:001",
        "payload_hash": "payload-hash-crypto",
        "priority": "normal",
        "deployment_id": "dep-crypto",
        "config_version": "cfg-crypto",
        "source": {"sink": "postgres"},
        "payload": {
            "event_metadata": {
                "bot_id": "crypto_trader",
                "strategy_id": "momentum",
                "exchange_timestamp": "2026-05-01T12:00:00+00:00",
                "local_timestamp": "2026-05-01T12:00:01+00:00",
                "data_source": "hyperliquid",
                "event_type": "trade",
                "payload_key": "trade-1",
            },
            "trade_id": "trade-1",
            "symbol": "BTC",
            "side": "LONG",
            "entry_time": "2026-05-01T12:00:00+00:00",
            "exit_time": "2026-05-01T13:00:00+00:00",
            "entry_price": 100.0,
            "exit_price": 103.0,
            "position_size": 2.0,
            "pnl": 6.0,
            "pnl_pct": 3.0,
            "entry_fill_ids": ["fill-entry"],
            "exit_fill_ids": ["fill-exit"],
            "intent_id": "intent-1",
            "portfolio_rule_event_id": "portfolio-rule-1",
            "risk_decision_id": "risk-1",
            "notional_usd": 200.0,
            "total_fees": 0.35,
            "realized_pnl_net": 5.65,
            "r_multiple": 1.4,
            "join_completeness": {"fills": True},
        },
    }

    trade = TradeEvent.model_validate(envelope)

    assert trade.bot_id == "crypto_trader"
    assert trade.strategy_id == "MomentumPullback_M15"
    assert trade.assistant_strategy_id == "MomentumPullback_M15"
    assert trade.family_id == "crypto_perps"
    assert trade.pair == "BTC"
    assert trade.logical_event_id == "crypto:momentum:BTC:001"
    assert trade.payload_hash == "payload-hash-crypto"
    assert trade.priority == "normal"
    assert trade.source == {"sink": "postgres"}
    assert trade.deployment_id == "dep-crypto"
    assert trade.config_version == "cfg-crypto"
    assert trade.entry_fill_ids == ["fill-entry"]
    assert trade.exit_fill_ids == ["fill-exit"]
    assert trade.intent_id == "intent-1"
    assert trade.portfolio_rule_event_id == "portfolio-rule-1"
    assert trade.risk_decision_id == "risk-1"
    assert trade.notional_usd == 200.0
    assert trade.total_fees == 0.35
    assert trade.realized_pnl_net == 5.65
    assert trade.r_multiple == 1.4
    assert trade.join_completeness == {"fills": True}
    assert trade.event_metadata is not None
    assert trade.event_metadata.data_source_id == "hyperliquid"


def test_k_stock_canonical_envelope_uses_source_strategy_and_krx_join_keys() -> None:
    envelope = {
        "event_id": "evt-kalcb-trade-1",
        "bot_id": "k_stock_trader",
        "event_type": "trade",
        "schema_version": "assistant_event_v1",
        "family_id": "krx_equity",
        "portfolio_id": "olr_kalcb",
        "account_alias": "kis-prod",
        "strategy_id": "KALCB",
        "exchange_timestamp": "2026-06-04T09:45:00+09:00",
        "logical_event_id": "KALCB:005930:20260604:trade",
        "event_ref": "kalcb-event-ref-1",
        "payload_hash": "payload-hash-krx",
        "priority": 2,
        "deployment_id": "dep-krx",
        "config_version": "cfg-krx",
        "kis_resource_plan_hash": "resource-hash",
        "portfolio_policy_hash": "policy-hash",
        "payload": json.dumps({
            "trade_id": "kalcb-trade-1",
            "symbol": "005930",
            "side": "LONG",
            "entry_time": "2026-06-04T09:35:00+09:00",
            "exit_time": "2026-06-04T14:55:00+09:00",
            "entry_price": 70000.0,
            "exit_price": 70700.0,
            "position_size": 10,
            "pnl": 7000.0,
            "pnl_pct": 1.0,
            "intent_id": "intent-krx-1",
            "entry_order_event_refs": ["entry-order-event-1"],
            "oms_order_id": "oms-1",
            "kis_order_id": "kis-1",
            "kis_exec_id": "exec-1",
        }),
    }

    trade = TradeEvent.model_validate(envelope)

    assert trade.bot_id == "k_stock_trader"
    assert trade.strategy_id == "KALCB"
    assert trade.family_id == "krx_equity"
    assert trade.portfolio_id == "olr_kalcb"
    assert trade.pair == "005930"
    assert trade.logical_event_id == "KALCB:005930:20260604:trade"
    assert trade.event_ref == "kalcb-event-ref-1"
    assert trade.payload_hash == "payload-hash-krx"
    assert trade.priority == 2
    assert trade.intent_id == "intent-krx-1"
    assert trade.entry_order_event_refs == ["entry-order-event-1"]
    assert trade.oms_order_id == "oms-1"
    assert trade.kis_order_id == "kis-1"
    assert trade.kis_exec_id == "exec-1"
    assert trade.kis_resource_plan_hash == "resource-hash"
    assert trade.portfolio_policy_hash == "policy-hash"
