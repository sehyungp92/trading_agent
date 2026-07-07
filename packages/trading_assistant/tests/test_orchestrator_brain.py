import json

import pytest

from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain, ActionType


@pytest.fixture
def brain() -> OrchestratorBrain:
    return OrchestratorBrain()


class TestOrchestratorBrain:
    def test_trade_event_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "t001",
            "bot_id": "bot1",
            "event_type": "trade",
            "payload": json.dumps({"trade_id": "t001", "pnl": 50.0}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_error_critical_triggers_immediate_alert(self, brain: OrchestratorBrain):
        event = {
            "event_id": "err001",
            "bot_id": "bot3",
            "event_type": "error",
            "payload": json.dumps({"severity": "CRITICAL", "message": "crash"}),
        }
        actions = brain.decide(event)
        assert any(a.type == ActionType.ALERT_IMMEDIATE for a in actions)

    def test_error_high_triggers_triage(self, brain: OrchestratorBrain):
        event = {
            "event_id": "err002",
            "bot_id": "bot2",
            "event_type": "error",
            "payload": json.dumps({"severity": "HIGH", "message": "repeated timeout"}),
        }
        actions = brain.decide(event)
        assert any(a.type == ActionType.SPAWN_TRIAGE for a in actions)

    def test_error_medium_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "err003",
            "bot_id": "bot1",
            "event_type": "error",
            "payload": json.dumps({"severity": "MEDIUM", "message": "timeout"}),
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details is not None
        assert actions[0].details["event_type"] == "error"

    def test_missed_opportunity_queued(self, brain: OrchestratorBrain):
        event = {
            "event_id": "m001",
            "bot_id": "bot1",
            "event_type": "missed_opportunity",
            "payload": json.dumps({"signal": "EMA cross"}),
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_heartbeat_updates_tracking(self, brain: OrchestratorBrain):
        event = {
            "event_id": "hb001",
            "bot_id": "bot1",
            "event_type": "heartbeat",
            "payload": "{}",
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.UPDATE_HEARTBEAT

    def test_coordinator_action_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "ca001",
            "bot_id": "swing_multi_01",
            "event_type": "coordinator_action",
            "source_strategy": "ATRSS",
            "target_strategy": "Helix",
            "action": "tighten_stops",
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].bot_id == "swing_multi_01"
        assert actions[0].details["event_type"] == "coordinator_action"
        assert actions[0].details["payload"]["action"] == "tighten_stops"

    def test_daily_snapshot_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "ds001",
            "bot_id": "bot1",
            "event_type": "daily_snapshot",
            "payload": json.dumps({"total_pnl": 120.0}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == "daily_snapshot"

    @pytest.mark.parametrize(
        ("relay_event_type", "canonical_event_type"),
        [
            ("instrumented_trades", "trade"),
            ("missed_opportunities", "missed_opportunity"),
            ("daily_snapshots", "daily_snapshot"),
            ("pipeline_funnels", "pipeline_funnel"),
            ("health_reports", "health_report"),
        ],
    )
    def test_relay_file_event_aliases_queue_for_daily(
        self,
        brain: OrchestratorBrain,
        relay_event_type: str,
        canonical_event_type: str,
    ):
        event = {
            "event_id": f"{relay_event_type}-001",
            "bot_id": "crypto_trader",
            "event_type": relay_event_type,
            "payload": json.dumps({"timestamp": "2026-05-10T00:00:00+00:00"}),
        }

        actions = brain.decide(event)

        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == canonical_event_type

    def test_order_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "ord001",
            "bot_id": "bot1",
            "event_type": "order",
            "payload": json.dumps({"order_id": "o1", "side": "BUY"}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == "order"
        assert actions[0].details["exchange_timestamp"] == "2026-03-01T14:00:00+00:00"
        assert json.loads(actions[0].details["payload"])["order_id"] == "o1"

    def test_fill_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "fill001",
            "bot_id": "crypto_trader",
            "event_type": "fill",
            "payload": json.dumps({"fill_id": "f1", "price": 65000.0}),
            "exchange_timestamp": "2026-03-01T14:00:01+00:00",
        }

        actions = brain.decide(event)

        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == "fill"
        assert actions[0].details["exchange_timestamp"] == "2026-03-01T14:00:01+00:00"
        assert json.loads(actions[0].details["payload"])["fill_id"] == "f1"

    def test_inferred_fill_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "ifill001",
            "bot_id": "stock_trader",
            "event_type": "inferred_fill",
            "payload": json.dumps({"fill_id": "if1", "price": 180.25}),
        }

        actions = brain.decide(event)

        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == "inferred_fill"

    def test_fill_payload_keeps_top_level_canonical_join_keys(self, brain: OrchestratorBrain):
        event = {
            "event_id": "fill002",
            "bot_id": "k_stock_trader",
            "event_type": "fill",
            "family_id": "krx_equity",
            "portfolio_id": "olr_kalcb",
            "strategy_id": "KALCB",
            "intent_id": "intent-1",
            "kis_order_id": "kis-1",
            "payload": json.dumps({"fill_id": "f2"}),
        }

        actions = brain.decide(event)

        payload = actions[0].details["payload"]
        assert isinstance(payload, dict)
        assert payload["fill_id"] == "f2"
        assert payload["family_id"] == "krx_equity"
        assert payload["portfolio_id"] == "olr_kalcb"
        assert payload["strategy_id"] == "KALCB"
        assert payload["intent_id"] == "intent-1"
        assert payload["kis_order_id"] == "kis-1"

    def test_process_quality_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "pq001",
            "bot_id": "bot2",
            "event_type": "process_quality",
            "payload": json.dumps({"score": 85}),
            "exchange_timestamp": "2026-03-01T14:05:00+00:00",
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == "process_quality"
        assert actions[0].details["exchange_timestamp"] == "2026-03-01T14:05:00+00:00"
        assert json.loads(actions[0].details["payload"])["score"] == 85

    def test_safety_critical_parameter_change_alerts_and_queues_for_daily(
        self, brain: OrchestratorBrain,
    ):
        event = {
            "event_id": "pc001",
            "bot_id": "bot1",
            "event_type": "parameter_change",
            "payload": json.dumps({"param_name": "risk_per_trade", "old_value": 0.01, "new_value": 0.02}),
            "exchange_timestamp": "2026-03-01T15:00:00+00:00",
        }

        actions = brain.decide(event)

        assert [action.type for action in actions] == [
            ActionType.ALERT_IMMEDIATE,
            ActionType.QUEUE_FOR_DAILY,
        ]
        assert actions[0].details["safety_critical"] is True
        assert actions[1].details["event_type"] == "parameter_change"
        assert actions[1].details["exchange_timestamp"] == "2026-03-01T15:00:00+00:00"

    def test_bot_error_critical_alerts_immediately(self, brain: OrchestratorBrain):
        event = {
            "event_id": "be001",
            "bot_id": "bot3",
            "event_type": "bot_error",
            "payload": json.dumps({"severity": "CRITICAL", "message": "OOM"}),
        }
        actions = brain.decide(event)
        assert any(a.type == ActionType.ALERT_IMMEDIATE for a in actions)

    def test_bot_error_high_triggers_triage(self, brain: OrchestratorBrain):
        event = {
            "event_id": "be002",
            "bot_id": "bot2",
            "event_type": "bot_error",
            "payload": json.dumps({"severity": "HIGH", "error_type": "api_timeout"}),
        }
        actions = brain.decide(event)
        assert any(a.type == ActionType.SPAWN_TRIAGE for a in actions)

    def test_post_exit_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "pe001",
            "bot_id": "bot1",
            "event_type": "post_exit",
            "payload": json.dumps({"trade_id": "t1", "post_exit_pnl": 15.0}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details is not None
        assert actions[0].details["event_type"] == "post_exit"

    def test_portfolio_rule_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "pr001",
            "bot_id": "bot1",
            "event_type": "portfolio_rule_check",
            "payload": json.dumps({"rule": "max_exposure", "triggered": True}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details is not None
        assert actions[0].details["event_type"] == "portfolio_rule_check"

    def test_portfolio_rule_alias_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "pr002",
            "bot_id": "crypto_trader",
            "event_type": "portfolio_rule",
            "payload": json.dumps({"rule": "max_exposure", "allowed": False}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details is not None
        assert actions[0].details["event_type"] == "portfolio_rule_check"

    def test_reference_bridge_telemetry_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "risk001",
            "bot_id": "crypto_trader",
            "event_type": "risk_decision",
            "payload": json.dumps({"risk_decision_id": "risk-1", "approved": False}),
        }

        actions = brain.decide(event)

        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == "risk_decision"

    def test_market_snapshot_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "ms001",
            "bot_id": "bot1",
            "event_type": "market_snapshot",
            "payload": json.dumps({"btc_price": 65000}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details is not None
        assert actions[0].details["event_type"] == "market_snapshot"

    def test_exit_movement_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "em001",
            "bot_id": "bot1",
            "event_type": "exit_movement",
            "payload": json.dumps({"trade_id": "t1", "movement_bps": 25}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details is not None
        assert actions[0].details["event_type"] == "exit_movement"

    def test_stop_adjustment_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "sa001",
            "bot_id": "bot1",
            "event_type": "stop_adjustment",
            "payload": json.dumps({"trade_id": "t1", "old_stop": 100.0, "new_stop": 105.0}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == "stop_adjustment"
        assert actions[0].details["exchange_timestamp"] == "2026-03-01T14:00:00+00:00"

    def test_trade_entry_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "te001",
            "bot_id": "bot1",
            "event_type": "trade_entry",
            "payload": json.dumps({"trade_id": "t1", "side": "LONG", "entry_price": 50000}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == "trade_entry"
        assert actions[0].details["exchange_timestamp"] == "2026-03-01T14:00:00+00:00"

    def test_error_low_queued_for_weekly(self, brain: OrchestratorBrain):
        event = {
            "event_id": "err004",
            "bot_id": "bot1",
            "event_type": "error",
            "payload": json.dumps({"severity": "LOW", "message": "minor warning"}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_WEEKLY
        assert actions[0].details is not None
        assert actions[0].details["event_type"] == "error"
        assert actions[0].details["exchange_timestamp"] == "2026-03-01T14:00:00+00:00"

    def test_error_high_suppressed_queued_for_daily(self, brain: OrchestratorBrain):
        """Second HIGH error with same error_type is suppressed → QUEUE_FOR_DAILY with details."""
        base = {
            "bot_id": "bot1",
            "event_type": "error",
            "payload": json.dumps({"severity": "HIGH", "error_type": "api_timeout"}),
        }
        # First HIGH error spawns triage
        brain.decide({"event_id": "sup001", **base})
        # Second is suppressed — must still populate details
        actions = brain.decide({"event_id": "sup002", **base})
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details is not None
        assert actions[0].details["event_type"] == "error"

    def test_non_safety_parameter_change_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "pc002",
            "bot_id": "bot1",
            "event_type": "parameter_change",
            "payload": json.dumps({"param_name": "ema_period", "old_value": 20, "new_value": 25}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == "parameter_change"

    def test_truly_unknown_type_still_logged(self, brain: OrchestratorBrain):
        """Unrecognized event types still get LOG_UNKNOWN."""
        event = {
            "event_id": "u002",
            "bot_id": "bot1",
            "event_type": "totally_new_type",
            "payload": "{}",
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.LOG_UNKNOWN

    def test_unknown_event_type_logged(self, brain: OrchestratorBrain):
        event = {
            "event_id": "u001",
            "bot_id": "bot1",
            "event_type": "something_new",
            "payload": "{}",
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.LOG_UNKNOWN
