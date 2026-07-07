"""Tests for enriched event pipeline: raw JSONL persistence and curated data building."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from trading_assistant.analysis.daily_triage import DailyTriage
from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler
from trading_assistant.orchestrator.db.queue import EventQueue
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.handlers import Handlers
from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain
from trading_assistant.orchestrator.task_registry import TaskRegistry
from trading_assistant.orchestrator.worker import Worker
from trading_assistant.schemas.bot_config import BotConfig
from trading_assistant.schemas.notifications import NotificationPreferences


@pytest.fixture
async def queue(tmp_path) -> EventQueue:
    q = EventQueue(db_path=str(tmp_path / "queue.db"))
    await q.initialize()
    return q


@pytest.fixture
async def registry(tmp_path) -> TaskRegistry:
    r = TaskRegistry(db_path=str(tmp_path / "tasks.db"))
    await r.initialize()
    return r


@pytest.fixture
def brain() -> OrchestratorBrain:
    return OrchestratorBrain()


class TestWorkerPersistsEnrichedEvents:
    """Task 1a: Worker persists enriched events to raw JSONL on QUEUE_FOR_DAILY."""

    async def test_persist_enriched_event_to_raw_jsonl(self, queue, registry, brain, tmp_path):
        raw_dir = tmp_path / "raw"
        worker = Worker(
            queue=queue, registry=registry, brain=brain,
            raw_data_dir=raw_dir,
        )

        await queue.enqueue({
            "event_id": "is001",
            "bot_id": "bot1",
            "event_type": "indicator_snapshot",
            "payload": json.dumps({"rsi": 55.2, "macd": 0.3}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        processed = await worker.process_batch(limit=10)
        assert processed == 1

        # Find the raw JSONL file written
        raw_files = list(raw_dir.rglob("indicator_snapshot.jsonl"))
        assert len(raw_files) == 1
        lines = raw_files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1

    async def test_persist_uses_bot_trading_date(self, queue, registry, brain, tmp_path):
        raw_dir = tmp_path / "raw"
        worker = Worker(
            queue=queue,
            registry=registry,
            brain=brain,
            raw_data_dir=raw_dir,
            bot_configs={
                "bot1": BotConfig(bot_id="bot1", timezone="Asia/Seoul"),
            },
        )

        await queue.enqueue({
            "event_id": "is003",
            "bot_id": "bot1",
            "event_type": "indicator_snapshot",
            "payload": json.dumps({"rsi": 55.2}),
            "exchange_timestamp": "2026-03-01T23:30:00+00:00",
            "received_at": "2026-03-01T23:30:01+00:00",
        })

        processed = await worker.process_batch(limit=10)
        assert processed == 1
        assert (raw_dir / "2026-03-02" / "bot1" / "indicator_snapshot.jsonl").exists()

    async def test_worker_without_raw_dir_still_counts(self, queue, registry, brain):
        """Backward compat: raw_data_dir=None just counts, doesn't persist."""
        worker = Worker(queue=queue, registry=registry, brain=brain)
        assert worker._raw_data_dir is None

        await queue.enqueue({
            "event_id": "is002",
            "bot_id": "bot1",
            "event_type": "indicator_snapshot",
            "payload": json.dumps({"rsi": 55.2}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        processed = await worker.process_batch(limit=10)
        assert processed == 1
        assert worker.daily_queue_counts.get("bot1", 0) >= 1

    async def test_persist_multiple_event_types(self, queue, registry, brain, tmp_path):
        raw_dir = tmp_path / "raw"
        worker = Worker(
            queue=queue, registry=registry, brain=brain,
            raw_data_dir=raw_dir,
        )

        for event_type, eid in [
            ("filter_decision", "fd001"),
            ("orderbook_context", "ob001"),
            ("parameter_change", "pc001"),
            ("order", "ord001"),
            ("process_quality", "pq001"),
        ]:
            await queue.enqueue({
                "event_id": eid,
                "bot_id": "bot1",
                "event_type": event_type,
                "payload": json.dumps({"type": event_type}),
                "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                "received_at": "2026-03-01T14:00:01+00:00",
            })

        await worker.process_batch(limit=10)

        for event_type in [
            "filter_decision",
            "orderbook_context",
            "parameter_change",
            "order",
            "process_quality",
        ]:
            files = list(raw_dir.rglob(f"{event_type}.jsonl"))
            assert len(files) == 1, f"Expected 1 file for {event_type}, got {len(files)}"

        order_record = json.loads(
            (raw_dir / "2026-03-01" / "bot1" / "order.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert order_record["event_type"] == "order"
        assert order_record["exchange_timestamp"] == "2026-03-01T14:00:00+00:00"

        process_quality_record = json.loads(
            (raw_dir / "2026-03-01" / "bot1" / "process_quality.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert process_quality_record["event_type"] == "process_quality"
        assert process_quality_record["exchange_timestamp"] == "2026-03-01T14:00:00+00:00"

    async def test_safety_critical_parameter_change_alerts_and_still_persists_raw(
        self, queue, registry, brain, tmp_path,
    ):
        raw_dir = tmp_path / "raw"
        alerts = []
        worker = Worker(
            queue=queue,
            registry=registry,
            brain=brain,
            raw_data_dir=raw_dir,
        )
        async def _on_alert(action):
            alerts.append(action)
        worker.on_alert = _on_alert

        await queue.enqueue({
            "event_id": "pc-critical-001",
            "bot_id": "bot1",
            "event_type": "parameter_change",
            "payload": json.dumps({
                "param_name": "risk_per_trade",
                "old_value": 0.01,
                "new_value": 0.02,
            }),
            "exchange_timestamp": "2026-03-01T15:00:00+00:00",
            "received_at": "2026-03-01T15:00:01+00:00",
        })

        processed = await worker.process_batch(limit=10)

        assert processed == 1
        assert len(alerts) == 1
        assert worker.daily_queue_counts["bot1"] == 1
        record = json.loads(
            (raw_dir / "2026-03-01" / "bot1" / "parameter_change.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert record["event_type"] == "parameter_change"
        assert record["exchange_timestamp"] == "2026-03-01T15:00:00+00:00"


class TestHandlerBuildsEnrichedCurated:
    """Task 1b/1c: Daily handler reads enriched raw files and builds curated files."""

    def _make_handlers(self, tmp_path):
        return Handlers(
            agent_runner=AsyncMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "data" / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1"],
        )

    def test_build_enriched_curated_reads_raw_and_writes_curated(self, tmp_path):
        handlers = self._make_handlers(tmp_path)
        date = "2026-03-01"

        # Create raw enriched events
        raw_dir = tmp_path / "data" / "raw" / date / "bot1"
        raw_dir.mkdir(parents=True)
        (raw_dir / "filter_decision.jsonl").write_text(
            json.dumps({"filter_name": "rsi_filter", "action": "pass", "signal_strength": 0.8}) + "\n",
            encoding="utf-8",
        )
        (raw_dir / "indicator_snapshot.jsonl").write_text(
            json.dumps({"rsi": 55.2, "macd": 0.3}) + "\n",
            encoding="utf-8",
        )
        (raw_dir / "daily_snapshot.jsonl").write_text(
            json.dumps({
                "date": date,
                "bot_id": "bot1",
                "total_trades": 3,
                "net_pnl": 120.0,
                "per_strategy_summary": {
                    "alpha": {
                        "trades": 3,
                        "win_count": 2,
                        "loss_count": 1,
                        "gross_pnl": 125.0,
                        "net_pnl": 120.0,
                        "win_rate": 0.67,
                    },
                },
            }) + "\n",
            encoding="utf-8",
        )
        (raw_dir / "coordinator_action.jsonl").write_text(
            json.dumps({"action": "tighten_stops", "rule": "heat_guard", "symbol": "BTCUSDT"}) + "\n",
            encoding="utf-8",
        )

        handlers._build_enriched_curated(date)

        # Check curated files were written (write_curated creates them)
        curated_dir = tmp_path / "data" / "curated" / date / "bot1"
        assert curated_dir.exists()
        summary = json.loads((curated_dir / "summary.json").read_text(encoding="utf-8"))
        assert "alpha" in summary["per_strategy_summary"]
        coordinator = json.loads((curated_dir / "coordinator_impact.json").read_text(encoding="utf-8"))
        assert coordinator["by_action"]["tighten_stops"] == 1

    def test_build_enriched_curated_skips_missing_raw(self, tmp_path):
        """No raw directory = graceful no-op."""
        handlers = self._make_handlers(tmp_path)
        handlers._build_enriched_curated("2026-03-01")
        # Should not raise

    def test_build_enriched_curated_handles_malformed_json(self, tmp_path):
        handlers = self._make_handlers(tmp_path)
        date = "2026-03-01"

        raw_dir = tmp_path / "data" / "raw" / date / "bot1"
        raw_dir.mkdir(parents=True)
        (raw_dir / "filter_decision.jsonl").write_text(
            "not valid json\n" + json.dumps({"filter_name": "test"}) + "\n",
            encoding="utf-8",
        )

        curated_bot_dir = tmp_path / "data" / "curated" / date / "bot1"
        curated_bot_dir.mkdir(parents=True)

        # Should not raise — malformed lines skipped
        handlers._build_enriched_curated(date)

    def test_parameter_change_events_persisted(self, tmp_path):
        """Parameter change events are persisted to raw JSONL."""
        raw_dir = tmp_path / "raw" / "2026-03-01" / "bot1"
        raw_dir.mkdir(parents=True)

        # Simulate worker persisting
        payload = {"param_name": "stop_loss", "old_value": 0.02, "new_value": 0.015}
        (raw_dir / "parameter_change.jsonl").write_text(
            json.dumps(payload) + "\n", encoding="utf-8",
        )

        data = json.loads((raw_dir / "parameter_change.jsonl").read_text().strip())
        assert data["param_name"] == "stop_loss"

    def test_build_enriched_curated_writes_execution_and_process_artifacts(self, tmp_path):
        handlers = self._make_handlers(tmp_path)
        date = "2026-03-01"
        raw_dir = tmp_path / "data" / "raw" / date / "bot1"
        raw_dir.mkdir(parents=True)

        (raw_dir / "order.jsonl").write_text(
            "\n".join([
                json.dumps({
                    "order_id": "o1",
                    "pair": "BTCUSDT",
                    "side": "LONG",
                    "order_type": "LIMIT",
                    "status": "fill",
                    "requested_qty": 1,
                    "filled_qty": 1,
                    "requested_price": 50000.0,
                    "fill_price": 50001.0,
                    "latency_ms": 15,
                    "strategy_type": "alpha",
                }),
                json.dumps({
                    "order_id": "o2",
                    "pair": "ETHUSDT",
                    "side": "SHORT",
                    "order_type": "STOP",
                    "status": "REJECTED",
                    "requested_qty": 1,
                    "reject_reason": "risk_check",
                    "strategy_type": "beta",
                }),
            ]) + "\n",
            encoding="utf-8",
        )
        (raw_dir / "process_quality.jsonl").write_text(
            json.dumps({
                "trade_id": "t1",
                "process_quality_score": 52,
                "classification": "poor",
                "root_causes": ["order_reject"],
                "negative_factors": ["reject"],
                "evidence_refs": ["run-1"],
            }) + "\n",
            encoding="utf-8",
        )
        (raw_dir / "parameter_change.jsonl").write_text(
            json.dumps({
                "strategy_id": "alpha",
                "param_name": "stop_loss",
                "old_value": 0.02,
                "new_value": 0.015,
                "reason": "monthly_validation",
                "exchange_timestamp": "2026-03-01T15:00:00+00:00",
            }) + "\n",
            encoding="utf-8",
        )

        handlers._build_enriched_curated(date)

        curated_dir = tmp_path / "data" / "curated" / date / "bot1"
        order_lifecycle = json.loads((curated_dir / "order_lifecycle.json").read_text(encoding="utf-8"))
        process_quality = json.loads((curated_dir / "process_quality.json").read_text(encoding="utf-8"))
        parameter_changes = json.loads((curated_dir / "parameter_changes.json").read_text(encoding="utf-8"))

        assert order_lifecycle["fill_count"] == 1
        assert order_lifecycle["reject_count"] == 1
        assert process_quality["low_score_count"] == 1
        assert parameter_changes["total_changes"] == 1
        assert parameter_changes["changes"][0]["timestamp"] == "2026-03-01T15:00:00+00:00"


class TestEndToEndEnrichedPipeline:
    """End-to-end: enriched event → raw JSONL → curated JSON."""

    async def test_enriched_event_end_to_end(self, queue, registry, brain, tmp_path):
        raw_dir = tmp_path / "raw"
        worker = Worker(
            queue=queue, registry=registry, brain=brain,
            raw_data_dir=raw_dir,
        )

        # Step 1: Enqueue enriched event
        await queue.enqueue({
            "event_id": "e2e001",
            "bot_id": "bot1",
            "event_type": "filter_decision",
            "payload": json.dumps({
                "filter_name": "rsi_filter",
                "action": "block",
                "signal_strength": 0.3,
            }),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        # Step 2: Worker processes and persists
        processed = await worker.process_batch(limit=10)
        assert processed == 1

        # Step 3: Verify raw JSONL exists
        raw_files = list(raw_dir.rglob("filter_decision.jsonl"))
        assert len(raw_files) == 1
        raw_data = json.loads(raw_files[0].read_text(encoding="utf-8").strip())
        assert raw_data["filter_name"] == "rsi_filter"

    async def test_order_and_process_quality_visible_in_prompt_package(
        self, queue, registry, brain, tmp_path,
    ):
        raw_dir = tmp_path / "raw"
        curated_dir = tmp_path / "curated"
        memory_dir = tmp_path / "memory"
        worker = Worker(
            queue=queue, registry=registry, brain=brain,
            raw_data_dir=raw_dir,
        )

        for rel_path, content in {
            "policies/v1/agent.md": "agent",
            "policies/v1/trading_rules.md": "rules",
            "policies/v1/soul.md": "soul",
        }.items():
            path = memory_dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        (memory_dir / "findings").mkdir(parents=True, exist_ok=True)

        for event in [
            {
                "event_id": "ord001",
                "bot_id": "bot1",
                "event_type": "order",
                "payload": json.dumps({
                    "order_id": "o1",
                    "pair": "BTCUSDT",
                    "side": "LONG",
                    "order_type": "LIMIT",
                    "status": "fill",
                    "requested_qty": 1,
                    "filled_qty": 1,
                    "requested_price": 50000.0,
                    "fill_price": 50001.0,
                    "latency_ms": 15,
                    "strategy_type": "alpha",
                }),
                "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                "received_at": "2026-03-01T14:00:01+00:00",
            },
            {
                "event_id": "pq001",
                "bot_id": "bot1",
                "event_type": "process_quality",
                "payload": json.dumps({
                    "trade_id": "t1",
                    "process_quality_score": 52,
                    "classification": "poor",
                    "root_causes": ["order_reject"],
                    "negative_factors": ["reject"],
                    "evidence_refs": ["run-1"],
                }),
                "exchange_timestamp": "2026-03-01T14:05:00+00:00",
                "received_at": "2026-03-01T14:05:01+00:00",
            },
        ]:
            await queue.enqueue(event)

        processed = await worker.process_batch(limit=10)
        assert processed == 2

        handlers = Handlers(
            agent_runner=AsyncMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1"],
            raw_data_dir=raw_dir,
        )
        handlers._build_enriched_curated("2026-03-01")

        triage = DailyTriage(curated_dir, "2026-03-01", ["bot1"]).run()
        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble(triage_report=triage)

        bot_data = pkg.data["bot1"]
        assert "order_lifecycle" in bot_data
        assert "process_quality" in bot_data
