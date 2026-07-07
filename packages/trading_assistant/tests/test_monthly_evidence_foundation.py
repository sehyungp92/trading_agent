from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.orchestrator.backtest_invocation import validate_backtest_repo
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.lineage_audit import LineageAuditor
from trading_assistant.orchestrator.market_data_jobs import MarketDataSyncJob
from trading_assistant.orchestrator.scheduler import SchedulerConfig, build_scheduled_job_specs
from trading_assistant.schemas.events import DailySnapshot, EventMetadata, TradeEvent
from trading_assistant.schemas.market_data_manifest import MarketDataManifest
from trading_assistant.schemas.monthly_validation import MonthlyValidationStatus
from trading_assistant.schemas.strategy_change_ledger import StrategyChangeRecord, StrategyChangeRecordType
from trading_assistant.schemas.strategy_profile import StrategyProfile, StrategyRegistry
from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder
from trading_assistant.skills.lineage_utils import build_lineage_summary
from trading_assistant.skills.monthly_validation_orchestrator import MonthlyValidationOrchestrator, MonthlyValidationRequest
from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger


def _trade(**overrides) -> TradeEvent:
    data = {
        "trade_id": "t1",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "pair": "AAPL",
        "side": "LONG",
        "entry_time": "2026-04-02T10:00:00Z",
        "exit_time": "2026-04-02T11:00:00Z",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "position_size": 1.0,
        "pnl": 1.0,
        "pnl_pct": 1.0,
        "strategy_version": "sv1",
        "config_version": "cv1",
        "deployment_id": "dep1",
        "parameter_set_id": "ps1",
        "experiment_id": "exp1",
    }
    data.update(overrides)
    return TradeEvent.model_validate(data)


def test_daily_snapshot_and_summary_lineage_back_compat() -> None:
    snap = DailySnapshot(date="2026-04-02", bot_id="bot1")
    assert snap.lineage_summary is None
    assert snap.lineage_gap is False

    builder = DailyMetricsBuilder("2026-04-02", "bot1")
    summary = builder.build_summary([
        _trade(trade_id="t1"),
        _trade(trade_id="t2", config_version=None),
    ])

    assert summary.lineage_summary is not None
    assert summary.lineage_summary["strategy_version_counts"] == {"sv1": 2}
    assert summary.lineage_summary["missing_field_counts"]["config_version"] == 1
    assert summary.lineage_gap is True


def test_lineage_summary_reads_nested_event_metadata() -> None:
    trade = _trade(
        event_metadata=EventMetadata(
            bot_id="bot1",
            exchange_timestamp=datetime(2026, 4, 2, 10, tzinfo=timezone.utc),
            local_timestamp=datetime(2026, 4, 2, 10, tzinfo=timezone.utc),
            data_source_id="feed1",
            event_type="trade",
            payload_key="t1",
            bar_id="bar1",
        ),
        code_sha="abc123",
        variant_id="control",
        signal_id="sig1",
    )

    summary = build_lineage_summary([trade])

    assert summary["data_source_id_counts"] == {"feed1": 1}
    assert summary["bar_id_counts"] == {"bar1": 1}
    assert summary["code_sha_counts"] == {"abc123": 1}
    assert summary["missing_field_counts"]["event_id"] == 0


def test_lineage_audit_writes_gap_and_telemetry_manifest(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    findings = tmp_path / "memory" / "findings"
    bot_dir = curated / "2026-04-02" / "bot1"
    bot_dir.mkdir(parents=True)
    (bot_dir / "trades.jsonl").write_text(_trade(config_version=None).model_dump_json() + "\n")

    auditor = LineageAuditor(curated, findings, required_lineage_ratio=0.95)
    reports = auditor.audit(
        bot_id="bot1",
        strategy_id="strat1",
        window_start=date(2026, 4, 1),
        window_end=date(2026, 4, 30),
    )
    assert reports[0].blocks_authoritative_validation
    assert (findings / "lineage_gaps.jsonl").exists()

    manifest_path = tmp_path / "telemetry_manifest.json"
    manifest = auditor.build_telemetry_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        window_start=date(2026, 4, 1),
        window_end=date(2026, 4, 30),
        output_path=manifest_path,
    )
    assert manifest.lineage_coverage_ratio < 1
    assert manifest_path.exists()


def test_strategy_change_ledger_and_context_loader(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    ledger = StrategyChangeLedger(findings)
    record = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.MONTHLY_REVIEW,
        run_month="2026-04",
        monthly_status="no_change",
        evidence_paths=["evidence.json"],
        decision_reason="clean shadow run",
    )
    assert ledger.record(record) is True
    assert ledger.record(record) is False
    assert ledger.get_by_id(record.record_id) is not None

    ctx = ContextBuilder(tmp_path / "memory")
    history = ctx.load_strategy_change_ledger(bot_id="bot1")
    assert history[0]["record_id"] == record.record_id


def test_strategy_change_ledger_projection_and_compaction(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    ledger = StrategyChangeLedger(findings)
    record = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.PROPOSED_CHANGE,
        run_month="2026-04",
        monthly_status="watch",
        evidence_paths=["before.json"],
        decision_reason="initial",
    )

    assert ledger.record(record) is True
    assert ledger.update(
        record.record_id,
        monthly_status="deployed",
        evidence_paths=["after.json"],
    )

    projection_path = Path(str(ledger.path) + ".index.json")
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    assert projection[record.record_id]["monthly_status"] == "deployed"

    assert ledger.compact() == 1
    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert ledger.get_by_id(record.record_id).evidence_paths == ["after.json"]


def test_strategy_change_monthly_review_ids_are_month_specific() -> None:
    april = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.MONTHLY_REVIEW,
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
    )
    march = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.MONTHLY_REVIEW,
        run_id="monthly-bot1-strat1-2026-03",
        run_month="2026-03",
    )

    assert april.record_id != march.record_id


def test_strategy_change_monthly_review_id_is_stable_across_rerun_days() -> None:
    from datetime import datetime, timezone

    first = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.MONTHLY_REVIEW,
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
        created_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
    )
    second = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.MONTHLY_REVIEW,
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
        created_at=datetime(2026, 5, 30, tzinfo=timezone.utc),
    )

    assert first.record_id == second.record_id


def test_monthly_orchestrator_blocks_missing_coverage_and_records_review(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    findings = tmp_path / "memory" / "findings"
    bot_dir = curated / "2026-04-02" / "bot1"
    bot_dir.mkdir(parents=True)
    (bot_dir / "trades.jsonl").write_text(_trade().model_dump_json() + "\n")

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=tmp_path / "market_data",
        backtest_repo_path=tmp_path / "missing_backtests",
        backtest_artifact_root=tmp_path / "artifacts",
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        optimizer_sequence_enabled=False,
    ))

    assert result.status == MonthlyValidationStatus.INSUFFICIENT_DATA
    assert result.strategy_change_record_id
    assert (findings / "strategy_change_ledger.jsonl").exists()
    assert Path(result.monthly_report_path).exists()


def test_monthly_orchestrator_runs_with_fixture_artifacts(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    findings = tmp_path / "memory" / "findings"
    bot_dir = curated / "2026-04-02" / "bot1"
    bot_dir.mkdir(parents=True)
    (bot_dir / "trades.jsonl").write_text(_trade().model_dump_json() + "\n")

    market_root = tmp_path / "market_data"
    manifest_path = market_root / "manifests" / "bot1" / "strat1" / "2026-04.coverage_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    MarketDataManifest(
        source="fixture",
        market="equity",
        symbol="AAPL",
        timeframe="1m",
        start_ts=datetime(2026, 4, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 4, 30, tzinfo=timezone.utc),
        expected_bars=10,
        actual_bars=10,
        usable_for_authoritative_validation=True,
    ).model_dump_json(indent=2)
    manifest = MarketDataManifest(
        source="fixture",
        market="equity",
        symbol="AAPL",
        timeframe="1m",
        start_ts=datetime(2026, 4, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 4, 30, tzinfo=timezone.utc),
        expected_bars=10,
        actual_bars=10,
        usable_for_authoritative_validation=True,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    repo = tmp_path / "backtests"
    repo.mkdir()
    script = repo / "fixture_runner.py"
    script.write_text(
        """
import json
from pathlib import Path
import sys
manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path(manifest["artifact_root"])
root.mkdir(parents=True, exist_ok=True)
(root / "coverage_manifest.json").write_text("{}")
(root / "incumbent_validation.json").write_text("{}")
(root / "gap_attribution.json").write_text("{}")
(root / "mode_decision.json").write_text("{}")
(root / "objective_breakdown.json").write_text("{}")
(root / "candidate_results.jsonl").write_text("")
(root / "selected_candidates.json").write_text("[]")
(root / "rejected_candidates.jsonl").write_text("")
(root / "monthly_report.md").write_text("ok")
(root / "replay_parity_report.json").write_text(json.dumps({
    "bot_id": manifest["bot_id"],
    "strategy_id": manifest["strategy_id"],
    "run_month": manifest["run_month"],
    "trade_count_live": 1,
    "trade_count_replay": 1,
    "entry_match_rate": 1.0,
    "exit_match_rate": 1.0,
    "side_quantity_match_rate": 1.0,
    "status": "pass"
}))
required = [
    "coverage_manifest.json",
    "incumbent_validation.json",
    "gap_attribution.json",
    "mode_decision.json",
    "replay_parity_report.json",
    "objective_breakdown.json",
    "candidate_results.jsonl",
    "selected_candidates.json",
    "rejected_candidates.jsonl",
    "monthly_report.md",
    "stdout.log",
    "stderr.log",
    "exit_status.json",
]
(root / "artifact_index.json").write_text(json.dumps({
    "run_id": manifest["run_id"],
    "artifact_root": str(root),
    "artifacts": {name: str(root / name) for name in required},
}))
""",
        encoding="utf-8",
    )

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.status == MonthlyValidationStatus.NO_CHANGE
    assert result.strategy_change_record_id


def test_backtest_path_and_artifact_index_fail_closed(tmp_path: Path) -> None:
    ok, reason = validate_backtest_repo("")
    assert ok is False
    assert "empty" in reason

    curated = tmp_path / "curated"
    findings = tmp_path / "memory" / "findings"
    bot_dir = curated / "2026-04-02" / "bot1"
    bot_dir.mkdir(parents=True)
    (bot_dir / "trades.jsonl").write_text(_trade().model_dump_json() + "\n")

    market_root = tmp_path / "market_data"
    manifest_path = market_root / "manifests" / "bot1" / "strat1" / "2026-04.coverage_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(MarketDataManifest(
        source="fixture",
        market="equity",
        symbol="AAPL",
        timeframe="1m",
        start_ts=datetime(2026, 4, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 4, 30, tzinfo=timezone.utc),
        expected_bars=10,
        actual_bars=10,
        usable_for_authoritative_validation=True,
    ).model_dump_json(indent=2))

    repo = tmp_path / "backtests"
    repo.mkdir()
    script = repo / "fixture_runner.py"
    script.write_text(
        "from pathlib import Path\n"
        "import json, sys\n"
        "manifest = json.loads(Path(sys.argv[1]).read_text())\n"
        "root = Path(manifest['artifact_root'])\n"
        "root.mkdir(parents=True, exist_ok=True)\n"
        "(root / 'monthly_report.md').write_text('no index')\n",
        encoding="utf-8",
    )

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.status == MonthlyValidationStatus.UNSUPPORTED_NO_REPLAY_PLUGIN
    assert any("missing artifact_index.json" in reason for reason in result.blocking_reasons)


def test_market_data_sync_job_writes_strategy_manifest_and_event(tmp_path: Path) -> None:
    market_root = tmp_path / "market_data"
    data_file = market_root / "filesystem" / "equity" / "AAPL" / "1d" / "2026-04.parquet"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("not parquet", encoding="utf-8")
    stream = EventStream()
    registry = StrategyRegistry(strategies={
        "strat1": StrategyProfile(
            bot_id="bot1",
            asset_class="equity",
            symbols=["AAPL"],
        ),
    })

    summary = MarketDataSyncJob(
        market_data_root=market_root,
        strategy_registry=registry,
        event_stream=stream,
    ).run(run_month="2026-04", bot_ids=["bot1"])

    assert summary["requirements"] == 1
    assert (market_root / "manifests" / "bot1" / "strat1" / "2026-04.coverage_manifest.json").exists()
    assert any(event.event_type == "market_data_sync_done" for event in stream.get_recent())


def test_scheduler_has_monthly_foundation_jobs() -> None:
    async def noop(_scheduled_for=None) -> None:
        return None

    specs = build_scheduled_job_specs(
        SchedulerConfig(),
        worker_fn=noop,
        monitoring_fn=noop,
        relay_fn=noop,
        lineage_audit_fn=noop,
        market_data_sync_fn=noop,
        monthly_validation_fn=noop,
    )
    by_name = {spec.name: spec for spec in specs}
    assert by_name["lineage_audit"].hour == 6
    assert by_name["market_data_sync"].day == 1
    assert by_name["monthly_validation"].day == 2
    assert by_name["monthly_validation"].awaits_completion is True

    scoped_specs = build_scheduled_job_specs(
        SchedulerConfig(),
        worker_fn=noop,
        monitoring_fn=noop,
        relay_fn=noop,
        monthly_validation_fns=[{
            "fn": noop,
            "name_suffix": "bot1",
            "scope_key": "bot:bot1",
        }],
    )
    scoped = {spec.name: spec for spec in scoped_specs}
    assert scoped["monthly_validation_bot1"].scope_key == "bot:bot1"
    assert scoped["monthly_validation_bot1"].awaits_completion is True
