from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from trading_assistant.analysis.discovery_prompt_assembler import DiscoveryPromptAssembler
from trading_assistant.schemas.monthly_candidates import MonthlyImprovementCandidate
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.skills.monthly_candidate_pipeline import MonthlyCandidatePipeline
from trading_assistant.skills.strategy_discovery_packet_builder import StrategyDiscoveryPacketBuilder


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_strategy_discovery_packet_clusters_missed_and_funnel_evidence(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    for day, suffix in [("2026-05-01", "a"), ("2026-05-02", "b")]:
        bot_dir = curated / day / "bot1"
        _write_jsonl(bot_dir / "missed.jsonl", [{
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "opportunity_id": f"miss-{suffix}",
            "pair": "AAPL",
            "signal": "opening_breakout",
            "market_regime": "trend_up",
            "would_have_pnl": 12.0,
            "signal_strength": 0.8,
            "blocked_by": "no_strategy_available",
        }])
        _write_jsonl(bot_dir / "trades.jsonl", [{
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "trade_id": f"trade-{suffix}",
            "pair": "AAPL",
            "market_regime": "trend_up",
            "net_pnl": 2.5,
        }])
        bot_dir.mkdir(parents=True, exist_ok=True)
        (bot_dir / "funnel_analysis.json").write_text(json.dumps({
            "stage_totals": {
                "setups_detected": 10,
                "confirmations": 4,
                "entries_attempted": 3,
                "fills": 2,
                "trades_closed": 2,
            },
        }), encoding="utf-8")
        (bot_dir / "regime_analysis.json").write_text(
            json.dumps({"dominant_regime": "trend_up"}),
            encoding="utf-8",
        )

    builder = StrategyDiscoveryPacketBuilder(curated, min_cluster_count=2)
    packet = builder.build(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
    )

    assert packet.authority == "diagnostics_only"
    assert packet.approval_gate_eligible is False
    assert len(packet.missed_opportunity_clusters) == 1
    assert packet.missed_opportunity_clusters[0].support_count == 2
    assert packet.missed_opportunity_clusters[0].estimated_after_cost_pnl == 24.0
    assert len(packet.denominator_clusters) == 1
    assert packet.control_slices
    assert packet.after_cost_estimates
    assert "Replay recurring clusters" in packet.replay_or_shadow_plan

    path = builder.write(packet, tmp_path / "artifacts")
    assert path is not None
    assert path.name == "strategy_discovery_packet.json"
    assert json.loads(path.read_text(encoding="utf-8"))["authority"] == "diagnostics_only"


def test_new_strategy_gate_requires_packet_and_cluster_citation(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    _write_jsonl(curated / "2026-05-01" / "bot1" / "trades.jsonl", [{
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "trade_id": "trade-control",
        "pair": "AAPL",
        "market_regime": "trend_up",
        "net_pnl": 2.5,
    }])
    _write_jsonl(curated / "2026-05-01" / "bot1" / "missed.jsonl", [
        {
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "opportunity_id": "miss-a",
            "pair": "AAPL",
            "signal": "opening_breakout",
            "market_regime": "trend_up",
            "would_have_pnl": 12.0,
        },
        {
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "opportunity_id": "miss-b",
            "pair": "AAPL",
            "signal": "opening_breakout",
            "market_regime": "trend_up",
            "would_have_pnl": 8.0,
        },
    ])
    packet = StrategyDiscoveryPacketBuilder(curated, min_cluster_count=2).build(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
    )
    packet_path = StrategyDiscoveryPacketBuilder(curated).write(packet, tmp_path / "artifacts")
    assert packet_path is not None
    cluster_id = packet.missed_opportunity_clusters[0].cluster_id
    monthly_result = MonthlyValidationResult(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        status=MonthlyValidationStatus.EXPERIMENT,
        strategy_discovery_packet_path=str(packet_path),
    )

    passing = MonthlyCandidatePipeline._new_strategy_discovery_gate(
        MonthlyImprovementCandidate.from_raw({
            "candidate_id": "new-1",
            "change_kind": "new_strategy",
            "strategy_discovery_cluster_ids": [cluster_id],
            "replay_or_experiment_plan": "Replay and then shadow for one completed month.",
        }),
        monthly_result,
    )
    missing_citation = MonthlyCandidatePipeline._new_strategy_discovery_gate(
        MonthlyImprovementCandidate.from_raw({
            "candidate_id": "new-2",
            "change_kind": "new_strategy",
            "replay_or_experiment_plan": "Replay and then shadow for one completed month.",
        }),
        monthly_result,
    )

    assert passing.passed is True
    assert missing_citation.passed is False
    assert "strategy_discovery_cluster_ids" in missing_citation.reason


def test_strategy_discovery_packet_is_not_written_for_below_threshold_clusters(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    _write_jsonl(curated / "2026-05-01" / "bot1" / "trades.jsonl", [{
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "trade_id": "trade-control",
        "pair": "AAPL",
        "market_regime": "trend_up",
        "net_pnl": 2.5,
    }])
    _write_jsonl(curated / "2026-05-01" / "bot1" / "missed.jsonl", [
        {
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "opportunity_id": "miss-a",
            "pair": "AAPL",
            "signal": "opening_breakout",
            "market_regime": "trend_up",
            "would_have_pnl": 0.1,
        },
        {
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "opportunity_id": "miss-b",
            "pair": "AAPL",
            "signal": "opening_breakout",
            "market_regime": "trend_up",
            "would_have_pnl": 0.1,
        },
    ])
    builder = StrategyDiscoveryPacketBuilder(curated, min_cluster_count=2, min_after_cost_estimate=1.0)
    packet = builder.build(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
    )
    stale_path = tmp_path / "artifacts" / "strategy_discovery_packet.json"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("{}", encoding="utf-8")

    written = builder.write(packet, tmp_path / "artifacts")

    assert written is None
    assert not stale_path.exists()


def test_new_strategy_gate_rejects_externally_supplied_below_threshold_packet(tmp_path: Path) -> None:
    packet_path = tmp_path / "strategy_discovery_packet.json"
    cluster_id = "weak-cluster"
    packet_path.write_text(json.dumps({
        "run_id": "monthly-bot1-strat1-2026-05",
        "run_month": "2026-05",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "missed_opportunity_clusters": [{
            "cluster_id": cluster_id,
            "source": "missed_opportunity",
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "support_count": 1,
            "control_count": 1,
            "estimated_after_cost_pnl": 10.0,
            "control_slice": {"control_count": 1},
        }],
        "control_slices": [{"control_count": 1}],
        "after_cost_estimates": [{"cluster_id": cluster_id, "estimated_after_cost_pnl": 10.0}],
        "replay_or_shadow_plan": "Replay and shadow before promotion.",
    }), encoding="utf-8")

    gate = MonthlyCandidatePipeline._new_strategy_discovery_gate(
        MonthlyImprovementCandidate.from_raw({
            "candidate_id": "new-weak",
            "change_kind": "new_strategy",
            "strategy_discovery_cluster_ids": [cluster_id],
            "replay_or_experiment_plan": "Replay and then shadow for one completed month.",
        }),
        MonthlyValidationResult(
            run_id="monthly-bot1-strat1-2026-05",
            run_month="2026-05",
            bot_id="bot1",
            strategy_id="strat1",
            status=MonthlyValidationStatus.EXPERIMENT,
            strategy_discovery_packet_path=str(packet_path),
        ),
    )

    assert gate.passed is False
    assert "no material after-cost clusters" in gate.reason


def test_structural_change_requires_discovery_packet_citation(tmp_path: Path) -> None:
    monthly_result = MonthlyValidationResult(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        status=MonthlyValidationStatus.EXPERIMENT,
    )

    gate = MonthlyCandidatePipeline._new_strategy_discovery_gate(
        MonthlyImprovementCandidate.from_raw({
            "candidate_id": "structural-1",
            "change_kind": "structural_change",
            "replay_or_experiment_plan": "Replay and then shadow for one completed month.",
        }),
        monthly_result,
    )

    assert gate.passed is False
    assert "strategy_discovery_packet.json" in gate.reason


def test_discovery_prompt_loads_strategy_discovery_packets(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    (memory / "policies" / "v1").mkdir(parents=True)
    curated = tmp_path / "curated"
    runs = tmp_path / "runs"
    packet_dir = runs / "monthly-bot1-strat1-2026-05"
    packet_dir.mkdir(parents=True)
    (packet_dir / "strategy_discovery_packet.json").write_text(json.dumps({
        "run_id": "monthly-bot1-strat1-2026-05",
        "run_month": "2026-05",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "authority": "diagnostics_only",
        "evidence_authority": "diagnostics_only",
        "approval_gate_eligible": False,
        "missed_opportunity_clusters": [{"cluster_id": "cluster-1"}],
        "control_slices": [{"control_count": 2}],
        "after_cost_estimates": [{"estimated_after_cost_pnl": 10.0}],
        "replay_or_shadow_plan": "Replay then shadow.",
    }), encoding="utf-8")

    package = DiscoveryPromptAssembler(
        date="2026-05-31",
        bots=["bot1"],
        curated_dir=curated,
        memory_dir=memory,
        runs_dir=runs,
    ).assemble()

    assert package.data["strategy_discovery_packets"][0]["authority"] == "diagnostics_only"
    assert "recurring opportunity clusters" in package.instructions
    assert "approval-grade evidence" in package.instructions
    assert str(packet_dir / "strategy_discovery_packet.json") in package.context_files
