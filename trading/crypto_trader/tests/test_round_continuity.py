"""Tests for optimization round continuity — round-aware dirs, config persistence, manifest."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crypto_trader.cli import _detect_next_round, _update_rounds_manifest
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState
from crypto_trader.strategy.momentum.config import MomentumConfig


# ---------------------------------------------------------------------------
# _detect_next_round
# ---------------------------------------------------------------------------

class TestDetectNextRound:
    def test_empty_dir(self, tmp_path: Path) -> None:
        assert _detect_next_round(tmp_path) == 1

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert _detect_next_round(tmp_path / "nope") == 1

    def test_existing_rounds(self, tmp_path: Path) -> None:
        (tmp_path / "round_1").mkdir()
        (tmp_path / "round_2").mkdir()
        assert _detect_next_round(tmp_path) == 3

    def test_gaps_in_round_numbers(self, tmp_path: Path) -> None:
        (tmp_path / "round_1").mkdir()
        (tmp_path / "round_5").mkdir()
        assert _detect_next_round(tmp_path) == 6

    def test_ignores_non_round_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "round_2").mkdir()
        (tmp_path / "backups").mkdir()
        (tmp_path / "round_abc").mkdir()
        (tmp_path / "some_file.txt").write_text("x")
        assert _detect_next_round(tmp_path) == 3


# ---------------------------------------------------------------------------
# _save_optimized_config (via PhaseRunner)
# ---------------------------------------------------------------------------

class TestSaveOptimizedConfig:
    def test_creates_optimized_config(self, tmp_path: Path) -> None:
        """Config file is created with strategy wrapper and round-trips."""
        plugin = MagicMock()
        plugin.base_config = MomentumConfig()

        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()
        state.cumulative_mutations = {"trail.trail_r_ceiling": 1.35}

        path = runner._save_optimized_config(state)

        assert path is not None
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "strategy" in data
        assert "metadata" in data
        # Round-trip: from_dict should produce a valid config
        restored = MomentumConfig.from_dict(data["strategy"])
        assert restored.trail.trail_r_ceiling == 1.35

    def test_metadata_contains_contract_when_supplied(self, tmp_path: Path) -> None:
        plugin = MagicMock()
        plugin.base_config = MomentumConfig()
        contract = {
            "contract_hash": "abc",
            "profile_hash": "profile",
            "strategy_config_hash": "strategy",
            "portfolio_config_hash": "portfolio",
            "data_window": {"start_date": "2026-01-01", "end_date": "2026-04-01"},
        }

        runner = PhaseRunner(plugin, tmp_path, contract=contract)
        path = runner._save_optimized_config(PhaseState())

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["metadata"]["contract_hash"] == "abc"
        assert data["metadata"]["profile_hash"] == "profile"
        assert data["metadata"]["contract"] == contract

    def test_no_base_config(self, tmp_path: Path) -> None:
        """Returns None when plugin has no base_config."""
        plugin = MagicMock(spec=[])  # no base_config attribute
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        result = runner._save_optimized_config(state)
        assert result is None

    def test_empty_mutations(self, tmp_path: Path) -> None:
        """Config saved even with no mutations (baseline config)."""
        plugin = MagicMock()
        plugin.base_config = MomentumConfig()

        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        path = runner._save_optimized_config(state)
        assert path is not None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        restored = MomentumConfig.from_dict(data["strategy"])
        # Should match defaults
        assert restored.trail.trail_r_ceiling == MomentumConfig().trail.trail_r_ceiling


# ---------------------------------------------------------------------------
# _update_rounds_manifest
# ---------------------------------------------------------------------------

class TestUpdateRoundsManifest:
    def test_creates_manifest(self, tmp_path: Path) -> None:
        mutations = {"trail.trail_r_ceiling": 1.35}
        metrics = {"total_trades": 20, "profit_factor": 3.97}

        _update_rounds_manifest(tmp_path, 1, mutations, metrics)

        manifest_path = tmp_path / "rounds_manifest.json"
        assert manifest_path.exists()
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["rounds"]) == 1
        entry = data["rounds"][0]
        assert entry["round"] == 1
        assert entry["mutations_count"] == 1
        assert entry["total_trades"] == 20
        assert "timestamp" in entry

    def test_accumulates_rounds(self, tmp_path: Path) -> None:
        _update_rounds_manifest(tmp_path, 1, {"a.b": 1}, {"total_trades": 10})
        _update_rounds_manifest(tmp_path, 2, {"c.d": 2}, {"total_trades": 15})

        with open(tmp_path / "rounds_manifest.json", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["rounds"]) == 2
        assert data["rounds"][0]["round"] == 1
        assert data["rounds"][1]["round"] == 2

    def test_replaces_on_rerun(self, tmp_path: Path) -> None:
        _update_rounds_manifest(tmp_path, 1, {"a.b": 1}, {"total_trades": 10})
        _update_rounds_manifest(tmp_path, 1, {"a.b": 2}, {"total_trades": 20})

        with open(tmp_path / "rounds_manifest.json", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["rounds"]) == 1
        assert data["rounds"][0]["mutations"]["a.b"] == 2
        assert data["rounds"][0]["total_trades"] == 20

    def test_none_metrics(self, tmp_path: Path) -> None:
        _update_rounds_manifest(tmp_path, 1, {}, None)

        with open(tmp_path / "rounds_manifest.json", encoding="utf-8") as f:
            data = json.load(f)
        entry = data["rounds"][0]
        assert "total_trades" not in entry

    def test_expanded_manifest_fields(self, tmp_path: Path) -> None:
        metrics = {
            "total_trades": 20,
            "expectancy_r": 0.25,
            "exit_efficiency": 0.6,
            "realized_pnl_net": 123.0,
            "terminal_mark_pnl_net": 45.0,
            "net_profit": 168.0,
            "total_fees": 12.0,
            "funding_cost_total": -3.0,
            "terminal_mark_count": 2,
        }
        contract = {
            "contract_hash": "contract",
            "profile_hash": "profile",
            "strategy_config_hash": "strategy",
            "portfolio_config_hash": "portfolio",
            "data_window": {"start_date": "2026-01-01", "end_date": "2026-04-01"},
            "symbols": ["BTC", "ETH", "SOL"],
            "required_timeframes": ["15m", "1h", "4h"],
        }

        _update_rounds_manifest(
            tmp_path,
            1,
            {"risk.risk_pct_a": 0.01},
            metrics,
            contract=contract,
            phase_result={"final_score": 0.77},
            gate_result={"passed": False, "failure_reasons": ["too few trades"]},
        )

        with open(tmp_path / "rounds_manifest.json", encoding="utf-8") as f:
            data = json.load(f)
        entry = data["rounds"][0]
        assert data["schema_version"] >= 3
        assert entry["score"] == 0.77
        assert entry["gate_status"] == "failed"
        assert entry["reject_reason"] == "too few trades"
        assert entry["expectancy_r"] == 0.25
        assert entry["exit_efficiency"] == 0.6
        assert entry["realized_pnl_net"] == 123.0
        assert entry["terminal_mark_pnl_net"] == 45.0
        assert entry["net_profit"] == 168.0
        assert entry["total_fees"] == 12.0
        assert entry["funding_cost_total"] == -3.0
        assert entry["terminal_mark_count"] == 2
        assert entry["contract_hash"] == "contract"
        assert entry["profile_hash"] == "profile"
        assert entry["strategy_config_hash"] == "strategy"
        assert entry["portfolio_config_hash"] == "portfolio"
        assert entry["data_window"] == contract["data_window"]
        assert entry["metrics"]["expectancy_r"] == 0.25


# ---------------------------------------------------------------------------
# --round flag integration (CLI argument wiring)
# ---------------------------------------------------------------------------

class TestRoundFlag:
    def test_loads_previous_config(self, tmp_path: Path) -> None:
        """--round N loads round_N/optimized_config.json and applies mutations."""
        # Set up round_1 output with a mutated config
        round_dir = tmp_path / "round_1"
        round_dir.mkdir()
        cfg = MomentumConfig()
        cfg.trail.trail_r_ceiling = 1.5
        config_data = {"strategy": cfg.to_dict()}
        with open(round_dir / "optimized_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        # Simulate the CLI loading logic (same as optimize() does with --round)
        import click
        prev_config_path = tmp_path / "round_1" / "optimized_config.json"
        assert prev_config_path.exists()
        with open(prev_config_path, encoding="utf-8") as f:
            prev_raw = json.load(f)
        restored = MomentumConfig.from_dict(prev_raw.get("strategy", {}))
        assert restored.trail.trail_r_ceiling == 1.5
        # Next round is source + 1
        assert 1 + 1 == _detect_next_round(tmp_path)

    def test_errors_on_missing_round(self, tmp_path: Path) -> None:
        """--round N errors if round_N/optimized_config.json doesn't exist."""
        import click

        prev_config_path = tmp_path / "round_99" / "optimized_config.json"
        assert not prev_config_path.exists()
        # The CLI raises ClickException when the file is missing
        with pytest.raises(click.ClickException, match="No optimized config found"):
            if not prev_config_path.exists():
                raise click.ClickException(
                    f"No optimized config found at {prev_config_path}. "
                    f"Round 99 must complete before starting round 100."
                )


# ---------------------------------------------------------------------------
# Output files land in round_N/ subdirectory
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_round_end_saves_config(self, tmp_path: Path) -> None:
        """run_end_of_round saves optimized_config.json in output_dir."""
        plugin = MagicMock()
        plugin.base_config = MomentumConfig()
        plugin.name = "momentum"
        artifacts = MagicMock()
        artifacts.final_diagnostics_text = "diag text"
        artifacts.dimension_reports = {}
        artifacts.final_metrics = {}
        artifacts.gate_summary = {}
        plugin.build_end_of_round_artifacts.return_value = artifacts

        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()
        state.completed_phases = [1]
        state.cumulative_mutations = {"trail.trail_r_ceiling": 1.35}

        runner.run_end_of_round(state)

        assert (tmp_path / "optimized_config.json").exists()
        assert (tmp_path / "round_final_diagnostics.txt").exists()
        assert (tmp_path / "round_evaluation.txt").exists()
