"""Tests for optimization contract hashing and preflight checks."""

from __future__ import annotations

from datetime import date
import json

import pandas as pd
import pytest

from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.optimize import contracts
from crypto_trader.optimize.contracts import (
    build_optimization_contract,
    check_stale_artifacts,
    data_snapshot_fingerprint,
    required_timeframes,
    run_optimization_preflight,
)
from crypto_trader.optimize.types import Experiment, GateCriterion, PhaseSpec
from crypto_trader.strategy.momentum.config import MomentumConfig


class MiniPlugin:
    @property
    def num_phases(self) -> int:
        return 1

    @property
    def initial_mutations(self) -> dict:
        return {}

    def get_phase_spec(self, phase, state):
        return PhaseSpec(
            phase_num=phase,
            name="mini",
            candidates=[Experiment("risk", {"risk.risk_pct_a": 0.01})],
            scoring_weights={"coverage": 1.0},
            hard_rejects={"max_drawdown_pct": ("<=", 20.0)},
            gate_criteria=[GateCriterion("total_trades", ">=", 10.0)],
            max_rounds=3,
        )


def _bt_config(start: date = date(2026, 1, 1)):
    return build_backtest_config_from_profile(
        start_date=start,
        end_date=date(2026, 4, 1),
    )


def _contract(tmp_path, *, config=None, bt_config=None, scoring_ceilings=None):
    return build_optimization_contract(
        strategy_type="momentum",
        strategy_config=config or MomentumConfig(),
        backtest_config=bt_config or _bt_config(),
        data_dir=tmp_path / "data",
        profile=LIVE_PARITY_PROFILE,
        plugin=MiniPlugin(),
        scoring_ceilings=scoring_ceilings or {"coverage": 100.0},
    )


def test_required_timeframes_by_strategy() -> None:
    assert required_timeframes("momentum") == ["15m", "1h", "4h"]
    assert required_timeframes("trend") == ["15m", "1h", "1d"]
    assert required_timeframes("breakout") == ["30m", "4h"]


def test_data_snapshot_fingerprint_is_file_metadata_sensitive(tmp_path) -> None:
    path = tmp_path / "data" / "candles" / "BTC"
    path.mkdir(parents=True)
    candle_path = path / "15m.parquet"
    candle_path.write_bytes(b"one")

    first = data_snapshot_fingerprint(
        tmp_path / "data",
        symbols=["BTC"],
        timeframes=["15m"],
        include_funding=False,
    )
    candle_path.write_bytes(b"two-two")
    second = data_snapshot_fingerprint(
        tmp_path / "data",
        symbols=["BTC"],
        timeframes=["15m"],
        include_funding=False,
    )

    assert first["hash"] != second["hash"]


def test_contract_hash_is_deterministic_and_sensitive(tmp_path) -> None:
    first = _contract(tmp_path)
    second = _contract(tmp_path)
    assert first["contract_hash"] == second["contract_hash"]

    changed_window = _contract(tmp_path, bt_config=_bt_config(date(2026, 1, 2)))
    assert changed_window["contract_hash"] != first["contract_hash"]

    changed_config = MomentumConfig()
    changed_config.risk.risk_pct_a = 0.03
    assert _contract(tmp_path, config=changed_config)["contract_hash"] != first["contract_hash"]

    changed_ceiling = _contract(tmp_path, scoring_ceilings={"coverage": 200.0})
    assert changed_ceiling["contract_hash"] != first["contract_hash"]


def test_stale_phase_state_and_checkpoint_are_refused(tmp_path) -> None:
    contract = _contract(tmp_path)
    state_path = tmp_path / "phase_state.json"
    state_path.write_text(json.dumps({"contract_hash": "old"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="phase_state.json"):
        check_stale_artifacts(tmp_path, contract_hash=contract["contract_hash"])

    state_path.write_text(
        json.dumps({"contract_hash": contract["contract_hash"]}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "phase_1_greedy_checkpoint.json"
    checkpoint.write_text(json.dumps({"identity": "legacy"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="phase_1_greedy_checkpoint.json"):
        check_stale_artifacts(tmp_path, contract_hash=contract["contract_hash"])


def test_stale_optimized_config_and_manifest_are_refused(tmp_path) -> None:
    contract = _contract(tmp_path)
    round_dir = tmp_path / "round_1"
    round_dir.mkdir()
    optimized = round_dir / "optimized_config.json"
    optimized.write_text(
        json.dumps({"strategy": {}, "metadata": {"contract_hash": "old"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="optimized_config.json"):
        check_stale_artifacts(round_dir, contract_hash=contract["contract_hash"])

    optimized.write_text(
        json.dumps({
            "strategy": {},
            "metadata": {"contract_hash": contract["contract_hash"]},
        }),
        encoding="utf-8",
    )
    (tmp_path / "rounds_manifest.json").write_text(
        json.dumps({"rounds": [{"round": 1, "contract_hash": "old"}]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="rounds_manifest.json"):
        check_stale_artifacts(round_dir, contract_hash=contract["contract_hash"])


def test_preflight_checks_missing_candles_and_funding(tmp_path, monkeypatch) -> None:
    class EmptyStore:
        def __init__(self, base_dir):
            self.base_dir = base_dir

        def load_candles(self, symbol, timeframe):
            return pd.DataFrame()

        def load_funding(self, symbol):
            return pd.DataFrame()

    monkeypatch.setattr(contracts, "ParquetStore", EmptyStore)
    contract = _contract(tmp_path)

    with pytest.raises(RuntimeError, match="missing required data"):
        run_optimization_preflight(
            contract=contract,
            backtest_config=_bt_config(),
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "round_1",
        )


def test_preflight_accepts_available_profile_data(tmp_path, monkeypatch) -> None:
    class FullStore:
        def __init__(self, base_dir):
            self.base_dir = base_dir

        def load_candles(self, symbol, timeframe):
            return pd.DataFrame({"close": [1.0]})

        def load_funding(self, symbol):
            return pd.DataFrame({"rate": [0.0]})

    monkeypatch.setattr(contracts, "ParquetStore", FullStore)
    contract = _contract(tmp_path)

    run_optimization_preflight(
        contract=contract,
        backtest_config=_bt_config(),
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "round_1",
    )
