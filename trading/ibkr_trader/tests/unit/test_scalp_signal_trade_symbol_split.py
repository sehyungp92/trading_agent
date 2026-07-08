from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from backtests.scalp.config_ivb_auction import IvbAuctionBacktestConfig
from backtests.scalp.config_po3_reversal import Po3ReversalBacktestConfig
from backtests.scalp.engine.ivb_auction_engine import load_ivb_auction_data
from backtests.scalp.engine.po3_reversal_engine import load_po3_reversal_data
from strategies.scalp.ivb_auction.config import EntryTrigger, IvbModule, TradeDirection as IvbDirection
from strategies.scalp.ivb_auction.core.logic import on_fill as ivb_on_fill
from strategies.scalp.ivb_auction.core.state import IvbAuctionCoreState, IvbFill
from strategies.scalp.ivb_auction.models import IvbSetup
from strategies.scalp.po3_reversal.config import EntryType, SetupTier, TradeDirection as Po3Direction
from strategies.scalp.po3_reversal.core.logic import on_fill as po3_on_fill
from strategies.scalp.po3_reversal.core.state import Po3Fill, Po3ReversalCoreState
from strategies.scalp.po3_reversal.models import Po3Setup


def test_ivb_loads_nq_analysis_data_but_defaults_to_mnq_execution(monkeypatch, tmp_path: Path) -> None:
    requested: list[str] = []

    def fake_load_bar_data(data_dir, symbol):
        requested.append(symbol)
        return {}

    monkeypatch.setattr("backtests.scalp.engine.ivb_auction_engine.load_bar_data", fake_load_bar_data)
    monkeypatch.setattr("backtests.scalp.engine.ivb_auction_engine.load_tick_data", lambda data_dir, symbol: None)
    config = IvbAuctionBacktestConfig(data_dir=tmp_path)

    data = load_ivb_auction_data(config)

    assert requested == ["NQ"]
    assert data.analysis_symbol == "NQ"
    assert config.trade_symbol == "MNQ"
    assert config.symbols == ["MNQ"]


def test_po3_loads_nq_analysis_data_and_es_confirmation_but_executes_mnq(monkeypatch, tmp_path: Path) -> None:
    requested: list[str] = []

    def fake_load_bar_data(data_dir, symbol):
        requested.append(symbol)
        return {}

    monkeypatch.setattr("backtests.scalp.data.multi_instrument.load_bar_data", fake_load_bar_data)
    config = Po3ReversalBacktestConfig(data_dir=tmp_path)

    data = load_po3_reversal_data(config)

    assert requested == ["NQ", "ES"]
    assert data.analysis_symbol == "NQ"
    assert data.trade_symbol == "NQ"
    assert data.confirmation_symbol == "ES"
    assert config.trade_symbol == "MNQ"
    assert config.symbols == ["MNQ"]


def test_ivb_core_realized_pnl_uses_mnq_contract_value() -> None:
    signal_time = datetime(2026, 4, 29, 14, 30, tzinfo=timezone.utc)
    setup = IvbSetup(
        setup_id="ivb-mnq",
        symbol="MNQ",
        module=IvbModule.A1_CONTINUATION,
        direction=IvbDirection.LONG,
        trigger=EntryTrigger.PROFILE_RELOAD,
        signal_time=signal_time,
        score=80.0,
        entry_price=100.0,
        stop_price=99.0,
        tp1_price=101.0,
        tp2_price=102.0,
        qty=1,
        size_multiplier=1.0,
        rr_to_tp1=1.0,
        qty_open=1,
        avg_entry=100.0,
        metadata={"_remaining_entry_commission": 0.0},
    )
    state = IvbAuctionCoreState(
        active_setups={setup.setup_id: setup},
        order_to_setup={"ivb-stop": setup.setup_id},
        order_kind={"ivb-stop": "stop"},
    )

    next_state, _, events = ivb_on_fill(
        state,
        IvbFill(
            oms_order_id="ivb-stop",
            fill_price=101.0,
            fill_qty=1,
            symbol="MNQ",
            fill_time=signal_time,
            commission=0.0,
            order_role="stop",
        ),
    )

    assert next_state.daily_pnl == pytest.approx(2.0)
    assert events[-1].symbol == "MNQ"


def test_po3_core_realized_pnl_uses_mnq_contract_value() -> None:
    signal_time = datetime(2026, 4, 29, 14, 30, tzinfo=timezone.utc)
    setup = Po3Setup(
        setup_id="po3-mnq",
        symbol="MNQ",
        direction=Po3Direction.LONG,
        tier=SetupTier.A,
        entry_type=EntryType.STOP_CONFIRMATION,
        signal_time=signal_time,
        score=80.0,
        entry_price=100.0,
        stop_price=99.0,
        target_price=101.0,
        qty=1,
        rr=1.0,
        qty_open=1,
        avg_entry=100.0,
        metadata={"_entry_commission": 0.0},
    )
    state = Po3ReversalCoreState(
        active_setup=setup,
        order_to_setup={"po3-target": setup.setup_id},
        order_kind={"po3-target": "target"},
    )

    next_state, _, events = po3_on_fill(
        state,
        Po3Fill(
            oms_order_id="po3-target",
            fill_price=101.0,
            fill_qty=1,
            symbol="MNQ",
            fill_time=signal_time,
            commission=0.0,
            order_role="target",
        ),
    )

    assert next_state.daily_pnl == pytest.approx(2.0)
    assert next_state.weekly_pnl == pytest.approx(2.0)
    assert events[-1].symbol == "MNQ"
