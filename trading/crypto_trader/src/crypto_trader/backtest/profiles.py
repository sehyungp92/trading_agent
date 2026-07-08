"""Shared backtest economic profiles for parity-sensitive runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from crypto_trader.backtest.config import BacktestConfig


@dataclass(frozen=True, slots=True)
class BacktestEconomicProfile:
    """Canonical economics and accounting assumptions for backtests."""

    profile_id: str
    symbols: tuple[str, ...] = ("BTC", "ETH", "SOL")
    initial_equity: float = 10_000.0
    taker_fee_bps: float = 4.5
    maker_fee_bps: float = 1.0
    slippage_bps: float = 2.0
    spread_bps: float = 1.0
    train_pct: float = 0.70
    apply_funding: bool = True
    warmup_days: int = 60
    terminal_accounting_mode: str = "terminal_mark"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        return payload


LIVE_PARITY_PROFILE = BacktestEconomicProfile(profile_id="live_parity_v1")


def build_backtest_config_from_profile(
    *,
    profile: BacktestEconomicProfile = LIVE_PARITY_PROFILE,
    symbols: list[str] | tuple[str, ...] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    initial_equity: float | None = None,
    warmup_days: int | None = None,
) -> BacktestConfig:
    """Build a ``BacktestConfig`` from a canonical profile plus run window."""
    run_symbols = [str(symbol).upper() for symbol in (symbols or profile.symbols)]
    return BacktestConfig(
        symbols=run_symbols,
        start_date=start_date,
        end_date=end_date,
        initial_equity=profile.initial_equity if initial_equity is None else initial_equity,
        taker_fee_bps=profile.taker_fee_bps,
        maker_fee_bps=profile.maker_fee_bps,
        slippage_bps=profile.slippage_bps,
        spread_bps=profile.spread_bps,
        train_pct=profile.train_pct,
        apply_funding=profile.apply_funding,
        warmup_days=profile.warmup_days if warmup_days is None else warmup_days,
    )


def assert_backtest_config_matches_profile(
    config: BacktestConfig,
    *,
    profile: BacktestEconomicProfile = LIVE_PARITY_PROFILE,
    terminal_accounting_mode: str | None = None,
) -> None:
    """Raise when a backtest config drifts from the declared economic profile."""
    expected = {
        "symbols": [symbol.upper() for symbol in profile.symbols],
        "initial_equity": profile.initial_equity,
        "taker_fee_bps": profile.taker_fee_bps,
        "maker_fee_bps": profile.maker_fee_bps,
        "slippage_bps": profile.slippage_bps,
        "spread_bps": profile.spread_bps,
        "train_pct": profile.train_pct,
        "apply_funding": profile.apply_funding,
        "warmup_days": profile.warmup_days,
    }
    mismatches = [
        f"{name}: {getattr(config, name)!r} != {value!r}"
        for name, value in expected.items()
        if getattr(config, name) != value
    ]
    if terminal_accounting_mode is not None and terminal_accounting_mode != profile.terminal_accounting_mode:
        mismatches.append(
            f"terminal_accounting_mode: {terminal_accounting_mode!r} != {profile.terminal_accounting_mode!r}"
        )
    if mismatches:
        raise ValueError(
            f"BacktestConfig does not match {profile.profile_id}: " + "; ".join(mismatches)
        )


def profile_hash(profile: BacktestEconomicProfile = LIVE_PARITY_PROFILE) -> str:
    """Return a stable hash for a profile definition."""
    payload = json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
