"""Portfolio phase-auto round 1.

This round optimizes the three latest strategy configs as a portfolio.  It is
intentionally conservative about inference: broad replay sweeps are scored on a
development window plus a forward holdout, and the selected policy is validated
with the real portfolio backtester before artifacts are written.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import structlog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.diagnostics import generate_diagnostics
from crypto_trader.backtest.metrics import metrics_to_dict
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.backtest.runner import run as run_individual
from crypto_trader.core.models import Side, Trade
from crypto_trader.optimize.config_mutator import apply_mutations, merge_mutations
from crypto_trader.portfolio.backtest_runner import run_portfolio_backtest
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation
from crypto_trader.strategy.breakout.config import BreakoutConfig
from crypto_trader.strategy.momentum.config import MomentumConfig
from crypto_trader.strategy.trend.config import TrendConfig


logging.basicConfig(level=logging.ERROR)
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))

SYMBOLS = ["BTC", "ETH", "SOL"]
STRATEGIES = ["momentum", "trend", "breakout"]
INITIAL_EQUITY = 25_000.0
DATA_DIR = ROOT / "data"
ROUND_DIR = ROOT / "output" / "portfolio" / "round_1"
RECOMMENDED_CONFIG_DIR = ROUND_DIR / "recommended_strategy_configs"

# The saved diagnostics end on 2026-04-18.  2026-04-19..2026-04-30 is a
# forward holdout from the perspective of those latest round artifacts.
DEV_START = date(2025, 12, 1)
DEV_END = date(2026, 4, 18)
HOLDOUT_START = date(2026, 4, 19)
HOLDOUT_END = date(2026, 4, 30)
FULL_START = DEV_START
FULL_END = HOLDOUT_END


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start: date
    end: date


DEV_WINDOW = WindowSpec("development", DEV_START, DEV_END)
HOLDOUT_WINDOW = WindowSpec("forward_holdout", HOLDOUT_START, HOLDOUT_END)
FULL_WINDOW = WindowSpec("full_refreshed", FULL_START, FULL_END)


@dataclass(frozen=True)
class PolicyDelta:
    name: str
    phase: int
    thesis: str
    strategy_mutations: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_scales: dict[str, float] = field(default_factory=dict)
    portfolio_overrides: dict[str, Any] = field(default_factory=dict)
    filter_rules: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PortfolioPolicy:
    name: str
    strategy_mutations: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_scales: dict[str, float] = field(
        default_factory=lambda: {sid: 1.0 for sid in STRATEGIES}
    )
    portfolio_overrides: dict[str, Any] = field(default_factory=dict)
    filter_rules: tuple[dict[str, Any], ...] = ()
    accepted_deltas: tuple[str, ...] = ()


@dataclass
class ReplayMetrics:
    trades: int = 0
    filtered: int = 0
    blocked: int = 0
    net_pnl: float = 0.0
    net_return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    total_r: float = 0.0
    per_strategy_trades: dict[str, int] = field(default_factory=dict)
    block_reasons: dict[str, int] = field(default_factory=dict)
    filter_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "filtered": self.filtered,
            "blocked": self.blocked,
            "net_pnl": self.net_pnl,
            "net_return_pct": self.net_return_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_r": self.total_r,
            "per_strategy_trades": dict(self.per_strategy_trades),
            "block_reasons": dict(self.block_reasons),
            "filter_reasons": dict(self.filter_reasons),
        }


@dataclass
class ReplayEvaluation:
    policy_name: str
    score: float
    rejected: bool
    reject_reason: str
    development: ReplayMetrics
    holdout: ReplayMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "score": self.score,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "development": self.development.to_dict(),
            "holdout": self.holdout.to_dict(),
        }


@dataclass
class TimelineEntry:
    strategy_id: str
    trade: Trade
    risk_R: float
    pnl_scale: float


@dataclass
class OpenReplayRisk:
    entry: TimelineEntry
    multiplier: float

    @property
    def risk_R(self) -> float:
        return self.entry.risk_R * self.multiplier


def _bt_config(window: WindowSpec) -> BacktestConfig:
    return build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=list(SYMBOLS),
        start_date=window.start,
        end_date=window.end,
        initial_equity=INITIAL_EQUITY,
    )


def _load_strategy_config(strategy_id: str) -> Any:
    path = ROOT / "output" / strategy_id / "round_3" / "optimized_config.json"
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)["strategy"]
    if strategy_id == "momentum":
        return MomentumConfig.from_dict(payload)
    if strategy_id == "trend":
        return TrendConfig.from_dict(payload)
    if strategy_id == "breakout":
        return BreakoutConfig.from_dict(payload)
    raise ValueError(f"Unknown strategy: {strategy_id}")


def _load_base_configs() -> dict[str, Any]:
    return {sid: _load_strategy_config(sid) for sid in STRATEGIES}


def _scale(value: float, multiplier: float, cap: float) -> float:
    return round(min(value * multiplier, cap), 8)


def _risk_scale_mutations(strategy_id: str, cfg: Any, multiplier: float) -> dict[str, Any]:
    if abs(multiplier - 1.0) < 1e-9:
        return {}

    if strategy_id == "momentum":
        risk = cfg.risk
        return {
            "risk.risk_pct_a": _scale(risk.risk_pct_a, multiplier, 0.030),
            "risk.risk_pct_b": _scale(risk.risk_pct_b, multiplier, 0.025),
            "risk.max_gross_risk": _scale(risk.max_gross_risk, max(1.0, multiplier), 0.060),
            "risk.max_correlated_risk": _scale(risk.max_correlated_risk, max(1.0, multiplier), 0.040),
        }

    if strategy_id == "trend":
        risk = cfg.risk
        limits = cfg.limits
        return {
            "risk.risk_pct_a": _scale(risk.risk_pct_a, multiplier, 0.028),
            "risk.risk_pct_b": _scale(risk.risk_pct_b, multiplier, 0.028),
            "risk.max_risk_pct": _scale(risk.max_risk_pct, max(1.0, multiplier), 0.030),
            "limits.max_correlated_risk_pct": _scale(
                limits.max_correlated_risk_pct, max(1.0, multiplier), 0.070
            ),
        }

    if strategy_id == "breakout":
        risk = cfg.risk
        limits = cfg.limits
        return {
            "risk.risk_pct_a_plus": _scale(risk.risk_pct_a_plus, multiplier, 0.018),
            "risk.risk_pct_a": _scale(risk.risk_pct_a, multiplier, 0.018),
            "risk.risk_pct_b": _scale(risk.risk_pct_b, multiplier, 0.032),
            "risk.max_risk_pct": _scale(risk.max_risk_pct, max(1.0, multiplier), 0.033),
            "limits.max_correlated_risk_pct": _scale(
                limits.max_correlated_risk_pct, max(1.0, multiplier), 0.030
            ),
        }

    raise ValueError(f"Unknown strategy: {strategy_id}")


def _apply_policy_configs(policy: PortfolioPolicy, base_configs: dict[str, Any]) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for sid, cfg in base_configs.items():
        risk_mut = _risk_scale_mutations(sid, cfg, policy.risk_scales.get(sid, 1.0))
        explicit = policy.strategy_mutations.get(sid, {})
        mutations = merge_mutations(risk_mut, explicit)
        new_cfg = apply_mutations(cfg, mutations) if mutations else apply_mutations(cfg, {})
        new_cfg.symbols = list(SYMBOLS)
        configs[sid] = new_cfg
    return configs


def _base_allocations(policy: PortfolioPolicy) -> tuple[StrategyAllocation, ...]:
    # Higher priority strategies keep access to directional headroom when a
    # headroom candidate is active: trend has the broadest sample, breakout the
    # highest expectancy but smaller sample, momentum the newest M15 sample.
    priority = {"trend": 0, "breakout": 1, "momentum": 2}
    return tuple(
        StrategyAllocation(
            strategy_id=sid,
            base_risk_pct=0.01 * policy.risk_scales.get(sid, 1.0),
            max_concurrent=5 if sid == "trend" else 3,
            daily_stop_R=4.0 if sid == "trend" else 3.0,
            priority=priority[sid],
        )
        for sid in STRATEGIES
    )


def _portfolio_config(policy: PortfolioPolicy) -> PortfolioConfig:
    kwargs = {
        "initial_equity": INITIAL_EQUITY,
        "strategies": _base_allocations(policy),
        "heat_cap_R": 6.0,
        "directional_cap_R": 4.0,
        "portfolio_daily_stop_R": 5.0,
        "max_total_positions": 9,
        "dd_tiers": (
            (0.08, 1.00),
            (0.12, 0.50),
            (0.15, 0.25),
            (1.00, 0.00),
        ),
        "symbol_collision": "cap",
        "symbol_exposure_cap_R": 3.0,
        "priority_headroom_R": 0.0,
        "priority_reserve_threshold": 0,
    }
    kwargs.update(policy.portfolio_overrides)
    if "dd_tiers" in kwargs:
        kwargs["dd_tiers"] = tuple(tuple(x) for x in kwargs["dd_tiers"])
    return PortfolioConfig(**kwargs)


def _merge_policy(base: PortfolioPolicy, delta: PolicyDelta) -> PortfolioPolicy:
    strategy_mutations = {
        sid: dict(muts) for sid, muts in base.strategy_mutations.items()
    }
    for sid, muts in delta.strategy_mutations.items():
        strategy_mutations[sid] = merge_mutations(strategy_mutations.get(sid, {}), muts)

    risk_scales = dict(base.risk_scales)
    for sid, mult in delta.risk_scales.items():
        risk_scales[sid] = round(risk_scales.get(sid, 1.0) * mult, 8)

    overrides = dict(base.portfolio_overrides)
    overrides.update(delta.portfolio_overrides)

    return PortfolioPolicy(
        name=f"{base.name}+{delta.name}" if base.name != "baseline" else delta.name,
        strategy_mutations=strategy_mutations,
        risk_scales=risk_scales,
        portfolio_overrides=overrides,
        filter_rules=(*base.filter_rules, *delta.filter_rules),
        accepted_deltas=(*base.accepted_deltas, delta.name),
    )


def _rule_direction(value: str | Side | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Side):
        return value.value
    return str(value).upper()


def _filter_reason(entry: TimelineEntry, rules: tuple[dict[str, Any], ...]) -> str | None:
    trade = entry.trade
    for rule in rules:
        sid = rule.get("strategy_id")
        if sid not in (None, entry.strategy_id):
            continue

        kind = rule["kind"]
        if kind == "symbol_direction":
            if trade.symbol == rule["symbol"] and trade.direction.value == _rule_direction(rule["direction"]):
                return rule.get("reason", f"{entry.strategy_id}_{trade.symbol}_{trade.direction.value}")

        elif kind == "entry_hour":
            hours = set(rule.get("hours", []))
            if trade.entry_time.hour in hours:
                return rule.get("reason", f"{entry.strategy_id}_hour_{trade.entry_time.hour}")

        elif kind == "confluence_lte":
            confluence_count = len(trade.confluences_used or [])
            if confluence_count <= int(rule["threshold"]):
                return rule.get("reason", f"{entry.strategy_id}_confluence_lte_{rule['threshold']}")

        elif kind == "confirmation_symbol":
            if trade.symbol == rule["symbol"] and trade.confirmation_type == rule["confirmation"]:
                return rule.get(
                    "reason",
                    f"{entry.strategy_id}_{trade.symbol}_{trade.confirmation_type}",
                )

    return None


def _build_timeline(
    trade_lists: dict[str, list[Trade]],
    policy: PortfolioPolicy,
) -> list[TimelineEntry]:
    entries: list[TimelineEntry] = []
    for sid, trades in trade_lists.items():
        scale = policy.risk_scales.get(sid, 1.0)
        for trade in trades:
            entries.append(TimelineEntry(
                strategy_id=sid,
                trade=trade,
                risk_R=scale,
                pnl_scale=scale,
            ))
    entries.sort(key=lambda item: item.trade.entry_time)
    return entries


def _dd_multiplier(dd: float, tiers: tuple[tuple[float, float], ...]) -> float:
    multiplier = 1.0
    for threshold, mult in tiers:
        if dd >= threshold:
            multiplier = mult
        else:
            break
    return multiplier


def _close_before(
    open_risks: list[OpenReplayRisk],
    before: datetime,
    equity_state: dict[str, float],
    daily_pnl: dict[str, float],
    closed_pnls: list[float],
    closed_rs: list[float],
) -> None:
    remaining: list[OpenReplayRisk] = []
    for risk in open_risks:
        trade = risk.entry.trade
        if trade.exit_time is None or trade.exit_time < before:
            r_mult = trade.r_multiple
            if r_mult is None:
                r_mult = trade.realized_r_multiple
            # A few historical Trade objects carry pathological realized-R
            # values from partial-fill accounting.  Portfolio stops should be
            # driven by bounded geometric R, not those artifacts.
            if r_mult is None or abs(r_mult) > 20.0:
                r_mult = 0.0
            pnl = trade.net_pnl * risk.entry.pnl_scale * risk.multiplier
            pnl_r = r_mult * risk.entry.risk_R * risk.multiplier
            equity_state["equity"] += pnl
            equity_state["peak"] = max(equity_state["peak"], equity_state["equity"])
            equity_state["max_dd"] = max(
                equity_state["max_dd"],
                (equity_state["peak"] - equity_state["equity"]) / equity_state["peak"],
            )
            daily_pnl["portfolio"] += pnl_r
            daily_pnl[risk.entry.strategy_id] += pnl_r
            closed_pnls.append(pnl)
            closed_rs.append(pnl_r)
        else:
            remaining.append(risk)
    open_risks[:] = remaining


def _replay_portfolio(
    policy: PortfolioPolicy,
    trade_lists: dict[str, list[Trade]],
) -> ReplayMetrics:
    config = _portfolio_config(policy)
    timeline = _build_timeline(trade_lists, policy)
    open_risks: list[OpenReplayRisk] = []
    equity_state = {
        "equity": INITIAL_EQUITY,
        "peak": INITIAL_EQUITY,
        "max_dd": 0.0,
    }
    current_day: date | None = None
    daily_pnl: defaultdict[str, float] = defaultdict(float)
    accepted_by_strategy: Counter[str] = Counter()
    block_reasons: Counter[str] = Counter()
    filter_reasons: Counter[str] = Counter()
    closed_pnls: list[float] = []
    closed_rs: list[float] = []
    filtered = 0
    blocked = 0

    for entry in timeline:
        trade = entry.trade
        if current_day != trade.entry_time.date():
            current_day = trade.entry_time.date()
            daily_pnl.clear()

        _close_before(open_risks, trade.entry_time, equity_state, daily_pnl, closed_pnls, closed_rs)

        reason = _filter_reason(entry, policy.filter_rules)
        if reason:
            filtered += 1
            filter_reasons[reason] += 1
            continue

        alloc = config.get_strategy(entry.strategy_id)
        if alloc is None or not alloc.enabled:
            blocked += 1
            block_reasons["strategy_disabled"] += 1
            continue

        total_positions = len(open_risks)
        strat_positions = sum(1 for r in open_risks if r.entry.strategy_id == entry.strategy_id)
        heat = sum(r.risk_R for r in open_risks)
        dir_heat = sum(r.risk_R for r in open_risks if r.entry.trade.direction == trade.direction)
        symbol_heat = sum(
            r.risk_R for r in open_risks
            if r.entry.trade.symbol == trade.symbol and r.entry.trade.direction == trade.direction
        )

        reason = ""
        if total_positions >= config.max_total_positions:
            reason = "max_total_positions"
        elif strat_positions >= alloc.max_concurrent:
            reason = f"{entry.strategy_id}_max_concurrent"
        elif heat + entry.risk_R > config.heat_cap_R:
            reason = "heat_cap_R"
        elif dir_heat + entry.risk_R > config.directional_cap_R:
            reason = "directional_cap_R"
        elif config.symbol_collision == "cap" and symbol_heat + entry.risk_R > config.symbol_exposure_cap_R:
            reason = "symbol_exposure_cap_R"
        elif daily_pnl["portfolio"] <= -config.portfolio_daily_stop_R:
            reason = "portfolio_daily_stop_R"
        elif daily_pnl[entry.strategy_id] <= -alloc.daily_stop_R:
            reason = f"{entry.strategy_id}_daily_stop_R"

        dd = (equity_state["peak"] - equity_state["equity"]) / equity_state["peak"]
        multiplier = _dd_multiplier(dd, config.dd_tiers)
        if not reason and multiplier <= 0.0:
            reason = "drawdown_tier_block"

        if reason:
            blocked += 1
            block_reasons[reason] += 1
            continue

        open_risks.append(OpenReplayRisk(entry=entry, multiplier=multiplier))
        accepted_by_strategy[entry.strategy_id] += 1

    _close_before(
        open_risks,
        datetime.max.replace(tzinfo=timezone.utc),
        equity_state,
        daily_pnl,
        closed_pnls,
        closed_rs,
    )

    wins = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    trades = len(closed_pnls)
    return ReplayMetrics(
        trades=trades,
        filtered=filtered,
        blocked=blocked,
        net_pnl=sum(closed_pnls),
        net_return_pct=(sum(closed_pnls) / INITIAL_EQUITY) * 100.0,
        win_rate=(len(wins) / trades * 100.0) if trades else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss else (math.inf if gross_win else 0.0),
        max_drawdown_pct=equity_state["max_dd"] * 100.0,
        total_r=sum(closed_rs),
        per_strategy_trades=dict(accepted_by_strategy),
        block_reasons=dict(block_reasons),
        filter_reasons=dict(filter_reasons),
    )


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def _score_metrics(dev: ReplayMetrics, holdout: ReplayMetrics) -> tuple[float, bool, str]:
    hard_failures = []
    if dev.trades < 76:
        hard_failures.append(f"development trades too low ({dev.trades} < 76)")
    if holdout.trades < 8:
        hard_failures.append(f"holdout trades too low ({holdout.trades} < 8)")
    if dev.profit_factor < 1.75:
        hard_failures.append(f"development PF too low ({dev.profit_factor:.2f} < 1.75)")
    if dev.max_drawdown_pct > 8.5:
        hard_failures.append(f"development DD too high ({dev.max_drawdown_pct:.2f}% > 8.5%)")
    if holdout.max_drawdown_pct > 7.5:
        hard_failures.append(f"holdout DD too high ({holdout.max_drawdown_pct:.2f}% > 7.5%)")
    if holdout.net_return_pct < -3.0:
        hard_failures.append(f"holdout return too weak ({holdout.net_return_pct:.2f}% < -3.0%)")

    ret_score = _clip(dev.net_return_pct / 100.0)
    freq_score = _clip(dev.trades / 95.0)
    pf_score = _clip((dev.profit_factor - 1.0) / 5.0)
    dd_score = _clip(1.0 - dev.max_drawdown_pct / 8.5)
    holdout_return_score = _clip((holdout.net_return_pct + 2.0) / 6.0)
    holdout_pf_score = _clip((holdout.profit_factor - 0.6) / 2.5)
    holdout_score = 0.65 * holdout_return_score + 0.35 * holdout_pf_score

    score = (
        0.34 * ret_score
        + 0.20 * freq_score
        + 0.18 * pf_score
        + 0.12 * dd_score
        + 0.16 * holdout_score
    )
    return score, bool(hard_failures), "; ".join(hard_failures)


def _evaluate_policy(
    policy: PortfolioPolicy,
    trade_windows: dict[str, dict[str, list[Trade]]],
) -> ReplayEvaluation:
    dev = _replay_portfolio(policy, trade_windows[DEV_WINDOW.name])
    holdout = _replay_portfolio(policy, trade_windows[HOLDOUT_WINDOW.name])
    score, rejected, reason = _score_metrics(dev, holdout)
    return ReplayEvaluation(
        policy_name=policy.name,
        score=score,
        rejected=rejected,
        reject_reason=reason,
        development=dev,
        holdout=holdout,
    )


def _run_individual_window(
    window: WindowSpec,
    configs: dict[str, Any],
) -> dict[str, list[Trade]]:
    result: dict[str, list[Trade]] = {}
    for sid in STRATEGIES:
        cfg = apply_mutations(configs[sid], {})
        cfg.symbols = list(SYMBOLS)
        bt_result = run_individual(
            strategy_config=cfg,
            backtest_config=_bt_config(window),
            data_dir=DATA_DIR,
            strategy_type=sid,
        )
        result[sid] = bt_result.trades
        print(f"  {window.name}: {sid} individual trades={len(bt_result.trades)}")
    return result


def _actual_portfolio_run(
    policy: PortfolioPolicy,
    window: WindowSpec,
    base_configs: dict[str, Any],
) -> tuple[dict[str, float], Any]:
    configs = _apply_policy_configs(policy, base_configs)
    result = run_portfolio_backtest(
        portfolio_config=_portfolio_config(policy),
        strategy_configs=configs,
        backtest_config=_bt_config(window),
        data_dir=DATA_DIR,
    )
    return metrics_to_dict(result.metrics), result


def _candidate_phases() -> dict[int, list[PolicyDelta]]:
    return {
        1: [
            PolicyDelta(
                name="breakout_block_sol_longs",
                phase=1,
                thesis="Breakout diagnostics show SOL longs 0/2 while SOL short is positive; block only that weak side.",
                strategy_mutations={"breakout": {"symbol_filter.sol_direction": "short_only"}},
                filter_rules=(
                    {
                        "kind": "symbol_direction",
                        "strategy_id": "breakout",
                        "symbol": "SOL",
                        "direction": "LONG",
                        "reason": "breakout_SOL_LONG_filter",
                    },
                ),
            ),
            PolicyDelta(
                name="breakout_block_eth_shorts",
                phase=1,
                thesis="Breakout ETH edge is concentrated in longs; ETH shorts are the recurring weak side.",
                strategy_mutations={
                    "breakout": {
                        "symbol_filter.eth_direction": "long_only",
                        "symbol_filter.eth_relaxed_body_direction": "long_only",
                    }
                },
                filter_rules=(
                    {
                        "kind": "symbol_direction",
                        "strategy_id": "breakout",
                        "symbol": "ETH",
                        "direction": "SHORT",
                        "reason": "breakout_ETH_SHORT_filter",
                    },
                ),
            ),
            PolicyDelta(
                name="breakout_block_sol_longs_eth_shorts",
                phase=1,
                thesis="Combine only the two breakout side filters with diagnostic support.",
                strategy_mutations={
                    "breakout": {
                        "symbol_filter.sol_direction": "short_only",
                        "symbol_filter.eth_direction": "long_only",
                        "symbol_filter.eth_relaxed_body_direction": "long_only",
                    }
                },
                filter_rules=(
                    {
                        "kind": "symbol_direction",
                        "strategy_id": "breakout",
                        "symbol": "SOL",
                        "direction": "LONG",
                        "reason": "breakout_SOL_LONG_filter",
                    },
                    {
                        "kind": "symbol_direction",
                        "strategy_id": "breakout",
                        "symbol": "ETH",
                        "direction": "SHORT",
                        "reason": "breakout_ETH_SHORT_filter",
                    },
                ),
            ),
            PolicyDelta(
                name="momentum_require_one_confluence",
                phase=1,
                thesis="Momentum zero-confluence trades are slightly negative; require one confluence for B setups.",
                strategy_mutations={"momentum": {"setup.min_confluences_b": 1}},
                filter_rules=(
                    {
                        "kind": "confluence_lte",
                        "strategy_id": "momentum",
                        "threshold": 0,
                        "reason": "momentum_zero_confluence_filter",
                    },
                ),
            ),
            PolicyDelta(
                name="trend_raise_weighted_b_score",
                phase=1,
                thesis="Nudge trend B setup quality higher without disabling a whole symbol or direction.",
                strategy_mutations={"trend": {"setup.min_setup_score_b": 1.45}},
            ),
        ],
        2: [
            PolicyDelta(
                name="trend_reentry_more_patient",
                phase=2,
                thesis="Trend has the broadest sample; allow one more controlled reentry window to lift frequency.",
                strategy_mutations={
                    "trend": {
                        "reentry.max_reentries": 2,
                        "reentry.max_wait_bars": 8,
                        "reentry.risk_scale": 0.65,
                    }
                },
            ),
            PolicyDelta(
                name="momentum_reentry_faster",
                phase=2,
                thesis="Momentum holds short but recovers quickly; reduce cooldown without increasing max reentries.",
                strategy_mutations={"momentum": {"reentry.cooldown_bars": 2}},
            ),
            PolicyDelta(
                name="breakout_relaxed_body_cautious_expand",
                phase=2,
                thesis="Probe more breakout frequency with lower relaxed-body confluence but smaller relaxed risk.",
                strategy_mutations={
                    "breakout": {
                        "setup.relaxed_body_min_confluences": 4,
                        "setup.relaxed_body_min_room_r": 1.6,
                        "setup.relaxed_body_risk_scale": 0.4,
                    }
                },
            ),
        ],
        3: [
            PolicyDelta(
                name="risk_all_115",
                phase=3,
                thesis="Uniform modest risk lift; tests whether the low DD headroom is real.",
                risk_scales={"momentum": 1.15, "trend": 1.15, "breakout": 1.15},
            ),
            PolicyDelta(
                name="risk_trend_core_130",
                phase=3,
                thesis="Overweight the most statistically supported strategy while keeping smaller-sample strategies near baseline.",
                risk_scales={"momentum": 0.95, "trend": 1.30, "breakout": 1.05},
            ),
            PolicyDelta(
                name="risk_trend_breakout_lean",
                phase=3,
                thesis="Lean into trend sample depth and breakout expectancy, with momentum slightly reduced.",
                risk_scales={"momentum": 0.90, "trend": 1.25, "breakout": 1.15},
            ),
            PolicyDelta(
                name="risk_frequency_lean",
                phase=3,
                thesis="Small momentum and trend lift for trade count, while leaving breakout concentration unchanged.",
                risk_scales={"momentum": 1.10, "trend": 1.20, "breakout": 1.00},
            ),
            PolicyDelta(
                name="risk_breakout_alpha_probe",
                phase=3,
                thesis="Probe breakout overweight, but only the holdout/scoring can approve it due low sample concentration.",
                risk_scales={"momentum": 0.85, "trend": 1.15, "breakout": 1.25},
            ),
        ],
        4: [
            PolicyDelta(
                name="caps_unlock_one_directional_unit",
                phase=4,
                thesis="The only saved portfolio block was directional_cap_R; loosen one unit to recover frequency.",
                portfolio_overrides={
                    "directional_cap_R": 5.0,
                    "heat_cap_R": 7.5,
                    "symbol_exposure_cap_R": 3.5,
                    "max_total_positions": 10,
                },
            ),
            PolicyDelta(
                name="caps_aggressive_with_dd_guard",
                phase=4,
                thesis="Allow more concurrent alpha, but cut size dynamically if drawdown starts compounding.",
                portfolio_overrides={
                    "directional_cap_R": 5.5,
                    "heat_cap_R": 8.0,
                    "symbol_exposure_cap_R": 4.0,
                    "max_total_positions": 10,
                    "dd_tiers": (
                        (0.05, 0.75),
                        (0.08, 0.50),
                        (0.11, 0.25),
                        (0.14, 0.00),
                    ),
                },
            ),
            PolicyDelta(
                name="trend_priority_headroom",
                phase=4,
                thesis="Reserve directional capacity for trend when lower-priority strategies crowd one side.",
                portfolio_overrides={
                    "priority_headroom_R": 1.0,
                    "priority_reserve_threshold": 1,
                    "directional_cap_R": 5.0,
                    "heat_cap_R": 7.5,
                },
            ),
        ],
        5: [
            PolicyDelta(
                name="postcap_risk_all_110",
                phase=5,
                thesis="After cap unlock, test a smaller uniform risk lift than the pre-cap 1.15 probe.",
                risk_scales={"momentum": 1.10, "trend": 1.10, "breakout": 1.10},
            ),
            PolicyDelta(
                name="postcap_risk_trend_core_120",
                phase=5,
                thesis="After cap unlock, overweight trend modestly while keeping lower-sample sleeves restrained.",
                risk_scales={"momentum": 0.95, "trend": 1.20, "breakout": 1.05},
            ),
            PolicyDelta(
                name="postcap_risk_trend_breakout_lean",
                phase=5,
                thesis="After cap unlock, lean into trend sample depth and breakout expectancy with moderate sizing.",
                risk_scales={"momentum": 0.90, "trend": 1.18, "breakout": 1.12},
            ),
            PolicyDelta(
                name="postcap_risk_frequency_lean",
                phase=5,
                thesis="After cap unlock, lift the higher-frequency sleeves without increasing breakout concentration.",
                risk_scales={"momentum": 1.08, "trend": 1.15, "breakout": 1.00},
            ),
        ],
    }


def _run_phase_auto(
    trade_windows: dict[str, dict[str, list[Trade]]],
) -> tuple[PortfolioPolicy, list[dict[str, Any]]]:
    current = PortfolioPolicy(name="baseline")
    phase_log: list[dict[str, Any]] = []
    current_eval = _evaluate_policy(current, trade_windows)
    min_delta = 0.005

    print(
        f"  baseline replay score={current_eval.score:.4f} "
        f"dev_ret={current_eval.development.net_return_pct:.2f}% "
        f"holdout_ret={current_eval.holdout.net_return_pct:.2f}%"
    )

    for phase, candidates in _candidate_phases().items():
        phase_results = []
        best_policy = current
        best_eval = current_eval
        for delta in candidates:
            policy = _merge_policy(current, delta)
            evaluation = _evaluate_policy(policy, trade_windows)
            phase_results.append({
                "delta": delta.name,
                "thesis": delta.thesis,
                "policy": policy.name,
                "evaluation": evaluation.to_dict(),
            })
            status = "REJECT" if evaluation.rejected else "ok"
            print(
                f"  phase {phase} {delta.name}: {status} score={evaluation.score:.4f} "
                f"dev_ret={evaluation.development.net_return_pct:.2f}% "
                f"holdout_ret={evaluation.holdout.net_return_pct:.2f}%"
            )
            if not evaluation.rejected and evaluation.score > best_eval.score + min_delta:
                best_policy = policy
                best_eval = evaluation

        accepted = best_policy is not current
        if accepted:
            current = best_policy
            current_eval = best_eval

        phase_log.append({
            "phase": phase,
            "accepted": accepted,
            "accepted_policy": current.name,
            "accepted_deltas": list(current.accepted_deltas),
            "current_evaluation": current_eval.to_dict(),
            "candidates": phase_results,
        })
        print(
            f"  phase {phase} {'accepted' if accepted else 'kept baseline/current'}: "
            f"{current.name} score={current_eval.score:.4f}"
        )

    return current, phase_log


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _actual_summary(metrics: dict[str, float]) -> dict[str, float]:
    keys = [
        "total_trades",
        "win_rate",
        "profit_factor",
        "net_return_pct",
        "max_drawdown_pct",
        "sharpe_ratio",
        "calmar_ratio",
    ]
    return {k: float(metrics.get(k, 0.0)) for k in keys}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _save_recommended_configs(policy: PortfolioPolicy, base_configs: dict[str, Any]) -> None:
    RECOMMENDED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    configs = _apply_policy_configs(policy, base_configs)
    for sid, cfg in configs.items():
        _write_json(RECOMMENDED_CONFIG_DIR / f"{sid}.json", {"strategy": cfg.to_dict()})
    _write_json(ROUND_DIR / "recommended_portfolio_config.json", _portfolio_config(policy).to_dict())


def _diagnostic_trades(trades: list[Trade]) -> list[Trade]:
    """Return copies that force diagnostics to use geometric R.

    Some partial-fill paths store extreme realized_r_multiple values even when
    dollar P&L is normal.  The diagnostics renderer prefers realized/economic R,
    so for this portfolio artifact we clear realized_r_multiple and leave net P&L
    untouched.
    """
    return [replace(trade, realized_r_multiple=None) for trade in trades]


def _build_report(
    selected: PortfolioPolicy,
    phase_log: list[dict[str, Any]],
    actual_baseline_full: dict[str, float],
    actual_selected_full: dict[str, float],
    actual_baseline_holdout: dict[str, float],
    actual_selected_holdout: dict[str, float],
) -> str:
    lines = []
    lines.append("PORTFOLIO PHASE-AUTO ROUND 1")
    lines.append("=" * 80)
    lines.append(f"Initial equity assumption: ${INITIAL_EQUITY:,.0f}")
    lines.append(f"Development: {DEV_START} to {DEV_END}")
    lines.append(f"Forward holdout: {HOLDOUT_START} to {HOLDOUT_END}")
    lines.append("Risk stance: aggressive-leaning, hard-gated at roughly 8.5% replay DD.")
    lines.append("")

    lines.append("Selected policy:")
    lines.append(f"  name: {selected.name}")
    lines.append(f"  accepted deltas: {', '.join(selected.accepted_deltas) or 'none'}")
    lines.append(f"  risk scales: {selected.risk_scales}")
    lines.append(f"  portfolio overrides: {selected.portfolio_overrides or '{}'}")
    if selected.strategy_mutations:
        lines.append("  strategy mutations:")
        for sid, muts in selected.strategy_mutations.items():
            if muts:
                lines.append(f"    {sid}: {muts}")
    lines.append("")

    lines.append("Actual portfolio validation:")
    lines.append("  Full refreshed window")
    lines.append(
        "    baseline: "
        f"trades={actual_baseline_full['total_trades']:.0f}, "
        f"return={_format_pct(actual_baseline_full['net_return_pct'])}, "
        f"PF={actual_baseline_full['profit_factor']:.2f}, "
        f"DD={_format_pct(actual_baseline_full['max_drawdown_pct'])}"
    )
    lines.append(
        "    selected: "
        f"trades={actual_selected_full['total_trades']:.0f}, "
        f"return={_format_pct(actual_selected_full['net_return_pct'])}, "
        f"PF={actual_selected_full['profit_factor']:.2f}, "
        f"DD={_format_pct(actual_selected_full['max_drawdown_pct'])}"
    )
    lines.append("  Forward holdout")
    lines.append(
        "    baseline: "
        f"trades={actual_baseline_holdout['total_trades']:.0f}, "
        f"return={_format_pct(actual_baseline_holdout['net_return_pct'])}, "
        f"PF={actual_baseline_holdout['profit_factor']:.2f}, "
        f"DD={_format_pct(actual_baseline_holdout['max_drawdown_pct'])}"
    )
    lines.append(
        "    selected: "
        f"trades={actual_selected_holdout['total_trades']:.0f}, "
        f"return={_format_pct(actual_selected_holdout['net_return_pct'])}, "
        f"PF={actual_selected_holdout['profit_factor']:.2f}, "
        f"DD={_format_pct(actual_selected_holdout['max_drawdown_pct'])}"
    )
    lines.append("")

    lines.append("Phase decisions:")
    for phase in phase_log:
        evaluation = phase["current_evaluation"]
        dev = evaluation["development"]
        holdout = evaluation["holdout"]
        lines.append(
            f"  Phase {phase['phase']}: "
            f"{'accepted' if phase['accepted'] else 'no change'} -> "
            f"{phase['accepted_policy']} "
            f"(score={evaluation['score']:.4f}, "
            f"dev_ret={dev['net_return_pct']:.2f}%, "
            f"holdout_ret={holdout['net_return_pct']:.2f}%)"
        )
    lines.append("")

    lines.append("Interpretation:")
    lines.append(
        "  The saved diagnostics show real portfolio-level edge, but the forward "
        "holdout after 2026-04-18 is weak. The round therefore rewards return and "
        "frequency only when the holdout degradation does not worsen and drawdown "
        "stays controlled."
    )
    lines.append(
        "  Treat signal-discrimination deltas as candidates, not final truth, when "
        "they rely on fewer than roughly 20 supporting trades. The risk/cap policy "
        "is more robust because it preserves the signal set and changes allocation."
    )
    return "\n".join(lines)


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    print("Portfolio phase-auto round 1")
    print(f"Output: {ROUND_DIR}")
    print("Loading latest round_3 strategy configs...")
    base_configs = _load_base_configs()

    print("Running individual strategy trade harvest for replay windows...")
    trade_windows = {
        DEV_WINDOW.name: _run_individual_window(DEV_WINDOW, base_configs),
        HOLDOUT_WINDOW.name: _run_individual_window(HOLDOUT_WINDOW, base_configs),
    }

    print("Running phased replay optimizer...")
    selected_policy, phase_log = _run_phase_auto(trade_windows)

    print("Running actual portfolio validation for baseline and selected policy...")
    baseline_policy = PortfolioPolicy(name="baseline")
    baseline_full_metrics, baseline_full_result = _actual_portfolio_run(
        baseline_policy, FULL_WINDOW, base_configs
    )
    selected_full_metrics, selected_full_result = _actual_portfolio_run(
        selected_policy, FULL_WINDOW, base_configs
    )
    baseline_holdout_metrics, _ = _actual_portfolio_run(
        baseline_policy, HOLDOUT_WINDOW, base_configs
    )
    selected_holdout_metrics, _ = _actual_portfolio_run(
        selected_policy, HOLDOUT_WINDOW, base_configs
    )

    print("Saving artifacts...")
    _save_recommended_configs(selected_policy, base_configs)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "initial_equity": INITIAL_EQUITY,
        "windows": {
            "development": {"start": str(DEV_START), "end": str(DEV_END)},
            "forward_holdout": {"start": str(HOLDOUT_START), "end": str(HOLDOUT_END)},
            "full_refreshed": {"start": str(FULL_START), "end": str(FULL_END)},
        },
        "selected_policy": {
            "name": selected_policy.name,
            "accepted_deltas": list(selected_policy.accepted_deltas),
            "strategy_mutations": selected_policy.strategy_mutations,
            "risk_scales": selected_policy.risk_scales,
            "portfolio_overrides": selected_policy.portfolio_overrides,
            "filter_rules": list(selected_policy.filter_rules),
        },
        "phase_log": phase_log,
        "actual_validation": {
            "baseline_full": _actual_summary(baseline_full_metrics),
            "selected_full": _actual_summary(selected_full_metrics),
            "baseline_holdout": _actual_summary(baseline_holdout_metrics),
            "selected_holdout": _actual_summary(selected_holdout_metrics),
        },
    }
    _write_json(ROUND_DIR / "phase_auto_results.json", payload)

    report = _build_report(
        selected_policy,
        phase_log,
        _actual_summary(baseline_full_metrics),
        _actual_summary(selected_full_metrics),
        _actual_summary(baseline_holdout_metrics),
        _actual_summary(selected_holdout_metrics),
    )
    (ROUND_DIR / "phase_auto_report.txt").write_text(report, encoding="utf-8")

    diagnostics = (
        "# R-multiple sections use geometric R; dollar P&L is from the actual "
        "selected portfolio backtest.\n\n"
    )
    diagnostics += generate_diagnostics(
        _diagnostic_trades(selected_full_result.all_trades),
        initial_equity=INITIAL_EQUITY,
    )
    (ROUND_DIR / "recommended_portfolio_diagnostics.txt").write_text(
        diagnostics,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
