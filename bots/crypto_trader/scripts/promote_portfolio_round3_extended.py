"""Promote the selected portfolio OOS repair candidate as round_3.

The promotion is intentionally snapshot-based.  It captures the pre-adoption
round_3 strategy configs and base portfolio config once, then uses those
snapshots for all reruns so the 1.4 risk scale cannot be applied twice.

Legacy guard: this original promotion script also mutates global strategy
round_3 lineage artifacts.  That is no longer the recommended adoption model
for portfolio-context candidates.  It now refuses to run unless explicitly
called with ``--allow-strategy-lineage-mutation``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import structlog

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_trader.backtest.analysis import (  # noqa: E402
    export_equity_curve,
    export_trade_journal,
    generate_report,
)
from crypto_trader.backtest.diagnostics import generate_diagnostics  # noqa: E402
from crypto_trader.backtest.metrics import (  # noqa: E402
    _trade_net_pnl,
    _trade_reporting_r,
    metrics_to_dict,
)
from crypto_trader.backtest.profiles import (  # noqa: E402
    LIVE_PARITY_PROFILE,
    build_backtest_config_from_profile,
    profile_hash,
)
from crypto_trader.backtest.runner import run  # noqa: E402
from crypto_trader.data.store import ParquetStore  # noqa: E402
from crypto_trader.optimize.config_mutator import apply_mutations, merge_mutations  # noqa: E402
from crypto_trader.optimize.contracts import (  # noqa: E402
    data_snapshot_fingerprint,
    portfolio_config_hash,
    required_timeframes,
    stable_hash,
    strategy_config_hash,
)
from crypto_trader.optimize.phase_state import _atomic_write_json  # noqa: E402
from crypto_trader.optimize.portfolio_round2_phased import (  # noqa: E402
    DEV_END,
    DEV_START,
    FULL_END,
    FULL_START,
    HARD_REJECTS,
    HOLDOUT_END,
    HOLDOUT_START,
    SCORING_WEIGHTS,
    STRATEGIES,
    SYMBOLS,
    _augment_metrics,
    _bt_config,
    _portfolio_config_from_mutations,
    _risk_scale_mutations,
    _score_metrics,
    _split_policy_mutations,
)
from crypto_trader.portfolio.backtest_runner import (  # noqa: E402
    PortfolioBacktestResult,
    run_portfolio_backtest,
)
from crypto_trader.portfolio.config import PortfolioConfig  # noqa: E402
from crypto_trader.strategy.breakout.config import BreakoutConfig  # noqa: E402
from crypto_trader.strategy.momentum.config import MomentumConfig  # noqa: E402
from crypto_trader.strategy.trend.config import TrendConfig  # noqa: E402


DATA_DIR = ROOT / "data"
PORTFOLIO_BASE_DIR = ROOT / "output" / "portfolio"
PORTFOLIO_ROUND_DIR = PORTFOLIO_BASE_DIR / "round_3"
SOURCE_DIR = PORTFOLIO_ROUND_DIR / "source_pre_adoption"
LIVE_PORTFOLIO_CONFIG_PATH = ROOT / "config" / "portfolio_config.json"
LIVE_STRATEGY_CONFIG_PATHS = {
    strategy_id: ROOT / "config" / "strategies" / f"{strategy_id}.json"
    for strategy_id in STRATEGIES
}
FOLLOWUP_CSV_PATH = (
    ROOT
    / "output"
    / "portfolio"
    / "round_2"
    / "oos_followup_second_phase"
    / "ranked_results.csv"
)

ROUND_NUM = 3
ADOPTED_LABEL = "extended_body108_scorea250_breakbody0715_riskall_1.4"
ADOPTED_POLICY: dict[str, Any] = {
    "strategy.momentum.symbol_filter.sol_direction": "short_only",
    "strategy.momentum.exits.proof_lock_trigger_r": 0.55,
    "strategy.momentum.exits.proof_lock_min_bars": 3,
    "portfolio.symbol_collision": "cap",
    "portfolio.symbol_exposure_cap_R": 2.5,
    "portfolio.dd_tiers": [
        [0.06, 0.75],
        [0.09, 0.50],
        [0.12, 0.25],
        [0.15, 0.00],
    ],
    "strategy.trend.setup.orderly_max_body_frac": 1.08,
    "strategy.trend.setup.min_setup_score_a": 2.5,
    "strategy.breakout.setup.body_ratio_min": 0.715,
    "risk_scale.momentum": 1.4,
    "risk_scale.trend": 1.4,
    "risk_scale.breakout": 1.4,
}
FOLLOWUP_EXPECTED = {
    "dev_return_pct": 73.78624361928698,
    "dev_trades": 46.0,
    "dev_pf": 6.01578975255237,
    "dev_dd": 3.901403566703282,
    "oos_return_pct": 48.242251586993405,
    "oos_trades": 42.0,
    "oos_pf": 2.7496577180615804,
    "oos_dd": 5.366676749628214,
}


def _quiet_logging() -> None:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _strategy_round3_path(strategy_id: str) -> Path:
    return ROOT / "output" / strategy_id / "round_3" / "optimized_config.json"


def _strategy_round_dir(strategy_id: str) -> Path:
    return ROOT / "output" / strategy_id / "round_3"


def _strategy_source_snapshot_path(strategy_id: str) -> Path:
    return SOURCE_DIR / f"{strategy_id}_round3_optimized_config.pre_adoption.json"


def _portfolio_source_snapshot_path() -> Path:
    return SOURCE_DIR / "portfolio_config.pre_adoption.json"


def _snapshot_once(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    payload = _read_json(src)
    _atomic_write_json(_json_ready(payload), dest)


def _backup_once(path: Path, backup_name: str) -> None:
    backup = path.with_name(backup_name)
    if backup.exists() or not path.exists():
        return
    _atomic_write_json(_json_ready(_read_json(path)), backup)


def _strategy_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    strategy = payload.get("strategy", payload)
    if not isinstance(strategy, dict):
        raise TypeError(f"Expected strategy payload in {path}")
    return strategy


def _load_strategy_config(strategy_id: str, path: Path) -> Any:
    payload = _strategy_payload(path)
    if strategy_id == "momentum":
        return MomentumConfig.from_dict(payload)
    if strategy_id == "trend":
        return TrendConfig.from_dict(payload)
    if strategy_id == "breakout":
        return BreakoutConfig.from_dict(payload)
    raise ValueError(f"Unknown strategy: {strategy_id}")


def _load_source_strategy_configs() -> dict[str, Any]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    configs: dict[str, Any] = {}
    for strategy_id in STRATEGIES:
        _snapshot_once(_strategy_round3_path(strategy_id), _strategy_source_snapshot_path(strategy_id))
        configs[strategy_id] = _load_strategy_config(
            strategy_id,
            _strategy_source_snapshot_path(strategy_id),
        )
    return configs


def _load_source_portfolio_config() -> PortfolioConfig:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _snapshot_once(LIVE_PORTFOLIO_CONFIG_PATH, _portfolio_source_snapshot_path())
    return PortfolioConfig.from_dict(_read_json(_portfolio_source_snapshot_path()))


def _apply_policy_to_strategy_configs(
    base_configs: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    strategy_mutations, risk_scales, _ = _split_policy_mutations(policy)
    configs: dict[str, Any] = {}
    for strategy_id, cfg in base_configs.items():
        risk_mutations = _risk_scale_mutations(strategy_id, cfg, risk_scales[strategy_id])
        merged = merge_mutations(risk_mutations, strategy_mutations.get(strategy_id, {}))
        next_cfg = apply_mutations(cfg, merged) if merged else apply_mutations(cfg, {})
        next_cfg.symbols = list(SYMBOLS)
        configs[strategy_id] = next_cfg
    return configs


def _portfolio_config_from_policy(
    base_portfolio_config: PortfolioConfig,
    policy: dict[str, Any],
) -> PortfolioConfig:
    config = _portfolio_config_from_mutations(base_portfolio_config, policy)
    if config.terminal_accounting_mode != LIVE_PARITY_PROFILE.terminal_accounting_mode:
        raise RuntimeError(
            f"Portfolio terminal accounting {config.terminal_accounting_mode!r} "
            f"does not match {LIVE_PARITY_PROFILE.terminal_accounting_mode!r}"
        )
    return config


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _diff_values(base: Any, candidate: Any) -> dict[str, dict[str, Any]]:
    base_flat = _flatten(base.to_dict())
    candidate_flat = _flatten(candidate.to_dict())
    return {
        key: {
            "base_value": base_flat.get(key),
            "candidate_value": candidate_flat.get(key),
        }
        for key in sorted(set(base_flat) | set(candidate_flat))
        if base_flat.get(key) != candidate_flat.get(key)
    }


def _mutation_values_from_diff(diff: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {key: item["candidate_value"] for key, item in diff.items()}


def _contract_hash_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["contract_hash"] = stable_hash(payload)
    return out


def _portfolio_contract(
    *,
    portfolio_config: PortfolioConfig,
    strategy_configs: dict[str, Any],
) -> dict[str, Any]:
    timeframes = ["15m", "30m", "1h", "4h", "1d"]
    payload = {
        "kind": "portfolio_round3_adoption",
        "schema_version": "portfolio_adoption_v1",
        "source_candidate": ADOPTED_LABEL,
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "profile_hash": profile_hash(LIVE_PARITY_PROFILE),
        "economic_profile": LIVE_PARITY_PROFILE.to_dict(),
        "symbols": list(SYMBOLS),
        "required_timeframes": timeframes,
        "data_window": {
            "start_date": FULL_START.isoformat(),
            "end_date": FULL_END.isoformat(),
        },
        "data_fingerprint": data_snapshot_fingerprint(
            DATA_DIR,
            symbols=list(SYMBOLS),
            timeframes=timeframes,
            include_funding=LIVE_PARITY_PROFILE.apply_funding,
        ),
        "strategy_config_hashes": {
            strategy_id: strategy_config_hash(config)
            for strategy_id, config in strategy_configs.items()
        },
        "portfolio_config_hash": portfolio_config_hash(portfolio_config),
        "terminal_accounting_mode": LIVE_PARITY_PROFILE.terminal_accounting_mode,
        "policy": ADOPTED_POLICY,
    }
    return _contract_hash_payload(payload)


def _strategy_contract(
    *,
    strategy_id: str,
    strategy_config: Any,
    portfolio_config: PortfolioConfig,
) -> dict[str, Any]:
    timeframes = required_timeframes(strategy_id)
    payload = {
        "kind": "strategy_round3_portfolio_adoption",
        "schema_version": "strategy_portfolio_adoption_v1",
        "source_candidate": ADOPTED_LABEL,
        "strategy_type": strategy_id,
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "profile_hash": profile_hash(LIVE_PARITY_PROFILE),
        "economic_profile": LIVE_PARITY_PROFILE.to_dict(),
        "symbols": list(SYMBOLS),
        "required_timeframes": timeframes,
        "data_window": {
            "start_date": FULL_START.isoformat(),
            "end_date": FULL_END.isoformat(),
        },
        "data_fingerprint": data_snapshot_fingerprint(
            DATA_DIR,
            symbols=list(SYMBOLS),
            timeframes=timeframes,
            include_funding=LIVE_PARITY_PROFILE.apply_funding,
        ),
        "strategy_config_hash": strategy_config_hash(strategy_config),
        "portfolio_config_hash": portfolio_config_hash(portfolio_config),
        "terminal_accounting_mode": LIVE_PARITY_PROFILE.terminal_accounting_mode,
    }
    return _contract_hash_payload(payload)


def _run_portfolio_window(
    *,
    portfolio_config: PortfolioConfig,
    strategy_configs: dict[str, Any],
    start: date,
    end: date,
    store: Any,
) -> tuple[dict[str, Any], PortfolioBacktestResult]:
    result = run_portfolio_backtest(
        portfolio_config=deepcopy(portfolio_config),
        strategy_configs={sid: deepcopy(cfg) for sid, cfg in strategy_configs.items()},
        backtest_config=_bt_config(start, end, portfolio_config.initial_equity),
        data_dir=DATA_DIR,
        store=store,
        terminal_accounting_mode=LIVE_PARITY_PROFILE.terminal_accounting_mode,
    )
    metrics = _augment_metrics(metrics_to_dict(result.metrics), result)
    score, rejected, reason, components = _score_metrics(
        metrics,
        dict(SCORING_WEIGHTS),
        dict(HARD_REJECTS),
    )
    metrics["immutable_score"] = score
    metrics["rejected"] = rejected
    metrics["reject_reason"] = reason
    metrics["score_components"] = components
    metrics["terminal_mark_count"] = sum(
        len(marks) for marks in result.terminal_marks.values()
    )
    return metrics, result


def _run_strategy_window(
    *,
    strategy_id: str,
    strategy_config: Any,
    start: date,
    end: date,
    store: Any,
) -> tuple[dict[str, Any], Any]:
    bt_config = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=list(SYMBOLS),
        start_date=start,
        end_date=end,
    )
    result = run(
        deepcopy(strategy_config),
        bt_config,
        data_dir=DATA_DIR,
        store=store,
        strategy_type=strategy_id,
    )
    metrics = metrics_to_dict(result.metrics)
    metrics["terminal_mark_count"] = len(result.terminal_marks)
    return metrics, result


def _metric_line(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"{label}: return {metrics.get('net_return_pct', 0.0):.4f}%, "
        f"trades {metrics.get('total_trades', 0.0):.0f}, "
        f"win rate {metrics.get('win_rate', 0.0):.2f}%, "
        f"PF {metrics.get('profit_factor', 0.0):.4f}, "
        f"expectancy {metrics.get('expectancy_r', 0.0):.4f}R, "
        f"DD {metrics.get('max_drawdown_pct', 0.0):.4f}%"
    )


def _followup_csv_row() -> dict[str, Any]:
    if not FOLLOWUP_CSV_PATH.exists():
        return {}
    with open(FOLLOWUP_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") == ADOPTED_LABEL:
                out: dict[str, Any] = dict(row)
                for key, value in row.items():
                    try:
                        out[key] = float(value)
                    except (TypeError, ValueError):
                        out[key] = value
                return out
    return {}


def _assert_followup_metrics(
    *,
    dev_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    row = _followup_csv_row()
    expected = dict(FOLLOWUP_EXPECTED)
    if row:
        expected.update(
            {
                "dev_return_pct": row.get("dev_return_pct", expected["dev_return_pct"]),
                "dev_trades": row.get("dev_trades", expected["dev_trades"]),
                "dev_pf": row.get("dev_pf", expected["dev_pf"]),
                "dev_dd": row.get("dev_dd", expected["dev_dd"]),
                "oos_return_pct": row.get("oos_return_pct", expected["oos_return_pct"]),
                "oos_trades": row.get("oos_trades", expected["oos_trades"]),
                "oos_pf": row.get("oos_pf", expected["oos_pf"]),
                "oos_dd": row.get("oos_dd", expected["oos_dd"]),
            }
        )
    actual = {
        "dev_return_pct": dev_metrics.get("net_return_pct"),
        "dev_trades": dev_metrics.get("total_trades"),
        "dev_pf": dev_metrics.get("profit_factor"),
        "dev_dd": dev_metrics.get("max_drawdown_pct"),
        "oos_return_pct": oos_metrics.get("net_return_pct"),
        "oos_trades": oos_metrics.get("total_trades"),
        "oos_pf": oos_metrics.get("profit_factor"),
        "oos_dd": oos_metrics.get("max_drawdown_pct"),
    }
    deltas = {
        key: float(actual[key] or 0.0) - float(expected[key] or 0.0)
        for key in actual
    }
    max_abs_delta = max(abs(value) for value in deltas.values())
    if max_abs_delta > tolerance:
        raise RuntimeError(
            "Adopted candidate no longer reconstructs follow-up metrics: "
            + json.dumps(
                {"expected": expected, "actual": actual, "deltas": deltas},
                indent=2,
                sort_keys=True,
            )
        )
    return {
        "status": "matched",
        "source_csv": str(FOLLOWUP_CSV_PATH),
        "expected": expected,
        "actual": actual,
        "deltas": deltas,
        "max_abs_delta": max_abs_delta,
    }


def _trade_row(strategy_id: str, trade: Any) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "trade_id": trade.trade_id,
        "symbol": trade.symbol,
        "direction": getattr(trade.direction, "name", str(trade.direction)),
        "entry_time": trade.entry_time.isoformat() if trade.entry_time else "",
        "exit_time": trade.exit_time.isoformat() if trade.exit_time else "",
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "qty": trade.qty,
        "pnl": trade.pnl,
        "net_pnl": _trade_net_pnl(trade),
        "r_multiple": _trade_reporting_r(trade),
        "commission": trade.commission,
        "funding_paid": trade.funding_paid,
        "bars_held": trade.bars_held,
        "setup_grade": getattr(trade.setup_grade, "name", str(trade.setup_grade)),
        "confirmation_type": trade.confirmation_type,
        "entry_method": trade.entry_method,
        "exit_reason": trade.exit_reason,
        "mae_r": trade.mae_r,
        "mfe_r": trade.mfe_r,
        "signal_variant": trade.signal_variant,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_ready(row), sort_keys=True) + "\n")
    tmp_path.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _portfolio_group_rows(result: PortfolioBacktestResult) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for strategy_id, trades in result.per_strategy_trades.items():
        for trade in trades:
            direction = getattr(trade.direction, "name", str(trade.direction))
            key = "|".join(
                [
                    strategy_id,
                    trade.symbol,
                    direction,
                    trade.confirmation_type or "none",
                ]
            )
            row = groups.setdefault(key, {"trades": 0, "net_profit": 0.0, "rs": [], "wins": 0})
            net_pnl = _trade_net_pnl(trade)
            row["trades"] += 1
            row["net_profit"] += net_pnl
            row["wins"] += 1 if net_pnl > 0 else 0
            rr = _trade_reporting_r(trade)
            if rr is not None:
                row["rs"].append(rr)
    return {
        key: {
            "trades": row["trades"],
            "net_profit": row["net_profit"],
            "win_rate": row["wins"] / row["trades"] * 100.0 if row["trades"] else 0.0,
            "avg_r": sum(row["rs"]) / len(row["rs"]) if row["rs"] else 0.0,
        }
        for key, row in sorted(groups.items())
    }


def _portfolio_worst_trades(result: PortfolioBacktestResult, n: int = 12) -> list[dict[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for strategy_id, trades in result.per_strategy_trades.items():
        rows.extend((strategy_id, trade) for trade in trades)
    rows.sort(key=lambda item: _trade_net_pnl(item[1]))
    return [_trade_row(strategy_id, trade) for strategy_id, trade in rows[:n]]


def _write_portfolio_artifacts(
    *,
    portfolio_config: PortfolioConfig,
    strategy_configs: dict[str, Any],
    contract: dict[str, Any],
    dev_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    full_metrics: dict[str, Any],
    full_result: PortfolioBacktestResult,
    followup_match: dict[str, Any],
) -> None:
    PORTFOLIO_ROUND_DIR.mkdir(parents=True, exist_ok=True)
    rec_dir = PORTFOLIO_ROUND_DIR / "recommended_strategy_configs"
    rec_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(_json_ready(portfolio_config.to_dict()), PORTFOLIO_ROUND_DIR / "recommended_portfolio_config.json")
    for strategy_id, config in strategy_configs.items():
        _atomic_write_json(
            {"strategy": _json_ready(config.to_dict())},
            rec_dir / f"{strategy_id}.json",
        )

    bundle = {
        "portfolio_config": portfolio_config.to_dict(),
        "strategy_configs": {
            strategy_id: config.to_dict()
            for strategy_id, config in strategy_configs.items()
        },
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "round": ROUND_NUM,
            "promotion_type": "portfolio_oos_repair_adoption",
            "source_candidate": ADOPTED_LABEL,
            "source_followup_csv": str(FOLLOWUP_CSV_PATH),
            "source_snapshots": {
                "portfolio": str(_portfolio_source_snapshot_path()),
                "strategies": {
                    strategy_id: str(_strategy_source_snapshot_path(strategy_id))
                    for strategy_id in STRATEGIES
                },
            },
            "policy": ADOPTED_POLICY,
            "contract": contract,
            "contract_hash": contract.get("contract_hash", ""),
            "profile_hash": contract.get("profile_hash", ""),
            "strategy_config_hashes": contract.get("strategy_config_hashes", {}),
            "portfolio_config_hash": contract.get("portfolio_config_hash", ""),
            "parity_alignment": {
                "profile_id": LIVE_PARITY_PROFILE.profile_id,
                "terminal_accounting_mode": LIVE_PARITY_PROFILE.terminal_accounting_mode,
                "followup_reconstruction": followup_match,
            },
        },
    }
    _atomic_write_json(_json_ready(bundle), PORTFOLIO_ROUND_DIR / "optimized_config.json")

    equity_rows = [
        {"timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts), "equity": f"{eq:.2f}"}
        for ts, eq in full_result.equity_curve
    ]
    _write_csv(PORTFOLIO_ROUND_DIR / "portfolio_equity_curve.csv", equity_rows, ["timestamp", "equity"])

    trade_rows: list[dict[str, Any]] = []
    for strategy_id, trades in full_result.per_strategy_trades.items():
        trade_rows.extend(_trade_row(strategy_id, trade) for trade in trades)
    _write_jsonl(PORTFOLIO_ROUND_DIR / "portfolio_trades.jsonl", trade_rows)

    rule_rows = [
        {
            "timestamp": event.timestamp.isoformat(),
            "strategy_id": event.strategy_id,
            "symbol": event.symbol,
            "direction": event.direction,
            "risk_R": event.risk_R,
            "approved": event.approved,
            "denial_reason": event.denial_reason,
            "size_multiplier": event.size_multiplier,
        }
        for event in full_result.rule_events
    ]
    _write_jsonl(PORTFOLIO_ROUND_DIR / "rule_events.jsonl", rule_rows)

    terminal_marks = [
        mark
        for marks in full_result.terminal_marks.values()
        for mark in marks
    ]
    diagnostics = generate_diagnostics(
        list(full_result.all_trades),
        initial_equity=portfolio_config.initial_equity,
        title="Portfolio Round 3 OOS Repair Final Diagnostics",
        terminal_marks=terminal_marks,
        performance_metrics=full_result.metrics,
        expected_symbols=list(SYMBOLS),
    )
    (PORTFOLIO_ROUND_DIR / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")

    evaluation_lines = [
        "Portfolio Round 3 OOS Repair Adoption",
        "=" * 39,
        "",
        f"Selected candidate: {ADOPTED_LABEL}",
        "",
        "Validation windows:",
        f"- In sample: {DEV_START.isoformat()} to {DEV_END.isoformat()}",
        f"- Out of sample: {HOLDOUT_START.isoformat()} to {HOLDOUT_END.isoformat()}",
        f"- Full promoted round: {FULL_START.isoformat()} to {FULL_END.isoformat()}",
        "",
        "Backtest results:",
        f"- {_metric_line('IS', dev_metrics)}",
        f"- {_metric_line('OOS', oos_metrics)}",
        f"- {_metric_line('Full', full_metrics)}",
        "",
        "Adopted policy:",
        json.dumps(_json_ready(ADOPTED_POLICY), indent=2, sort_keys=True),
        "",
        "OOS group attribution:",
        json.dumps(_json_ready(_portfolio_group_rows(full_result)), indent=2, sort_keys=True),
        "",
        "Worst full-period trades:",
        json.dumps(_json_ready(_portfolio_worst_trades(full_result)), indent=2, sort_keys=True),
        "",
        "Parity contract:",
        f"- profile_id: {LIVE_PARITY_PROFILE.profile_id}",
        f"- terminal_accounting_mode: {LIVE_PARITY_PROFILE.terminal_accounting_mode}",
        f"- contract_hash: {contract.get('contract_hash', '')}",
        f"- portfolio_config_hash: {contract.get('portfolio_config_hash', '')}",
        f"- strategy_config_hashes: {json.dumps(contract.get('strategy_config_hashes', {}), sort_keys=True)}",
        f"- followup_reconstruction: {followup_match.get('status', 'unknown')}",
        "",
    ]
    (PORTFOLIO_ROUND_DIR / "round_evaluation.txt").write_text(
        "\n".join(evaluation_lines),
        encoding="utf-8",
    )

    summary = {
        "round": ROUND_NUM,
        "source_candidate": ADOPTED_LABEL,
        "promotion_type": "portfolio_oos_repair_adoption",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "windows": {
            "in_sample": {"start": DEV_START.isoformat(), "end": DEV_END.isoformat()},
            "out_of_sample": {"start": HOLDOUT_START.isoformat(), "end": HOLDOUT_END.isoformat()},
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
        },
        "policy": ADOPTED_POLICY,
        "metrics": {
            "in_sample": dev_metrics,
            "out_of_sample": oos_metrics,
            "full": full_metrics,
            "per_strategy_full": {
                strategy_id: metrics_to_dict(metrics) if metrics is not None else None
                for strategy_id, metrics in full_result.per_strategy_metrics.items()
            },
        },
        "followup_reconstruction": followup_match,
        "contract": contract,
        "artifacts": {
            "optimized_config": str(PORTFOLIO_ROUND_DIR / "optimized_config.json"),
            "recommended_portfolio_config": str(PORTFOLIO_ROUND_DIR / "recommended_portfolio_config.json"),
            "recommended_strategy_configs": str(rec_dir),
            "round_final_diagnostics": str(PORTFOLIO_ROUND_DIR / "round_final_diagnostics.txt"),
            "round_evaluation": str(PORTFOLIO_ROUND_DIR / "round_evaluation.txt"),
            "portfolio_trades": str(PORTFOLIO_ROUND_DIR / "portfolio_trades.jsonl"),
            "portfolio_equity_curve": str(PORTFOLIO_ROUND_DIR / "portfolio_equity_curve.csv"),
            "rule_events": str(PORTFOLIO_ROUND_DIR / "rule_events.jsonl"),
        },
    }
    _atomic_write_json(_json_ready(summary), PORTFOLIO_ROUND_DIR / "portfolio_summary.json")
    _atomic_write_json(
        _json_ready(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "selection_basis": "user_approved_followup_oos_repair",
                "source_candidate": ADOPTED_LABEL,
                "holdout_policy": "post-2026-04-20 used as OOS validation evidence after repair selection",
                "mutations": ADOPTED_POLICY,
                "metrics": summary["metrics"],
                "contract": contract,
                "followup_reconstruction": followup_match,
            }
        ),
        PORTFOLIO_ROUND_DIR / "phase_auto_results.json",
    )
    _atomic_write_json(
        _json_ready(
            {
                "round": ROUND_NUM,
                "strategy_type": "portfolio",
                "promotion_type": "portfolio_oos_repair_adoption",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_candidate": ADOPTED_LABEL,
                "source_followup_csv": str(FOLLOWUP_CSV_PATH),
                "source_snapshots": {
                    "portfolio": str(_portfolio_source_snapshot_path()),
                    "strategies": {
                        strategy_id: str(_strategy_source_snapshot_path(strategy_id))
                        for strategy_id in STRATEGIES
                    },
                },
                "policy": ADOPTED_POLICY,
                "windows": summary["windows"],
                "contract_hash": contract.get("contract_hash", ""),
                "profile_id": LIVE_PARITY_PROFILE.profile_id,
                "profile_hash": contract.get("profile_hash", ""),
                "required_timeframes": contract.get("required_timeframes", []),
                "symbols": list(SYMBOLS),
            }
        ),
        PORTFOLIO_ROUND_DIR / "run_spec.json",
    )


def _write_strategy_artifacts(
    *,
    strategy_id: str,
    source_config: Any,
    promoted_config: Any,
    portfolio_config: PortfolioConfig,
    contract: dict[str, Any],
    is_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    full_metrics: dict[str, Any],
    full_result: Any,
) -> dict[str, Any]:
    round_dir = _strategy_round_dir(strategy_id)
    round_dir.mkdir(parents=True, exist_ok=True)
    diff = _diff_values(source_config, promoted_config)
    mutation_values = _mutation_values_from_diff(diff)
    payload = {
        "strategy": promoted_config.to_dict(),
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "round": ROUND_NUM,
            "promotion_type": "portfolio_round3_adoption",
            "source_candidate": ADOPTED_LABEL,
            "source_config_snapshot": str(_strategy_source_snapshot_path(strategy_id)),
            "source_portfolio_round": str(PORTFOLIO_ROUND_DIR),
            "contract": contract,
            "contract_hash": contract.get("contract_hash", ""),
            "profile_hash": contract.get("profile_hash", ""),
            "strategy_config_hash": contract.get("strategy_config_hash", ""),
            "portfolio_config_hash": contract.get("portfolio_config_hash", ""),
            "strategy_adoption_mutations": mutation_values,
            "portfolio_policy": ADOPTED_POLICY,
            "parity_alignment": {
                "profile_id": LIVE_PARITY_PROFILE.profile_id,
                "terminal_accounting_mode": LIVE_PARITY_PROFILE.terminal_accounting_mode,
                "config_round_trip": "matched",
            },
        },
    }
    _backup_once(
        _strategy_round3_path(strategy_id),
        "optimized_config.pre_portfolio_round3_adoption.json",
    )
    _atomic_write_json(_json_ready(payload), _strategy_round3_path(strategy_id))

    generate_report(full_result, round_dir)
    export_equity_curve(full_result, round_dir)
    export_trade_journal(full_result, round_dir)
    diagnostics = generate_diagnostics(
        list(full_result.trades),
        initial_equity=LIVE_PARITY_PROFILE.initial_equity,
        title=f"{strategy_id.title()} Round 3 Portfolio Adoption Final Diagnostics",
        terminal_marks=list(full_result.terminal_marks),
        performance_metrics=full_result.metrics,
        expected_symbols=list(SYMBOLS),
        diagnostic_context=full_result.diagnostic_context,
    )
    (round_dir / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")

    evaluation_lines = [
        f"{strategy_id.title()} Round 3 Portfolio Adoption",
        "=" * (len(strategy_id) + 29),
        "",
        f"Selected portfolio candidate: {ADOPTED_LABEL}",
        "",
        "Validation windows:",
        f"- In sample: {DEV_START.isoformat()} to {DEV_END.isoformat()}",
        f"- Out of sample: {HOLDOUT_START.isoformat()} to {HOLDOUT_END.isoformat()}",
        f"- Full promoted round: {FULL_START.isoformat()} to {FULL_END.isoformat()}",
        "",
        "Backtest results:",
        f"- {_metric_line('IS', is_metrics)}",
        f"- {_metric_line('OOS', oos_metrics)}",
        f"- {_metric_line('Full', full_metrics)}",
        "",
        "Pre-adoption round_3 -> promoted round_3 parameter delta:",
        json.dumps(_json_ready(diff), indent=2, sort_keys=True),
        "",
        "Parity contract:",
        f"- profile_id: {LIVE_PARITY_PROFILE.profile_id}",
        f"- terminal_accounting_mode: {LIVE_PARITY_PROFILE.terminal_accounting_mode}",
        f"- contract_hash: {contract.get('contract_hash', '')}",
        f"- strategy_config_hash: {contract.get('strategy_config_hash', '')}",
        f"- portfolio_config_hash: {contract.get('portfolio_config_hash', '')}",
        "",
    ]
    (round_dir / "round_evaluation.txt").write_text(
        "\n".join(evaluation_lines),
        encoding="utf-8",
    )

    summary = {
        "round": ROUND_NUM,
        "strategy_type": strategy_id,
        "promotion_type": "portfolio_round3_adoption",
        "source_candidate": ADOPTED_LABEL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_config_snapshot": str(_strategy_source_snapshot_path(strategy_id)),
        "mutation_diff_vs_pre_adoption_round3": diff,
        "mutations": mutation_values,
        "windows": {
            "in_sample": {"start": DEV_START.isoformat(), "end": DEV_END.isoformat()},
            "out_of_sample": {"start": HOLDOUT_START.isoformat(), "end": HOLDOUT_END.isoformat()},
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
        },
        "final_metrics": full_metrics,
        "in_sample_metrics": is_metrics,
        "out_of_sample_metrics": oos_metrics,
        "contract": contract,
        "artifacts": {
            "optimized_config": str(_strategy_round3_path(strategy_id)),
            "backtest_report": str(round_dir / "backtest_report.md"),
            "equity_curve": str(round_dir / "equity_curve.csv"),
            "journal": str(round_dir / "journal.csv"),
            "round_final_diagnostics": str(round_dir / "round_final_diagnostics.txt"),
            "round_evaluation": str(round_dir / "round_evaluation.txt"),
        },
    }
    _atomic_write_json(_json_ready(summary), round_dir / "round3_summary.json")
    _atomic_write_json(
        _json_ready(
            {
                "round": ROUND_NUM,
                "strategy_type": strategy_id,
                "promotion_type": "portfolio_round3_adoption",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_candidate": ADOPTED_LABEL,
                "source_config_snapshot": str(_strategy_source_snapshot_path(strategy_id)),
                "mutations": mutation_values,
                "windows": summary["windows"],
                "contract_hash": contract.get("contract_hash", ""),
                "profile_id": LIVE_PARITY_PROFILE.profile_id,
                "profile_hash": contract.get("profile_hash", ""),
                "required_timeframes": contract.get("required_timeframes", []),
                "symbols": list(SYMBOLS),
            }
        ),
        round_dir / "run_spec.json",
    )
    return {
        "diff": diff,
        "mutations": mutation_values,
        "summary": summary,
    }


def _write_live_configs(
    *,
    portfolio_config: PortfolioConfig,
    strategy_configs: dict[str, Any],
) -> dict[str, Any]:
    _backup_once(
        LIVE_PORTFOLIO_CONFIG_PATH,
        "portfolio_config.pre_portfolio_round3_adoption.json",
    )
    _atomic_write_json(_json_ready(portfolio_config.to_dict()), LIVE_PORTFOLIO_CONFIG_PATH)

    live_hashes: dict[str, Any] = {
        "portfolio": {
            "path": str(LIVE_PORTFOLIO_CONFIG_PATH),
            "hash": portfolio_config_hash(PortfolioConfig.from_dict(_read_json(LIVE_PORTFOLIO_CONFIG_PATH))),
            "expected_hash": portfolio_config_hash(portfolio_config),
            "status": "pending",
        },
        "strategies": {},
    }
    if live_hashes["portfolio"]["hash"] != live_hashes["portfolio"]["expected_hash"]:
        raise RuntimeError("Live portfolio config hash does not match promoted portfolio config.")
    live_hashes["portfolio"]["status"] = "matched"

    for strategy_id, config in strategy_configs.items():
        live_path = LIVE_STRATEGY_CONFIG_PATHS[strategy_id]
        _backup_once(
            live_path,
            f"{strategy_id}.pre_portfolio_round3_adoption.json",
        )
        _atomic_write_json({"strategy": _json_ready(config.to_dict())}, live_path)
        reloaded = _load_strategy_config(strategy_id, live_path)
        actual_hash = strategy_config_hash(reloaded)
        expected_hash = strategy_config_hash(config)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Live {strategy_id} config hash does not match promoted config.")
        live_hashes["strategies"][strategy_id] = {
            "path": str(live_path),
            "hash": actual_hash,
            "expected_hash": expected_hash,
            "status": "matched",
        }
    return live_hashes


def _load_live_portfolio_configs() -> tuple[PortfolioConfig, dict[str, Any]]:
    portfolio_config = PortfolioConfig.from_dict(_read_json(LIVE_PORTFOLIO_CONFIG_PATH))
    strategy_configs = {
        strategy_id: _load_strategy_config(strategy_id, LIVE_STRATEGY_CONFIG_PATHS[strategy_id])
        for strategy_id in STRATEGIES
    }
    return portfolio_config, strategy_configs


def _compare_metrics(
    expected: dict[str, Any],
    actual: dict[str, Any],
    keys: list[str],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    deltas = {
        key: float(actual.get(key, 0.0)) - float(expected.get(key, 0.0))
        for key in keys
    }
    max_abs_delta = max((abs(value) for value in deltas.values()), default=0.0)
    return {
        "status": "matched" if max_abs_delta <= tolerance else "mismatch",
        "deltas": deltas,
        "max_abs_delta": max_abs_delta,
        "tolerance": tolerance,
    }


def _write_parity_alignment(
    *,
    live_config_hashes: dict[str, Any],
    portfolio_contract: dict[str, Any],
    portfolio_full_metrics: dict[str, Any],
    live_full_metrics: dict[str, Any],
    followup_match: dict[str, Any],
    strategy_artifact_info: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metric_compare = _compare_metrics(
        portfolio_full_metrics,
        live_full_metrics,
        [
            "net_return_pct",
            "net_profit",
            "realized_pnl_net",
            "terminal_mark_pnl_net",
            "total_trades",
            "win_rate",
            "profit_factor",
            "expectancy_r",
            "max_drawdown_pct",
            "sharpe_ratio",
            "calmar_ratio",
            "exit_efficiency",
        ],
    )
    if metric_compare["status"] != "matched":
        raise RuntimeError(
            "Live config portfolio replay does not match promoted artifacts: "
            + json.dumps(metric_compare, indent=2, sort_keys=True)
        )
    payload = {
        "status": "matched",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": LIVE_PARITY_PROFILE.to_dict(),
        "terminal_accounting_mode": LIVE_PARITY_PROFILE.terminal_accounting_mode,
        "source_candidate": ADOPTED_LABEL,
        "followup_reconstruction": followup_match,
        "live_config_hashes": live_config_hashes,
        "portfolio_contract_hash": portfolio_contract.get("contract_hash", ""),
        "portfolio_metric_replay": metric_compare,
        "strategy_artifacts": {
            strategy_id: {
                "mutations": info["mutations"],
                "summary_path": info["summary"]["artifacts"]["round_final_diagnostics"],
            }
            for strategy_id, info in strategy_artifact_info.items()
        },
    }
    _atomic_write_json(_json_ready(payload), PORTFOLIO_ROUND_DIR / "parity_alignment.json")
    for strategy_id in STRATEGIES:
        _atomic_write_json(
            _json_ready(
                {
                    "status": "matched",
                    "created_at": payload["created_at"],
                    "profile": LIVE_PARITY_PROFILE.to_dict(),
                    "terminal_accounting_mode": LIVE_PARITY_PROFILE.terminal_accounting_mode,
                    "source_candidate": ADOPTED_LABEL,
                    "live_config_hash": live_config_hashes["strategies"][strategy_id],
                    "portfolio_metric_replay": metric_compare,
                    "strategy_mutations": strategy_artifact_info[strategy_id]["mutations"],
                }
            ),
            _strategy_round_dir(strategy_id) / "parity_alignment.json",
        )
    return payload


def _metric_entry_fields(entry: dict[str, Any], metrics: dict[str, Any]) -> None:
    for key in [
        "total_trades",
        "win_rate",
        "profit_factor",
        "max_drawdown_pct",
        "sharpe_ratio",
        "calmar_ratio",
        "net_return_pct",
        "expectancy_r",
        "exit_efficiency",
        "realized_pnl_net",
        "terminal_mark_pnl_net",
        "net_profit",
        "total_fees",
        "funding_cost_total",
        "terminal_mark_count",
    ]:
        if key in metrics:
            entry[key] = metrics[key]


def _update_manifest(base_dir: Path, round_num: int, entry: dict[str, Any]) -> None:
    manifest_path = base_dir / "rounds_manifest.json"
    if manifest_path.exists():
        _backup_once(manifest_path, "rounds_manifest.pre_portfolio_round3_adoption.json")
        manifest = _read_json(manifest_path)
    else:
        manifest = {"rounds": []}
    manifest.setdefault("rounds", [])
    try:
        manifest["schema_version"] = max(int(manifest.get("schema_version", 1)), 3)
    except (TypeError, ValueError):
        manifest["schema_version"] = 3
    manifest["rounds"] = [r for r in manifest["rounds"] if r.get("round") != round_num]
    manifest["rounds"].append(entry)
    manifest["rounds"].sort(key=lambda r: int(r.get("round", 0)))
    _atomic_write_json(_json_ready(manifest), manifest_path)


def _portfolio_round2_manifest_entry_from_artifacts() -> dict[str, Any]:
    phase_path = PORTFOLIO_BASE_DIR / "round_2" / "phase_auto_results.json"
    portfolio_path = PORTFOLIO_BASE_DIR / "round_2" / "recommended_portfolio_config.json"
    strategy_dir = PORTFOLIO_BASE_DIR / "round_2" / "recommended_strategy_configs"
    phase = _read_json(phase_path)
    portfolio_config = PortfolioConfig.from_dict(_read_json(portfolio_path))
    strategy_classes = {
        "momentum": MomentumConfig,
        "trend": TrendConfig,
        "breakout": BreakoutConfig,
    }
    strategy_configs = {
        strategy_id: cls.from_dict(_strategy_payload(strategy_dir / f"{strategy_id}.json"))
        for strategy_id, cls in strategy_classes.items()
    }
    timeframes = ["15m", "30m", "1h", "4h", "1d"]
    contract = {
        "kind": "portfolio_round2_phased_auto_backfill",
        "schema_version": "portfolio_manifest_backfill_v1",
        "source_artifact": str(phase_path),
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "profile_hash": profile_hash(LIVE_PARITY_PROFILE),
        "economic_profile": LIVE_PARITY_PROFILE.to_dict(),
        "symbols": list(SYMBOLS),
        "required_timeframes": timeframes,
        "data_window": {
            "start_date": DEV_START.isoformat(),
            "end_date": FULL_END.isoformat(),
        },
        "data_fingerprint": data_snapshot_fingerprint(
            DATA_DIR,
            symbols=list(SYMBOLS),
            timeframes=timeframes,
            include_funding=LIVE_PARITY_PROFILE.apply_funding,
        ),
        "strategy_config_hashes": {
            strategy_id: strategy_config_hash(config)
            for strategy_id, config in strategy_configs.items()
        },
        "portfolio_config_hash": portfolio_config_hash(portfolio_config),
        "terminal_accounting_mode": LIVE_PARITY_PROFILE.terminal_accounting_mode,
        "source_contract": phase.get("contract", {}),
    }
    contract["contract_hash"] = stable_hash(contract)
    metrics = phase["metrics"]["full"]
    gate_passed = not bool(phase.get("rejected"))
    gate_result = {
        "passed": gate_passed,
        "failure_reasons": [] if gate_passed else [phase.get("reject_reason", "")],
        "validation_mode": "strict",
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "acceptance_basis": "portfolio_round2_phased_auto_backfill",
        "source_artifact": str(phase_path),
    }
    entry = {
        "round": 2,
        "timestamp": phase.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "baseline_type": "portfolio_round2_phased_auto_backfill",
        "mutations_count": len(phase.get("mutations", {})),
        "mutations": phase.get("mutations", {}),
        "score": phase.get("immutable_score"),
        "gate_status": "passed" if gate_passed else "failed",
        "gate_passed": gate_passed,
        "gate_failure_reasons": gate_result["failure_reasons"],
        "reject_reason": phase.get("reject_reason", ""),
        "contract_hash": contract["contract_hash"],
        "profile_hash": contract["profile_hash"],
        "strategy_config_hashes": contract["strategy_config_hashes"],
        "portfolio_config_hash": contract["portfolio_config_hash"],
        "data_window": contract["data_window"],
        "symbols": contract["symbols"],
        "required_timeframes": contract["required_timeframes"],
        "data_fingerprint": contract["data_fingerprint"],
        "economic_profile": contract["economic_profile"],
        "contract": contract,
        "metrics": metrics,
        "gate_result": gate_result,
        "final_validation": {
            "status": "passed" if gate_passed else "failed",
            "source": "portfolio_round2_phase_auto_results_backfill",
            "in_sample": {"start": DEV_START.isoformat(), "end": DEV_END.isoformat()},
            "out_of_sample": {"start": HOLDOUT_START.isoformat(), "end": HOLDOUT_END.isoformat()},
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
        },
        "accepted_count": len(phase.get("mutations", {})),
        "new_mutations": phase.get("mutations", {}),
    }
    _metric_entry_fields(entry, metrics)
    return entry


def _ensure_portfolio_round2_manifest_entry() -> None:
    manifest_path = PORTFOLIO_BASE_DIR / "rounds_manifest.json"
    if not manifest_path.exists():
        return
    manifest = _read_json(manifest_path)
    rounds = manifest.get("rounds", [])
    if any(item.get("round") == 2 for item in rounds):
        return
    rounds.append(_portfolio_round2_manifest_entry_from_artifacts())
    rounds.sort(key=lambda r: int(r.get("round", 0)))
    manifest["rounds"] = rounds
    _atomic_write_json(_json_ready(manifest), manifest_path)


def _portfolio_manifest_entry(
    *,
    contract: dict[str, Any],
    full_metrics: dict[str, Any],
    parity_alignment: dict[str, Any],
) -> dict[str, Any]:
    gate_result = {
        "passed": True,
        "failure_reasons": [],
        "validation_mode": "strict",
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "acceptance_basis": "user_approved_followup_oos_repair",
        "source_candidate": ADOPTED_LABEL,
        "live_backtest_parity": parity_alignment["status"],
        "followup_reconstruction": parity_alignment["followup_reconstruction"]["status"],
    }
    phase_result = {
        "final_score": full_metrics.get("immutable_score"),
        "accepted_count": len(ADOPTED_POLICY),
        "new_mutations": ADOPTED_POLICY,
        "final_validation": {
            "status": "passed",
            "source": "portfolio_round3_oos_repair_adoption",
            "in_sample": {"start": DEV_START.isoformat(), "end": DEV_END.isoformat()},
            "out_of_sample": {"start": HOLDOUT_START.isoformat(), "end": HOLDOUT_END.isoformat()},
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
            "live_backtest_parity": parity_alignment["status"],
        },
    }
    entry = {
        "round": ROUND_NUM,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_type": "portfolio_round3_oos_repair_adoption",
        "mutations_count": len(ADOPTED_POLICY),
        "mutations": ADOPTED_POLICY,
        "score": phase_result["final_score"],
        "gate_status": "passed",
        "gate_passed": True,
        "gate_failure_reasons": [],
        "reject_reason": "",
        "contract_hash": contract.get("contract_hash", ""),
        "profile_hash": contract.get("profile_hash", ""),
        "strategy_config_hashes": contract.get("strategy_config_hashes", {}),
        "portfolio_config_hash": contract.get("portfolio_config_hash", ""),
        "data_window": contract.get("data_window", {}),
        "symbols": contract.get("symbols", []),
        "required_timeframes": contract.get("required_timeframes", []),
        "data_fingerprint": contract.get("data_fingerprint", {}),
        "economic_profile": contract.get("economic_profile", {}),
        "contract": contract,
        "metrics": full_metrics,
        "gate_result": gate_result,
        "final_validation": phase_result["final_validation"],
        "accepted_count": phase_result["accepted_count"],
        "new_mutations": phase_result["new_mutations"],
    }
    _metric_entry_fields(entry, full_metrics)
    return entry


def _strategy_manifest_entry(
    *,
    strategy_id: str,
    contract: dict[str, Any],
    full_metrics: dict[str, Any],
    mutations: dict[str, Any],
    parity_alignment: dict[str, Any],
) -> dict[str, Any]:
    gate_result = {
        "passed": True,
        "failure_reasons": [],
        "validation_mode": "strict",
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "acceptance_basis": "portfolio_round3_adoption",
        "source_candidate": ADOPTED_LABEL,
        "live_backtest_parity": parity_alignment["status"],
    }
    phase_result = {
        "final_score": None,
        "accepted_count": len(mutations),
        "new_mutations": mutations,
        "final_validation": {
            "status": "passed",
            "source": "portfolio_round3_adoption",
            "in_sample": {"start": DEV_START.isoformat(), "end": DEV_END.isoformat()},
            "out_of_sample": {"start": HOLDOUT_START.isoformat(), "end": HOLDOUT_END.isoformat()},
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
            "live_backtest_parity": parity_alignment["status"],
        },
    }
    entry = {
        "round": ROUND_NUM,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_type": "portfolio_round3_adoption",
        "mutations_count": len(mutations),
        "mutations": mutations,
        "score": phase_result["final_score"],
        "gate_status": "passed",
        "gate_passed": True,
        "gate_failure_reasons": [],
        "reject_reason": "",
        "contract_hash": contract.get("contract_hash", ""),
        "profile_hash": contract.get("profile_hash", ""),
        "strategy_config_hash": contract.get("strategy_config_hash", ""),
        "portfolio_config_hash": contract.get("portfolio_config_hash", ""),
        "data_window": contract.get("data_window", {}),
        "symbols": contract.get("symbols", []),
        "required_timeframes": contract.get("required_timeframes", []),
        "data_fingerprint": contract.get("data_fingerprint", {}),
        "economic_profile": contract.get("economic_profile", {}),
        "contract": contract,
        "metrics": full_metrics,
        "gate_result": gate_result,
        "final_validation": phase_result["final_validation"],
        "accepted_count": phase_result["accepted_count"],
        "new_mutations": phase_result["new_mutations"],
        "source_candidate": ADOPTED_LABEL,
        "strategy_type": strategy_id,
    }
    _metric_entry_fields(entry, full_metrics)
    return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--followup-tolerance",
        type=float,
        default=1e-6,
        help="Maximum allowed absolute delta when reconstructing follow-up IS/OOS metrics.",
    )
    parser.add_argument(
        "--allow-strategy-lineage-mutation",
        action="store_true",
        help=(
            "Legacy escape hatch. Allows this script to overwrite global "
            "output/{strategy}/round_3 lineage artifacts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _quiet_logging()
    if not args.allow_strategy_lineage_mutation:
        raise SystemExit(
            "Refusing to run legacy portfolio promotion because it mutates global "
            "strategy round_3 lineage artifacts. Keep output/portfolio/round_3 as "
            "the adopted portfolio bundle and use scripts/correct_portfolio_round3_lineage.py "
            "to verify/restabilize lineage and live-config separation. Pass "
            "--allow-strategy-lineage-mutation only for an intentional legacy rerun."
        )
    source_strategy_configs = _load_source_strategy_configs()
    source_portfolio_config = _load_source_portfolio_config()
    promoted_strategy_configs = _apply_policy_to_strategy_configs(
        source_strategy_configs,
        ADOPTED_POLICY,
    )
    promoted_portfolio_config = _portfolio_config_from_policy(
        source_portfolio_config,
        ADOPTED_POLICY,
    )

    store = ParquetStore(base_dir=DATA_DIR)
    dev_metrics, _ = _run_portfolio_window(
        portfolio_config=promoted_portfolio_config,
        strategy_configs=promoted_strategy_configs,
        start=DEV_START,
        end=DEV_END,
        store=store,
    )
    oos_metrics, _ = _run_portfolio_window(
        portfolio_config=promoted_portfolio_config,
        strategy_configs=promoted_strategy_configs,
        start=HOLDOUT_START,
        end=HOLDOUT_END,
        store=store,
    )
    followup_match = _assert_followup_metrics(
        dev_metrics=dev_metrics,
        oos_metrics=oos_metrics,
        tolerance=args.followup_tolerance,
    )
    full_metrics, full_result = _run_portfolio_window(
        portfolio_config=promoted_portfolio_config,
        strategy_configs=promoted_strategy_configs,
        start=FULL_START,
        end=FULL_END,
        store=store,
    )
    portfolio_contract = _portfolio_contract(
        portfolio_config=promoted_portfolio_config,
        strategy_configs=promoted_strategy_configs,
    )
    _write_portfolio_artifacts(
        portfolio_config=promoted_portfolio_config,
        strategy_configs=promoted_strategy_configs,
        contract=portfolio_contract,
        dev_metrics=dev_metrics,
        oos_metrics=oos_metrics,
        full_metrics=full_metrics,
        full_result=full_result,
        followup_match=followup_match,
    )

    strategy_artifact_info: dict[str, dict[str, Any]] = {}
    strategy_contracts: dict[str, dict[str, Any]] = {}
    for strategy_id in STRATEGIES:
        config = promoted_strategy_configs[strategy_id]
        is_metrics, _ = _run_strategy_window(
            strategy_id=strategy_id,
            strategy_config=config,
            start=DEV_START,
            end=DEV_END,
            store=store,
        )
        oos_strategy_metrics, _ = _run_strategy_window(
            strategy_id=strategy_id,
            strategy_config=config,
            start=HOLDOUT_START,
            end=HOLDOUT_END,
            store=store,
        )
        full_strategy_metrics, full_strategy_result = _run_strategy_window(
            strategy_id=strategy_id,
            strategy_config=config,
            start=FULL_START,
            end=FULL_END,
            store=store,
        )
        contract = _strategy_contract(
            strategy_id=strategy_id,
            strategy_config=config,
            portfolio_config=promoted_portfolio_config,
        )
        strategy_contracts[strategy_id] = contract
        strategy_artifact_info[strategy_id] = _write_strategy_artifacts(
            strategy_id=strategy_id,
            source_config=source_strategy_configs[strategy_id],
            promoted_config=config,
            portfolio_config=promoted_portfolio_config,
            contract=contract,
            is_metrics=is_metrics,
            oos_metrics=oos_strategy_metrics,
            full_metrics=full_strategy_metrics,
            full_result=full_strategy_result,
        )

    live_config_hashes = _write_live_configs(
        portfolio_config=promoted_portfolio_config,
        strategy_configs=promoted_strategy_configs,
    )
    live_portfolio_config, live_strategy_configs = _load_live_portfolio_configs()
    live_full_metrics, _ = _run_portfolio_window(
        portfolio_config=live_portfolio_config,
        strategy_configs=live_strategy_configs,
        start=FULL_START,
        end=FULL_END,
        store=store,
    )
    parity_alignment = _write_parity_alignment(
        live_config_hashes=live_config_hashes,
        portfolio_contract=portfolio_contract,
        portfolio_full_metrics=full_metrics,
        live_full_metrics=live_full_metrics,
        followup_match=followup_match,
        strategy_artifact_info=strategy_artifact_info,
    )

    _ensure_portfolio_round2_manifest_entry()
    _update_manifest(
        PORTFOLIO_BASE_DIR,
        ROUND_NUM,
        _portfolio_manifest_entry(
            contract=portfolio_contract,
            full_metrics=full_metrics,
            parity_alignment=parity_alignment,
        ),
    )
    for strategy_id in STRATEGIES:
        _update_manifest(
            ROOT / "output" / strategy_id,
            ROUND_NUM,
            _strategy_manifest_entry(
                strategy_id=strategy_id,
                contract=strategy_contracts[strategy_id],
                full_metrics=strategy_artifact_info[strategy_id]["summary"]["final_metrics"],
                mutations=strategy_artifact_info[strategy_id]["mutations"],
                parity_alignment=parity_alignment,
            ),
        )

    summary = {
        "status": "completed",
        "source_candidate": ADOPTED_LABEL,
        "portfolio_round_dir": str(PORTFOLIO_ROUND_DIR),
        "portfolio_metrics": {
            "in_sample": dev_metrics,
            "out_of_sample": oos_metrics,
            "full": full_metrics,
        },
        "followup_reconstruction": followup_match,
        "parity_alignment": parity_alignment,
        "strategy_mutations": {
            strategy_id: info["mutations"]
            for strategy_id, info in strategy_artifact_info.items()
        },
    }
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
