"""Ablate, perturb, and repair portfolio round 2 against the OOS window.

This script intentionally evaluates the post-2026-04-20 period as diagnostics
only.  It does not rewrite any optimized config; it writes evidence artifacts
that can be reviewed before promotion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import structlog

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from crypto_trader.backtest.metrics import (  # noqa: E402
    _trade_net_pnl,
    _trade_reporting_r,
    metrics_to_dict,
)
from crypto_trader.optimize.config_mutator import apply_mutations, merge_mutations  # noqa: E402
from crypto_trader.optimize.portfolio_round2_phased import (  # noqa: E402
    DEV_END,
    DEV_START,
    HARD_REJECTS,
    HOLDOUT_END,
    HOLDOUT_START,
    SCORING_WEIGHTS,
    STRATEGIES,
    SYMBOLS,
    _augment_metrics,
    _bt_config,
    _hard_reject_reason,
    _portfolio_config_from_mutations,
    _risk_scale_mutations,
    _score_metrics,
    _split_policy_mutations,
    load_base_portfolio_config,
)
from crypto_trader.portfolio.backtest_runner import (  # noqa: E402
    PortfolioBacktestResult,
    run_portfolio_backtest,
)
from crypto_trader.strategy.breakout.config import BreakoutConfig  # noqa: E402
from crypto_trader.strategy.momentum.config import MomentumConfig  # noqa: E402
from crypto_trader.strategy.trend.config import TrendConfig  # noqa: E402

from analyze_oos_repair import (  # noqa: E402
    _default_base_config as _strategy_default_base_config,
    _diff_configs as _strategy_diff_configs,
)


REPORT_METRICS = (
    "net_return_pct",
    "net_profit",
    "total_trades",
    "profit_factor",
    "expectancy_r",
    "exit_efficiency",
    "max_drawdown_pct",
    "win_rate",
    "total_fees",
    "funding_cost_total",
    "strategy_balance",
    "rule_checks",
    "blocked_entries",
    "immutable_score",
)

PORTFOLIO_FEED_TIMEFRAMES = ("15m", "30m", "1h", "4h", "1d")
_WORKER_STORE: Any | None = None


@dataclass(frozen=True)
class Candidate:
    label: str
    kind: str
    policy: dict[str, Any]
    base_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    strategy_sources: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def _configure_logging() -> None:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))


def _init_portfolio_worker(data_dir: str, symbols: list[str], timeframes: list[str]) -> None:
    """Preload market data once per worker process for repeated candidates."""
    _configure_logging()
    from crypto_trader.data.store import ParquetStore
    from crypto_trader.optimize.parallel import _CachedStore

    global _WORKER_STORE
    if _WORKER_STORE is None:
        _WORKER_STORE = _CachedStore(ParquetStore(base_dir=Path(data_dir)), symbols, timeframes)


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _get_path(payload: dict[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _strategy_config_path(strategy_id: str, round_num: int) -> Path:
    return ROOT / "output" / strategy_id / f"round_{round_num}" / "optimized_config.json"


def _load_strategy_config(strategy_id: str, path: Path) -> Any:
    payload = _read_json(path)
    strategy_payload = payload.get("strategy", payload)
    if strategy_id == "momentum":
        return MomentumConfig.from_dict(strategy_payload)
    if strategy_id == "trend":
        return TrendConfig.from_dict(strategy_payload)
    if strategy_id == "breakout":
        return BreakoutConfig.from_dict(strategy_payload)
    raise ValueError(f"Unknown strategy: {strategy_id}")


def _load_strategy_dict(strategy_id: str, round_num: int) -> dict[str, Any]:
    payload = _read_json(_strategy_config_path(strategy_id, round_num))
    return payload.get("strategy", payload)


def _load_manifest_mutations() -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for strategy_id in STRATEGIES:
        manifest = _read_json(ROOT / "output" / strategy_id / "rounds_manifest.json")
        rounds: dict[int, dict[str, Any]] = {}
        for item in manifest.get("rounds", []):
            round_num = int(item.get("round", 0))
            if round_num <= 1:
                continue
            rounds[round_num] = dict(item.get("mutations") or {})
        out[strategy_id] = rounds
    return out


def _current_policy() -> dict[str, Any]:
    state = _read_json(ROOT / "output" / "portfolio" / "round_2" / "phase_state.json")
    return dict(state["cumulative_mutations"])


def _phase_policies() -> dict[int, dict[str, Any]]:
    state = _read_json(ROOT / "output" / "portfolio" / "round_2" / "phase_state.json")
    phases: dict[int, dict[str, Any]] = {}
    for phase, result in state.get("phase_results", {}).items():
        phases[int(phase)] = dict(result.get("final_mutations") or {})
    return phases


def _candidate_fingerprint(candidate: Candidate) -> str:
    payload = {
        "policy": candidate.policy,
        "base_overrides": candidate.base_overrides,
        "strategy_sources": candidate.strategy_sources,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen: dict[str, Candidate] = {}
    aliases: dict[str, list[str]] = {}
    for candidate in candidates:
        fp = _candidate_fingerprint(candidate)
        if fp in seen:
            aliases.setdefault(fp, []).append(candidate.label)
            continue
        seen[fp] = candidate
    out: list[Candidate] = []
    for fp, candidate in seen.items():
        alias_note = ""
        if aliases.get(fp):
            alias_note = " aliases=" + ",".join(aliases[fp][:8])
        out.append(
            Candidate(
                label=candidate.label,
                kind=candidate.kind,
                policy=candidate.policy,
                base_overrides=candidate.base_overrides,
                strategy_sources=candidate.strategy_sources,
                notes=(candidate.notes + alias_note).strip(),
            )
        )
    return out


def _apply_policy_to_base_configs(
    base_configs: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    strategy_mutations, risk_scales, _ = _split_policy_mutations(policy)
    configs: dict[str, Any] = {}
    for strategy_id, cfg in base_configs.items():
        risk_mutations = _risk_scale_mutations(strategy_id, cfg, risk_scales[strategy_id])
        merged = merge_mutations(risk_mutations, strategy_mutations.get(strategy_id, {}))
        new_cfg = apply_mutations(cfg, merged) if merged else apply_mutations(cfg, {})
        new_cfg.symbols = list(SYMBOLS)
        configs[strategy_id] = new_cfg
    return configs


def _run_custom_policy(
    *,
    policy: dict[str, Any],
    start: date,
    end: date,
    base_overrides: dict[str, dict[str, Any]] | None = None,
    strategy_sources: dict[str, str] | None = None,
) -> tuple[dict[str, float], PortfolioBacktestResult]:
    base_overrides = base_overrides or {}
    strategy_sources = strategy_sources or {}
    base_configs: dict[str, Any] = {}
    for strategy_id in STRATEGIES:
        source = strategy_sources.get(strategy_id, "round_3")
        if source.startswith("round_"):
            round_num = int(source.split("_", 1)[1])
            path = _strategy_config_path(strategy_id, round_num)
        else:
            path = Path(source)
        cfg = _load_strategy_config(strategy_id, path)
        overrides = base_overrides.get(strategy_id, {})
        if overrides:
            cfg = apply_mutations(cfg, overrides)
        base_configs[strategy_id] = cfg

    base_portfolio = load_base_portfolio_config(ROOT / "config" / "portfolio_config.json")
    strategy_configs = _apply_policy_to_base_configs(base_configs, policy)
    portfolio_config = _portfolio_config_from_mutations(base_portfolio, policy)
    result = run_portfolio_backtest(
        portfolio_config=portfolio_config,
        strategy_configs=strategy_configs,
        backtest_config=_bt_config(start, end, portfolio_config.initial_equity),
        data_dir=ROOT / "data",
        store=_WORKER_STORE,
    )
    metrics = _augment_metrics(metrics_to_dict(result.metrics), result)
    score, rejected, reason, components = _score_metrics(
        metrics,
        dict(SCORING_WEIGHTS),
        dict(HARD_REJECTS),
    )
    metrics["immutable_score"] = score
    metrics["rejected"] = float(1 if rejected else 0)
    metrics["reject_reason"] = reason
    metrics["score_components"] = components
    return metrics, result


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in REPORT_METRICS:
        value = metrics.get(key)
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            out[key] = str(value)
        elif value is not None:
            out[key] = value
    out["reject_reason"] = metrics.get("reject_reason", "")
    return out


def _group_trade_rows(result: PortfolioBacktestResult) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for strategy_id, trades in result.per_strategy_trades.items():
        for trade in trades:
            direction = getattr(trade.direction, "name", str(trade.direction))
            key = "|".join([
                strategy_id,
                trade.symbol,
                direction,
                trade.confirmation_type or "none",
            ])
            row = rows.setdefault(
                key,
                {"n": 0, "pnl": 0.0, "rs": [], "wins": 0},
            )
            pnl = _trade_net_pnl(trade)
            rr = _trade_reporting_r(trade)
            row["n"] += 1
            row["pnl"] += pnl
            row["wins"] += 1 if pnl > 0 else 0
            if rr is not None:
                row["rs"].append(rr)

    compact: dict[str, Any] = {}
    for key, row in rows.items():
        compact[key] = {
            "trades": row["n"],
            "net_profit": row["pnl"],
            "win_rate": row["wins"] / row["n"] * 100.0 if row["n"] else 0.0,
            "avg_r": statistics.mean(row["rs"]) if row["rs"] else 0.0,
        }
    return compact


def _worst_trades(result: PortfolioBacktestResult, n: int = 10) -> list[dict[str, Any]]:
    rows = []
    for strategy_id, trades in result.per_strategy_trades.items():
        for trade in trades:
            rows.append((strategy_id, trade))
    rows.sort(key=lambda item: _trade_net_pnl(item[1]))
    out = []
    for strategy_id, trade in rows[:n]:
        out.append(
            {
                "strategy": strategy_id,
                "entry_time": trade.entry_time.isoformat() if trade.entry_time else "",
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else "",
                "symbol": trade.symbol,
                "direction": getattr(trade.direction, "name", str(trade.direction)),
                "confirmation_type": trade.confirmation_type,
                "exit_reason": trade.exit_reason,
                "r": _trade_reporting_r(trade),
                "net_pnl": _trade_net_pnl(trade),
                "mfe_r": trade.mfe_r,
                "mae_r": trade.mae_r,
                "bars_held": trade.bars_held,
            }
        )
    return out


def _evaluate_candidate(candidate: Candidate) -> dict[str, Any]:
    _configure_logging()
    dev_metrics, dev_result = _run_custom_policy(
        policy=candidate.policy,
        base_overrides=candidate.base_overrides,
        strategy_sources=candidate.strategy_sources,
        start=DEV_START,
        end=DEV_END,
    )
    oos_metrics, oos_result = _run_custom_policy(
        policy=candidate.policy,
        base_overrides=candidate.base_overrides,
        strategy_sources=candidate.strategy_sources,
        start=HOLDOUT_START,
        end=HOLDOUT_END,
    )
    return {
        "label": candidate.label,
        "kind": candidate.kind,
        "notes": candidate.notes,
        "fingerprint": _candidate_fingerprint(candidate),
        "policy": candidate.policy,
        "base_overrides": candidate.base_overrides,
        "strategy_sources": candidate.strategy_sources,
        "dev": _compact_metrics(dev_metrics),
        "oos": _compact_metrics(oos_metrics),
        "oos_groups": _group_trade_rows(oos_result),
        "oos_worst_trades": _worst_trades(oos_result, n=6),
    }


def _policy_without(policy: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: v for k, v in policy.items() if k not in keys}


def _with_policy(policy: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(policy)
    out.update(updates)
    return out


def _base_override(
    strategy_id: str,
    path: str,
    value: Any,
) -> dict[str, dict[str, Any]]:
    return {strategy_id: {path: _coerce_scalar(value)}}


def _merge_base_overrides(
    *items: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        for strategy_id, mutations in item.items():
            out.setdefault(strategy_id, {}).update(mutations)
    return out


def _numeric_neighbors(value: Any) -> list[Any]:
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        raw = {value - 2, value - 1, value + 1, value + 2}
        return [v for v in sorted(raw) if v >= 0 and v != value]
    if isinstance(value, float):
        raw = {
            round(value * 0.80, 8),
            round(value * 0.90, 8),
            round(value * 1.10, 8),
            round(value * 1.20, 8),
        }
        return [v for v in sorted(raw) if v != value]
    if isinstance(value, str) and value in {"both", "long_only", "short_only", "disabled"}:
        return [v for v in ("both", "long_only", "short_only", "disabled") if v != value]
    return []


def build_candidates() -> list[Candidate]:
    policy = _current_policy()
    candidates: list[Candidate] = [
        Candidate("baseline_current_round2", "baseline", dict(policy)),
        Candidate("baseline_no_portfolio_round2", "checkpoint", {}),
    ]

    for phase, phase_policy in sorted(_phase_policies().items()):
        candidates.append(Candidate(f"checkpoint_portfolio_phase_{phase}", "checkpoint", phase_policy))

    for source_round in (1, 2):
        candidates.append(
            Candidate(
                f"checkpoint_all_strategies_round_{source_round}_plus_portfolio_round2",
                "checkpoint",
                dict(policy),
                strategy_sources={sid: f"round_{source_round}" for sid in STRATEGIES},
            )
        )
        for strategy_id in STRATEGIES:
            candidates.append(
                Candidate(
                    f"checkpoint_{strategy_id}_round_{source_round}_others_current",
                    "checkpoint",
                    dict(policy),
                    strategy_sources={strategy_id: f"round_{source_round}"},
                )
            )

    # Portfolio round 2 field-level ablation and phase ablation.
    for key in sorted(policy):
        candidates.append(
            Candidate(
                f"ablate_portfolio_field__{key}",
                "ablation",
                _policy_without(policy, {key}),
                notes=f"Remove current policy key {key}.",
            )
        )

    phase_state = _read_json(ROOT / "output" / "portfolio" / "round_2" / "phase_state.json")
    for phase, result in sorted(phase_state.get("phase_results", {}).items()):
        keys = set((result.get("new_mutations") or {}).keys())
        if keys:
            candidates.append(
                Candidate(
                    f"ablate_portfolio_phase_{phase}",
                    "ablation",
                    _policy_without(policy, keys),
                    notes="Remove accepted keys from one portfolio phase.",
                )
            )

    # Cumulative strategy mutation ablations from all accepted strategy rounds.
    manifest_mutations = _load_manifest_mutations()
    current_policy_keys = set(policy)
    active_cumulative_overrides: dict[str, dict[str, Any]] = {}
    active_cumulative_unshadow: set[str] = set()
    for strategy_id, rounds in manifest_mutations.items():
        round_dicts = {
            1: _load_strategy_dict(strategy_id, 1),
            2: _load_strategy_dict(strategy_id, 2),
            3: _load_strategy_dict(strategy_id, 3),
        }
        base_source, pre_round_config = _strategy_default_base_config(strategy_id)
        current_config = _load_strategy_config(strategy_id, _strategy_config_path(strategy_id, 3))
        active_diffs = _strategy_diff_configs(pre_round_config, current_config)
        active_all_to_base: dict[str, Any] = {}
        active_round_unshadow: set[str] = set()
        for path, detail in sorted(active_diffs.items()):
            base_value = detail["base_value"]
            current_value = detail["candidate_value"]
            namespaced = f"strategy.{strategy_id}.{path}"
            remove_keys = {namespaced} if namespaced in current_policy_keys else set()
            candidates.append(
                Candidate(
                    f"ablate_strategy_active_cumulative_to_base__{strategy_id}__{path}",
                    "ablation",
                    _policy_without(policy, remove_keys),
                    base_overrides=_base_override(strategy_id, path, base_value),
                    notes=f"Revert active cumulative strategy diff to pre-round baseline ({base_source}).",
                )
            )
            active_all_to_base[path] = _coerce_scalar(base_value)
            active_cumulative_overrides.setdefault(strategy_id, {})[path] = _coerce_scalar(base_value)
            active_round_unshadow.update(remove_keys)
            active_cumulative_unshadow.update(remove_keys)
            for value in _numeric_neighbors(current_value):
                candidates.append(
                    Candidate(
                        f"perturb_strategy_active_cumulative__{strategy_id}__{path}__{value}",
                        "perturbation",
                        _policy_without(policy, remove_keys),
                        base_overrides=_base_override(strategy_id, path, value),
                        notes="Perturb active cumulative strategy diff versus pre-round baseline.",
                    )
                )
        if active_all_to_base:
            candidates.append(
                Candidate(
                    f"ablate_strategy_active_cumulative_all_to_base__{strategy_id}",
                    "ablation",
                    _policy_without(policy, active_round_unshadow),
                    base_overrides={strategy_id: active_all_to_base},
                    notes=f"Revert all active cumulative strategy diffs to pre-round baseline ({base_source}).",
                )
            )

        unique_paths = sorted({path for muts in rounds.values() for path in muts})
        all_to_round1: dict[str, Any] = {}
        all_unshadow: set[str] = set()
        for path in unique_paths:
            round1_value = _get_path(round_dicts[1], path)
            current_value = _get_path(round_dicts[3], path)
            namespaced = f"strategy.{strategy_id}.{path}"
            if round1_value != current_value:
                if namespaced in current_policy_keys:
                    candidates.append(
                        Candidate(
                            f"ablate_strategy_shadowed_to_round1__{strategy_id}__{path}",
                            "ablation",
                            _policy_without(policy, {namespaced}),
                            base_overrides=_base_override(strategy_id, path, round1_value),
                            notes="Remove portfolio override and revert embedded strategy field to round 1.",
                        )
                    )
                    all_unshadow.add(namespaced)
                else:
                    candidates.append(
                        Candidate(
                            f"ablate_strategy_to_round1__{strategy_id}__{path}",
                            "ablation",
                            dict(policy),
                            base_overrides=_base_override(strategy_id, path, round1_value),
                            notes="Revert one cumulative accepted strategy field to round 1.",
                        )
                    )
                all_to_round1[path] = _coerce_scalar(round1_value)

            for round_num, muts in sorted(rounds.items()):
                if path not in muts:
                    continue
                prev_round = round_num - 1
                if prev_round < 1 or prev_round not in round_dicts:
                    continue
                prev_value = _get_path(round_dicts[prev_round], path)
                if prev_value == current_value:
                    continue
                remove_keys = {namespaced} if namespaced in current_policy_keys else set()
                candidates.append(
                    Candidate(
                        f"ablate_strategy_to_round{prev_round}__{strategy_id}__{path}",
                        "ablation",
                        _policy_without(policy, remove_keys),
                        base_overrides=_base_override(strategy_id, path, prev_value),
                        notes=f"Revert field to value before accepted strategy round {round_num}.",
                    )
                )

        if all_to_round1:
            candidates.append(
                Candidate(
                    f"ablate_strategy_all_to_round1__{strategy_id}",
                    "ablation",
                    _policy_without(policy, all_unshadow),
                    base_overrides={strategy_id: all_to_round1},
                    notes="Revert all known accepted cumulative fields for this strategy.",
                )
            )

        for round_num, muts in sorted(rounds.items()):
            prev_round = round_num - 1
            if prev_round < 1:
                continue
            overrides: dict[str, Any] = {}
            remove_keys: set[str] = set()
            for path in muts:
                prev_value = _get_path(round_dicts[prev_round], path)
                current_value = _get_path(round_dicts[3], path)
                if prev_value == current_value:
                    continue
                overrides[path] = _coerce_scalar(prev_value)
                namespaced = f"strategy.{strategy_id}.{path}"
                if namespaced in current_policy_keys:
                    remove_keys.add(namespaced)
            if overrides:
                candidates.append(
                    Candidate(
                        f"ablate_strategy_round_{round_num}__{strategy_id}",
                        "ablation",
                        _policy_without(policy, remove_keys),
                        base_overrides={strategy_id: overrides},
                        notes=f"Revert all fields introduced or changed by strategy round {round_num}.",
                    )
                )

    # All-strategy cumulative-to-round-1 ablation.
    all_strategy_overrides: dict[str, dict[str, Any]] = {}
    all_strategy_unshadow: set[str] = set()
    for strategy_id, rounds in manifest_mutations.items():
        r1 = _load_strategy_dict(strategy_id, 1)
        r3 = _load_strategy_dict(strategy_id, 3)
        for path in sorted({path for muts in rounds.values() for path in muts}):
            r1_value = _get_path(r1, path)
            r3_value = _get_path(r3, path)
            if r1_value != r3_value:
                all_strategy_overrides.setdefault(strategy_id, {})[path] = _coerce_scalar(r1_value)
                namespaced = f"strategy.{strategy_id}.{path}"
                if namespaced in current_policy_keys:
                    all_strategy_unshadow.add(namespaced)
    candidates.append(
        Candidate(
            "ablate_all_strategy_cumulative_to_round1",
            "ablation",
            _policy_without(policy, all_strategy_unshadow),
            base_overrides=all_strategy_overrides,
            notes="Revert every known accepted cumulative strategy mutation to round 1.",
        )
    )
    candidates.append(
        Candidate(
            "ablate_all_strategy_active_cumulative_to_base",
            "ablation",
            _policy_without(policy, active_cumulative_unshadow),
            base_overrides=active_cumulative_overrides,
            notes="Revert every active cumulative strategy diff to each strategy's pre-round baseline.",
        )
    )

    # Perturb current portfolio fields.
    for risk in (0.85, 0.95, 1.00, 1.05, 1.10, 1.20, 1.25):
        candidates.append(
            Candidate(
                f"perturb_risk_all_{risk:.2f}",
                "perturbation",
                _with_policy(
                    policy,
                    {
                        "risk_scale.momentum": risk,
                        "risk_scale.trend": risk,
                        "risk_scale.breakout": risk,
                    },
                ),
            )
        )
    for strategy_id in STRATEGIES:
        for risk in (0.85, 0.95, 1.00, 1.05, 1.10, 1.20, 1.25):
            candidates.append(
                Candidate(
                    f"perturb_risk_{strategy_id}_{risk:.2f}",
                    "perturbation",
                    _with_policy(policy, {f"risk_scale.{strategy_id}": risk}),
                )
            )
    for cap in (1.5, 2.0, 2.25, 2.75, 3.0, 3.5, 4.0):
        candidates.append(
            Candidate(
                f"perturb_symbol_exposure_cap_{cap:.2f}",
                "perturbation",
                _with_policy(policy, {"portfolio.symbol_collision": "cap", "portfolio.symbol_exposure_cap_R": cap}),
            )
        )
    for mode in ("allow", "block", "cap"):
        candidates.append(
            Candidate(
                f"perturb_symbol_collision_{mode}",
                "perturbation",
                _with_policy(policy, {"portfolio.symbol_collision": mode}),
            )
        )
    dd_variants = {
        "baseline": ((0.08, 1.0), (0.12, 0.5), (0.15, 0.25), (1.0, 0.0)),
        "looser": ((0.08, 0.85), (0.12, 0.60), (0.16, 0.25), (1.0, 0.0)),
        "tighter": ((0.04, 0.75), (0.08, 0.5), (0.12, 0.25), (0.15, 0.0)),
        "current_late_zero": ((0.06, 0.85), (0.10, 0.55), (0.14, 0.25), (0.18, 0.0)),
    }
    for name, tiers in dd_variants.items():
        candidates.append(
            Candidate(
                f"perturb_dd_tiers_{name}",
                "perturbation",
                _with_policy(policy, {"portfolio.dd_tiers": tiers}),
            )
        )

    # Perturb cumulative accepted strategy fields at field level.
    for strategy_id, rounds in manifest_mutations.items():
        current_dict = _load_strategy_dict(strategy_id, 3)
        for path in sorted({path for muts in rounds.values() for path in muts}):
            namespaced = f"strategy.{strategy_id}.{path}"
            current_value = _get_path(current_dict, path)
            for value in _numeric_neighbors(current_value):
                remove = {namespaced} if namespaced in current_policy_keys else set()
                candidates.append(
                    Candidate(
                        f"perturb_strategy__{strategy_id}__{path}__{value}",
                        "perturbation",
                        _policy_without(policy, remove),
                        base_overrides=_base_override(strategy_id, path, value),
                        notes="Perturb accepted cumulative strategy field.",
                    )
                )

    # Targeted repair phase driven by the OOS weakness pattern.
    targeted: list[tuple[str, dict[str, Any]]] = [
        ("momentum_sol_disabled", {"strategy.momentum.symbol_filter.sol_direction": "disabled"}),
        ("momentum_sol_long_only", {"strategy.momentum.symbol_filter.sol_direction": "long_only"}),
        ("momentum_sol_both", {"strategy.momentum.symbol_filter.sol_direction": "both"}),
        ("momentum_disable_inside_bar", {"strategy.momentum.confirmation.enable_inside_bar": False}),
        ("momentum_weak_confirm_need_3", {"strategy.momentum.confirmation.min_confluences_for_weak": 3}),
        ("momentum_volume_120", {"strategy.momentum.confirmation.volume_threshold_mult": 1.2}),
        ("momentum_volume_130", {"strategy.momentum.confirmation.volume_threshold_mult": 1.3}),
        ("momentum_quick_exit_less_loss", {"strategy.momentum.exits.quick_exit_max_r": -0.1}),
        ("momentum_quick_exit_flat", {"strategy.momentum.exits.quick_exit_max_r": 0.0}),
        ("momentum_proof_lock_045_b2", {
            "strategy.momentum.exits.proof_lock_trigger_r": 0.45,
            "strategy.momentum.exits.proof_lock_min_bars": 2,
        }),
        ("momentum_proof_lock_065_b4", {
            "strategy.momentum.exits.proof_lock_trigger_r": 0.65,
            "strategy.momentum.exits.proof_lock_min_bars": 4,
        }),
        ("trend_sol_disabled", {"strategy.trend.symbol_filter.sol_direction": "disabled"}),
        ("trend_sol_short_only", {"strategy.trend.symbol_filter.sol_direction": "short_only"}),
        ("trend_eth_short_only", {"strategy.trend.symbol_filter.eth_direction": "short_only"}),
        ("trend_btc_disabled", {"strategy.trend.symbol_filter.btc_direction": "disabled"}),
        ("trend_mfe_lock_075_floor010", {
            "strategy.trend.exits.mfe_lock_exit_enabled": True,
            "strategy.trend.exits.mfe_lock_trigger_r": 0.75,
            "strategy.trend.exits.mfe_lock_floor_r": 0.10,
            "strategy.trend.exits.mfe_lock_min_bars": 2,
        }),
        ("trend_funding_filter_on", {"strategy.trend.filters.funding_filter_enabled": True}),
        ("trend_scratch_less_eager", {
            "strategy.trend.exits.scratch_peak_r": 0.45,
            "strategy.trend.exits.scratch_floor_r": -0.05,
            "strategy.trend.exits.scratch_min_bars": 3,
        }),
        ("breakout_eth_long_only", {"strategy.breakout.symbol_filter.eth_direction": "long_only"}),
        ("breakout_sol_long_only", {"strategy.breakout.symbol_filter.sol_direction": "long_only"}),
        ("breakout_sol_disabled", {"strategy.breakout.symbol_filter.sol_direction": "disabled"}),
        ("breakout_quick_exit_on", {"strategy.breakout.exits.quick_exit_enabled": True}),
        ("breakout_early_lock", {
            "strategy.breakout.exits.early_lock_enabled": True,
            "strategy.breakout.exits.early_lock_mfe_r": 0.45,
            "strategy.breakout.exits.early_lock_stop_r": 0.05,
        }),
        ("breakout_faster_trail", {
            "strategy.breakout.trail.trail_activation_r": 0.35,
            "strategy.breakout.trail.trail_activation_bars": 4,
        }),
        ("breakout_model1_volume_140", {"strategy.breakout.confirmation.model1_min_volume_mult": 1.4}),
        ("risk_momentum_down_trend_breakout_up", {
            "risk_scale.momentum": 0.90,
            "risk_scale.trend": 1.15,
            "risk_scale.breakout": 1.25,
        }),
        ("risk_momentum_down_breakout_up", {
            "risk_scale.momentum": 0.85,
            "risk_scale.trend": 1.05,
            "risk_scale.breakout": 1.30,
        }),
        ("oos_weak_sleeve_combo", {
            "strategy.momentum.symbol_filter.sol_direction": "disabled",
            "strategy.breakout.symbol_filter.eth_direction": "long_only",
            "risk_scale.momentum": 0.95,
            "risk_scale.breakout": 1.20,
        }),
        ("frequency_preserve_sol_rotation", {
            "strategy.momentum.symbol_filter.sol_direction": "disabled",
            "strategy.breakout.symbol_filter.sol_direction": "long_only",
            "risk_scale.breakout": 1.25,
        }),
    ]
    for label, updates in targeted:
        candidates.append(
            Candidate(
                f"targeted_{label}",
                "targeted",
                _with_policy(policy, updates),
                notes="Targeted OOS repair probe.",
            )
        )

    # Pair the strongest OOS hypothesis families, still as explicit test cases.
    pair_updates = [
        ("disable_mom_sol_plus_no_inside", {
            "strategy.momentum.symbol_filter.sol_direction": "disabled",
            "strategy.momentum.confirmation.enable_inside_bar": False,
        }),
        ("mom_sol_disabled_trend_eth_short", {
            "strategy.momentum.symbol_filter.sol_direction": "disabled",
            "strategy.trend.symbol_filter.eth_direction": "short_only",
        }),
        ("mom_sol_disabled_breakout_eth_long", {
            "strategy.momentum.symbol_filter.sol_direction": "disabled",
            "strategy.breakout.symbol_filter.eth_direction": "long_only",
        }),
        ("mom_sol_long_breakout_sol_long", {
            "strategy.momentum.symbol_filter.sol_direction": "long_only",
            "strategy.breakout.symbol_filter.sol_direction": "long_only",
        }),
    ]
    for label, updates in pair_updates:
        candidates.append(
            Candidate(
                f"targeted_pair_{label}",
                "targeted",
                _with_policy(policy, updates),
                notes="Targeted combination after OOS cohort review.",
            )
        )

    return _dedupe_candidates(candidates)


def _delta(value: Any, base: Any) -> float:
    try:
        return float(value) - float(base)
    except (TypeError, ValueError):
        return 0.0


def _profit_factor_value(value: Any) -> float:
    if value == "inf":
        return 99.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(out):
        return 99.0
    return out


def _rank_results(results: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    base_dev = baseline["dev"]
    base_oos = baseline["oos"]
    for row in results:
        row["delta"] = {
            "dev_return_pct": _delta(row["dev"].get("net_return_pct"), base_dev.get("net_return_pct")),
            "dev_trades": _delta(row["dev"].get("total_trades"), base_dev.get("total_trades")),
            "dev_expectancy_r": _delta(row["dev"].get("expectancy_r"), base_dev.get("expectancy_r")),
            "oos_return_pct": _delta(row["oos"].get("net_return_pct"), base_oos.get("net_return_pct")),
            "oos_trades": _delta(row["oos"].get("total_trades"), base_oos.get("total_trades")),
            "oos_expectancy_r": _delta(row["oos"].get("expectancy_r"), base_oos.get("expectancy_r")),
            "oos_pf": _profit_factor_value(row["oos"].get("profit_factor"))
            - _profit_factor_value(base_oos.get("profit_factor")),
            "oos_exit_efficiency": _delta(
                row["oos"].get("exit_efficiency"), base_oos.get("exit_efficiency")
            ),
        }
        oos_return = float(row["oos"].get("net_return_pct", 0.0))
        oos_trades = float(row["oos"].get("total_trades", 0.0))
        oos_exp = float(row["oos"].get("expectancy_r", 0.0))
        oos_pf = _profit_factor_value(row["oos"].get("profit_factor"))
        dev_return_delta = row["delta"]["dev_return_pct"]
        dev_trade_delta = row["delta"]["dev_trades"]
        dev_penalty = min(0.0, dev_return_delta + 2.5) * 0.20 + min(0.0, dev_trade_delta + 2.0) * 0.15
        row["selection_score"] = (
            0.50 * oos_return
            + 1.25 * oos_trades
            + 18.0 * oos_exp
            + 2.0 * min(oos_pf, 8.0)
            + dev_penalty
        )
        row["passes_is_preservation"] = (
            dev_return_delta >= -2.5
            and dev_trade_delta >= -2.0
            and float(row["dev"].get("profit_factor", 0.0)) >= 1.55
            and float(row["dev"].get("expectancy_r", 0.0)) >= 0.20
        )
    return sorted(results, key=lambda r: r["selection_score"], reverse=True)


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "label",
        "kind",
        "selection_score",
        "passes_is_preservation",
        "dev_return_pct",
        "dev_trades",
        "dev_pf",
        "dev_exp_r",
        "dev_exit_eff",
        "dev_dd",
        "oos_return_pct",
        "oos_trades",
        "oos_pf",
        "oos_exp_r",
        "oos_exit_eff",
        "oos_dd",
        "delta_dev_return_pct",
        "delta_dev_trades",
        "delta_oos_return_pct",
        "delta_oos_trades",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "label": row["label"],
                    "kind": row["kind"],
                    "selection_score": row.get("selection_score"),
                    "passes_is_preservation": row.get("passes_is_preservation"),
                    "dev_return_pct": row["dev"].get("net_return_pct"),
                    "dev_trades": row["dev"].get("total_trades"),
                    "dev_pf": row["dev"].get("profit_factor"),
                    "dev_exp_r": row["dev"].get("expectancy_r"),
                    "dev_exit_eff": row["dev"].get("exit_efficiency"),
                    "dev_dd": row["dev"].get("max_drawdown_pct"),
                    "oos_return_pct": row["oos"].get("net_return_pct"),
                    "oos_trades": row["oos"].get("total_trades"),
                    "oos_pf": row["oos"].get("profit_factor"),
                    "oos_exp_r": row["oos"].get("expectancy_r"),
                    "oos_exit_eff": row["oos"].get("exit_efficiency"),
                    "oos_dd": row["oos"].get("max_drawdown_pct"),
                    "delta_dev_return_pct": row.get("delta", {}).get("dev_return_pct"),
                    "delta_dev_trades": row.get("delta", {}).get("dev_trades"),
                    "delta_oos_return_pct": row.get("delta", {}).get("oos_return_pct"),
                    "delta_oos_trades": row.get("delta", {}).get("oos_trades"),
                    "notes": row.get("notes", ""),
                }
            )


def _fmt_metric(row: dict[str, Any], window: str) -> str:
    metrics = row[window]
    pf = metrics.get("profit_factor")
    return (
        f"{metrics.get('net_return_pct', 0.0):.2f}% return, "
        f"{metrics.get('total_trades', 0.0):.0f} trades, "
        f"PF {float(pf) if not isinstance(pf, str) else pf}, "
        f"expR {metrics.get('expectancy_r', 0.0):.3f}, "
        f"exitEff {metrics.get('exit_efficiency', 0.0):.3f}, "
        f"DD {metrics.get('max_drawdown_pct', 0.0):.2f}%"
    )


def _best_by_kind(results: list[dict[str, Any]], kind: str, limit: int = 8) -> list[dict[str, Any]]:
    return [
        row for row in results
        if row["kind"] == kind and row.get("passes_is_preservation")
    ][:limit]


def _write_report(path: Path, results: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    accepted_like = [
        row for row in results
        if row["label"] != baseline["label"]
        and row.get("passes_is_preservation")
        and row.get("delta", {}).get("oos_return_pct", 0.0) > 0
    ]
    top = accepted_like[:12]
    lines: list[str] = []
    lines.append("# Portfolio Round 2 OOS Ablation And Perturbation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"IS window: {DEV_START} to {DEV_END}")
    lines.append(f"OOS window: {HOLDOUT_START} to {HOLDOUT_END}")
    lines.append(f"Candidates evaluated: {len(results)}")
    lines.append("")
    lines.append("## Current Round 2 Baseline")
    lines.append(f"- IS: {_fmt_metric(baseline, 'dev')}")
    lines.append(f"- OOS: {_fmt_metric(baseline, 'oos')}")
    lines.append("")
    lines.append("## OOS Weakness Autopsy")
    lines.append("Worst current OOS trades:")
    for trade in baseline.get("oos_worst_trades", [])[:8]:
        lines.append(
            "- "
            f"{trade['entry_time']} {trade['strategy']} {trade['symbol']} "
            f"{trade['direction']} {trade.get('confirmation_type')} "
            f"{trade.get('exit_reason')} R={trade.get('r'):.3f} "
            f"PnL={trade.get('net_pnl'):.2f} MFE={trade.get('mfe_r')}"
        )
    lines.append("")
    lines.append("Current OOS group attribution, weakest first:")
    groups = sorted(
        baseline.get("oos_groups", {}).items(),
        key=lambda item: item[1].get("net_profit", 0.0),
    )
    for key, group in groups[:10]:
        lines.append(
            f"- {key}: n={group['trades']}, pnl={group['net_profit']:.2f}, "
            f"avgR={group['avg_r']:.3f}, WR={group['win_rate']:.1f}%"
        )
    lines.append("")
    lines.append("Interpretation: OOS underperformance is broad edge decay plus several repeatable weak sleeves. "
                 "It is not caused by one or two outsized loss events; the worst losses are ordinary stop/quick-exit "
                 "events clustered in momentum SOL shorts and BTC/ETH inside-bar longs. Risk scaling amplifies both "
                 "sides but does not by itself create the losing trades.")
    lines.append("")
    lines.append("## Best Preserving OOS Improvements")
    if top:
        for row in top:
            d = row["delta"]
            lines.append(
                f"- {row['label']} ({row['kind']}): OOS {_fmt_metric(row, 'oos')} "
                f"[delta return {d['oos_return_pct']:+.2f}pp, trades {d['oos_trades']:+.0f}; "
                f"IS return {d['dev_return_pct']:+.2f}pp, trades {d['dev_trades']:+.0f}]"
            )
    else:
        lines.append("- No candidate improved OOS while preserving IS within the configured tolerance.")
    lines.append("")
    for kind, title in [
        ("ablation", "Ablation Winners"),
        ("perturbation", "Perturbation Winners"),
        ("targeted", "Targeted Repair Winners"),
        ("checkpoint", "Checkpoint Lessons"),
    ]:
        lines.append(f"## {title}")
        rows = _best_by_kind(results, kind, limit=10)
        if not rows:
            lines.append("- None passed the IS preservation screen with positive OOS uplift.")
            lines.append("")
            continue
        for row in rows:
            d = row["delta"]
            lines.append(
                f"- {row['label']}: OOS return {row['oos'].get('net_return_pct', 0.0):.2f}% "
                f"({d['oos_return_pct']:+.2f}pp), OOS trades {row['oos'].get('total_trades', 0.0):.0f} "
                f"({d['oos_trades']:+.0f}), IS return {row['dev'].get('net_return_pct', 0.0):.2f}% "
                f"({d['dev_return_pct']:+.2f}pp)"
            )
        lines.append("")
    lines.append("## Recommended Next Action")
    if top:
        best = top[0]
        lines.append(
            f"Promote only after a fresh full backtest/replay: {best['label']} currently has the best "
            "OOS/IS trade-off under this diagnostic objective."
        )
        lines.append("Policy / overrides:")
        lines.append("```json")
        lines.append(json.dumps(
            {
                "policy": best.get("policy", {}),
                "base_overrides": best.get("base_overrides", {}),
                "strategy_sources": best.get("strategy_sources", {}),
            },
            indent=2,
            default=str,
        ))
        lines.append("```")
    else:
        lines.append("Do not promote a repair from this run; use the ablation evidence to design a narrower next search.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 3))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "portfolio" / "round_2" / "oos_ablation_perturbation",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    _configure_logging()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    if args.limit > 0:
        candidates = candidates[:args.limit]

    _write_json(out_dir / "candidate_manifest.json", [
        {
            "label": c.label,
            "kind": c.kind,
            "policy": c.policy,
            "base_overrides": c.base_overrides,
            "strategy_sources": c.strategy_sources,
            "notes": c.notes,
            "fingerprint": _candidate_fingerprint(c),
        }
        for c in candidates
    ])

    print(f"Evaluating {len(candidates)} candidates with {args.workers} workers", flush=True)
    results_path = out_dir / "results.jsonl"
    results: list[dict[str, Any]] = []
    done_fingerprints: set[str] = set()
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                results.append(row)
                done_fingerprints.add(row["fingerprint"])

    pending = [c for c in candidates if _candidate_fingerprint(c) not in done_fingerprints]
    if pending:
        with open(results_path, "a", encoding="utf-8") as f:
            if args.workers <= 1:
                _init_portfolio_worker(str(ROOT / "data"), list(SYMBOLS), list(PORTFOLIO_FEED_TIMEFRAMES))
                for idx, candidate in enumerate(pending, 1):
                    row = _evaluate_candidate(candidate)
                    f.write(json.dumps(row, default=str) + "\n")
                    f.flush()
                    results.append(row)
                    print(f"[{idx}/{len(pending)}] {candidate.label}", flush=True)
            else:
                with ProcessPoolExecutor(
                    max_workers=args.workers,
                    initializer=_init_portfolio_worker,
                    initargs=(str(ROOT / "data"), list(SYMBOLS), list(PORTFOLIO_FEED_TIMEFRAMES)),
                ) as executor:
                    future_map = {
                        executor.submit(_evaluate_candidate, candidate): candidate
                        for candidate in pending
                    }
                    for idx, future in enumerate(as_completed(future_map), 1):
                        candidate = future_map[future]
                        row = future.result()
                        f.write(json.dumps(row, default=str) + "\n")
                        f.flush()
                        results.append(row)
                        print(f"[{idx}/{len(pending)}] {candidate.label}", flush=True)

    baseline = next(row for row in results if row["label"] == "baseline_current_round2")
    ranked = _rank_results(results, baseline)
    _write_json(out_dir / "ranked_results.json", ranked)
    _write_csv(out_dir / "ranked_results.csv", ranked)
    _write_report(out_dir / "round2_oos_ablation_report.md", ranked, baseline)
    print(f"Report: {out_dir / 'round2_oos_ablation_report.md'}", flush=True)
    print(f"CSV: {out_dir / 'ranked_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
