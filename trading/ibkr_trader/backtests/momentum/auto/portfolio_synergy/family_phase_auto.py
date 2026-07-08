from __future__ import annotations

import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from backtests.momentum.engine.family_portfolio_engine import (
    FamilyPortfolioBacktestConfig,
    FamilyPortfolioBacktester,
    FamilyPortfolioReplayBundle,
    MOMENTUM_FAMILY_STRATEGY_IDS,
    build_family_replay_bundle,
    family_config_to_dict,
    make_controlled_aggressive_family_config,
    update_allocation,
)
from backtests.shared.auto.round_manager import RoundManager


_PRICE_BARS_UNSET = object()

SCORE_WEIGHTS: dict[str, float] = {
    "expected_return": 0.24,
    "trade_frequency": 0.18,
    "drawdown_control": 0.18,
    "profit_quality": 0.13,
    "risk_efficiency": 0.12,
    "strategy_balance": 0.10,
    "live_rule_health": 0.05,
}

TARGETS = {
    "net_profit": 220_000.0,
    "trades_per_month": 40.0,
    "max_drawdown_pct": 0.18,
    "profit_factor": 2.8,
    "calmar": 8.0,
    "min_strategy_trades": 80.0,
    "max_block_rate": 0.15,
}


@dataclass(frozen=True)
class PortfolioCandidate:
    name: str
    description: str
    mutations: dict[str, Any]


@dataclass
class PortfolioEvaluation:
    name: str
    score: float
    rejected: bool
    reject_reason: str
    soft_warnings: list[str]
    components: dict[str, float]
    metrics: dict[str, float]
    config: FamilyPortfolioBacktestConfig


def run_family_phase_auto(
    *,
    trades_by_strategy: dict[str, list],
    output_dir: Path,
    initial_equity: float = 50_000.0,
    max_workers: int = 2,
    min_delta: float = 0.00001,
    data_dir: Path = Path("backtests/momentum/data/raw"),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_bundle = build_family_replay_bundle(trades_by_strategy)
    price_bars = _load_mtm_price_bars_for_scoring(data_dir)
    current_config = make_controlled_aggressive_family_config(initial_equity)
    current_eval = evaluate_portfolio_config(
        "BASELINE",
        current_config,
        replay_bundle,
        price_bars=price_bars,
    )
    phase_records: list[dict[str, Any]] = []

    _write_json(output_dir / "baseline.json", _evaluation_record(current_eval))
    for phase in sorted(PHASE_CANDIDATES):
        candidates = PHASE_CANDIDATES[phase]
        evaluations = _evaluate_candidates(
            current_config,
            candidates,
            replay_bundle,
            max_workers=max_workers,
            price_bars=price_bars,
        )
        viable = [item for item in evaluations if not item.rejected]
        best = max(viable, key=_portfolio_selection_key, default=None)
        accepted = bool(best and best.score > current_eval.score + min_delta)
        if accepted and best is not None:
            current_config = best.config
            current_eval = best
        record = {
            "phase": phase,
            "accepted": accepted,
            "accepted_candidate": best.name if accepted and best is not None else None,
            "current_score": current_eval.score,
            "evaluations": [_evaluation_record(item) for item in evaluations],
        }
        phase_records.append(record)
        _write_json(output_dir / f"phase_{phase}.json", record)

    final_result = FamilyPortfolioBacktester(current_config).run_bundle(replay_bundle)
    final_metric_package = headline_mtm_metric_package(
        current_config,
        final_result,
        data_dir=data_dir,
        price_bars=price_bars,
    )
    final_eval = score_metrics(final_metric_package["headline_metrics"])
    summary = {
        "score_components": SCORE_WEIGHTS,
        "score_component_count": len(SCORE_WEIGHTS),
        "targets": TARGETS,
        "initial_equity": initial_equity,
        "max_workers": max_workers,
        "min_delta": min_delta,
        "replay_architecture": final_result.replay_architecture,
        "replay_source_fingerprint": replay_bundle.source_fingerprint,
        "trade_outcome_count": len(replay_bundle.trade_outcomes),
        "decision_count": len(replay_bundle.decisions),
        "phases": phase_records,
        "final_score": final_eval["score"],
        "final_components": final_eval["components"],
        "final_rejected": final_eval["rejected"],
        "final_reject_reason": final_eval["reject_reason"],
        "final_soft_warnings": final_eval["soft_warnings"],
        "final_metrics": final_metric_package["headline_metrics"],
        "final_metrics_realized": final_metric_package["realized_metrics"],
        "final_diagnostic_equity": final_metric_package["diagnostic_equity"],
        "final_config": family_config_to_dict(current_config),
        "strategy_trade_counts": final_result.strategy_trade_counts,
        "strategy_blocked_counts": final_result.strategy_blocked_counts,
        "rule_blocks": final_result.rule_blocks,
    }
    _write_json(output_dir / "run_summary.json", summary)
    _write_json(output_dir / "optimized_portfolio_config.json", summary["final_config"])
    return summary


def headline_mtm_metric_package(
    config: FamilyPortfolioBacktestConfig,
    result,
    *,
    data_dir: Path = Path("backtests/momentum/data/raw"),
    price_bars: Any = _PRICE_BARS_UNSET,
) -> dict[str, Any]:
    realized_metrics = dict(result.metrics)
    from backtests.momentum.analysis.family_portfolio_diagnostics import _portfolio_mtm_metrics

    if price_bars is _PRICE_BARS_UNSET:
        price_bars = _load_mtm_price_bars_for_scoring(data_dir)

    diagnostic_equity = _portfolio_mtm_metrics(
        config,
        result,
        price_bars,
    )
    realized_dd = float(realized_metrics.get("max_drawdown_pct", 0.0) or 0.0)
    realized_calmar = float(realized_metrics.get("calmar", 0.0) or 0.0)
    realized_return = float(realized_metrics.get("net_return_pct", 0.0) or 0.0)
    mtm_dd = float(diagnostic_equity.get("max_drawdown_pct", realized_dd) or 0.0)
    mtm_calmar = float(diagnostic_equity.get("calmar", realized_calmar) or 0.0)
    diagnostic_return = float(diagnostic_equity.get("net_return_pct", realized_return) or 0.0)
    headline_metrics = {
        **realized_metrics,
        "risk_basis": diagnostic_equity.get("risk_basis", "realized_daily"),
        "final_equity": diagnostic_equity.get("final_equity"),
        "net_return_pct": diagnostic_return,
        "net_return_pct_realized": realized_return,
        "max_drawdown_pct": mtm_dd,
        "calmar": mtm_calmar,
        "max_drawdown_pct_mtm": mtm_dd,
        "calmar_mtm": mtm_calmar,
        "max_drawdown_pct_realized": realized_dd,
        "calmar_realized": realized_calmar,
    }
    return {
        "headline_metrics": headline_metrics,
        "realized_metrics": realized_metrics,
        "diagnostic_equity": diagnostic_equity,
    }


def evaluate_portfolio_config(
    name: str,
    config: FamilyPortfolioBacktestConfig,
    replay_bundle: FamilyPortfolioReplayBundle,
    *,
    price_bars: Any = _PRICE_BARS_UNSET,
) -> PortfolioEvaluation:
    result = FamilyPortfolioBacktester(config).run_bundle(replay_bundle)
    metric_package = headline_mtm_metric_package(config, result, price_bars=price_bars)
    headline_metrics = metric_package["headline_metrics"]
    scored = score_metrics(headline_metrics)
    return PortfolioEvaluation(
        name=name,
        score=scored["score"],
        rejected=scored["rejected"],
        reject_reason=scored["reject_reason"],
        soft_warnings=scored["soft_warnings"],
        components=scored["components"],
        metrics=headline_metrics,
        config=config,
    )


def score_metrics(metrics: dict[str, float]) -> dict[str, Any]:
    components = {
        "expected_return": _cap(metrics.get("net_profit", 0.0) / TARGETS["net_profit"], 1.30),
        "trade_frequency": _cap(metrics.get("trades_per_month", 0.0) / TARGETS["trades_per_month"], 1.25),
        "drawdown_control": _drawdown_component(metrics.get("max_drawdown_pct", 1.0)),
        "profit_quality": _cap(metrics.get("profit_factor", 0.0) / TARGETS["profit_factor"], 1.20),
        "risk_efficiency": _cap(metrics.get("calmar", 0.0) / TARGETS["calmar"], 1.25),
        "strategy_balance": _strategy_balance_component(metrics),
        "live_rule_health": _live_rule_health_component(metrics),
    }
    reject_reason = _reject_reason(metrics)
    score = 0.0 if reject_reason else sum(SCORE_WEIGHTS[key] * components[key] for key in SCORE_WEIGHTS)
    return {
        "score": float(score),
        "rejected": bool(reject_reason),
        "reject_reason": reject_reason,
        "soft_warnings": _soft_warnings(metrics),
        "components": components,
    }


def apply_portfolio_mutations(
    config: FamilyPortfolioBacktestConfig,
    mutations: dict[str, Any],
) -> FamilyPortfolioBacktestConfig:
    updated = config
    rule_changes: dict[str, Any] = {}
    config_changes: dict[str, Any] = {}
    dynamic_changes: dict[str, Any] = {}
    allocation_changes: dict[str, dict[str, Any]] = {}

    for key, value in mutations.items():
        if key.startswith("rules."):
            rule_changes[key.split(".", 1)[1]] = _tupleify(value)
        elif key.startswith("config."):
            config_changes[key.split(".", 1)[1]] = _tupleify(value)
        elif key.startswith("dynamic."):
            dynamic_changes[key.split(".", 1)[1]] = _tupleify(value)
        elif key.startswith("allocation."):
            strategy_id, field_name = key.removeprefix("allocation.").rsplit(".", 1)
            allocation_changes.setdefault(strategy_id, {})[field_name] = value
        else:
            raise ValueError(f"Unknown portfolio mutation key: {key}")

    if rule_changes:
        updated = replace(updated, rules=replace(updated.rules, **rule_changes))
    if "signal_filter_rules" in config_changes:
        config_changes["signal_filter_rules"] = tuple(config_changes["signal_filter_rules"])
    if config_changes:
        updated = replace(updated, **config_changes)
    if dynamic_changes:
        updated = replace(updated, dynamic_risk=replace(updated.dynamic_risk, **dynamic_changes))
    for strategy_id, changes in allocation_changes.items():
        updated = update_allocation(updated, strategy_id, **changes)
    return updated


def load_or_build_latest_strategy_trades(
    *,
    data_dir: Path,
    output_dir: Path,
    initial_equity: float,
    force: bool = False,
) -> dict[str, list]:
    cache_path = output_dir / "strategy_trades.pkl"
    manifest_path = output_dir / "strategy_trade_manifest.json"
    source_manifest = _strategy_source_manifest()
    if cache_path.exists() and manifest_path.exists() and not force:
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == source_manifest:
                with cache_path.open("rb") as fh:
                    return pickle.load(fh)
        except Exception:
            pass

    trades_by_strategy = {
        "NQDTC_v2.1": _run_nqdtc(data_dir, initial_equity),
        "VdubusNQ_v4": _run_vdubus(data_dir, initial_equity),
        "DownturnDominator_v1": _run_downturn(data_dir, initial_equity),
        "NQ_REGIME": _run_nq_regime(data_dir, initial_equity),
    }
    with cache_path.open("wb") as fh:
        pickle.dump(trades_by_strategy, fh)
    _write_json(manifest_path, source_manifest)
    return trades_by_strategy


def _evaluate_candidates(
    current_config: FamilyPortfolioBacktestConfig,
    candidates: list[PortfolioCandidate],
    replay_bundle: FamilyPortfolioReplayBundle,
    *,
    max_workers: int,
    price_bars: Any,
) -> list[PortfolioEvaluation]:
    worker_count = max(1, min(int(max_workers), 2))
    evaluations: list[PortfolioEvaluation] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                evaluate_portfolio_config,
                candidate.name,
                apply_portfolio_mutations(current_config, candidate.mutations),
                replay_bundle,
                price_bars=price_bars,
            ): candidate
            for candidate in candidates
        }
        for future in as_completed(future_map):
            evaluations.append(future.result())
    evaluations.sort(key=_portfolio_selection_key, reverse=True)
    return evaluations


def _run_nqdtc(data_dir: Path, initial_equity: float) -> list:
    from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
    from backtests.momentum.auto.nqdtc.worker import load_worker_data
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.data.replay_cache import replay_engine_kwargs
    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine

    mutations = _latest_optimized_mutations("nqdtc")
    cfg = NQDTCBacktestConfig(
        initial_equity=initial_equity,
        data_dir=data_dir,
        fixed_qty=10,
        track_signals=False,
        track_shadows=False,
        scoring_mode=True,
        max_dd_abort=0.50,
    )
    cfg = mutate_nqdtc_config(cfg, mutations)
    bundle = load_worker_data("NQ", data_dir)
    result = NQDTCEngine("MNQ", cfg).run(**replay_engine_kwargs(bundle))
    return result.trades


def _run_vdubus(data_dir: Path, initial_equity: float) -> list:
    from backtests.momentum.auto.config_mutator import mutate_vdubus_config
    from backtests.momentum.auto.vdubus.worker import load_worker_data
    from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
    from backtests.momentum.data.replay_cache import replay_engine_kwargs
    from backtests.momentum.engine.vdubus_engine import VdubusEngine

    mutations = _latest_optimized_mutations("vdubus")
    cfg = VdubusBacktestConfig(
        initial_equity=initial_equity,
        data_dir=data_dir,
        fixed_qty=10,
        flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
        track_signals=False,
        track_shadows=False,
    )
    cfg = mutate_vdubus_config(cfg, mutations)
    bundle = load_worker_data("NQ", data_dir)
    result = VdubusEngine("NQ", cfg).run(**replay_engine_kwargs(bundle))
    return result.trades


def _run_downturn(data_dir: Path, initial_equity: float) -> list:
    from backtests.momentum.auto.downturn.config_mutator import mutate_downturn_config
    from backtests.momentum.auto.downturn.worker import load_worker_data
    from backtests.momentum.config_downturn import DownturnBacktestConfig
    from backtests.momentum.data.replay_cache import replay_engine_kwargs
    from backtests.momentum.engine.downturn_engine import DownturnEngine

    mutations = _latest_optimized_mutations("downturn")
    cfg = DownturnBacktestConfig(
        initial_equity=initial_equity,
        data_dir=data_dir,
        track_signals=False,
        skip_parity_output=True,
        max_dd_abort=0.50,
    )
    cfg = mutate_downturn_config(cfg, mutations)
    bundle = load_worker_data("NQ", data_dir)
    result = DownturnEngine("NQ", cfg).run(**replay_engine_kwargs(bundle))
    return result.trades


def _run_nq_regime(data_dir: Path, initial_equity: float) -> list:
    from backtests.momentum.auto.nq_regime.worker import mutate_config
    from backtests.momentum.config_regime import NqRegimeBacktestConfig
    from backtests.momentum.engine.regime_engine import (
        load_nq_regime_data,
        run_nq_regime_backtest,
    )

    mutations = _latest_optimized_mutations("nq_regime")
    cfg = NqRegimeBacktestConfig(
        data_dir=data_dir,
        initial_equity=initial_equity,
        analysis_symbol="NQ",
        trade_symbol="MNQ",
    )
    cfg = mutate_config(cfg, mutations)
    result = run_nq_regime_backtest(load_nq_regime_data(cfg), cfg)
    return result.trades


def _latest_optimized_mutations(strategy_dir_name: str) -> dict[str, Any]:
    path, _ = _latest_optimized_config_path(strategy_dir_name)
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_source_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for name in ("nqdtc", "vdubus", "downturn", "nq_regime"):
        try:
            latest, selection_method = _latest_optimized_config_path(name)
        except FileNotFoundError:
            manifest[name] = None
            continue
        stat = latest.stat()
        manifest[name] = {
            "path": str(latest),
            "round": _round_num_from_config_path(latest),
            "selection_method": selection_method,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "sha256": _file_sha256(latest),
            "source_provenance": _strategy_artifact_provenance(name, latest),
        }
    return manifest


def _latest_optimized_config_path(strategy_dir_name: str) -> tuple[Path, str]:
    manager = RoundManager("momentum", strategy_dir_name, base_dir=Path("backtests/output"))
    if manager.manifest_path.exists():
        latest_round = manager.get_latest_round()
        if latest_round < 1:
            raise FileNotFoundError(f"No active manifest round for momentum/{strategy_dir_name}.")
        path = manager.optimized_config_path(manager.round_path(latest_round))
        if not path.exists():
            raise FileNotFoundError(f"Active manifest round {latest_round} is missing {path}.")
        return path, "active_manifest_latest"

    root = Path("backtests/output/momentum") / strategy_dir_name
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("round_*/optimized_config.json"):
        round_num = _round_num_from_config_path(path)
        if round_num is not None:
            candidates.append((round_num, path))
    if not candidates:
        raise FileNotFoundError(f"No optimized momentum config found for {strategy_dir_name} under {root}.")
    return max(candidates, key=lambda item: item[0])[1], "filesystem_latest_round_fallback"


def _round_num_from_config_path(path: Path) -> int | None:
    try:
        return int(path.parent.name.removeprefix("round_"))
    except ValueError:
        return None


def _strategy_artifact_provenance(strategy_name: str, config_path: Path) -> dict[str, Any]:
    round_dir = config_path.parent
    runner_paths = {
        "downturn": Path("backtests/momentum/auto/downturn/plugin.py"),
        "nq_regime": Path("backtests/momentum/auto/nq_regime/current_oos_frequency_repair.py"),
    }
    record: dict[str, Any] = {
        "contract": "momentum_source_artifact_provenance.v1",
        "strategy": strategy_name,
        "optimized_config": _artifact_record(config_path),
        "run_summary": _artifact_record(round_dir / "run_summary.json"),
        "round_final_diagnostics": _artifact_record(round_dir / "round_final_diagnostics.txt"),
    }
    summary_path = round_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if isinstance(summary, dict) and isinstance(summary.get("provenance"), dict):
            record["saved_provenance"] = summary["provenance"]
            record["saved_provenance_status"] = summary.get("provenance_status", "complete")
        else:
            record["saved_provenance_status"] = "artifact_hash_recorded_no_saved_provenance"
    else:
        record["saved_provenance_status"] = "missing_run_summary"

    runner_path = runner_paths.get(strategy_name)
    if runner_path is not None:
        record["runner"] = _artifact_record(runner_path)
        if strategy_name == "nq_regime":
            record["runner_role"] = "actual_is_oos_split_repair_runner"
    return record


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _file_sha256(path) if path.exists() and path.is_file() else "",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mtm_price_bars_for_scoring(data_dir: Path) -> dict[str, Any] | None:
    from backtests.momentum.analysis.family_portfolio_diagnostics import _load_mtm_price_bars

    return _load_mtm_price_bars(data_dir)


def _phase_candidates() -> dict[int, list[PortfolioCandidate]]:
    return {
        1: [
            PortfolioCandidate("nq_regime_70bp", "Push clean frequency leader", {"allocation.NQ_REGIME.base_risk_pct": 0.0070}),
            PortfolioCandidate("vdubus_65bp", "Push Vdubus participation", {"allocation.VdubusNQ_v4.base_risk_pct": 0.0065}),
            PortfolioCandidate("nqdtc_55bp", "Lift NQDTC confirmer risk", {"allocation.NQDTC_v2.1.base_risk_pct": 0.0055}),
            PortfolioCandidate("downturn_50bp", "Lift downturn hedge/range ballast", {"allocation.DownturnDominator_v1.base_risk_pct": 0.0050}),
            PortfolioCandidate(
                "balanced_plus_10pct",
                "Increase all allocations by a controlled step",
                {
                    "allocation.NQ_REGIME.base_risk_pct": 0.0066,
                    "allocation.VdubusNQ_v4.base_risk_pct": 0.0061,
                    "allocation.NQDTC_v2.1.base_risk_pct": 0.0050,
                    "allocation.DownturnDominator_v1.base_risk_pct": 0.0044,
                },
            ),
        ],
        2: [
            PortfolioCandidate("heat_5_25", "Raise shared heat cap", {"config.heat_cap_R": 5.25}),
            PortfolioCandidate("max_positions_6", "Allow six simultaneous family positions", {"config.max_total_positions": 6}),
            PortfolioCandidate(
                "dir_caps_asym_4_75_5_25",
                "More short-side capacity for downturn/NQ regime coexistence",
                {"rules.directional_cap_long_R": 4.75, "rules.directional_cap_short_R": 5.25},
            ),
            PortfolioCandidate("contracts_18", "Raise live family MNQ-equivalent cap", {"rules.max_family_contracts_mnq_eq": 18}),
            PortfolioCandidate(
                "heat_5_75_contracts_20",
                "Controlled capacity lift with live family cap intact",
                {"config.heat_cap_R": 5.75, "rules.max_family_contracts_mnq_eq": 20},
            ),
            PortfolioCandidate(
                "heat_6_25_contracts_22",
                "Upper controlled-aggressive capacity probe",
                {"config.heat_cap_R": 6.25, "rules.max_family_contracts_mnq_eq": 22},
            ),
            PortfolioCandidate("priority_headroom_050", "Loosen priority reservation", {"rules.priority_headroom_R": 0.50}),
            PortfolioCandidate(
                "capacity_combo",
                "Controlled combined capacity lift",
                {
                    "config.heat_cap_R": 5.25,
                    "rules.directional_cap_long_R": 4.75,
                    "rules.directional_cap_short_R": 5.25,
                    "rules.max_family_contracts_mnq_eq": 18,
                },
            ),
        ],
        3: [
            PortfolioCandidate("oppose_block", "Block Vdubus against NQDTC direction", {"rules.nqdtc_oppose_size_mult": 0.0}),
            PortfolioCandidate("oppose_quarter", "Quarter-size Vdubus against NQDTC", {"rules.nqdtc_oppose_size_mult": 0.25}),
            PortfolioCandidate("agree_150", "Boost Vdubus when NQDTC agrees", {"rules.nqdtc_agree_size_mult": 1.50}),
            PortfolioCandidate("direction_filter_off", "Disable Vdubus/NQDTC direction filter", {"rules.nqdtc_direction_filter_enabled": False}),
        ],
        4: [
            PortfolioCandidate("portfolio_daily_2_25", "Tighter hostile-day cutoff", {"config.portfolio_daily_stop_R": 2.25}),
            PortfolioCandidate("portfolio_daily_3_25", "Looser daily cutoff for more frequency", {"config.portfolio_daily_stop_R": 3.25}),
            PortfolioCandidate("weekly_6_0", "Tighter weekly stop", {"config.portfolio_weekly_stop_R": 6.0}),
            PortfolioCandidate("weekly_9_0", "Looser weekly stop", {"config.portfolio_weekly_stop_R": 9.0}),
            PortfolioCandidate(
                "dd_tiers_tighter",
                "Earlier derisking before max DD accelerates",
                {"rules.dd_tiers": ((0.08, 1.00), (0.12, 0.55), (0.17, 0.25), (1.00, 0.00))},
            ),
            PortfolioCandidate(
                "dd_tiers_looser",
                "Controlled-aggressive drawdown breathing room",
                {"rules.dd_tiers": ((0.12, 1.00), (0.17, 0.65), (0.22, 0.30), (1.00, 0.00))},
            ),
        ],
        5: [
            PortfolioCandidate(
                "alpha_frequency_combo",
                "Lean into NQ regime and Vdubus while retaining guards",
                {
                    "allocation.NQ_REGIME.base_risk_pct": 0.0070,
                    "allocation.VdubusNQ_v4.base_risk_pct": 0.0062,
                    "config.heat_cap_R": 5.25,
                    "rules.max_family_contracts_mnq_eq": 18,
                },
            ),
            PortfolioCandidate(
                "balanced_aggressive_combo",
                "Broad risk lift with asymmetric capacity",
                {
                    "allocation.NQ_REGIME.base_risk_pct": 0.0066,
                    "allocation.VdubusNQ_v4.base_risk_pct": 0.0061,
                    "allocation.NQDTC_v2.1.base_risk_pct": 0.0050,
                    "allocation.DownturnDominator_v1.base_risk_pct": 0.0044,
                    "config.heat_cap_R": 5.25,
                    "rules.directional_cap_long_R": 4.75,
                    "rules.directional_cap_short_R": 5.25,
                },
            ),
            PortfolioCandidate(
                "dd_guarded_frequency_combo",
                "More frequency with tighter daily loss control",
                {
                    "allocation.NQ_REGIME.base_risk_pct": 0.0068,
                    "allocation.VdubusNQ_v4.base_risk_pct": 0.0060,
                    "config.portfolio_daily_stop_R": 2.25,
                },
            ),
            PortfolioCandidate(
                "guarded_capacity_plus",
                "Capacity lift with tighter daily guard and modest leader risk",
                {
                    "allocation.NQ_REGIME.base_risk_pct": 0.0068,
                    "allocation.VdubusNQ_v4.base_risk_pct": 0.0060,
                    "config.heat_cap_R": 5.75,
                    "config.max_total_positions": 6,
                    "config.portfolio_daily_stop_R": 2.25,
                    "rules.directional_cap_long_R": 5.25,
                    "rules.directional_cap_short_R": 5.75,
                    "rules.max_family_contracts_mnq_eq": 20,
                },
            ),
            PortfolioCandidate(
                "frequency_frontier",
                "Higher trade-through capacity while keeping original sizing",
                {
                    "config.heat_cap_R": 6.25,
                    "rules.max_family_contracts_mnq_eq": 22,
                },
            ),
        ],
    }


PHASE_CANDIDATES = _phase_candidates()


def _evaluation_record(evaluation: PortfolioEvaluation) -> dict[str, Any]:
    return {
        "name": evaluation.name,
        "score": evaluation.score,
        "rejected": evaluation.rejected,
        "reject_reason": evaluation.reject_reason,
        "soft_warnings": evaluation.soft_warnings,
        "components": evaluation.components,
        "metrics": evaluation.metrics,
        "config": family_config_to_dict(evaluation.config),
    }


def _reject_reason(metrics: dict[str, float]) -> str:
    if metrics.get("net_profit", 0.0) <= 0:
        return "negative_or_flat_net_profit"
    if metrics.get("max_drawdown_pct", 1.0) > 0.20:
        return "max_drawdown_above_20pct"
    if metrics.get("profit_factor", 0.0) < 1.35:
        return "profit_factor_below_1_35"
    if metrics.get("active_strategies", 0.0) < len(MOMENTUM_FAMILY_STRATEGY_IDS):
        return "not_all_family_strategies_active"
    return ""


def _soft_warnings(metrics: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    if metrics.get("trades_per_month", 0.0) < 18.0:
        warnings.append("frequency_below_18_trades_per_month")
    elif metrics.get("trades_per_month", 0.0) < TARGETS["trades_per_month"]:
        warnings.append("frequency_below_target")
    if metrics.get("block_rate", 1.0) > TARGETS["max_block_rate"]:
        warnings.append("block_rate_above_target")
    if metrics.get("max_drawdown_pct", 1.0) > TARGETS["max_drawdown_pct"]:
        warnings.append("drawdown_above_target")
    if metrics.get("net_profit", 0.0) < TARGETS["net_profit"]:
        warnings.append("net_profit_below_target")
    if metrics.get("profit_factor", 0.0) < TARGETS["profit_factor"]:
        warnings.append("profit_factor_below_target")
    if metrics.get("min_strategy_trades", 0.0) < TARGETS["min_strategy_trades"]:
        warnings.append("strategy_balance_below_target")
    return warnings


def _strategy_balance_component(metrics: dict[str, float]) -> float:
    active = metrics.get("active_strategies", 0.0) / len(MOMENTUM_FAMILY_STRATEGY_IDS)
    min_trades = metrics.get("min_strategy_trades", 0.0) / TARGETS["min_strategy_trades"]
    return _cap(0.65 * active + 0.35 * min(min_trades, 1.0), 1.0)


def _live_rule_health_component(metrics: dict[str, float]) -> float:
    block_rate = metrics.get("block_rate", 1.0)
    block_score = max(0.0, 1.0 - block_rate / TARGETS["max_block_rate"])
    concurrency = min(metrics.get("max_concurrent", 0.0) / 4.0, 1.0)
    return 0.65 * block_score + 0.35 * concurrency


def _portfolio_selection_key(evaluation: PortfolioEvaluation) -> tuple[float, float, float, float, float, float]:
    metrics = evaluation.metrics
    return (
        evaluation.score,
        float(metrics.get("net_profit", 0.0) or 0.0),
        float(metrics.get("trades_per_month", 0.0) or 0.0),
        -float(metrics.get("max_drawdown_pct", 1.0) or 1.0),
        -float(metrics.get("block_rate", 1.0) or 1.0),
        float(metrics.get("profit_factor", 0.0) or 0.0),
    )


def _drawdown_component(max_drawdown_pct: float) -> float:
    if max_drawdown_pct <= 0:
        return 1.2
    if max_drawdown_pct > 0.20:
        return 0.0
    return _cap(1.0 - max(0.0, max_drawdown_pct - 0.08) / 0.12, 1.2)


def _cap(value: float, cap: float) -> float:
    return float(max(0.0, min(value, cap)))


def _tupleify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleify(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_tupleify(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")
