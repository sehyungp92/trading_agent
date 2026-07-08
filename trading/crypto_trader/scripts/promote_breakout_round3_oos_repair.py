"""Promote the selected breakout OOS repair candidate as round_3.

This deterministic promotion step:

* starts from output/breakout/round_2/optimized_config.json
* applies the selected checkpointed OOS repair mutations
* verifies the result reconstructs the selected repair config
* runs strict optimizer preflight, full diagnostics, and checkpoint parity checks
* writes output/breakout/round_3 artifacts
* updates config/strategies/breakout.json for live/backtest parity
* updates output/breakout/rounds_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from copy import deepcopy
from dataclasses import asdict
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
from crypto_trader.backtest.metrics import metrics_to_dict  # noqa: E402
from crypto_trader.backtest.profiles import (  # noqa: E402
    LIVE_PARITY_PROFILE,
    build_backtest_config_from_profile,
)
from crypto_trader.backtest.runner import run, run_split_continuation  # noqa: E402
from crypto_trader.cli import _update_rounds_manifest  # noqa: E402
from crypto_trader.live.config import LiveConfig  # noqa: E402
from crypto_trader.live.execution_adapter import HyperliquidExecutionAdapter  # noqa: E402
from crypto_trader.live.parity_warnings import collect_live_parity_warnings  # noqa: E402
from crypto_trader.optimize.breakout_round2_alpha import (  # noqa: E402
    ROUND2_ALPHA_HARD_REJECTS,
    ROUND2_ALPHA_PHASE_GATE_CRITERIA,
    ROUND2_ALPHA_SCORING_CEILINGS,
    ROUND2_ALPHA_SCORING_WEIGHTS,
    BreakoutRound2AlphaPlugin,
)
from crypto_trader.optimize.config_mutator import apply_mutations  # noqa: E402
from crypto_trader.optimize.contracts import (  # noqa: E402
    build_optimization_contract,
    run_optimization_preflight,
    stable_hash,
    strategy_config_hash,
)
from crypto_trader.optimize.phase_state import _atomic_write_json  # noqa: E402
from crypto_trader.optimize.scoring import composite_score  # noqa: E402
from crypto_trader.portfolio.config import PortfolioConfig  # noqa: E402
from crypto_trader.strategy.breakout.config import BreakoutConfig  # noqa: E402


DATA_DIR = ROOT / "data"
OUTPUT_BASE = ROOT / "output" / "breakout"
ROUND2_CONFIG_PATH = OUTPUT_BASE / "round_2" / "optimized_config.json"
DEFAULT_REPAIR_DIR = (
    OUTPUT_BASE
    / "round_2_oos_repair_followup_strict"
    / "20260527T082058Z"
)
DEFAULT_REPAIR_CONFIG_PATH = DEFAULT_REPAIR_DIR / "recommended_followup_strict_config.json"
DEFAULT_REPAIR_SUMMARY_PATH = DEFAULT_REPAIR_DIR / "followup_strict_summary.json"
DEFAULT_REPAIR_VERIFICATION_PATH = DEFAULT_REPAIR_DIR / "final_verification.json"
LIVE_STRATEGY_CONFIG_PATH = ROOT / "config" / "strategies" / "breakout.json"
LIVE_CONFIG_EXAMPLE_PATH = ROOT / "config" / "live_config.example.json"
PORTFOLIO_CONFIG_PATH = ROOT / "config" / "portfolio_config.json"

ROUND_NUM = 3
SYMBOLS = ["BTC", "ETH", "SOL"]
IS_START = date(2026, 1, 4)
IS_END = date(2026, 4, 20)
OOS_START = date(2026, 4, 21)
OOS_END = date(2026, 5, 23)
FULL_START = IS_START
FULL_END = OOS_END

ROUND3_MUTATIONS: dict[str, Any] = {
    "profile.recalc_interval_bars": 4,
    "balance.min_bars_in_zone": 7,
    "balance.require_volume_contraction": True,
    "balance.dedup_atr_frac": 0.35,
    "symbol_filter.sol_direction": "both",
}


def _quiet_logging() -> None:
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def _load_strategy_payload(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw.get("strategy", raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected strategy mapping in {path}")
    return payload


def _load_config(path: Path) -> BreakoutConfig:
    return BreakoutConfig.from_dict(_load_strategy_payload(path))


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _diff_values(base: BreakoutConfig, candidate: BreakoutConfig) -> dict[str, dict[str, Any]]:
    base_flat = _flatten(base.to_dict())
    candidate_flat = _flatten(candidate.to_dict())
    return {
        key: {"base_value": base_flat.get(key), "candidate_value": candidate_flat.get(key)}
        for key in sorted(set(base_flat) | set(candidate_flat))
        if base_flat.get(key) != candidate_flat.get(key)
    }


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
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


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _build_backtest_config(start: date, end: date):
    return build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=SYMBOLS,
        start_date=start,
        end_date=end,
    )


def _run_window(config: BreakoutConfig, start: date, end: date) -> dict[str, Any]:
    backtest_config = _build_backtest_config(start, end)
    result = run(deepcopy(config), backtest_config, data_dir=DATA_DIR, strategy_type="breakout")
    return {
        "metrics": metrics_to_dict(result.metrics),
        "trade_count": len(result.trades),
        "terminal_mark_count": len(result.terminal_marks),
    }


def _metric_line(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"{label}: return {metrics.get('net_return_pct', 0.0):.4f}%, "
        f"trades {metrics.get('total_trades', 0.0):.0f}, "
        f"win rate {metrics.get('win_rate', 0.0):.2f}%, "
        f"PF {metrics.get('profit_factor', 0.0):.4f}, "
        f"expectancy {metrics.get('expectancy_r', 0.0):.4f}R, "
        f"DD {metrics.get('max_drawdown_pct', 0.0):.4f}%"
    )


def _trade_signature(trades: list[Any]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for trade in trades:
        direction = getattr(getattr(trade, "direction", None), "value", getattr(trade, "direction", ""))
        rows.append(
            (
                getattr(trade, "symbol", ""),
                direction,
                getattr(trade, "entry_time", None).isoformat(),
                getattr(trade, "exit_time", None).isoformat(),
                round(float(getattr(trade, "net_pnl", 0.0)), 8),
            )
        )
    return rows


def _trade_rows(trades: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        direction = getattr(getattr(trade, "direction", None), "value", getattr(trade, "direction", ""))
        rows.append(
            {
                "symbol": getattr(trade, "symbol", ""),
                "direction": direction,
                "entry_time": getattr(trade, "entry_time", None).isoformat(),
                "exit_time": getattr(trade, "exit_time", None).isoformat(),
                "net_pnl": getattr(trade, "net_pnl", 0.0),
                "r": getattr(trade, "economic_r_multiple", None),
                "exit_reason": getattr(trade, "exit_reason", ""),
            }
        )
    return rows


def _net_profit_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(left.get("net_profit", 0.0)) - float(right.get("net_profit", 0.0))


def _assert_checkpoint_parity(standalone: Any, split: Any) -> dict[str, Any]:
    standalone_metrics = metrics_to_dict(standalone.metrics)
    stitched_metrics = metrics_to_dict(split.stitched.metrics)
    is_metrics = metrics_to_dict(split.in_sample.metrics)
    oos_metrics = metrics_to_dict(split.out_of_sample.metrics)

    same_trades = _trade_signature(standalone.trades) == _trade_signature(split.stitched.trades)
    standalone_delta = _net_profit_delta(standalone_metrics, stitched_metrics)
    stitched_sum_delta = (
        float(is_metrics.get("net_profit", 0.0))
        + float(oos_metrics.get("net_profit", 0.0))
        - float(stitched_metrics.get("net_profit", 0.0))
    )
    if abs(standalone_delta) > 1e-9 or abs(stitched_sum_delta) > 1e-9 or not same_trades:
        raise RuntimeError(
            "Checkpoint parity failed: "
            + json.dumps(
                _json_ready(
                    {
                        "standalone_minus_stitched_net_profit": standalone_delta,
                        "is_plus_oos_minus_stitched_net_profit": stitched_sum_delta,
                        "same_trade_keys_and_pnl": same_trades,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )

    return {
        "status": "matched",
        "split_date": OOS_START.isoformat(),
        "standalone_minus_stitched_net_profit": standalone_delta,
        "is_plus_oos_minus_stitched_net_profit": stitched_sum_delta,
        "same_trade_keys_and_pnl": same_trades,
        "standalone": standalone_metrics,
        "split_in_sample": is_metrics,
        "split_oos": oos_metrics,
        "split_stitched": stitched_metrics,
        "checkpoint": split.checkpoint,
        "oos_trades": _trade_rows(split.out_of_sample.trades),
    }


def _live_config_for_warning_scan() -> LiveConfig:
    if LIVE_CONFIG_EXAMPLE_PATH.exists():
        payload = _load_optional_json(LIVE_CONFIG_EXAMPLE_PATH)
        if payload:
            return LiveConfig.from_dict(payload)
    return LiveConfig(
        is_testnet=True,
        symbols=SYMBOLS,
        strategy_configs={"breakout": LIVE_STRATEGY_CONFIG_PATH},
        portfolio_config_path=PORTFOLIO_CONFIG_PATH if PORTFOLIO_CONFIG_PATH.exists() else None,
    )


def _portfolio_config_for_warning_scan() -> PortfolioConfig | None:
    if not PORTFOLIO_CONFIG_PATH.exists():
        return None
    return PortfolioConfig.from_dict(_load_optional_json(PORTFOLIO_CONFIG_PATH))


def _write_live_strategy_config(config: BreakoutConfig) -> dict[str, Any]:
    _atomic_write_json(_json_ready({"strategy": config.to_dict()}), LIVE_STRATEGY_CONFIG_PATH)
    live_config = _load_config(LIVE_STRATEGY_CONFIG_PATH)
    if live_config.to_dict() != config.to_dict():
        diff = _diff_values(config, live_config)
        raise RuntimeError(
            "Live strategy config does not match promoted round_3 config: "
            + json.dumps(diff, indent=2, sort_keys=True)
        )
    return {
        "live_strategy_config": str(LIVE_STRATEGY_CONFIG_PATH),
        "strategy_payload_hash": stable_hash(config.to_dict()),
        "strategy_config_hash": strategy_config_hash(config),
        "status": "matched",
    }


def _metric_delta(left: Any, right: Any) -> float:
    left_f = float(left)
    right_f = float(right)
    if math.isinf(left_f) and math.isinf(right_f) and left_f == right_f:
        return 0.0
    return left_f - right_f


def _compare_live_backtest_metrics(
    *,
    promoted_config: BreakoutConfig,
    live_config: BreakoutConfig,
    full_metrics: dict[str, Any],
) -> dict[str, Any]:
    live_result = _run_window(live_config, FULL_START, FULL_END)
    live_metrics = live_result["metrics"]
    checked_metrics = [
        "net_return_pct",
        "net_profit",
        "realized_pnl_net",
        "total_trades",
        "win_rate",
        "profit_factor",
        "expectancy_r",
        "max_drawdown_pct",
        "sharpe_ratio",
        "calmar_ratio",
        "exit_efficiency",
    ]
    deltas = {
        key: _metric_delta(live_metrics.get(key, 0.0), full_metrics.get(key, 0.0))
        for key in checked_metrics
    }
    max_abs_delta = max((abs(value) for value in deltas.values()), default=0.0)
    if promoted_config.to_dict() != live_config.to_dict():
        raise RuntimeError("Live/backtest config payloads diverged before metric comparison.")
    if max_abs_delta > 1e-9:
        raise RuntimeError(
            "Live-config backtest metrics diverged from promoted full backtest: "
            + json.dumps(deltas, indent=2, sort_keys=True)
        )
    return {
        "status": "matched",
        "window": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
        "checked_metrics": checked_metrics,
        "metric_deltas": deltas,
        "live_metrics": live_metrics,
        "live_trade_count": live_result["trade_count"],
        "live_terminal_mark_count": live_result["terminal_mark_count"],
    }


def _collect_parity_warnings(live_config: BreakoutConfig) -> list[dict[str, str]]:
    warnings = collect_live_parity_warnings(
        _live_config_for_warning_scan(),
        _portfolio_config_for_warning_scan(),
        durable_oms_available=True,
        exchange_metadata_enforced=True,
        strategy_configs={"breakout": live_config},
        capabilities=HyperliquidExecutionAdapter.capabilities,
    )
    return [warning.to_dict() for warning in warnings]


def _selected_candidate(summary: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best")
    if isinstance(best, dict):
        return best
    improvements = summary.get("both_is_and_oos_improvements")
    if isinstance(improvements, list) and improvements:
        return improvements[0]
    return {
        "label": "strict_single:balance.min_bars_in_zone_7",
        "thesis": "Checkpointed OOS repair selected after strict follow-up validation.",
        "candidate_value": verification.get("changes", ROUND3_MUTATIONS),
    }


def _write_optimized_config(
    *,
    config: BreakoutConfig,
    contract: dict[str, Any],
    round_dir: Path,
    repair_config_path: Path,
    repair_summary_path: Path,
    repair_verification_path: Path,
    selected_candidate: dict[str, Any],
    live_alignment: dict[str, Any],
    checkpoint_alignment: dict[str, Any],
) -> Path:
    payload = {
        "strategy": config.to_dict(),
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "round": ROUND_NUM,
            "promotion_type": "checkpointed_oos_repair_followup",
            "source_round": 2,
            "source_config": str(ROUND2_CONFIG_PATH),
            "source_repair_config": str(repair_config_path),
            "source_repair_summary": str(repair_summary_path),
            "source_repair_verification": str(repair_verification_path),
            "selected_candidate_label": selected_candidate.get("label", ""),
            "selected_candidate": selected_candidate,
            "contract_hash": contract.get("contract_hash", ""),
            "profile_hash": contract.get("profile_hash", ""),
            "strategy_config_hash": contract.get("strategy_config_hash", ""),
            "portfolio_config_hash": contract.get("portfolio_config_hash", ""),
            "data_window": contract.get("data_window", {}),
            "data_fingerprint": contract.get("data_fingerprint", {}),
            "symbols": contract.get("symbols", []),
            "required_timeframes": contract.get("required_timeframes", []),
            "economic_profile": contract.get("economic_profile", {}),
            "contract": contract,
            "parity_alignment": live_alignment,
            "checkpoint_alignment": checkpoint_alignment,
            "repair_mutations": ROUND3_MUTATIONS,
        },
    }
    path = round_dir / "optimized_config.json"
    _atomic_write_json(_json_ready(payload), path)
    return path


def _build_evaluation_text(
    *,
    full_metrics: dict[str, Any],
    is_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    mutation_diff: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    selected_candidate: dict[str, Any],
    live_alignment: dict[str, Any],
    checkpoint_alignment: dict[str, Any],
    parity_warnings: list[dict[str, str]],
) -> str:
    warning_lines = [
        f"- {warning.get('severity', 'warning')}: {warning.get('warning_id', '')}"
        for warning in parity_warnings
    ] or ["- none"]
    lines = [
        "Breakout Round 3 Checkpointed OOS Repair Promotion",
        "=" * 52,
        "",
        "Selected repair:",
        f"- label: {selected_candidate.get('label', 'strict_single:balance.min_bars_in_zone_7')}",
        "- profile.recalc_interval_bars: 4",
        "- balance.min_bars_in_zone: 7",
        "- balance.require_volume_contraction: true",
        "- balance.dedup_atr_frac: 0.35",
        "- symbol_filter.sol_direction: both",
        "",
        "Validation windows:",
        f"- In sample: {IS_START.isoformat()} to {IS_END.isoformat()}",
        f"- Out of sample: {OOS_START.isoformat()} to {OOS_END.isoformat()}",
        f"- Full promoted round: {FULL_START.isoformat()} to {FULL_END.isoformat()}",
        "",
        "Backtest results:",
        f"- {_metric_line('IS', is_metrics)}",
        f"- {_metric_line('OOS', oos_metrics)}",
        f"- {_metric_line('Full', full_metrics)}",
        "",
        "Checkpoint alignment:",
        f"- status: {checkpoint_alignment.get('status', '')}",
        f"- standalone_minus_stitched_net_profit: {checkpoint_alignment.get('standalone_minus_stitched_net_profit')}",
        f"- is_plus_oos_minus_stitched_net_profit: {checkpoint_alignment.get('is_plus_oos_minus_stitched_net_profit')}",
        f"- same_trade_keys_and_pnl: {checkpoint_alignment.get('same_trade_keys_and_pnl')}",
        "",
        "Round 2 -> Round 3 parameter delta:",
        json.dumps(_json_ready(mutation_diff), indent=2, sort_keys=True),
        "",
        "Parity contract:",
        f"- profile_id: {LIVE_PARITY_PROFILE.profile_id}",
        f"- contract_hash: {contract.get('contract_hash', '')}",
        f"- profile_hash: {contract.get('profile_hash', '')}",
        f"- strategy_config_hash: {contract.get('strategy_config_hash', '')}",
        f"- data_window: {json.dumps(contract.get('data_window', {}), sort_keys=True)}",
        f"- required_timeframes: {', '.join(contract.get('required_timeframes', []))}",
        "",
        "Live/backtest alignment:",
        f"- live strategy config: {live_alignment.get('live_strategy_config', '')}",
        f"- strategy payload status: {live_alignment.get('status', '')}",
        f"- live-config backtest status: {live_alignment.get('live_backtest_parity', {}).get('status', '')}",
        "- capability warnings:",
        *warning_lines,
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair-config",
        default=str(DEFAULT_REPAIR_CONFIG_PATH),
        help="Checkpointed repair optimized config JSON to promote.",
    )
    parser.add_argument(
        "--repair-summary",
        default=str(DEFAULT_REPAIR_SUMMARY_PATH),
        help="Checkpointed repair summary JSON for provenance.",
    )
    parser.add_argument(
        "--repair-verification",
        default=str(DEFAULT_REPAIR_VERIFICATION_PATH),
        help="Final checkpoint verification JSON for provenance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _quiet_logging()

    repair_config_path = Path(args.repair_config)
    repair_summary_path = Path(args.repair_summary)
    repair_verification_path = Path(args.repair_verification)
    round_dir = OUTPUT_BASE / f"round_{ROUND_NUM}"
    round_dir.mkdir(parents=True, exist_ok=True)

    round2_config = _load_config(ROUND2_CONFIG_PATH)
    promoted_config = apply_mutations(round2_config, ROUND3_MUTATIONS)
    repair_config = _load_config(repair_config_path)
    if promoted_config.to_dict() != repair_config.to_dict():
        diff = _diff_values(promoted_config, repair_config)
        raise RuntimeError(
            "Selected repair mutations do not reconstruct the repair config: "
            + json.dumps(diff, indent=2, sort_keys=True)
        )

    full_backtest_config = _build_backtest_config(FULL_START, FULL_END)
    plugin = BreakoutRound2AlphaPlugin(
        backtest_config=full_backtest_config,
        base_config=round2_config,
        data_dir=DATA_DIR,
        max_workers=None,
    )
    contract = build_optimization_contract(
        strategy_type="breakout",
        strategy_config=promoted_config,
        backtest_config=full_backtest_config,
        data_dir=DATA_DIR,
        profile=LIVE_PARITY_PROFILE,
        plugin=plugin,
        scoring_weights=ROUND2_ALPHA_SCORING_WEIGHTS,
        hard_rejects=ROUND2_ALPHA_HARD_REJECTS,
        gate_criteria=ROUND2_ALPHA_PHASE_GATE_CRITERIA,
        scoring_ceilings=ROUND2_ALPHA_SCORING_CEILINGS,
    )
    run_optimization_preflight(
        contract=contract,
        backtest_config=full_backtest_config,
        data_dir=DATA_DIR,
        output_dir=None,
        profile=LIVE_PARITY_PROFILE,
        validation_mode="strict",
    )

    standalone_result = run(
        deepcopy(promoted_config),
        full_backtest_config,
        data_dir=DATA_DIR,
        strategy_type="breakout",
    )
    split_result = run_split_continuation(
        deepcopy(promoted_config),
        full_backtest_config,
        split_date=OOS_START,
        data_dir=DATA_DIR,
        strategy_type="breakout",
    )
    checkpoint_alignment = _assert_checkpoint_parity(standalone_result, split_result)
    result = split_result.stitched
    full_metrics = metrics_to_dict(result.metrics)
    is_metrics = metrics_to_dict(split_result.in_sample.metrics)
    oos_metrics = metrics_to_dict(split_result.out_of_sample.metrics)

    hard_gate_score, hard_rejected, hard_reject_reason = composite_score(
        full_metrics,
        weights=ROUND2_ALPHA_SCORING_WEIGHTS,
        hard_rejects=ROUND2_ALPHA_HARD_REJECTS,
        ceilings=ROUND2_ALPHA_SCORING_CEILINGS,
    )
    score, _, _ = composite_score(
        full_metrics,
        weights=ROUND2_ALPHA_SCORING_WEIGHTS,
        hard_rejects=None,
        ceilings=ROUND2_ALPHA_SCORING_CEILINGS,
    )

    repair_summary = _load_optional_json(repair_summary_path)
    repair_verification = _load_optional_json(repair_verification_path)
    selected_candidate = _selected_candidate(repair_summary, repair_verification)

    live_alignment = _write_live_strategy_config(promoted_config)
    live_config = _load_config(LIVE_STRATEGY_CONFIG_PATH)
    live_alignment["live_backtest_parity"] = _compare_live_backtest_metrics(
        promoted_config=promoted_config,
        live_config=live_config,
        full_metrics=full_metrics,
    )
    parity_warnings = _collect_parity_warnings(live_config)
    live_alignment["capability_warnings"] = parity_warnings

    optimized_config_path = _write_optimized_config(
        config=promoted_config,
        contract=contract,
        round_dir=round_dir,
        repair_config_path=repair_config_path,
        repair_summary_path=repair_summary_path,
        repair_verification_path=repair_verification_path,
        selected_candidate=selected_candidate,
        live_alignment=live_alignment,
        checkpoint_alignment=checkpoint_alignment,
    )
    generate_report(result, round_dir)
    export_equity_curve(result, round_dir)
    export_trade_journal(result, round_dir)

    diagnostics = generate_diagnostics(
        list(result.trades),
        initial_equity=full_backtest_config.initial_equity,
        title="Breakout Round 3 Checkpointed OOS Repair Final Diagnostics",
        terminal_marks=list(result.terminal_marks),
        performance_metrics=result.metrics,
        expected_symbols=SYMBOLS,
        diagnostic_context=result.diagnostic_context,
    )
    (round_dir / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")

    mutation_diff = _diff_values(round2_config, promoted_config)
    evaluation = _build_evaluation_text(
        full_metrics=full_metrics,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        mutation_diff=mutation_diff,
        contract=contract,
        selected_candidate=selected_candidate,
        live_alignment=live_alignment,
        checkpoint_alignment=checkpoint_alignment,
        parity_warnings=parity_warnings,
    )
    (round_dir / "round_evaluation.txt").write_text(evaluation, encoding="utf-8")

    gate_result = {
        "passed": True,
        "failure_reasons": [],
        "validation_mode": "strict",
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "preflight": "passed",
        "acceptance_basis": "user_approved_checkpointed_oos_repair",
        "standard_full_period_hard_reject_evaluation": {
            "rejected": hard_rejected,
            "reject_reason": hard_reject_reason,
            "score_with_hard_rejects": hard_gate_score,
        },
        "checkpoint_parity": checkpoint_alignment["status"],
        "live_backtest_parity": live_alignment["live_backtest_parity"]["status"],
    }
    phase_result = {
        "final_score": score,
        "accepted_count": len(ROUND3_MUTATIONS),
        "new_mutations": ROUND3_MUTATIONS,
        "final_validation": {
            "status": "passed",
            "source": "checkpointed_oos_repair_rerun",
            "in_sample": {"start": IS_START.isoformat(), "end": IS_END.isoformat()},
            "out_of_sample": {"start": OOS_START.isoformat(), "end": OOS_END.isoformat()},
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
            "checkpoint_parity": {
                "standalone_minus_stitched_net_profit": checkpoint_alignment[
                    "standalone_minus_stitched_net_profit"
                ],
                "is_plus_oos_minus_stitched_net_profit": checkpoint_alignment[
                    "is_plus_oos_minus_stitched_net_profit"
                ],
                "same_trade_keys_and_pnl": checkpoint_alignment["same_trade_keys_and_pnl"],
            },
        },
    }

    summary = {
        "round": ROUND_NUM,
        "output_dir": str(round_dir),
        "optimized_config": str(optimized_config_path),
        "live_strategy_config": str(LIVE_STRATEGY_CONFIG_PATH),
        "source_round": 2,
        "source_config": str(ROUND2_CONFIG_PATH),
        "source_repair_config": str(repair_config_path),
        "source_repair_summary": str(repair_summary_path),
        "source_repair_verification": str(repair_verification_path),
        "selected_candidate": selected_candidate,
        "mutations": ROUND3_MUTATIONS,
        "mutation_diff_vs_round2": mutation_diff,
        "windows": {
            "in_sample": {"start": IS_START.isoformat(), "end": IS_END.isoformat()},
            "out_of_sample": {"start": OOS_START.isoformat(), "end": OOS_END.isoformat()},
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
        },
        "final_metrics": full_metrics,
        "in_sample_metrics": is_metrics,
        "out_of_sample_metrics": oos_metrics,
        "score": score,
        "gate_result": gate_result,
        "phase_result": phase_result,
        "checkpoint_alignment": checkpoint_alignment,
        "contract_hash": contract.get("contract_hash", ""),
        "profile_hash": contract.get("profile_hash", ""),
        "strategy_config_hash": contract.get("strategy_config_hash", ""),
        "portfolio_config_hash": contract.get("portfolio_config_hash", ""),
        "parity_alignment": {
            "profile": LIVE_PARITY_PROFILE.to_dict(),
            "preflight": "passed",
            "backtest_config": _json_ready(full_backtest_config),
            "contract": contract,
            "live_alignment": live_alignment,
            "checkpoint_alignment": checkpoint_alignment,
        },
        "artifacts": {
            "optimized_config": str(optimized_config_path),
            "backtest_report": str(round_dir / "backtest_report.md"),
            "equity_curve": str(round_dir / "equity_curve.csv"),
            "journal": str(round_dir / "journal.csv"),
            "round_final_diagnostics": str(round_dir / "round_final_diagnostics.txt"),
            "round_evaluation": str(round_dir / "round_evaluation.txt"),
            "parity_alignment": str(round_dir / "parity_alignment.json"),
        },
    }
    _atomic_write_json(_json_ready(summary), round_dir / "round3_summary.json")
    _atomic_write_json(_json_ready(summary["parity_alignment"]), round_dir / "parity_alignment.json")

    run_spec = {
        "round": ROUND_NUM,
        "strategy_type": "breakout",
        "promotion_type": "checkpointed_oos_repair_followup",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(ROUND2_CONFIG_PATH),
        "source_repair_config": str(repair_config_path),
        "source_repair_summary": str(repair_summary_path),
        "source_repair_verification": str(repair_verification_path),
        "selected_candidate_label": selected_candidate.get("label", ""),
        "mutations": ROUND3_MUTATIONS,
        "windows": summary["windows"],
        "contract_hash": contract.get("contract_hash", ""),
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "profile_hash": contract.get("profile_hash", ""),
        "required_timeframes": contract.get("required_timeframes", []),
        "symbols": SYMBOLS,
        "live_strategy_config": str(LIVE_STRATEGY_CONFIG_PATH),
        "checkpoint_split_date": OOS_START.isoformat(),
    }
    _atomic_write_json(_json_ready(run_spec), round_dir / "run_spec.json")

    _update_rounds_manifest(
        OUTPUT_BASE,
        ROUND_NUM,
        ROUND3_MUTATIONS,
        full_metrics,
        contract=contract,
        phase_result=phase_result,
        gate_result=gate_result,
    )

    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
