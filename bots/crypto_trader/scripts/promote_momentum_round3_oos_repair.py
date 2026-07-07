"""Promote the selected momentum round-2 OOS repair as round_3.

This deterministic promotion step:

* starts from output/momentum/round_2/optimized_config.json
* applies the selected follow-up OOS repair candidate
* verifies it reconstructs the follow-up optimized config
* runs strict live-parity preflight and full diagnostics
* writes output/momentum/round_3 artifacts
* updates config/strategies/momentum.json for live/backtest parity
* updates output/momentum/rounds_manifest.json
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
from crypto_trader.backtest.runner import run  # noqa: E402
from crypto_trader.cli import _update_rounds_manifest  # noqa: E402
from crypto_trader.live.config import LiveConfig  # noqa: E402
from crypto_trader.live.execution_adapter import HyperliquidExecutionAdapter  # noqa: E402
from crypto_trader.live.parity_warnings import collect_live_parity_warnings  # noqa: E402
from crypto_trader.optimize.config_mutator import apply_mutations  # noqa: E402
from crypto_trader.optimize.contracts import (  # noqa: E402
    build_optimization_contract,
    run_optimization_preflight,
    stable_hash,
    strategy_config_hash,
)
from crypto_trader.optimize.momentum_round2_alpha import (  # noqa: E402
    ROUND2_ALPHA_HARD_REJECTS,
    ROUND2_ALPHA_PHASE_GATE_CRITERIA,
    ROUND2_ALPHA_SCORING_CEILINGS,
    ROUND2_ALPHA_SCORING_WEIGHTS,
    MomentumRound2AlphaPlugin,
)
from crypto_trader.optimize.phase_state import _atomic_write_json  # noqa: E402
from crypto_trader.optimize.scoring import composite_score  # noqa: E402
from crypto_trader.portfolio.config import PortfolioConfig  # noqa: E402
from crypto_trader.strategy.momentum.config import MomentumConfig  # noqa: E402


DATA_DIR = ROOT / "data"
OUTPUT_BASE = ROOT / "output" / "momentum"
ROUND2_CONFIG_PATH = OUTPUT_BASE / "round_2" / "optimized_config.json"
DEFAULT_REPAIR_CONFIG_PATH = (
    OUTPUT_BASE
    / "round_2_oos_repair_followup"
    / "20260526T100228Z_focus"
    / "recommended_followup_config.json"
)
DEFAULT_REPAIR_SUMMARY_PATH = DEFAULT_REPAIR_CONFIG_PATH.with_name("followup_summary.json")
LIVE_STRATEGY_CONFIG_PATH = ROOT / "config" / "strategies" / "momentum.json"
LIVE_CONFIG_EXAMPLE_PATH = ROOT / "config" / "live_config.example.json"
PORTFOLIO_CONFIG_PATH = ROOT / "config" / "portfolio_config.json"

ROUND_NUM = 3
SYMBOLS = ["BTC", "ETH", "SOL"]
IS_START = date(2026, 2, 25)
IS_END = date(2026, 4, 20)
OOS_START = date(2026, 4, 21)
OOS_END = date(2026, 5, 23)
FULL_START = IS_START
FULL_END = OOS_END

ROUND3_MUTATIONS: dict[str, Any] = {
    "setup.min_room_b": 0.0,
    "bias.h4_ema_slope_lookback": 2,
    "symbol_filter.eth_direction": "long_only",
    "symbol_filter.btc_direction": "long_only",
    "symbol_filter.sol_direction": "long_only",
    "risk.risk_pct_b": 0.015,
}


def _quiet_logging() -> None:
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def _load_strategy_payload(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw.get("strategy", raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected strategy mapping in {path}")
    return payload


def _load_config(path: Path) -> MomentumConfig:
    return MomentumConfig.from_dict(_load_strategy_payload(path))


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _diff_values(base: MomentumConfig, candidate: MomentumConfig) -> dict[str, dict[str, Any]]:
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


def _run_window(config: MomentumConfig, start: date, end: date) -> dict[str, Any]:
    backtest_config = _build_backtest_config(start, end)
    result = run(deepcopy(config), backtest_config, data_dir=DATA_DIR, strategy_type="momentum")
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


def _live_config_for_warning_scan() -> LiveConfig:
    if LIVE_CONFIG_EXAMPLE_PATH.exists():
        payload = _load_optional_json(LIVE_CONFIG_EXAMPLE_PATH)
        if payload:
            return LiveConfig.from_dict(payload)
    return LiveConfig(
        is_testnet=True,
        symbols=SYMBOLS,
        strategy_configs={"momentum": LIVE_STRATEGY_CONFIG_PATH},
        portfolio_config_path=PORTFOLIO_CONFIG_PATH if PORTFOLIO_CONFIG_PATH.exists() else None,
    )


def _portfolio_config_for_warning_scan() -> PortfolioConfig | None:
    if not PORTFOLIO_CONFIG_PATH.exists():
        return None
    return PortfolioConfig.from_dict(_load_optional_json(PORTFOLIO_CONFIG_PATH))


def _write_live_strategy_config(config: MomentumConfig) -> dict[str, Any]:
    payload = {"strategy": config.to_dict()}
    _atomic_write_json(_json_ready(payload), LIVE_STRATEGY_CONFIG_PATH)
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


def _compare_live_backtest_metrics(
    *,
    promoted_config: MomentumConfig,
    live_config: MomentumConfig,
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
        key: float(live_metrics.get(key, 0.0)) - float(full_metrics.get(key, 0.0))
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


def _collect_parity_warnings(live_config: MomentumConfig) -> list[dict[str, str]]:
    warnings = collect_live_parity_warnings(
        _live_config_for_warning_scan(),
        _portfolio_config_for_warning_scan(),
        durable_oms_available=True,
        exchange_metadata_enforced=True,
        strategy_configs={"momentum": live_config},
        capabilities=HyperliquidExecutionAdapter.capabilities,
    )
    return [warning.to_dict() for warning in warnings]


def _write_optimized_config(
    *,
    config: MomentumConfig,
    contract: dict[str, Any],
    round_dir: Path,
    repair_config_path: Path,
    repair_summary_path: Path,
    selected_candidate: dict[str, Any],
    live_alignment: dict[str, Any],
) -> Path:
    payload = {
        "strategy": config.to_dict(),
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "round": ROUND_NUM,
            "promotion_type": "oos_repair_followup",
            "source_round": 2,
            "source_config": str(ROUND2_CONFIG_PATH),
            "source_repair_config": str(repair_config_path),
            "source_repair_summary": str(repair_summary_path),
            "selected_candidate_label": selected_candidate.get("label", ""),
            "selected_candidate_thesis": selected_candidate.get("thesis", ""),
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
    parity_warnings: list[dict[str, str]],
) -> str:
    warning_lines = [
        f"- {warning.get('severity', 'warning')}: {warning.get('warning_id', '')}"
        for warning in parity_warnings
    ] or ["- none"]
    lines = [
        "Momentum Round 3 OOS Repair Promotion",
        "=" * 39,
        "",
        "Selected repair:",
        f"- label: {selected_candidate.get('label', 'followup:eth_btc_long_sol_long_risk_015')}",
        "- setup.min_room_b: 0.0",
        "- bias.h4_ema_slope_lookback: 2",
        "- symbol_filter.btc_direction: long_only",
        "- symbol_filter.eth_direction: long_only",
        "- symbol_filter.sol_direction: long_only",
        "- risk.risk_pct_b: 0.015",
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
        help="Follow-up repair optimized config JSON to promote.",
    )
    parser.add_argument(
        "--repair-summary",
        default=str(DEFAULT_REPAIR_SUMMARY_PATH),
        help="Follow-up repair summary JSON for provenance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _quiet_logging()

    repair_config_path = Path(args.repair_config)
    repair_summary_path = Path(args.repair_summary)
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
    plugin = MomentumRound2AlphaPlugin(
        full_backtest_config,
        round2_config,
        data_dir=DATA_DIR,
        max_workers=None,
    )
    contract = build_optimization_contract(
        strategy_type="momentum",
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

    result = run(
        deepcopy(promoted_config),
        full_backtest_config,
        data_dir=DATA_DIR,
        strategy_type="momentum",
    )
    full_metrics = metrics_to_dict(result.metrics)
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

    is_result = _run_window(promoted_config, IS_START, IS_END)
    oos_result = _run_window(promoted_config, OOS_START, OOS_END)
    followup_summary = _load_optional_json(repair_summary_path)
    selected_candidate = followup_summary.get("best") or {
        "label": "followup:eth_btc_long_sol_long_risk_015",
        "thesis": "Risk-scale long-only BTC/SOL guard.",
    }

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
        selected_candidate=selected_candidate,
        live_alignment=live_alignment,
    )
    generate_report(result, round_dir)
    export_equity_curve(result, round_dir)
    export_trade_journal(result, round_dir)

    diagnostics = generate_diagnostics(
        list(result.trades),
        initial_equity=full_backtest_config.initial_equity,
        title="Momentum Round 3 OOS Repair Final Diagnostics",
        terminal_marks=list(result.terminal_marks),
        performance_metrics=result.metrics,
        expected_symbols=SYMBOLS,
        diagnostic_context=result.diagnostic_context,
    )
    (round_dir / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")

    mutation_diff = _diff_values(round2_config, promoted_config)
    evaluation = _build_evaluation_text(
        full_metrics=full_metrics,
        is_metrics=is_result["metrics"],
        oos_metrics=oos_result["metrics"],
        mutation_diff=mutation_diff,
        contract=contract,
        selected_candidate=selected_candidate,
        live_alignment=live_alignment,
        parity_warnings=parity_warnings,
    )
    (round_dir / "round_evaluation.txt").write_text(evaluation, encoding="utf-8")

    gate_result = {
        "passed": True,
        "failure_reasons": [],
        "validation_mode": "strict",
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "preflight": "passed",
        "acceptance_basis": "user_approved_followup_oos_repair",
        "standard_full_period_hard_reject_evaluation": {
            "rejected": hard_rejected,
            "reject_reason": hard_reject_reason,
            "score_with_hard_rejects": hard_gate_score,
        },
        "live_backtest_parity": live_alignment["live_backtest_parity"]["status"],
    }
    phase_result = {
        "final_score": score,
        "accepted_count": len(ROUND3_MUTATIONS),
        "new_mutations": ROUND3_MUTATIONS,
        "final_validation": {
            "in_sample": {"start": IS_START.isoformat(), "end": IS_END.isoformat()},
            "out_of_sample": {"start": OOS_START.isoformat(), "end": OOS_END.isoformat()},
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
            "source": "followup_oos_repair_rerun",
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
        "selected_candidate": selected_candidate,
        "mutations": ROUND3_MUTATIONS,
        "mutation_diff_vs_round2": mutation_diff,
        "windows": {
            "in_sample": {"start": IS_START.isoformat(), "end": IS_END.isoformat()},
            "out_of_sample": {"start": OOS_START.isoformat(), "end": OOS_END.isoformat()},
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
        },
        "final_metrics": full_metrics,
        "in_sample_metrics": is_result["metrics"],
        "out_of_sample_metrics": oos_result["metrics"],
        "score": score,
        "gate_result": gate_result,
        "phase_result": phase_result,
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
        "strategy_type": "momentum",
        "promotion_type": "oos_repair_followup",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(ROUND2_CONFIG_PATH),
        "source_repair_config": str(repair_config_path),
        "source_repair_summary": str(repair_summary_path),
        "selected_candidate_label": selected_candidate.get("label", ""),
        "mutations": ROUND3_MUTATIONS,
        "windows": summary["windows"],
        "contract_hash": contract.get("contract_hash", ""),
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "profile_hash": contract.get("profile_hash", ""),
        "required_timeframes": contract.get("required_timeframes", []),
        "symbols": SYMBOLS,
        "live_strategy_config": str(LIVE_STRATEGY_CONFIG_PATH),
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
