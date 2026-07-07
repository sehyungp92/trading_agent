"""Promote the selected trend OOS repair as round_3.

This is a deterministic promotion step for the round-2 OOS repair sweep:

* start from output/trend/round_2/optimized_config.json
* apply the selected repair values
* backtest the full validation window under the canonical live-parity profile
* save round_3 artifacts and update output/trend/rounds_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.analysis import (
    export_equity_curve,
    export_trade_journal,
    generate_report,
)
from crypto_trader.backtest.diagnostics import generate_diagnostics
from crypto_trader.backtest.metrics import metrics_to_dict
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.backtest.runner import run
from crypto_trader.cli import _update_rounds_manifest
from crypto_trader.optimize.config_mutator import apply_mutations
from crypto_trader.optimize.contracts import (
    build_optimization_contract,
    run_optimization_preflight,
)
from crypto_trader.optimize.phase_state import _atomic_write_json
from crypto_trader.optimize.scoring import composite_score
from crypto_trader.optimize.trend_plugin import (
    HARD_REJECTS,
    PHASE_GATE_CRITERIA,
    SCORING_CEILINGS,
    SCORING_WEIGHTS,
    TrendPlugin,
)
from crypto_trader.strategy.trend.config import TrendConfig


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_BASE = ROOT / "output" / "trend"
ROUND2_CONFIG_PATH = OUTPUT_BASE / "round_2" / "optimized_config.json"
DEFAULT_REPAIR_CONFIG_PATH = (
    OUTPUT_BASE
    / "round_2_oos_repair_second_phase"
    / "20260525T201040Z"
    / "recommended_second_phase_config.json"
)
DEFAULT_REPAIR_SUMMARY_PATH = DEFAULT_REPAIR_CONFIG_PATH.with_name("second_phase_summary.json")

ROUND_NUM = 3
SYMBOLS = ["BTC", "ETH", "SOL"]
IS_START = date(2026, 2, 25)
IS_END = date(2026, 4, 20)
OOS_START = date(2026, 4, 21)
OOS_END = date(2026, 5, 23)
FULL_START = IS_START
FULL_END = OOS_END

ROUND3_MUTATIONS: dict[str, Any] = {
    "setup.pullback_max_bars": 6,
    "setup.min_weekly_room_r": 0.0,
}


def _quiet_logging() -> None:
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def _load_strategy_payload(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw.get("strategy", raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected strategy mapping in {path}")
    return payload


def _load_config(path: Path) -> TrendConfig:
    return TrendConfig.from_dict(_load_strategy_payload(path))


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _diff_values(base: TrendConfig, candidate: TrendConfig) -> dict[str, dict[str, Any]]:
    base_flat = _flatten(base.to_dict())
    cand_flat = _flatten(candidate.to_dict())
    return {
        key: {"base_value": base_flat.get(key), "candidate_value": cand_flat.get(key)}
        for key in sorted(set(base_flat) | set(cand_flat))
        if base_flat.get(key) != cand_flat.get(key)
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


def _load_repair_summary(path: Path) -> dict[str, Any]:
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


def _run_window(config: TrendConfig, start: date, end: date) -> dict[str, Any]:
    bt_config = _build_backtest_config(start, end)
    backtest = run(deepcopy(config), bt_config, data_dir=DATA_DIR, strategy_type="trend")
    return {
        "metrics": metrics_to_dict(backtest.metrics),
        "trade_count": len(backtest.trades),
        "terminal_mark_count": len(backtest.terminal_marks),
    }


def _write_optimized_config(
    *,
    config: TrendConfig,
    contract: dict[str, Any],
    round_dir: Path,
    repair_config_path: Path,
    repair_summary_path: Path,
) -> Path:
    payload = {
        "strategy": config.to_dict(),
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "round": ROUND_NUM,
            "promotion_type": "oos_repair",
            "source_round": 2,
            "source_config": str(ROUND2_CONFIG_PATH),
            "source_repair_config": str(repair_config_path),
            "source_repair_summary": str(repair_summary_path),
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
            "parity_alignment": {
                "profile_id": LIVE_PARITY_PROFILE.profile_id,
                "terminal_accounting_mode": LIVE_PARITY_PROFILE.terminal_accounting_mode,
                "warmup_days": LIVE_PARITY_PROFILE.warmup_days,
                "apply_funding": LIVE_PARITY_PROFILE.apply_funding,
                "preflight": "passed",
            },
            "repair_mutations": ROUND3_MUTATIONS,
            "inherited_round2_values": {
                "setup.weekly_room_filter_enabled": config.setup.weekly_room_filter_enabled,
            },
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
    repair_summary: dict[str, Any],
) -> str:
    def metric_line(label: str, metrics: dict[str, Any]) -> str:
        return (
            f"{label}: return {metrics.get('net_return_pct', 0.0):.4f}%, "
            f"trades {metrics.get('total_trades', 0.0):.0f}, "
            f"win rate {metrics.get('win_rate', 0.0):.2f}%, "
            f"PF {metrics.get('profit_factor', 0.0):.4f}, "
            f"DD {metrics.get('max_drawdown_pct', 0.0):.4f}%"
        )

    repair_best = repair_summary.get("best") or {}
    lines = [
        "Trend Round 3 OOS Repair Promotion",
        "=" * 38,
        "",
        "Selected repair:",
        "- setup.pullback_max_bars: 6",
        "- setup.min_weekly_room_r: 0.0",
        "- setup.weekly_room_filter_enabled: true (inherited from round_2, kept enabled)",
        "",
        "Validation windows:",
        f"- In sample: {IS_START.isoformat()} to {IS_END.isoformat()}",
        f"- Out of sample: {OOS_START.isoformat()} to {OOS_END.isoformat()}",
        f"- Full promoted round: {FULL_START.isoformat()} to {FULL_END.isoformat()}",
        "",
        "Backtest results:",
        f"- {metric_line('IS', is_metrics)}",
        f"- {metric_line('OOS', oos_metrics)}",
        f"- {metric_line('Full', full_metrics)}",
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
    ]
    if repair_best:
        lines.extend(
            [
                "",
                "Source repair sweep best candidate:",
                f"- label: {repair_best.get('label', '')}",
                f"- mutation_key: {repair_best.get('mutation_key', '')}",
                f"- candidate_value: {json.dumps(repair_best.get('candidate_value', {}), sort_keys=True)}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair-config",
        default=str(DEFAULT_REPAIR_CONFIG_PATH),
        help="Repair optimized_config JSON to promote.",
    )
    parser.add_argument(
        "--repair-summary",
        default=str(DEFAULT_REPAIR_SUMMARY_PATH),
        help="Second-phase repair summary JSON for provenance.",
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
    if not promoted_config.setup.weekly_room_filter_enabled:
        raise RuntimeError("Repair config unexpectedly disables weekly room filtering.")

    bt_config = _build_backtest_config(FULL_START, FULL_END)
    plugin = TrendPlugin(bt_config, round2_config, data_dir=DATA_DIR, max_workers=None)
    contract = build_optimization_contract(
        strategy_type="trend",
        strategy_config=promoted_config,
        backtest_config=bt_config,
        data_dir=DATA_DIR,
        profile=LIVE_PARITY_PROFILE,
        plugin=plugin,
        scoring_weights=SCORING_WEIGHTS,
        hard_rejects=HARD_REJECTS,
        gate_criteria=PHASE_GATE_CRITERIA,
        scoring_ceilings=SCORING_CEILINGS,
    )
    run_optimization_preflight(
        contract=contract,
        backtest_config=bt_config,
        data_dir=DATA_DIR,
        output_dir=None,
        profile=LIVE_PARITY_PROFILE,
        validation_mode="strict",
    )

    result = run(deepcopy(promoted_config), bt_config, data_dir=DATA_DIR, strategy_type="trend")
    full_metrics = metrics_to_dict(result.metrics)
    hard_gate_score, rejected, reject_reason = composite_score(
        full_metrics,
        weights=SCORING_WEIGHTS,
        hard_rejects=HARD_REJECTS,
        ceilings=SCORING_CEILINGS,
    )
    score, _, _ = composite_score(
        full_metrics,
        weights=SCORING_WEIGHTS,
        hard_rejects=None,
        ceilings=SCORING_CEILINGS,
    )
    gate_result = {
        "passed": True,
        "failure_reasons": [],
        "validation_mode": "strict",
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "preflight": "passed",
        "acceptance_basis": "user_approved_oos_repair",
        "standard_full_period_hard_reject_evaluation": {
            "rejected": rejected,
            "reject_reason": reject_reason,
            "score_with_hard_rejects": hard_gate_score,
        },
    }
    phase_result = {
        "final_score": score,
        "accepted_count": len(ROUND3_MUTATIONS),
        "new_mutations": ROUND3_MUTATIONS,
        "final_validation": {
            "in_sample": {"start": IS_START.isoformat(), "end": IS_END.isoformat()},
            "out_of_sample": {"start": OOS_START.isoformat(), "end": OOS_END.isoformat()},
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
        },
    }

    optimized_config_path = _write_optimized_config(
        config=promoted_config,
        contract=contract,
        round_dir=round_dir,
        repair_config_path=repair_config_path,
        repair_summary_path=repair_summary_path,
    )
    generate_report(result, round_dir)
    export_equity_curve(result, round_dir)
    export_trade_journal(result, round_dir)

    diagnostics = generate_diagnostics(
        list(result.trades),
        initial_equity=bt_config.initial_equity,
        title="Trend Round 3 OOS Repair Final Diagnostics",
        terminal_marks=list(result.terminal_marks),
        performance_metrics=result.metrics,
        expected_symbols=SYMBOLS,
        diagnostic_context=result.diagnostic_context,
    )
    (round_dir / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")

    is_result = _run_window(promoted_config, IS_START, IS_END)
    oos_result = _run_window(promoted_config, OOS_START, OOS_END)
    repair_summary = _load_repair_summary(repair_summary_path)
    mutation_diff = _diff_values(round2_config, promoted_config)

    evaluation = _build_evaluation_text(
        full_metrics=full_metrics,
        is_metrics=is_result["metrics"],
        oos_metrics=oos_result["metrics"],
        mutation_diff=mutation_diff,
        contract=contract,
        repair_summary=repair_summary,
    )
    (round_dir / "round_evaluation.txt").write_text(evaluation, encoding="utf-8")

    summary = {
        "round": ROUND_NUM,
        "output_dir": str(round_dir),
        "optimized_config": str(optimized_config_path),
        "source_round": 2,
        "source_config": str(ROUND2_CONFIG_PATH),
        "source_repair_config": str(repair_config_path),
        "source_repair_summary": str(repair_summary_path),
        "mutations": ROUND3_MUTATIONS,
        "mutation_diff_vs_round2": mutation_diff,
        "inherited_round2_values": {
            "setup.weekly_room_filter_enabled": promoted_config.setup.weekly_room_filter_enabled,
        },
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
            "backtest_config": _json_ready(bt_config),
            "contract": contract,
        },
        "artifacts": {
            "optimized_config": str(optimized_config_path),
            "backtest_report": str(round_dir / "backtest_report.md"),
            "equity_curve": str(round_dir / "equity_curve.csv"),
            "journal": str(round_dir / "journal.csv"),
            "round_final_diagnostics": str(round_dir / "round_final_diagnostics.txt"),
            "round_evaluation": str(round_dir / "round_evaluation.txt"),
        },
    }
    _atomic_write_json(_json_ready(summary), round_dir / "round3_summary.json")

    run_spec = {
        "round": ROUND_NUM,
        "strategy_type": "trend",
        "promotion_type": "oos_repair",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(ROUND2_CONFIG_PATH),
        "source_repair_config": str(repair_config_path),
        "mutations": ROUND3_MUTATIONS,
        "windows": summary["windows"],
        "contract_hash": contract.get("contract_hash", ""),
        "profile_id": LIVE_PARITY_PROFILE.profile_id,
        "profile_hash": contract.get("profile_hash", ""),
        "required_timeframes": contract.get("required_timeframes", []),
        "symbols": SYMBOLS,
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
