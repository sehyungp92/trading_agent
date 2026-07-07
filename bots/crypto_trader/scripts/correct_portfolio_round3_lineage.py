"""Restore strategy lineage after portfolio round_3 adoption.

Portfolio round_3 remains the promoted deployment bundle.  This script restores
global strategy round_3 artifacts from pre-adoption backups and points the live
example config at the portfolio-specific bundle, keeping deployment separate
from strategy optimization lineage.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DATA_DIR = ROOT / "data"
PORTFOLIO_R3 = ROOT / "output" / "portfolio" / "round_3"
DEPLOYMENT_MANIFEST = PORTFOLIO_R3 / "deployment_manifest.json"
PORTFOLIO_MANIFEST = ROOT / "output" / "portfolio" / "rounds_manifest.json"
SOURCE_DIR = PORTFOLIO_R3 / "source_pre_adoption"
LIVE_EXAMPLE = ROOT / "config" / "live_config.example.json"
BASELINE_PORTFOLIO_CONFIG = ROOT / "config" / "portfolio_config.json"
BASELINE_STRATEGY_DIR = ROOT / "config" / "strategies"
ROUND_LABEL = "extended_body108_scorea250_breakbody0715_riskall_1.4"
STRATEGIES = ("momentum", "trend", "breakout")


def _quiet_logging() -> None:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    import structlog

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


def _strategy_class(strategy_id: str) -> type[Any]:
    from crypto_trader.strategy.breakout.config import BreakoutConfig
    from crypto_trader.strategy.momentum.config import MomentumConfig
    from crypto_trader.strategy.trend.config import TrendConfig

    if strategy_id == "momentum":
        return MomentumConfig
    if strategy_id == "trend":
        return TrendConfig
    if strategy_id == "breakout":
        return BreakoutConfig
    raise ValueError(strategy_id)


def _strategy_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    strategy = payload.get("strategy", payload)
    if not isinstance(strategy, dict):
        raise TypeError(f"Expected strategy payload in {path}")
    return strategy


def _load_strategy_config(strategy_id: str, path: Path) -> Any:
    return _strategy_class(strategy_id).from_dict(_strategy_payload(path))


def _copy_json_backup(src: Path, dest: Path) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    before_hash = None
    if dest.exists():
        before_hash = _hash_json(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return {
        "source": str(src),
        "destination": str(dest),
        "before_hash": before_hash,
        "after_hash": _hash_json(dest),
    }


def _pre_adoption_backup(base_dir: Path, filename: str) -> Path:
    direct = base_dir / filename
    if direct.exists():
        return direct
    archived = sorted((base_dir / "archive").glob(f"*/{filename}"))
    if archived:
        return archived[-1]
    return direct


def _portfolio_manifest_backup() -> Path:
    archive_root = ROOT / "output" / "portfolio" / "archive"
    archived = sorted(archive_root.glob("*/rounds_manifest.json"))
    if archived:
        return archived[-1]
    return PORTFOLIO_MANIFEST


def _hash_json(path: Path) -> str:
    from crypto_trader.optimize.contracts import stable_hash

    return stable_hash(_read_json(path))


def _restore_strategy_lineage() -> dict[str, Any]:
    restored: dict[str, Any] = {}
    for strategy_id in STRATEGIES:
        round_dir = ROOT / "output" / strategy_id / "round_3"
        base_dir = ROOT / "output" / strategy_id
        restored[strategy_id] = {
            "optimized_config": _copy_json_backup(
                round_dir / "optimized_config.pre_portfolio_round3_adoption.json",
                round_dir / "optimized_config.json",
            ),
            "rounds_manifest": _copy_json_backup(
                _pre_adoption_backup(
                    base_dir,
                    "rounds_manifest.pre_portfolio_round3_adoption.json",
                ),
                base_dir / "rounds_manifest.json",
            ),
        }
    return restored


def _restore_baseline_config_materialization() -> dict[str, Any]:
    restored: dict[str, Any] = {
        "portfolio_config": _copy_json_backup(
            ROOT / "config" / "portfolio_config.pre_portfolio_round3_adoption.json",
            BASELINE_PORTFOLIO_CONFIG,
        ),
        "strategies": {},
    }
    for strategy_id in STRATEGIES:
        restored["strategies"][strategy_id] = _copy_json_backup(
            BASELINE_STRATEGY_DIR / f"{strategy_id}.pre_portfolio_round3_adoption.json",
            BASELINE_STRATEGY_DIR / f"{strategy_id}.json",
        )
    return restored


def _restore_portfolio_manifest() -> dict[str, Any]:
    return _copy_json_backup(_portfolio_manifest_backup(), PORTFOLIO_MANIFEST)


def _update_live_example_to_portfolio_bundle() -> dict[str, Any]:
    from crypto_trader.optimize.phase_state import _atomic_write_json

    payload = _read_json(LIVE_EXAMPLE)
    expected_paths = _expected_live_config_paths()
    payload["strategy_configs"] = expected_paths["strategy_configs"]
    payload["portfolio_config_path"] = expected_paths["portfolio_config_path"]
    payload["deployment_manifest_path"] = expected_paths["deployment_manifest_path"]
    _atomic_write_json(_json_ready(payload), LIVE_EXAMPLE)
    return {
        "path": str(LIVE_EXAMPLE),
        "strategy_configs": payload["strategy_configs"],
        "portfolio_config_path": payload["portfolio_config_path"],
        "deployment_manifest_path": payload["deployment_manifest_path"],
    }


def _expected_live_config_paths() -> dict[str, Any]:
    return {
        "strategy_configs": {
            strategy_id: str(
                Path("output")
                / "portfolio"
                / "round_3"
                / "recommended_strategy_configs"
                / f"{strategy_id}.json"
            ).replace("\\", "/")
            for strategy_id in STRATEGIES
        },
        "portfolio_config_path": str(
            Path("output") / "portfolio" / "round_3" / "recommended_portfolio_config.json"
        ).replace("\\", "/"),
        "deployment_manifest_path": str(
            Path("output") / "portfolio" / "round_3" / "deployment_manifest.json"
        ).replace("\\", "/"),
    }


def _check_json_match(
    *,
    label: str,
    actual: Path,
    expected: Path,
    failures: list[str],
) -> dict[str, Any]:
    result = {
        "label": label,
        "actual": str(actual),
        "expected": str(expected),
        "status": "missing",
    }
    if not actual.exists():
        failures.append(f"{label} is missing: {actual}")
        return result
    if not expected.exists():
        failures.append(f"{label} reference is missing: {expected}")
        return result
    actual_hash = _hash_json(actual)
    expected_hash = _hash_json(expected)
    result.update({
        "actual_hash": actual_hash,
        "expected_hash": expected_hash,
        "status": "matched" if actual_hash == expected_hash else "mismatch",
    })
    if actual_hash != expected_hash:
        failures.append(f"{label} does not match {expected}")
    return result


def _manifest_rounds(path: Path, failures: list[str], label: str) -> list[int]:
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"{label} could not be read: {exc}")
        return []
    rounds = payload.get("rounds") if isinstance(payload, dict) else None
    if not isinstance(rounds, list):
        failures.append(f"{label} does not contain a rounds list")
        return []
    return [item.get("round") for item in rounds if isinstance(item, dict)]


def _check_deployment_manifest(expected_live_paths: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(DEPLOYMENT_MANIFEST), "status": "missing"}
    if not DEPLOYMENT_MANIFEST.exists():
        failures.append(f"deployment manifest is missing: {DEPLOYMENT_MANIFEST}")
        return result
    try:
        payload = _read_json(DEPLOYMENT_MANIFEST)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"deployment manifest could not be read: {exc}")
        result["status"] = "unreadable"
        return result

    status = {
        "required_strategy_ids": payload.get("required_strategy_ids") == list(STRATEGIES),
        "portfolio_config_path": payload.get("portfolio_config_path") == expected_live_paths["portfolio_config_path"],
        "strategy_configs": payload.get("strategy_configs") == expected_live_paths["strategy_configs"],
        "portfolio_rounds_manifest_path": payload.get("portfolio_rounds_manifest_path")
        == "output/portfolio/rounds_manifest.json",
        "required_portfolio_rounds": payload.get("required_portfolio_rounds") == [1, 2, 3],
        "parity_alignment_path": payload.get("parity_alignment_path")
        == "output/portfolio/round_3/parity_alignment.json",
    }
    result.update({"status": "passed" if all(status.values()) else "failed", "fields": status})
    if not all(status.values()):
        failures.append("output/portfolio/round_3/deployment_manifest.json does not describe the expected round 3 bundle")
    return result


def _check_lineage_state() -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, Any] = {
        "strategy_lineage": {},
        "baseline_config_materialization": {},
        "portfolio_manifest": {},
        "deployment_manifest": {},
        "live_example": {},
        "portfolio_parity": {},
    }

    for strategy_id in STRATEGIES:
        round_dir = ROOT / "output" / strategy_id / "round_3"
        base_dir = ROOT / "output" / strategy_id
        checks["strategy_lineage"][strategy_id] = {
            "optimized_config": _check_json_match(
                label=f"{strategy_id} optimized_config.json",
                actual=round_dir / "optimized_config.json",
                expected=round_dir / "optimized_config.pre_portfolio_round3_adoption.json",
                failures=failures,
            ),
            "rounds_manifest": _check_json_match(
                label=f"{strategy_id} rounds_manifest.json",
                actual=base_dir / "rounds_manifest.json",
                expected=_pre_adoption_backup(
                    base_dir,
                    "rounds_manifest.pre_portfolio_round3_adoption.json",
                ),
                failures=failures,
            ),
        }

    checks["baseline_config_materialization"]["portfolio_config"] = _check_json_match(
        label="config/portfolio_config.json",
        actual=BASELINE_PORTFOLIO_CONFIG,
        expected=ROOT / "config" / "portfolio_config.pre_portfolio_round3_adoption.json",
        failures=failures,
    )
    checks["baseline_config_materialization"]["strategies"] = {}
    for strategy_id in STRATEGIES:
        checks["baseline_config_materialization"]["strategies"][strategy_id] = _check_json_match(
            label=f"config/strategies/{strategy_id}.json",
            actual=BASELINE_STRATEGY_DIR / f"{strategy_id}.json",
            expected=BASELINE_STRATEGY_DIR / f"{strategy_id}.pre_portfolio_round3_adoption.json",
            failures=failures,
        )

    checks["portfolio_manifest"] = _check_json_match(
        label="output/portfolio/rounds_manifest.json",
        actual=PORTFOLIO_MANIFEST,
        expected=_portfolio_manifest_backup(),
        failures=failures,
    )
    if PORTFOLIO_MANIFEST.exists():
        rounds = _manifest_rounds(PORTFOLIO_MANIFEST, failures, "output/portfolio/rounds_manifest.json")
        checks["portfolio_manifest"]["rounds"] = rounds
        if rounds != [1, 2, 3]:
            failures.append(f"output/portfolio/rounds_manifest.json rounds {rounds} do not equal [1, 2, 3]")

    expected_live_paths = _expected_live_config_paths()
    live_payload = _read_json(LIVE_EXAMPLE)
    live_path_status = {
        "strategy_configs": live_payload.get("strategy_configs") == expected_live_paths["strategy_configs"],
        "portfolio_config_path": live_payload.get("portfolio_config_path") == expected_live_paths["portfolio_config_path"],
        "deployment_manifest_path": live_payload.get("deployment_manifest_path") == expected_live_paths["deployment_manifest_path"],
    }
    if not all(live_path_status.values()):
        failures.append("config/live_config.example.json does not point to the portfolio round 3 deployment bundle")
    checks["live_example"] = {
        "path": str(LIVE_EXAMPLE),
        "expected_paths": expected_live_paths,
        "status": live_path_status,
    }

    checks["deployment_manifest"] = _check_deployment_manifest(expected_live_paths, failures)

    for strategy_id, path_text in expected_live_paths["strategy_configs"].items():
        checks["live_example"][f"{strategy_id}_bundle"] = _check_json_match(
            label=f"live example {strategy_id} bundle path",
            actual=_resolve_config_path(path_text),
            expected=PORTFOLIO_R3 / "recommended_strategy_configs" / f"{strategy_id}.json",
            failures=failures,
        )
    checks["live_example"]["portfolio_bundle"] = _check_json_match(
        label="live example portfolio bundle path",
        actual=_resolve_config_path(expected_live_paths["portfolio_config_path"]),
        expected=PORTFOLIO_R3 / "recommended_portfolio_config.json",
        failures=failures,
    )

    parity_path = PORTFOLIO_R3 / "parity_alignment.json"
    if parity_path.exists():
        parity = _read_json(parity_path)
        replay = parity.get("portfolio_metric_replay", {})
        checks["portfolio_parity"] = {
            "path": str(parity_path),
            "status": replay.get("status"),
            "max_abs_delta": replay.get("max_abs_delta"),
            "tolerance": replay.get("tolerance"),
        }
        if replay.get("status") != "matched":
            failures.append("portfolio round 3 parity_alignment.json is not matched")
    else:
        failures.append(f"portfolio parity evidence is missing: {parity_path}")
        checks["portfolio_parity"] = {"path": str(parity_path), "status": "missing"}

    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "checks": checks,
    }


def _run_check() -> None:
    result = _check_lineage_state()
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    if result["failures"]:
        raise SystemExit(1)


def _resolve_config_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_live_example_bundle() -> tuple[Any, dict[str, Any], dict[str, str]]:
    from crypto_trader.portfolio.config import PortfolioConfig

    payload = _read_json(LIVE_EXAMPLE)
    portfolio_path = _resolve_config_path(payload["portfolio_config_path"])
    strategy_paths = {
        strategy_id: _resolve_config_path(path_text)
        for strategy_id, path_text in payload["strategy_configs"].items()
    }
    portfolio_config = PortfolioConfig.from_dict(_read_json(portfolio_path))
    strategy_configs = {
        strategy_id: _load_strategy_config(strategy_id, path)
        for strategy_id, path in strategy_paths.items()
    }
    return portfolio_config, strategy_configs, {
        "portfolio_config_path": str(portfolio_path),
        "strategy_config_paths": {
            strategy_id: str(path) for strategy_id, path in strategy_paths.items()
        },
    }


def _run_strategy_full_diagnostics(strategy_id: str, store: Any) -> dict[str, Any]:
    from crypto_trader.backtest.analysis import (
        export_equity_curve,
        export_trade_journal,
        generate_report,
    )
    from crypto_trader.backtest.diagnostics import generate_diagnostics
    from crypto_trader.backtest.metrics import metrics_to_dict
    from crypto_trader.backtest.profiles import (
        LIVE_PARITY_PROFILE,
        build_backtest_config_from_profile,
    )
    from crypto_trader.backtest.runner import run
    from crypto_trader.optimize.contracts import strategy_config_hash
    from crypto_trader.optimize.phase_state import _atomic_write_json
    from crypto_trader.optimize.portfolio_round2_phased import FULL_END, FULL_START, SYMBOLS

    round_dir = ROOT / "output" / strategy_id / "round_3"
    config = _load_strategy_config(strategy_id, round_dir / "optimized_config.json")
    bt_config = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=list(SYMBOLS),
        start_date=FULL_START,
        end_date=FULL_END,
    )
    result = run(
        deepcopy(config),
        bt_config,
        data_dir=DATA_DIR,
        store=store,
        strategy_type=strategy_id,
    )
    generate_report(result, round_dir)
    export_equity_curve(result, round_dir)
    export_trade_journal(result, round_dir)
    diagnostics = generate_diagnostics(
        list(result.trades),
        initial_equity=LIVE_PARITY_PROFILE.initial_equity,
        title=f"{strategy_id.title()} Round 3 Restored Strategy-Lineage Final Diagnostics",
        terminal_marks=list(result.terminal_marks),
        performance_metrics=result.metrics,
        expected_symbols=list(SYMBOLS),
        diagnostic_context=result.diagnostic_context,
    )
    (round_dir / "round_final_diagnostics.txt").write_text(diagnostics, encoding="utf-8")
    metrics = metrics_to_dict(result.metrics)
    metrics["terminal_mark_count"] = len(result.terminal_marks)
    config_hash = strategy_config_hash(config)
    evaluation = [
        f"{strategy_id.title()} Round 3 Strategy-Lineage Restore",
        "=" * (len(strategy_id) + 34),
        "",
        "Correction:",
        "- Restored optimized_config.json from the pre-portfolio-adoption backup.",
        "- Portfolio round_3 deployment overlays remain only under output/portfolio/round_3.",
        "- Live example now points to the portfolio round_3 bundle instead of this strategy lineage folder.",
        "",
        "Full restored round diagnostics:",
        (
            f"- Full: return {metrics.get('net_return_pct', 0.0):.4f}%, "
            f"trades {metrics.get('total_trades', 0.0):.0f}, "
            f"win rate {metrics.get('win_rate', 0.0):.2f}%, "
            f"PF {metrics.get('profit_factor', 0.0):.4f}, "
            f"expectancy {metrics.get('expectancy_r', 0.0):.4f}R, "
            f"DD {metrics.get('max_drawdown_pct', 0.0):.4f}%"
        ),
        "",
        "Lineage:",
        f"- restored_from: {round_dir / 'optimized_config.pre_portfolio_round3_adoption.json'}",
        f"- strategy_config_hash: {config_hash}",
        f"- live_deployment_bundle: {PORTFOLIO_R3}",
        "",
    ]
    (round_dir / "round_evaluation.txt").write_text("\n".join(evaluation), encoding="utf-8")
    summary = {
        "round": 3,
        "strategy_type": strategy_id,
        "lineage_status": "restored_pre_portfolio_adoption_strategy_round3",
        "correction_context": {
            "portfolio_candidate": ROUND_LABEL,
            "portfolio_bundle": str(PORTFOLIO_R3),
            "reason": "Portfolio-context overlay must not mutate strategy lineage artifacts.",
        },
        "source_config_backup": str(round_dir / "optimized_config.pre_portfolio_round3_adoption.json"),
        "strategy_config_hash": config_hash,
        "windows": {
            "full": {"start": FULL_START.isoformat(), "end": FULL_END.isoformat()},
        },
        "final_metrics": metrics,
        "artifacts": {
            "optimized_config": str(round_dir / "optimized_config.json"),
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
                "round": 3,
                "strategy_type": strategy_id,
                "lineage_status": "restored_pre_portfolio_adoption_strategy_round3",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_config_backup": str(round_dir / "optimized_config.pre_portfolio_round3_adoption.json"),
                "symbols": list(SYMBOLS),
                "windows": summary["windows"],
                "strategy_config_hash": config_hash,
                "live_deployment_bundle": str(PORTFOLIO_R3),
            }
        ),
        round_dir / "run_spec.json",
    )
    _atomic_write_json(
        _json_ready(
            {
                "status": "strategy_lineage_restored",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "strategy_type": strategy_id,
                "strategy_config_hash": config_hash,
                "profile": LIVE_PARITY_PROFILE.to_dict(),
                "note": (
                    "This strategy round_3 folder is no longer the live deployment "
                    "materialization for the portfolio candidate. Live example paths "
                    "point to output/portfolio/round_3 instead."
                ),
                "live_deployment_bundle": str(PORTFOLIO_R3),
            }
        ),
        round_dir / "parity_alignment.json",
    )
    return {
        "metrics": metrics,
        "strategy_config_hash": config_hash,
        "round_final_diagnostics": str(round_dir / "round_final_diagnostics.txt"),
    }


def _run_portfolio_bundle_parity(store: Any) -> dict[str, Any]:
    from crypto_trader.backtest.metrics import metrics_to_dict
    from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE
    from crypto_trader.optimize.contracts import portfolio_config_hash, strategy_config_hash
    from crypto_trader.optimize.phase_state import _atomic_write_json
    from crypto_trader.optimize.portfolio_round2_phased import (
        FULL_END,
        FULL_START,
        _augment_metrics,
        _bt_config,
    )
    from crypto_trader.portfolio.backtest_runner import run_portfolio_backtest

    portfolio_config, strategy_configs, paths = _load_live_example_bundle()
    result = run_portfolio_backtest(
        portfolio_config=deepcopy(portfolio_config),
        strategy_configs={sid: deepcopy(cfg) for sid, cfg in strategy_configs.items()},
        backtest_config=_bt_config(FULL_START, FULL_END, portfolio_config.initial_equity),
        data_dir=DATA_DIR,
        store=store,
        terminal_accounting_mode=LIVE_PARITY_PROFILE.terminal_accounting_mode,
    )
    metrics = _augment_metrics(metrics_to_dict(result.metrics), result)
    metrics["terminal_mark_count"] = sum(len(marks) for marks in result.terminal_marks.values())
    expected = _read_json(PORTFOLIO_R3 / "portfolio_summary.json")["metrics"]["full"]
    keys = [
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
    ]
    deltas = {
        key: float(metrics.get(key, 0.0)) - float(expected.get(key, 0.0))
        for key in keys
    }
    max_abs_delta = max(abs(value) for value in deltas.values())
    status = "matched" if max_abs_delta <= 1e-9 else "mismatch"
    if status != "matched":
        raise RuntimeError(json.dumps({"status": status, "deltas": deltas}, indent=2))
    parity = {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": LIVE_PARITY_PROFILE.to_dict(),
        "terminal_accounting_mode": LIVE_PARITY_PROFILE.terminal_accounting_mode,
        "source_candidate": ROUND_LABEL,
        "live_config_example": str(LIVE_EXAMPLE),
        "deployment_model": "live_config_example_points_to_portfolio_round_3_bundle",
        "bundle_paths": paths,
        "portfolio_metric_replay": {
            "status": status,
            "deltas": deltas,
            "max_abs_delta": max_abs_delta,
            "tolerance": 1e-9,
        },
        "live_config_hashes": {
            "portfolio": {
                "path": paths["portfolio_config_path"],
                "hash": portfolio_config_hash(portfolio_config),
                "status": "matched",
            },
            "strategies": {
                strategy_id: {
                    "path": paths["strategy_config_paths"][strategy_id],
                    "hash": strategy_config_hash(config),
                    "status": "matched",
                }
                for strategy_id, config in strategy_configs.items()
            },
        },
    }
    _atomic_write_json(_json_ready(parity), PORTFOLIO_R3 / "parity_alignment.json")
    return parity


def _write_lineage_correction_summary(
    *,
    restored_lineage: dict[str, Any],
    restored_baseline_configs: dict[str, Any],
    restored_portfolio_manifest: dict[str, Any],
    live_example: dict[str, Any],
    strategy_diagnostics: dict[str, Any],
    portfolio_parity: dict[str, Any],
) -> None:
    from crypto_trader.optimize.phase_state import _atomic_write_json

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_candidate": ROUND_LABEL,
        "correction": "Restored global strategy round_3 lineage and moved live deployment reference to the portfolio round_3 bundle.",
        "portfolio_round_3_preserved": str(PORTFOLIO_R3),
        "restored_strategy_lineage": restored_lineage,
        "restored_baseline_config_materialization": restored_baseline_configs,
        "restored_portfolio_manifest": restored_portfolio_manifest,
        "live_example": live_example,
        "strategy_diagnostics_after_restore": strategy_diagnostics,
        "portfolio_bundle_parity": portfolio_parity,
    }
    _atomic_write_json(_json_ready(summary), PORTFOLIO_R3 / "lineage_correction.json")


def _run_fix() -> None:
    from crypto_trader.data.store import ParquetStore

    restored_lineage = _restore_strategy_lineage()
    restored_baseline_configs = _restore_baseline_config_materialization()
    restored_portfolio_manifest = _restore_portfolio_manifest()
    live_example = _update_live_example_to_portfolio_bundle()
    store = ParquetStore(base_dir=DATA_DIR)
    strategy_diagnostics = {
        strategy_id: _run_strategy_full_diagnostics(strategy_id, store)
        for strategy_id in STRATEGIES
    }
    portfolio_parity = _run_portfolio_bundle_parity(store)
    _write_lineage_correction_summary(
        restored_lineage=restored_lineage,
        restored_baseline_configs=restored_baseline_configs,
        restored_portfolio_manifest=restored_portfolio_manifest,
        live_example=live_example,
        strategy_diagnostics=strategy_diagnostics,
        portfolio_parity=portfolio_parity,
    )
    rows = [
        {
            "strategy": strategy_id,
            "net_return_pct": info["metrics"]["net_return_pct"],
            "total_trades": info["metrics"]["total_trades"],
            "profit_factor": info["metrics"]["profit_factor"],
            "max_drawdown_pct": info["metrics"]["max_drawdown_pct"],
        }
        for strategy_id, info in strategy_diagnostics.items()
    ]
    csv_path = PORTFOLIO_R3 / "lineage_correction_strategy_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "net_return_pct",
                "total_trades",
                "profit_factor",
                "max_drawdown_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            _json_ready(
                {
                    "status": "completed",
                    "live_example": live_example,
                    "portfolio_parity": portfolio_parity["portfolio_metric_replay"],
                    "strategy_diagnostics": rows,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Run read-only lineage and deployment bundle checks. This is the default.",
    )
    mode.add_argument(
        "--fix",
        action="store_true",
        help="Restore lineage artifacts and regenerate diagnostics/parity evidence.",
    )
    return parser.parse_args()


def main() -> None:
    _quiet_logging()
    args = _parse_args()
    if args.fix:
        _run_fix()
        return
    _run_check()


if __name__ == "__main__":
    main()
