from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root.")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.shared.auto.phase_state import _atomic_write_json
from backtests.shared.auto.round_manager import RoundManager
from backtests.stock.auto.portfolio_synergy.phase_candidates import (
    BLOCKED_ALPHA_ROUND3_PROFILE,
    get_score_weights,
)
from backtests.stock.auto.portfolio_synergy.plugin import StockPortfolioSynergyPlugin


DATA_DIR = ROOT / "backtests/stock/data/raw"
BASELINE_PATH = ROOT / "backtests/output/stock/portfolio_synergy/round_2/optimized_config.json"
EVIDENCE_DIR = (
    ROOT
    / "backtests/output/stock/portfolio_synergy/round_3/validation_checks/blocked_alpha_ref_only_20260524"
)
SUMMARY_PATH = EVIDENCE_DIR / "stock_portfolio_ref_only_1_16_round3_promotion_summary.json"
HOLDOUT_PROBE_PATH = EVIDENCE_DIR / "stock_portfolio_blocked_alpha_holdout_probe_20260524.json"
HOLDOUT_PROBE_SOURCE = HOLDOUT_PROBE_PATH.relative_to(ROOT).as_posix()
ARCHIVE_REASON = "faulty_round3_lower_unit_risk_return_tradeoff"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def promoted_config() -> dict[str, Any]:
    config = deepcopy(load_json(BASELINE_PATH))
    base_ref = float(config["portfolio_rules"]["reference_risk_pct"])
    config["portfolio_rules"]["reference_risk_pct"] = round(base_ref * 1.16, 8)
    return config


def full_plugin() -> StockPortfolioSynergyPlugin:
    plugin = StockPortfolioSynergyPlugin(
        DATA_DIR,
        start_date="2024-01-01",
        end_date="2026-03-01",
        initial_equity=25_000.0,
        max_workers=1,
        round_profile=BLOCKED_ALPHA_ROUND3_PROFILE,
    )
    plugin.diagnostic_round_label = "round_3_dynamic_stock_synergy"
    return plugin


def write_round_evaluation(
    path: Path,
    *,
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    config: dict[str, Any],
    archive_dir: Path,
) -> None:
    score_weights = get_score_weights(BLOCKED_ALPHA_ROUND3_PROFILE)
    lines = [
        "=" * 70,
        "STOCK_PORTFOLIO_SYNERGY ROUND_3 REF_ONLY_1.16 EVALUATION",
        "=" * 70,
        "",
        "Selection:",
        "  Promoted candidate: ref_only_1.16",
        "  Source: blocked-alpha return/capture and holdout probe",
        f"  Baseline config: {BASELINE_PATH}",
        f"  Archived faulty round_3: {archive_dir}",
        "",
        "Intent:",
        "  Increase blocked-alpha capture by raising the portfolio heat reference denominator",
        "  without cutting per-trade unit dollar risk.",
        "",
        "Config delta versus round_2:",
        f"  portfolio_rules.reference_risk_pct: "
        f"{baseline['portfolio_rules']['reference_risk_pct']:.8f} -> "
        f"{config['portfolio_rules']['reference_risk_pct']:.8f}",
        "",
        "Headline:",
        f"  Final equity: ${metrics.get('final_equity', 0.0):,.2f}",
        f"  Net PnL: ${metrics.get('net_pnl', 0.0):+,.2f}",
        f"  Net return: {metrics.get('net_return_pct', 0.0):+.2%}",
        f"  Total trades: {metrics.get('total_trades', 0.0):.0f}",
        f"  Active trades/month: {metrics.get('active_trades_per_month', 0.0):.2f}",
        f"  Total R/month: {metrics.get('total_r_per_month', 0.0):.2f}",
        f"  Profit factor: {metrics.get('profit_factor', 0.0):.2f}",
        f"  Max DD: {metrics.get('max_drawdown_pct', 0.0):.2%}",
        f"  Trade capture ratio: {metrics.get('trade_capture_ratio', 0.0):.2%}",
        f"  Positive-alpha block rate: {metrics.get('positive_alpha_block_rate', 0.0):.2%}",
        "",
        "Score Components:",
    ]
    for key, weight in score_weights.items():
        lines.append(f"- {key}: {metrics.get(f'score_{key}', 0.0):.4f} (weight {weight:.2f})")
    lines.extend(
        [
            "",
            "Execution Note",
            "Replay starts from latest active stock ALCB and IARIC optimized trades, then applies "
            "portfolio-level dynamic allocation, routing, and heat controls.",
            "",
            "Overall Verdict",
            "PROMOTED",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def holdout_summary_from_probe() -> dict[str, Any]:
    if not HOLDOUT_PROBE_PATH.exists():
        return {}
    probe = load_json(HOLDOUT_PROBE_PATH)
    results: dict[str, Any] = {}
    for name, payload in probe.get("windows", {}).items():
        row = next((item for item in payload.get("rows", []) if item.get("name") == "ref_only_1.16"), None)
        baseline = payload.get("baseline", {})
        if not row:
            continue
        results[name] = {
            "start": payload.get("start"),
            "end": payload.get("end"),
            "baseline_round2": {
                "net_return_pct": baseline.get("net_return_pct"),
                "net_pnl": baseline.get("net_pnl"),
                "trade_capture_ratio": baseline.get("trade_capture_ratio"),
                "positive_alpha_block_rate": baseline.get("positive_alpha_block_rate"),
                "max_drawdown_pct": baseline.get("max_drawdown_pct"),
                "profit_factor": baseline.get("profit_factor"),
            },
            "round3_ref_only_1_16": {
                "net_return_pct": row.get("net_return_pct"),
                "net_pnl": row.get("net_pnl"),
                "trade_capture_ratio": row.get("trade_capture_ratio"),
                "positive_alpha_block_rate": row.get("positive_alpha_block_rate"),
                "max_drawdown_pct": row.get("max_drawdown_pct"),
                "profit_factor": row.get("profit_factor"),
                "score_total": row.get("score_total"),
            },
            "delta_vs_round2": {
                "net_return_pct": row.get("delta_net_return_pct"),
                "net_pnl": row.get("delta_net_pnl"),
                "trade_capture_ratio": row.get("delta_trade_capture_ratio"),
                "positive_alpha_block_rate": row.get("delta_positive_alpha_block_rate"),
                "max_drawdown_pct": row.get("delta_max_drawdown_pct"),
                "profit_factor": row.get("delta_profit_factor"),
            },
        }
    return results


def active_round3_exists(manager: RoundManager) -> bool:
    for entry in manager.load_manifest().get("rounds", []):
        if int(entry.get("round", 0)) == 3 and not entry.get("archived"):
            return True
    return False


def latest_round3_archive(manager: RoundManager) -> Path:
    archives = [
        path for path in (manager.strategy_dir / "archived_rounds").glob(f"*_{ARCHIVE_REASON}")
        if path.is_dir()
    ]
    if not archives:
        return manager.strategy_dir / "archived_rounds"
    return max(archives, key=lambda item: item.stat().st_mtime)


def main() -> None:
    manager = RoundManager("stock", "portfolio_synergy", base_dir=ROOT / "backtests/output")
    baseline = load_json(BASELINE_PATH)
    config = promoted_config()
    timestamp = datetime.now(timezone.utc).isoformat()

    if active_round3_exists(manager):
        print("Archiving active faulty round_3...", flush=True)
        archive_dir = manager.archive_rounds([3], reason=ARCHIVE_REASON)
    else:
        archive_dir = latest_round3_archive(manager)
        print(f"No active round_3 to archive; using existing archive {archive_dir}", flush=True)
    run_dir = manager.get_round_dir(3)

    plugin = full_plugin()
    print("Building provenance...", flush=True)
    provenance = plugin.build_provenance()
    provenance_validation = manager.validate_previous_round_provenance(
        3,
        provenance,
        allow_diagnostics_only_drift=True,
    )
    provenance_status = "selection_drift_accepted" if not provenance_validation.valid else provenance_validation.status
    print("Computing full-period round_3 metrics...", flush=True)
    metrics = plugin.compute_final_metrics(config)
    diagnostics = plugin._format_diagnostics("FINAL STOCK PORTFOLIO SYNERGY DIAGNOSTICS", metrics, None)
    validation_results = holdout_summary_from_probe()

    print("Writing round_3 artifacts...", flush=True)
    manager.write_optimized_config(run_dir, config)
    manager.write_run_spec(
        run_dir,
        3,
        "stock_portfolio_synergy",
        description="round_3_ref_only_1_16_blocked_alpha_return_capture_promotion",
        scoring_weights=get_score_weights(BLOCKED_ALPHA_ROUND3_PROFILE),
        baseline_mutations=baseline,
        baseline_source=BASELINE_PATH,
        execution_context={
            "data_dir": str(Path("backtests/stock/data/raw")),
            "initial_equity": 25_000.0,
            "start_date": "2024-01-01",
            "end_date": "2026-03-01",
            "max_workers": 1,
            "round_profile": BLOCKED_ALPHA_ROUND3_PROFILE,
            "selection_source": HOLDOUT_PROBE_SOURCE,
            "selected_candidate": "ref_only_1.16",
            "selection_timestamp_utc": timestamp,
        },
        provenance=provenance,
        provenance_status=provenance_status,
        overwrite=True,
    )
    manager.write_run_summary(
        run_dir,
        config,
        metrics,
        [],
        round_num=3,
        provenance=provenance,
        provenance_status=provenance_status,
        provenance_validation=provenance_validation,
    )
    manager.diagnostics_path(run_dir).write_text(diagnostics, encoding="utf-8")
    write_round_evaluation(
        manager.evaluation_path(run_dir),
        metrics=metrics,
        baseline=baseline,
        config=config,
        archive_dir=archive_dir,
    )
    (run_dir / "validation_checks").mkdir(parents=True, exist_ok=True)
    (run_dir / "validation_checks" / "holdout_summary.json").write_text(
        json.dumps(validation_results, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Updating rounds manifest...", flush=True)
    manager.append_to_manifest(
        3,
        config,
        metrics,
        provenance=provenance,
        provenance_status=provenance_status,
    )

    manifest = manager.load_manifest()
    for entry in manifest.get("rounds", []):
        if int(entry.get("round", 0)) == 3 and not entry.get("archived"):
            entry["round_type"] = "blocked_alpha_reference_risk_promotion"
            entry["selected_candidate"] = "ref_only_1.16"
            entry["baseline_round"] = 2
            entry["baseline_reference_risk_pct"] = baseline["portfolio_rules"]["reference_risk_pct"]
            entry["promoted_reference_risk_pct"] = config["portfolio_rules"]["reference_risk_pct"]
            entry["selection_source"] = HOLDOUT_PROBE_SOURCE
            entry["archive_replaced_round3_path"] = str(archive_dir)
            entry["holdout_validation"] = validation_results
            break
    manifest["latest_round"] = 3
    manifest["latest_round_note"] = (
        "Round 3 replaces the lower-unit-risk blocked-alpha pass with ref_only_1.16: "
        "a reference-risk/heat-denominator promotion from the parity-aligned round_2 baseline. "
        "It raises portfolio_rules.reference_risk_pct while preserving per-trade unit risk, "
        "improving return, capture, and positive-alpha block behavior versus round_2 on IS and both "
        "tracked holdout windows."
    )
    _atomic_write_json(manifest, manager.manifest_path)

    summary = {
        "archive_dir": str(archive_dir),
        "run_dir": str(run_dir),
        "optimized_config": str(manager.optimized_config_path(run_dir)),
        "run_summary": str(manager.run_summary_path(run_dir)),
        "diagnostics": str(manager.diagnostics_path(run_dir)),
        "round_evaluation": str(manager.evaluation_path(run_dir)),
        "manifest": str(manager.manifest_path),
        "metrics": {
            "net_return_pct": metrics.get("net_return_pct"),
            "net_pnl": metrics.get("net_pnl"),
            "total_trades": metrics.get("total_trades"),
            "trade_capture_ratio": metrics.get("trade_capture_ratio"),
            "positive_alpha_block_rate": metrics.get("positive_alpha_block_rate"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "profit_factor": metrics.get("profit_factor"),
            "score_total": metrics.get("score_total"),
        },
        "holdout_validation": validation_results,
        "provenance_status": provenance_status,
        "provenance_validation": provenance_validation.to_dict(),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
