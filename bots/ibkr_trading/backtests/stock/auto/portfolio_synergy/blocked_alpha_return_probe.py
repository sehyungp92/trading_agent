from __future__ import annotations

import json
import sys
from copy import deepcopy
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

from backtests.stock.auto.portfolio_synergy.plugin import StockPortfolioSynergyPlugin


DATA_DIR = ROOT / "backtests/stock/data/raw"
EVIDENCE_DIR = (
    ROOT
    / "backtests/output/stock/portfolio_synergy/round_3/validation_checks/blocked_alpha_ref_only_20260524"
)
OUT = EVIDENCE_DIR / "stock_portfolio_blocked_alpha_return_probe_20260524.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def set_path(config: dict[str, Any], path: str, value: Any) -> None:
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def with_updates(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(base)
    for path, value in updates.items():
        set_path(config, path, value)
    return config


def summarize(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "net_return_pct",
        "net_pnl",
        "total_trades",
        "active_trades_per_month",
        "total_r_per_month",
        "profit_factor",
        "max_drawdown_pct",
        "max_drawdown_pct_realized",
        "sharpe",
        "calmar",
        "trade_capture_ratio",
        "positive_alpha_block_rate",
        "candidate_discrimination",
        "max_daily_loss_R",
        "max_weekly_loss_R",
        "max_strategy_trade_share",
        "max_strategy_risk_share",
        "blocked_reason_portfolio_heat_cap",
        "blocked_reason_strategy_heat_cap",
        "blocked_reason_sector_heat_cap",
        "blocked_reason_strategy_trade_share_cap",
        "pnl_IARIC_V5R1",
        "pnl_ALCB_R3",
        "trades_IARIC_V5R1",
        "trades_ALCB_R3",
        "score_total",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def main() -> None:
    round2 = load_json(ROOT / "backtests/output/stock/portfolio_synergy/round_2/optimized_config.json")
    round3 = load_json(ROOT / "backtests/output/stock/portfolio_synergy/round_3/optimized_config.json")
    base_ref = float(round2["portfolio_rules"]["reference_risk_pct"])
    base_iaric_risk = float(round2["strategy_allocations"]["IARIC_V5R1"]["unit_risk_pct"])
    base_alcb_risk = float(round2["strategy_allocations"]["ALCB_R3"]["unit_risk_pct"])

    configs: list[tuple[str, dict[str, Any], str]] = [
        ("round2_current", round2, "Current high-dollar return baseline."),
        ("round3_current", round3, "Accepted lower-risk, higher-capture round_3 result."),
    ]

    for ref_mult in (1.04, 1.08, 1.12, 1.16, 1.20):
        configs.append(
            (
                f"ref_only_{ref_mult:.2f}",
                with_updates(round2, {"portfolio_rules.reference_risk_pct": round(base_ref * ref_mult, 8)}),
                "Raises heat reference only; should admit more trades without cutting per-trade dollar risk.",
            )
        )

    for ref_mult, unit_mult in ((1.12, 1.03), (1.16, 1.05), (1.20, 1.08), (1.24, 1.10)):
        configs.append(
            (
                f"ref_{ref_mult:.2f}_unit_{unit_mult:.2f}",
                with_updates(
                    round2,
                    {
                        "portfolio_rules.reference_risk_pct": round(base_ref * ref_mult, 8),
                        "strategy_allocations.IARIC_V5R1.unit_risk_pct": round(base_iaric_risk * unit_mult, 8),
                        "strategy_allocations.ALCB_R3.unit_risk_pct": round(base_alcb_risk * unit_mult, 8),
                    },
                ),
                "Raises reference risk more than unit risk to seek higher capture and higher dollars together.",
            )
        )

    capacity_updates = [
        ("strategy_heat_5_7_4_2", {
            "strategy_allocations.IARIC_V5R1.max_heat_R": 5.7,
            "strategy_allocations.ALCB_R3.max_heat_R": 4.2,
        }),
        ("strategy_heat_6_0_4_4", {
            "strategy_allocations.IARIC_V5R1.max_heat_R": 6.0,
            "strategy_allocations.ALCB_R3.max_heat_R": 4.4,
        }),
        ("portfolio_heat_6_8", {"portfolio_rules.heat_cap_R": 6.8}),
        ("sector_heat_4_4", {"cross_strategy_rules.same_sector_heat_cap_R": 4.4}),
        ("symbol_heat_2_6", {"portfolio_rules.max_symbol_heat_R": 2.6}),
    ]
    for name, updates in capacity_updates:
        configs.append((name, with_updates(round2, updates), "Direct capacity candidate from blocked-alpha phase 2."))
        ref_updates = {"portfolio_rules.reference_risk_pct": round(base_ref * 1.12, 8), **updates}
        configs.append((f"ref_1.12_plus_{name}", with_updates(round2, ref_updates), "Reference-risk plus direct capacity."))

    quality_updates = [
        ("iaric_gap_0_50", {"strategy_filters.IARIC_V5R1.gap_up_size_mult": 0.5}),
        ("iaric_gap_0_70", {"strategy_filters.IARIC_V5R1.gap_up_size_mult": 0.7}),
        ("alcb_financials_0_55", {"strategy_filters.ALCB_R3.financials_size_mult": 0.55}),
        ("pdh_1_25_score_0_45", {
            "strategy_filters.ALCB_R3.pdh_size_mult": 1.25,
            "strategy_filters.ALCB_R3.score5_no_surge_mult": 0.45,
        }),
    ]
    for name, updates in quality_updates:
        configs.append((name, with_updates(round2, updates), "Quality/routing dampener candidate from blocked-alpha phase 5."))
        ref_updates = {"portfolio_rules.reference_risk_pct": round(base_ref * 1.12, 8), **updates}
        configs.append((f"ref_1.12_plus_{name}", with_updates(round2, ref_updates), "Reference-risk plus quality dampener."))

    plugin = StockPortfolioSynergyPlugin(
        DATA_DIR,
        start_date="2024-01-01",
        end_date="2026-03-01",
        initial_equity=25_000.0,
        max_workers=1,
        round_profile="blocked_alpha_round3",
    )

    rows = []
    for index, (name, config, rationale) in enumerate(configs, start=1):
        metrics = plugin.compute_final_metrics(config)
        row = {"name": name, "rationale": rationale, **summarize(metrics)}
        rows.append(row)
        print(
            f"[{index:02d}/{len(configs):02d}] {name}: "
            f"ret={row['net_return_pct'] * 100:.2f}% "
            f"capture={row['trade_capture_ratio']:.4f} "
            f"pos_block={row['positive_alpha_block_rate']:.4f} "
            f"dd={row['max_drawdown_pct'] * 100:.2f}% "
            f"score={row['score_total']:.4f}",
            flush=True,
        )

    baseline = next(row for row in rows if row["name"] == "round2_current")
    def improves_capture_and_return(row: dict[str, Any]) -> bool:
        return (
            row["net_return_pct"] > baseline["net_return_pct"]
            and row["trade_capture_ratio"] > baseline["trade_capture_ratio"]
            and row["positive_alpha_block_rate"] < baseline["positive_alpha_block_rate"]
        )

    result = {
        "baseline": baseline,
        "rows": rows,
        "higher_return_and_capture": [row for row in rows if improves_capture_and_return(row)],
        "top_by_return": sorted(rows, key=lambda item: item["net_return_pct"], reverse=True)[:10],
        "top_by_capture": sorted(rows, key=lambda item: item["trade_capture_ratio"], reverse=True)[:10],
        "top_by_score": sorted(rows, key=lambda item: item["score_total"], reverse=True)[:10],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"RESULT_PATH={OUT}", flush=True)


if __name__ == "__main__":
    main()
