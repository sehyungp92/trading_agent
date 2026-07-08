"""Focused second-pass portfolio round 2 OOS repair experiments.

This follow-up starts from the broad ablation report and tests local values and
combinations around the strongest preserving mechanisms, without changing any
promoted config.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_portfolio_round2_oos import (
    PORTFOLIO_FEED_TIMEFRAMES,
    SYMBOLS,
    Candidate,
    _candidate_fingerprint,
    _current_policy,
    _evaluate_candidate,
    _init_portfolio_worker,
)


DEFAULT_OUTPUT_DIR = ROOT / "output" / "portfolio" / "round_2" / "oos_followup_second_phase"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _merge_overrides(*items: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        for strategy_id, mutations in item.items():
            out.setdefault(strategy_id, {}).update(mutations)
    return out


def _trend(**mutations: Any) -> dict[str, dict[str, Any]]:
    return {"trend": dict(mutations)}


def _momentum(**mutations: Any) -> dict[str, dict[str, Any]]:
    return {"momentum": dict(mutations)}


def _breakout(**mutations: Any) -> dict[str, dict[str, Any]]:
    return {"breakout": dict(mutations)}


def _with_policy(base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    out = dict(base)
    out.update(updates)
    return out


def _add(
    candidates: list[Candidate],
    label: str,
    *,
    kind: str,
    policy: dict[str, Any],
    overrides: dict[str, dict[str, Any]] | None = None,
    notes: str = "",
) -> None:
    candidates.append(
        Candidate(
            label=label,
            kind=kind,
            policy=policy,
            base_overrides=overrides or {},
            notes=notes,
        )
    )


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for candidate in candidates:
        fp = _candidate_fingerprint(candidate)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(candidate)
    return out


def build_candidates() -> list[Candidate]:
    policy = _current_policy()
    candidates: list[Candidate] = []

    _add(candidates, "baseline_current_round2", kind="baseline", policy=policy)

    # Local grid around the broad sweep winner.  The first sweep only tested
    # coarse 20% steps; the response was discontinuous around 1.0, so inspect
    # the boundary and higher values directly.
    top = _trend(**{"setup.orderly_max_body_frac": 1.08})
    _add(
        candidates,
        "recommended_orderly_body_1_08",
        kind="anchor",
        policy=policy,
        overrides=top,
        notes="First-pass recommended candidate.",
    )

    body_values = [
        0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06,
        1.10, 1.12, 1.15, 1.18, 1.20, 1.25,
    ]
    for value in body_values:
        _add(
            candidates,
            f"local_trend_orderly_body_{value:g}",
            kind="local",
            policy=policy,
            overrides=_trend(**{"setup.orderly_max_body_frac": value}),
            notes="Local grid around first-pass OOS winner.",
        )

    # Combine the trend-body uplift with first-pass candidates that improved
    # both IS and OOS, plus the best stricter-OOS filters.
    for value in [0.016, 0.0165, 0.01725, 0.018, 0.019, 0.020]:
        _add(
            candidates,
            f"combo_body108_momentum_risk_b_{value:g}",
            kind="combo",
            policy=policy,
            overrides=_merge_overrides(top, _momentum(**{"risk.risk_pct_b": value})),
        )

    for value in [1.18, 1.20, 1.22, 1.25, 1.28, 1.30]:
        updates = {
            "risk_scale.momentum": value,
            "risk_scale.trend": value,
            "risk_scale.breakout": value,
        }
        _add(
            candidates,
            f"combo_body108_risk_all_{value:g}",
            kind="combo",
            policy=_with_policy(policy, **updates),
            overrides=top,
        )

    for value in [1.20, 1.25, 1.30]:
        _add(
            candidates,
            f"combo_body108_risk_momentum_{value:g}",
            kind="combo",
            policy=_with_policy(policy, **{"risk_scale.momentum": value}),
            overrides=top,
        )
        _add(
            candidates,
            f"combo_body108_risk_breakout_{value:g}",
            kind="combo",
            policy=_with_policy(policy, **{"risk_scale.breakout": value}),
            overrides=top,
        )

    for value in [2.20, 2.255, 2.35, 2.46, 2.50, 2.60]:
        _add(
            candidates,
            f"combo_body108_trend_min_score_a_{value:g}",
            kind="combo",
            policy=policy,
            overrides=_merge_overrides(top, _trend(**{"setup.min_setup_score_a": value})),
        )

    for value in [1.30, 1.35, 1.40]:
        _add(
            candidates,
            f"combo_body108_trend_min_stop_atr_{value:g}",
            kind="combo",
            policy=policy,
            overrides=_merge_overrides(top, _trend(**{"stops.min_stop_atr": value})),
        )

    for value in [1.44, 1.62, 1.98]:
        _add(
            candidates,
            f"combo_body108_trend_tp2_r_{value:g}",
            kind="combo",
            policy=policy,
            overrides=_merge_overrides(top, _trend(**{"exits.tp2_r": value})),
        )

    for value in [0.48, 0.54, 0.66]:
        _add(
            candidates,
            f"combo_body108_trend_tp2_frac_{value:g}",
            kind="combo",
            policy=policy,
            overrides=_merge_overrides(top, _trend(**{"exits.tp2_frac": value})),
        )

    for value in [0.68, 0.70, 0.715, 0.73, 0.75]:
        _add(
            candidates,
            f"combo_body108_breakout_body_ratio_{value:g}",
            kind="combo",
            policy=policy,
            overrides=_merge_overrides(top, _breakout(**{"setup.body_ratio_min": value})),
        )

    for value in [0.68, 0.70, 0.715, 0.73, 0.75]:
        _add(
            candidates,
            f"local_breakout_body_ratio_{value:g}",
            kind="local",
            policy=policy,
            overrides=_breakout(**{"setup.body_ratio_min": value}),
            notes="Local grid around first-pass breakout body-ratio winner.",
        )

    _add(
        candidates,
        "combo_body108_momentum_trail_bars_8",
        kind="combo",
        policy=policy,
        overrides=_merge_overrides(top, _momentum(**{"trail.trail_activation_bars": 8})),
    )
    _add(
        candidates,
        "combo_body108_breakout_faster_trail",
        kind="combo",
        policy=policy,
        overrides=_merge_overrides(top, _breakout(**{"trail.trail_buffer_tight": 0.048})),
    )
    _add(
        candidates,
        "combo_body108_breakout_eth_long_only",
        kind="combo",
        policy=policy,
        overrides=_merge_overrides(top, _breakout(**{"symbol_filter.eth_direction": "long_only"})),
    )
    _add(
        candidates,
        "combo_body108_breakout_body0715_eth_long",
        kind="combo",
        policy=policy,
        overrides=_merge_overrides(
            top,
            _breakout(
                **{
                    "setup.body_ratio_min": 0.715,
                    "symbol_filter.eth_direction": "long_only",
                }
            ),
        ),
    )

    # Test whether the extra trend SOL short edge from the anchor can be kept
    # while dropping weak SOL long exposure.
    for value in ["short_only", "disabled", "long_only"]:
        _add(
            candidates,
            f"combo_body108_trend_sol_{value}",
            kind="targeted",
            policy=policy,
            overrides=_merge_overrides(top, _trend(**{"symbol_filter.sol_direction": value})),
        )

    # Strong OOS filters from the first pass were too trade-suppressive alone;
    # test whether the body uplift offsets their IS/frequency cost.
    for value in ["disabled", "long_only", "both"]:
        _add(
            candidates,
            f"combo_body108_momentum_sol_{value}",
            kind="targeted",
            policy=_with_policy(policy, **{"strategy.momentum.symbol_filter.sol_direction": value}),
            overrides=top,
        )

    # Pairwise/triple combinations among the most plausible positive mechanisms.
    _add(
        candidates,
        "combo_body108_scorea250_momrisk018",
        kind="combo",
        policy=policy,
        overrides=_merge_overrides(
            top,
            _trend(**{"setup.min_setup_score_a": 2.50}),
            _momentum(**{"risk.risk_pct_b": 0.018}),
        ),
    )
    _add(
        candidates,
        "combo_body108_scorea250_riskall125",
        kind="combo",
        policy=_with_policy(
            policy,
            **{
                "risk_scale.momentum": 1.25,
                "risk_scale.trend": 1.25,
                "risk_scale.breakout": 1.25,
            },
        ),
        overrides=_merge_overrides(top, _trend(**{"setup.min_setup_score_a": 2.50})),
    )
    _add(
        candidates,
        "combo_body108_breakbody0715_momrisk018",
        kind="combo",
        policy=policy,
        overrides=_merge_overrides(
            top,
            _breakout(**{"setup.body_ratio_min": 0.715}),
            _momentum(**{"risk.risk_pct_b": 0.018}),
        ),
    )
    _add(
        candidates,
        "combo_body108_breakbody0715_riskall125",
        kind="combo",
        policy=_with_policy(
            policy,
            **{
                "risk_scale.momentum": 1.25,
                "risk_scale.trend": 1.25,
                "risk_scale.breakout": 1.25,
            },
        ),
        overrides=_merge_overrides(top, _breakout(**{"setup.body_ratio_min": 0.715})),
    )
    _add(
        candidates,
        "combo_body108_scorea250_breakbody0715",
        kind="combo",
        policy=policy,
        overrides=_merge_overrides(
            top,
            _trend(**{"setup.min_setup_score_a": 2.50}),
            _breakout(**{"setup.body_ratio_min": 0.715}),
        ),
    )
    _add(
        candidates,
        "combo_body108_scorea250_breakbody0715_momrisk018",
        kind="combo",
        policy=policy,
        overrides=_merge_overrides(
            top,
            _trend(**{"setup.min_setup_score_a": 2.50}),
            _breakout(**{"setup.body_ratio_min": 0.715}),
            _momentum(**{"risk.risk_pct_b": 0.018}),
        ),
    )
    _add(
        candidates,
        "combo_body108_scorea250_breakbody0715_riskall125",
        kind="combo",
        policy=_with_policy(
            policy,
            **{
                "risk_scale.momentum": 1.25,
                "risk_scale.trend": 1.25,
                "risk_scale.breakout": 1.25,
            },
        ),
        overrides=_merge_overrides(
            top,
            _trend(**{"setup.min_setup_score_a": 2.50}),
            _breakout(**{"setup.body_ratio_min": 0.715}),
        ),
    )

    # Third-pass extension: the first follow-up winner was at the top of the
    # momentum risk grid, so bracket the boundary and combine it with the best
    # lower-variance breakout/score filters.  These are still diagnostics only.
    for body_value in [1.08, 1.10, 1.12]:
        body = _trend(**{"setup.orderly_max_body_frac": body_value})
        body_tag = str(body_value).replace(".", "p")
        for risk_b in [0.0195, 0.020, 0.0205, 0.021, 0.022, 0.023, 0.024]:
            _add(
                candidates,
                f"extended_body{body_tag}_momentum_risk_b_{risk_b:g}",
                kind="extended",
                policy=policy,
                overrides=_merge_overrides(body, _momentum(**{"risk.risk_pct_b": risk_b})),
            )

    for risk_b in [0.019, 0.020, 0.021, 0.022, 0.023]:
        _add(
            candidates,
            f"extended_body108_scorea250_breakbody0715_momrisk_{risk_b:g}",
            kind="extended",
            policy=policy,
            overrides=_merge_overrides(
                top,
                _trend(**{"setup.min_setup_score_a": 2.50}),
                _breakout(**{"setup.body_ratio_min": 0.715}),
                _momentum(**{"risk.risk_pct_b": risk_b}),
            ),
        )
        _add(
            candidates,
            f"extended_body108_breakbody0715_momrisk_{risk_b:g}",
            kind="extended",
            policy=policy,
            overrides=_merge_overrides(
                top,
                _breakout(**{"setup.body_ratio_min": 0.715}),
                _momentum(**{"risk.risk_pct_b": risk_b}),
            ),
        )

    for value in [1.32, 1.35, 1.40, 1.45]:
        risk_updates = {
            "risk_scale.momentum": value,
            "risk_scale.trend": value,
            "risk_scale.breakout": value,
        }
        _add(
            candidates,
            f"extended_body108_risk_all_{value:g}",
            kind="extended",
            policy=_with_policy(policy, **risk_updates),
            overrides=top,
        )
        _add(
            candidates,
            f"extended_body108_scorea250_breakbody0715_riskall_{value:g}",
            kind="extended",
            policy=_with_policy(policy, **risk_updates),
            overrides=_merge_overrides(
                top,
                _trend(**{"setup.min_setup_score_a": 2.50}),
                _breakout(**{"setup.body_ratio_min": 0.715}),
            ),
        )

    for value in [1.20, 1.25, 1.30]:
        _add(
            candidates,
            f"extended_body108_momrisk020_riskall_{value:g}",
            kind="extended",
            policy=_with_policy(
                policy,
                **{
                    "risk_scale.momentum": value,
                    "risk_scale.trend": value,
                    "risk_scale.breakout": value,
                },
            ),
            overrides=_merge_overrides(top, _momentum(**{"risk.risk_pct_b": 0.020})),
        )
        _add(
            candidates,
            f"extended_body108_scorea250_breakbody0715_momrisk020_riskall_{value:g}",
            kind="extended",
            policy=_with_policy(
                policy,
                **{
                    "risk_scale.momentum": value,
                    "risk_scale.trend": value,
                    "risk_scale.breakout": value,
                },
            ),
            overrides=_merge_overrides(
                top,
                _trend(**{"setup.min_setup_score_a": 2.50}),
                _breakout(**{"setup.body_ratio_min": 0.715}),
                _momentum(**{"risk.risk_pct_b": 0.020}),
            ),
        )

    return _dedupe(candidates)


def _delta(row: dict[str, Any], baseline: dict[str, Any], split: str, key: str) -> float:
    return float(row[split].get(key, 0.0)) - float(baseline[split].get(key, 0.0))


def _rank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(row for row in results if row["label"] == "baseline_current_round2")
    out: list[dict[str, Any]] = []
    for row in results:
        item = dict(row)
        item["delta"] = {
            "dev_return_pct": _delta(row, baseline, "dev", "net_return_pct"),
            "dev_trades": _delta(row, baseline, "dev", "total_trades"),
            "oos_return_pct": _delta(row, baseline, "oos", "net_return_pct"),
            "oos_trades": _delta(row, baseline, "oos", "total_trades"),
        }
        item["passes_dev_minus_2_trade_preserve"] = (
            item["delta"]["oos_return_pct"] > 0
            and item["delta"]["dev_return_pct"] >= -2.0
            and item["delta"]["oos_trades"] >= 0
        )
        item["passes_strict_is"] = (
            item["delta"]["oos_return_pct"] > 0
            and item["delta"]["dev_return_pct"] >= 0
            and item["delta"]["oos_trades"] >= 0
        )
        item["selection_score"] = (
            item["delta"]["oos_return_pct"] * 4.0
            + item["delta"]["oos_trades"] * 0.6
            + min(item["delta"]["dev_return_pct"], 3.0)
            - max(-item["delta"]["dev_return_pct"] - 2.0, 0.0) * 3.0
            - max(float(row["oos"].get("max_drawdown_pct", 0.0)) - float(baseline["oos"].get("max_drawdown_pct", 0.0)), 0.0)
        )
        out.append(item)
    return sorted(out, key=lambda item: item["selection_score"], reverse=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "label", "kind", "selection_score", "passes_dev_minus_2_trade_preserve", "passes_strict_is",
        "dev_return_pct", "dev_trades", "dev_pf", "dev_dd",
        "oos_return_pct", "oos_trades", "oos_pf", "oos_dd",
        "delta_dev_return_pct", "delta_dev_trades", "delta_oos_return_pct", "delta_oos_trades",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "kind": row["kind"],
                    "selection_score": row["selection_score"],
                    "passes_dev_minus_2_trade_preserve": row["passes_dev_minus_2_trade_preserve"],
                    "passes_strict_is": row["passes_strict_is"],
                    "dev_return_pct": row["dev"].get("net_return_pct"),
                    "dev_trades": row["dev"].get("total_trades"),
                    "dev_pf": row["dev"].get("profit_factor"),
                    "dev_dd": row["dev"].get("max_drawdown_pct"),
                    "oos_return_pct": row["oos"].get("net_return_pct"),
                    "oos_trades": row["oos"].get("total_trades"),
                    "oos_pf": row["oos"].get("profit_factor"),
                    "oos_dd": row["oos"].get("max_drawdown_pct"),
                    "delta_dev_return_pct": row["delta"]["dev_return_pct"],
                    "delta_dev_trades": row["delta"]["dev_trades"],
                    "delta_oos_return_pct": row["delta"]["oos_return_pct"],
                    "delta_oos_trades": row["delta"]["oos_trades"],
                    "notes": row.get("notes", ""),
                }
            )


def _line(row: dict[str, Any]) -> str:
    return (
        f"- {row['label']} ({row['kind']}): OOS {row['oos'].get('net_return_pct'):.2f}% "
        f"[{row['delta']['oos_return_pct']:+.2f}pp], trades {row['oos'].get('total_trades'):.0f} "
        f"[{row['delta']['oos_trades']:+.0f}], IS {row['dev'].get('net_return_pct'):.2f}% "
        f"[{row['delta']['dev_return_pct']:+.2f}pp], PF {row['oos'].get('profit_factor'):.3f}, "
        f"DD {row['oos'].get('max_drawdown_pct'):.2f}%"
    )


def _write_report(path: Path, ranked: list[dict[str, Any]]) -> None:
    baseline = next(row for row in ranked if row["label"] == "baseline_current_round2")
    anchor = next(row for row in ranked if row["label"] == "recommended_orderly_body_1_08")
    preserving = [row for row in ranked if row["passes_dev_minus_2_trade_preserve"]]
    strict = [row for row in ranked if row["passes_strict_is"]]
    beat_anchor = [
        row for row in preserving
        if row["oos"].get("net_return_pct", -999) > anchor["oos"].get("net_return_pct", -999)
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Portfolio Round 2 Follow-Up OOS Repair Report\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Candidates evaluated: {len(ranked)}\n\n")
        f.write("## Baseline And First-Pass Anchor\n")
        f.write(_line(baseline) + "\n")
        f.write(_line(anchor) + "\n\n")
        f.write("## Best Dev-minus-2 / OOS-trade-preserving Candidates\n")
        for row in sorted(
            preserving,
            key=lambda item: (item["oos"].get("net_return_pct", -999), item["dev"].get("net_return_pct", -999)),
            reverse=True,
        )[:15]:
            f.write(_line(row) + "\n")
        f.write("\n## Best Strict-IS Candidates\n")
        for row in sorted(
            strict,
            key=lambda item: (item["oos"].get("net_return_pct", -999), item["dev"].get("net_return_pct", -999)),
            reverse=True,
        )[:15]:
            f.write(_line(row) + "\n")
        f.write("\n## Candidates Beating First-Pass Anchor Under Preservation Rule\n")
        if beat_anchor:
            for row in sorted(beat_anchor, key=lambda item: item["oos"].get("net_return_pct", -999), reverse=True):
                f.write(_line(row) + "\n")
        else:
            f.write("- None.\n")
        f.write("\n## Top Selection Score\n")
        for row in ranked[:15]:
            f.write(_line(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    if args.limit > 0:
        candidates = candidates[: args.limit]

    _write_json(
        out_dir / "candidate_manifest.json",
        [
            {
                "label": item.label,
                "kind": item.kind,
                "policy": item.policy,
                "base_overrides": item.base_overrides,
                "strategy_sources": item.strategy_sources,
                "notes": item.notes,
                "fingerprint": _candidate_fingerprint(item),
            }
            for item in candidates
        ],
    )

    results_path = out_dir / "results.jsonl"
    results: list[dict[str, Any]] = []
    done: set[str] = set()
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                results.append(row)
                done.add(row["fingerprint"])

    pending = [item for item in candidates if _candidate_fingerprint(item) not in done]
    print(f"Evaluating {len(candidates)} follow-up candidates with {args.workers} workers", flush=True)
    print(f"Pending: {len(pending)}", flush=True)
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

    ranked = _rank(results)
    _write_json(out_dir / "ranked_results.json", ranked)
    _write_csv(out_dir / "ranked_results.csv", ranked)
    _write_report(out_dir / "followup_report.md", ranked)
    print(f"Report: {out_dir / 'followup_report.md'}", flush=True)
    print(f"CSV: {out_dir / 'ranked_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
