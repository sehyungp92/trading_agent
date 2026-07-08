"""Daily Aggregate Snapshots — end-of-day rollup computed locally.

Reads today's trade events, missed opportunities, and process scores,
then computes the daily aggregate for the central analysis system.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import lineage_from_config

logger = logging.getLogger("instrumentation.daily_snapshot")


@dataclass
class DailySnapshot:
    """End-of-day aggregate for a single bot."""
    date: str
    bot_id: str
    strategy_type: str

    # Trade counts
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0

    # PnL
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_fees: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Risk
    max_drawdown_pct: float = 0.0
    max_exposure: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    exposure_pct: float = 0.0

    # Rolling metrics
    sharpe_rolling_30d: Optional[float] = None
    sortino_rolling_30d: Optional[float] = None
    calmar_rolling_30d: Optional[float] = None

    # Missed opportunities
    missed_count: int = 0
    missed_would_have_won: int = 0
    missed_potential_pnl: float = 0.0
    top_missed_filter: str = ""

    # Process quality
    avg_process_quality: float = 0.0
    process_scores_distribution: Dict[str, int] = field(default_factory=dict)
    root_cause_distribution: Dict[str, int] = field(default_factory=dict)

    # Regime breakdown
    regime_breakdown: Dict[str, dict] = field(default_factory=dict)

    # Excursion & efficiency aggregates (Task 19)
    avg_mfe_pct: Optional[float] = None
    avg_mae_pct: Optional[float] = None
    avg_exit_efficiency: Optional[float] = None
    session_breakdown: Dict[str, dict] = field(default_factory=dict)

    # Per-strategy breakdown (Gap #4c)
    per_strategy_summary: Optional[dict] = field(default=None)
    overlay_state_summary: Optional[dict] = field(default=None)

    # Execution quality
    avg_entry_slippage_bps: Optional[float] = None
    avg_exit_slippage_bps: Optional[float] = None
    avg_entry_latency_ms: Optional[float] = None

    # Execution cascade
    avg_signal_to_fill_ms: Optional[float] = None

    # Experiment A/B tracking
    experiment_breakdown: Optional[dict] = field(default=None)
    active_experiments: Dict[str, dict] = field(default_factory=dict)

    # Regime
    regime_context: Optional[dict] = None
    applied_regime_config: Optional[dict] = None

    # Health
    error_count: int = 0
    uptime_pct: float = 100.0
    data_gaps: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class DailySnapshotBuilder:
    """Reads today's events and computes the daily aggregate.

    Usage::

        builder = DailySnapshotBuilder(config)
        snapshot = builder.build(date_str="2026-03-01")
        builder.save(snapshot)
    """

    def __init__(self, config: dict, experiment_registry=None, get_regime_ctx=None, get_applied_config=None):
        self.bot_id = config["bot_id"]
        self.strategy_type = config.get("strategy_type", "multi_strategy")
        self.data_dir = Path(config["data_dir"])
        self.experiment_registry = experiment_registry
        self._get_regime_ctx = get_regime_ctx
        self._get_applied_config = get_applied_config
        self._lineage = lineage_from_config(
            config,
            family_id="swing",
            strategy_id=config.get("strategy_id", ""),
        )

    def build(self, date_str: str = None) -> DailySnapshot:
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        trades = self._load_trades(date_str)
        missed = self._load_missed(date_str)
        scores = self._load_scores(date_str)
        errors = self._load_errors(date_str)

        snapshot = DailySnapshot(
            date=date_str,
            bot_id=self.bot_id,
            strategy_type=self.strategy_type,
        )

        # --- TRADE AGGREGATES ---
        completed = [t for t in trades if t.get("stage") == "exit" and t.get("pnl") is not None]
        snapshot.total_trades = len(completed)

        if completed:
            pnls = [t["pnl"] for t in completed]
            fees = [t.get("fees_paid", 0) or 0 for t in completed]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            snapshot.win_count = len(wins)
            snapshot.loss_count = len(losses)
            snapshot.breakeven_count = len([p for p in pnls if p == 0])
            snapshot.gross_pnl = round(sum(pnls) + sum(fees), 4)
            snapshot.net_pnl = round(sum(pnls), 4)
            snapshot.total_fees = round(sum(fees), 4)
            snapshot.best_trade_pnl = round(max(pnls), 4)
            snapshot.worst_trade_pnl = round(min(pnls), 4)
            snapshot.avg_win = round(sum(wins) / len(wins), 4) if wins else 0
            snapshot.avg_loss = round(sum(losses) / len(losses), 4) if losses else 0
            snapshot.win_rate = round(len(wins) / len(completed), 4)

            gross_wins = sum(wins) if wins else 0
            gross_losses = abs(sum(losses)) if losses else 0
            snapshot.profit_factor = round(gross_wins / gross_losses, 4) if gross_losses > 0 else float("inf")

            # Slippage averages
            entry_slips = [t.get("entry_slippage_bps") for t in completed if t.get("entry_slippage_bps") is not None]
            exit_slips = [t.get("exit_slippage_bps") for t in completed if t.get("exit_slippage_bps") is not None]
            latencies = [t.get("entry_latency_ms") for t in completed if t.get("entry_latency_ms") is not None]

            snapshot.avg_entry_slippage_bps = round(sum(entry_slips) / len(entry_slips), 2) if entry_slips else None
            snapshot.avg_exit_slippage_bps = round(sum(exit_slips) / len(exit_slips), 2) if exit_slips else None
            snapshot.avg_entry_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else None

            # Regime breakdown
            regime_data: Dict[str, dict] = {}
            for t in completed:
                regime = t.get("market_regime", "unknown")
                if regime not in regime_data:
                    regime_data[regime] = {"trades": 0, "pnl": 0.0, "wins": 0}
                regime_data[regime]["trades"] += 1
                regime_data[regime]["pnl"] += t["pnl"]
                if t["pnl"] > 0:
                    regime_data[regime]["wins"] += 1
            for data in regime_data.values():
                data["pnl"] = round(data["pnl"], 4)
                data["win_rate"] = round(data["wins"] / data["trades"], 4) if data["trades"] > 0 else 0
            snapshot.regime_breakdown = regime_data

            # Excursion & efficiency aggregates
            mfe_pcts = [t.get("mfe_pct") for t in completed if t.get("mfe_pct") is not None]
            mae_pcts = [t.get("mae_pct") for t in completed if t.get("mae_pct") is not None]
            efficiencies = [t.get("exit_efficiency") for t in completed if t.get("exit_efficiency") is not None]

            snapshot.avg_mfe_pct = round(sum(mfe_pcts) / len(mfe_pcts), 6) if mfe_pcts else None
            snapshot.avg_mae_pct = round(sum(mae_pcts) / len(mae_pcts), 6) if mae_pcts else None
            snapshot.avg_exit_efficiency = round(sum(efficiencies) / len(efficiencies), 4) if efficiencies else None

            # Session breakdown
            session_data: Dict[str, dict] = {}
            for t in completed:
                session = t.get("market_session", "unknown")
                if session not in session_data:
                    session_data[session] = {"trades": 0, "pnl": 0.0, "wins": 0}
                session_data[session]["trades"] += 1
                session_data[session]["pnl"] += t["pnl"]
                if t["pnl"] > 0:
                    session_data[session]["wins"] += 1
            for data in session_data.values():
                data["pnl"] = round(data["pnl"], 4)
                data["win_rate"] = round(data["wins"] / data["trades"], 4) if data["trades"] > 0 else 0
            snapshot.session_breakdown = session_data

            # Execution cascade: avg signal-to-fill latency
            stf_deltas = []
            for t in completed:
                tl = t.get("execution_timeline")
                if isinstance(tl, dict):
                    sig = tl.get("signal_generated_at")
                    fill = tl.get("fill_confirmed_at")
                    if sig and fill:
                        try:
                            from datetime import datetime as _dt
                            t0 = _dt.fromisoformat(str(sig))
                            t1 = _dt.fromisoformat(str(fill))
                            delta_ms = (t1 - t0).total_seconds() * 1000
                            if delta_ms >= 0:
                                stf_deltas.append(delta_ms)
                        except (ValueError, TypeError):
                            pass
            snapshot.avg_signal_to_fill_ms = (
                round(sum(stf_deltas) / len(stf_deltas), 1) if stf_deltas else None
            )

            # Experiment A/B breakdown
            exp_groups: Dict[str, dict] = {}
            exp_strategy_ids: Dict[str, List[str]] = {}
            for t in completed:
                eid = t.get("experiment_id") or ""
                evar = t.get("experiment_variant") or ""
                if not eid:
                    continue
                key = f"{eid}:{evar}" if evar else eid
                if key not in exp_groups:
                    exp_groups[key] = {"trades": 0, "pnl": 0.0, "wins": 0}
                    exp_strategy_ids[key] = []
                exp_groups[key]["trades"] += 1
                exp_groups[key]["pnl"] += t.get("pnl", 0) or 0
                if (t.get("pnl") or 0) > 0:
                    exp_groups[key]["wins"] += 1
                sid = t.get("strategy_id", "")
                if sid:
                    exp_strategy_ids[key].append(sid)
            for key, data in exp_groups.items():
                data["pnl"] = round(data["pnl"], 4)
                data["win_rate"] = round(data["wins"] / data["trades"], 4) if data["trades"] > 0 else 0
                sids = exp_strategy_ids.get(key, [])
                data["strategy_id"] = max(set(sids), key=sids.count) if sids else ""
                data["strategy_ids"] = sorted(set(sids))
            snapshot.experiment_breakdown = exp_groups or None

        # --- PER-STRATEGY SUMMARY ---
        strategy_ids = sorted({t.get("strategy_id", "") for t in completed
                                if t.get("strategy_id") and t.get("strategy_id") != "OVERLAY"})
        per_strategy_summary: dict = {}
        for sid in strategy_ids:
            st = [t for t in completed if t.get("strategy_id") == sid]
            wins_st   = [t for t in st if (t.get("pnl") or 0) > 0]
            losses_st = [t for t in st if (t.get("pnl") or 0) < 0]
            mfes   = [t["mfe_pct"] for t in st if t.get("mfe_pct") is not None]
            maes   = [t["mae_pct"] for t in st if t.get("mae_pct") is not None]
            effs   = [t["exit_efficiency"] for t in st if t.get("exit_efficiency") is not None]
            per_strategy_summary[sid] = {
                "trades":              len(st),
                "win_count":           len(wins_st),
                "loss_count":          len(losses_st),
                "gross_pnl":           round(sum(t.get("pnl") or 0 for t in st), 2),
                "net_pnl":             round(sum((t.get("pnl") or 0) - (t.get("fees_paid") or 0) for t in st), 2),
                "win_rate":            round(len(wins_st) / len(st), 3) if st else None,
                "avg_mfe_pct":         round(sum(mfes) / len(mfes), 4) if mfes else None,
                "avg_mae_pct":         round(sum(maes) / len(maes), 4) if maes else None,
                "avg_exit_efficiency": round(sum(effs) / len(effs), 3) if effs else None,
                "symbols_traded":      sorted({t.get("pair") for t in st if t.get("pair")}),
            }
        snapshot.per_strategy_summary = per_strategy_summary or None

        # --- OVERLAY STATE SUMMARY ---
        overlay_all = [t for t in trades if t.get("strategy_id") == "OVERLAY"]
        overlay_in  = [t for t in overlay_all if t.get("stage") == "entry"]
        overlay_out = [t for t in overlay_all if t.get("stage") == "exit"]
        open_syms = {t.get("pair") for t in overlay_in} - {t.get("pair") for t in overlay_out}

        # Load coordinator actions for today
        coordinator_actions = self._load_jsonl("coordination", "coordination", date_str)

        snapshot.overlay_state_summary = {
            "entry_count_today":  len(overlay_in),
            "exit_count_today":   len(overlay_out),
            "qqq_bullish":        "QQQ" in open_syms,
            "gld_bullish":        "GLD" in open_syms,
            "active_symbols":     sorted(open_syms),
            "coordinator_actions_today": len(coordinator_actions),
            "coordinator_action_types": dict(Counter(
                a.get("action") for a in coordinator_actions
            )),
        }

        # --- MISSED OPPORTUNITIES ---
        snapshot.missed_count = len(missed)
        missed_winners = [m for m in missed if m.get("first_hit") == "TP"]
        snapshot.missed_would_have_won = len(missed_winners)

        if missed:
            filter_win_counts: Counter = Counter()
            for m in missed_winners:
                filter_win_counts[m.get("blocked_by", "unknown")] += 1
            if filter_win_counts:
                snapshot.top_missed_filter = filter_win_counts.most_common(1)[0][0]

        # --- PROCESS QUALITY ---
        if scores:
            quality_scores = [s.get("process_quality_score", 50) for s in scores]
            snapshot.avg_process_quality = round(sum(quality_scores) / len(quality_scores), 1)

            classifications = Counter(s.get("classification", "neutral") for s in scores)
            snapshot.process_scores_distribution = dict(classifications)

            all_causes: List[str] = []
            for s in scores:
                all_causes.extend(s.get("root_causes", []))
            snapshot.root_cause_distribution = dict(Counter(all_causes))

        # --- ERRORS ---
        snapshot.error_count = len(errors)

        # --- ACTIVE EXPERIMENTS ---
        if self.experiment_registry is not None:
            try:
                snapshot.active_experiments = self.experiment_registry.export_active()
            except Exception:
                snapshot.active_experiments = self._load_active_experiments()
        else:
            snapshot.active_experiments = self._load_active_experiments()

        # --- REGIME CONTEXT ---
        if self._get_regime_ctx is not None:
            try:
                rctx = self._get_regime_ctx()
                if rctx is not None:
                    snapshot.regime_context = rctx.to_snapshot_dict()
            except Exception:
                pass
        if self._get_applied_config is not None:
            try:
                from regime.context import serialize_applied_config
                snapshot.applied_regime_config = serialize_applied_config(self._get_applied_config())
            except Exception:
                pass

        return snapshot

    def save(self, snapshot: DailySnapshot) -> None:
        daily_dir = self.data_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        filepath = daily_dir / f"daily_{snapshot.date}.json"
        payload = enrich_payload(
            snapshot.to_dict(),
            lineage=self._lineage,
            event_type="daily_snapshot",
            scope="strategy",
        )
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def _load_jsonl(self, directory: str, prefix: str, date_str: str) -> list:
        filepath = self.data_dir / directory / f"{prefix}_{date_str}.jsonl"
        if not filepath.exists():
            return []
        events = []
        for line in filepath.read_text().strip().split("\n"):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events

    def _load_active_experiments(self) -> dict:
        """Load active experiment metadata from config."""
        config_path = Path(self.data_dir).parent / "config" / "experiments.yaml"
        if not config_path.exists():
            return {}
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                experiments = yaml.safe_load(f) or {}
            return {
                exp_id: {
                    "hypothesis": exp.get("hypothesis", ""),
                    "variants": exp.get("variants", []),
                    "primary_metric": exp.get("primary_metric", "sharpe"),
                    "start_date": exp.get("start_date", ""),
                }
                for exp_id, exp in experiments.items()
                if not exp.get("concluded", False)
            }
        except Exception:
            return {}

    def _load_trades(self, date_str: str) -> list:
        return self._load_jsonl("trades", "trades", date_str)

    def _load_missed(self, date_str: str) -> list:
        return self._load_jsonl("missed", "missed", date_str)

    def _load_scores(self, date_str: str) -> list:
        return self._load_jsonl("scores", "scores", date_str)

    def _load_errors(self, date_str: str) -> list:
        return self._load_jsonl("errors", "instrumentation_errors", date_str)
