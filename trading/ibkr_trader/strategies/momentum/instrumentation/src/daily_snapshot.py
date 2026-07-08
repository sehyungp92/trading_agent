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

    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0

    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_fees: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    max_drawdown_pct: float = 0.0
    max_exposure: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    exposure_pct: float = 0.0

    sharpe_rolling_30d: Optional[float] = None
    sortino_rolling_30d: Optional[float] = None
    calmar_rolling_30d: Optional[float] = None

    missed_count: int = 0
    missed_would_have_won: int = 0
    missed_potential_pnl: float = 0.0
    top_missed_filter: str = ""

    avg_process_quality: float = 0.0
    process_scores_distribution: Dict[str, int] = field(default_factory=dict)
    root_cause_distribution: Dict[str, int] = field(default_factory=dict)

    regime_breakdown: Dict[str, dict] = field(default_factory=dict)

    per_strategy_summary: Dict[str, dict] = field(default_factory=dict)

    per_engine_stats: Dict[str, dict] = field(default_factory=dict)

    experiment_breakdown: Dict[str, dict] = field(default_factory=dict)
    active_experiments: Dict[str, dict] = field(default_factory=dict)

    avg_entry_slippage_bps: Optional[float] = None
    avg_exit_slippage_bps: Optional[float] = None
    avg_entry_latency_ms: Optional[float] = None

    # Regime
    regime_context: Optional[dict] = None
    applied_regime_config: Optional[dict] = None

    error_count: int = 0
    uptime_pct: float = 100.0
    data_gaps: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class DailySnapshotBuilder:
    """
    Reads today's trade events, missed opportunities, and process scores,
    then computes the daily aggregate.

    Usage:
        builder = DailySnapshotBuilder(config)
        snapshot = builder.build(date_str="2026-03-01")
        builder.save(snapshot)
    """

    def __init__(self, config: dict, experiment_registry=None, get_regime_ctx=None, get_applied_config=None):
        self.bot_id = config["bot_id"]
        self.strategy_type = config.get("strategy_type", "unknown")
        self.data_dir = Path(config["data_dir"])
        self._experiment_registry = experiment_registry
        self._get_regime_ctx = get_regime_ctx
        self._get_applied_config = get_applied_config
        self._lineage = lineage_from_config(
            config,
            family_id="momentum",
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
            stats = self._compute_trade_stats(completed)
            snapshot.win_count = stats["win_count"]
            snapshot.loss_count = stats["loss_count"]
            snapshot.breakeven_count = stats["breakeven_count"]
            snapshot.gross_pnl = stats["gross_pnl"]
            snapshot.net_pnl = stats["net_pnl"]
            snapshot.total_fees = stats["total_fees"]
            snapshot.best_trade_pnl = stats["best_trade_pnl"]
            snapshot.worst_trade_pnl = stats["worst_trade_pnl"]
            snapshot.avg_win = stats["avg_win"]
            snapshot.avg_loss = stats["avg_loss"]
            snapshot.win_rate = stats["win_rate"]
            snapshot.profit_factor = stats["profit_factor"]
            snapshot.avg_entry_slippage_bps = stats["avg_entry_slippage_bps"]
            snapshot.avg_exit_slippage_bps = stats["avg_exit_slippage_bps"]
            snapshot.avg_entry_latency_ms = stats["avg_entry_latency_ms"]

            regime_data = {}
            for t in completed:
                regime = t.get("market_regime", "unknown")
                if regime not in regime_data:
                    regime_data[regime] = {"trades": 0, "pnl": 0, "wins": 0}
                regime_data[regime]["trades"] += 1
                regime_data[regime]["pnl"] += t["pnl"]
                if t["pnl"] > 0:
                    regime_data[regime]["wins"] += 1
            for regime, data in regime_data.items():
                data["pnl"] = round(data["pnl"], 4)
                data["win_rate"] = round(data["wins"] / data["trades"], 4) if data["trades"] > 0 else 0
            snapshot.regime_breakdown = regime_data

            # --- PER-STRATEGY SUMMARY ---
            by_strategy = {}
            for t in completed:
                st = t.get("strategy_type") or self.strategy_type
                by_strategy.setdefault(st, []).append(t)
            for st, st_trades in by_strategy.items():
                s = self._compute_trade_stats(st_trades)
                snapshot.per_strategy_summary[st] = {
                    "trades": len(st_trades),
                    "win_count": s["win_count"],
                    "loss_count": s["loss_count"],
                    "gross_pnl": s["gross_pnl"],
                    "net_pnl": s["net_pnl"],
                    "win_rate": s["win_rate"],
                    "avg_win": s["avg_win"],
                    "avg_loss": s["avg_loss"],
                    "best_trade_pnl": s["best_trade_pnl"],
                    "worst_trade_pnl": s["worst_trade_pnl"],
                    "avg_entry_slippage_bps": s["avg_entry_slippage_bps"],
                }

            # --- PER-ENGINE STATS (Downturn engine_tag breakdown) ---
            by_engine: dict[str, list] = {}
            for t in completed:
                tag = (t.get("strategy_params_at_entry") or {}).get("engine_tag", "")
                if not tag:
                    sig = t.get("entry_signal", "")
                    tag = sig.split("_")[0] if sig else ""
                if tag:
                    by_engine.setdefault(tag, []).append(t)
            for tag, trades in by_engine.items():
                s = self._compute_trade_stats(trades)
                snapshot.per_engine_stats[tag] = {
                    "trades": len(trades),
                    "pnl": s["net_pnl"],
                    "win_rate": s["win_rate"],
                }

        # --- MISSED OPPORTUNITIES ---
        snapshot.missed_count = len(missed)
        missed_winners = [m for m in missed if m.get("first_hit") == "TP"]
        snapshot.missed_would_have_won = len(missed_winners)

        if missed:
            filter_win_counts = Counter()
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

            all_causes = []
            for s in scores:
                all_causes.extend(s.get("root_causes", []))
            snapshot.root_cause_distribution = dict(Counter(all_causes))

        # --- ERRORS ---
        snapshot.error_count = len(errors)

        # --- EXPERIMENT BREAKDOWN ---
        snapshot.experiment_breakdown = self._build_experiment_breakdown(completed)

        if self._experiment_registry:
            try:
                snapshot.active_experiments = self._experiment_registry.export_active(
                    as_of=date_str
                )
            except Exception:
                pass

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

    @staticmethod
    def _build_experiment_breakdown(completed_trades: list) -> dict:
        """Group completed trades by experiment_id:experiment_variant."""
        groups: dict[str, list[dict]] = {}
        for t in completed_trades:
            exp_id = t.get("experiment_id") or ""
            exp_var = t.get("experiment_variant") or ""
            if not exp_id:
                continue
            key = f"{exp_id}:{exp_var}"
            groups.setdefault(key, []).append(t)

        breakdown = {}
        for key, trades in groups.items():
            exp_id, exp_var = key.split(":", 1)
            wins = [t for t in trades if (t.get("pnl") or 0) > 0]
            losses = [t for t in trades if (t.get("pnl") or 0) < 0]
            gross_pnl = sum(t.get("pnl", 0) + (t.get("fees_paid", 0) or 0) for t in trades)
            net_pnl = sum(t.get("pnl", 0) for t in trades)
            scores = [t.get("process_quality_score", 0) for t in trades
                      if t.get("process_quality_score") is not None]
            slippage_vals = [t["entry_slippage_bps"] for t in trades
                             if t.get("entry_slippage_bps") is not None]

            strategy_types = [t.get("strategy_type", "") for t in trades if t.get("strategy_type")]
            param_set_ids = [t.get("param_set_id", "") for t in trades if t.get("param_set_id")]

            breakdown[key] = {
                "experiment_id": exp_id,
                "experiment_variant": exp_var,
                "trades": len(trades),
                "win_count": len(wins),
                "loss_count": len(losses),
                "gross_pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
                "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
                "avg_win": round(
                    sum(t["pnl"] for t in wins) / len(wins), 2
                ) if wins else 0.0,
                "avg_loss": round(
                    sum(t["pnl"] for t in losses) / len(losses), 2
                ) if losses else 0.0,
                "avg_process_quality": round(
                    sum(scores) / len(scores), 1
                ) if scores else 0.0,
                "strategy_type": max(set(strategy_types), key=strategy_types.count)
                    if strategy_types else "",
                "param_set_id": param_set_ids[0] if len(set(param_set_ids)) == 1
                    else "",
                "avg_slippage_bps": round(
                    sum(slippage_vals) / len(slippage_vals), 2
                ) if slippage_vals else None,
            }

        return breakdown

    @staticmethod
    def _compute_trade_stats(completed: list) -> dict:
        """Compute core trade stats from a list of completed trade dicts."""
        pnls = [t["pnl"] for t in completed]
        fees = [t.get("fees_paid", 0) or 0 for t in completed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        gross_wins = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 0

        entry_slips = [t.get("entry_slippage_bps") for t in completed if t.get("entry_slippage_bps") is not None]
        exit_slips = [t.get("exit_slippage_bps") for t in completed if t.get("exit_slippage_bps") is not None]
        latencies = [t.get("entry_latency_ms") for t in completed if t.get("entry_latency_ms") is not None]

        return {
            "win_count": len(wins),
            "loss_count": len(losses),
            "breakeven_count": len([p for p in pnls if p == 0]),
            "gross_pnl": round(sum(pnls) + sum(fees), 4),
            "net_pnl": round(sum(pnls), 4),
            "total_fees": round(sum(fees), 4),
            "best_trade_pnl": round(max(pnls), 4),
            "worst_trade_pnl": round(min(pnls), 4),
            "avg_win": round(sum(wins) / len(wins), 4) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 4) if losses else 0,
            "win_rate": round(len(wins) / len(completed), 4) if completed else 0,
            "profit_factor": round(gross_wins / gross_losses, 4) if gross_losses > 0 else float('inf'),
            "avg_entry_slippage_bps": round(sum(entry_slips) / len(entry_slips), 2) if entry_slips else None,
            "avg_exit_slippage_bps": round(sum(exit_slips) / len(exit_slips), 2) if exit_slips else None,
            "avg_entry_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        }

    def save(self, snapshot: DailySnapshot):
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

    def _load_trades(self, date_str: str) -> list:
        return self._load_jsonl("trades", "trades", date_str)

    def _load_missed(self, date_str: str) -> list:
        return self._load_jsonl("missed", "missed", date_str)

    def _load_scores(self, date_str: str) -> list:
        return self._load_jsonl("scores", "scores", date_str)

    def _load_errors(self, date_str: str) -> list:
        return self._load_jsonl("errors", "instrumentation_errors", date_str)
