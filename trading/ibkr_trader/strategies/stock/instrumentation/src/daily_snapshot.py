import json
import math
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
    timestamp: str = ""
    snapshot_kind: str = "final"

    total_trades: int = 0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0

    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    avg_pnl: float = 0.0
    total_fees: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    max_drawdown_pct: float = 0.0
    max_drawdown: float = 0.0
    drawdown: float = 0.0
    max_exposure: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    exposure_pct: float = 0.0

    sharpe_rolling_30d: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sharpe: Optional[float] = None
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
    heartbeat_count: int = 0
    heartbeat_gap_count: int = 0
    allocated_nav: Optional[float] = None

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
        self._heartbeat_interval_seconds = int(config.get("heartbeat_interval_seconds", 30) or 30)
        self._heartbeat_gap_multiplier = float(config.get("heartbeat_gap_multiplier", 2.5) or 2.5)
        self._lineage = lineage_from_config(
            config,
            family_id="stock",
            strategy_id=config.get("strategy_id", ""),
        )

    def build(self, date_str: str = None, snapshot_kind: str = "final") -> DailySnapshot:
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        trades = self._load_trades(date_str)
        missed = self._load_missed(date_str)
        scores = self._load_scores(date_str)
        errors = self._load_errors(date_str)
        heartbeats = self._load_heartbeats(date_str)

        snapshot = DailySnapshot(
            date=date_str,
            bot_id=self.bot_id,
            strategy_type=self.strategy_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            snapshot_kind=snapshot_kind,
        )

        # --- TRADE AGGREGATES ---
        completed = [t for t in trades if t.get("stage") == "exit" and t.get("pnl") is not None]
        snapshot.total_trades = len(completed)
        snapshot.trade_count = snapshot.total_trades

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
            snapshot.avg_pnl = stats["avg_pnl"]

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

        # --- MISSED OPPORTUNITIES ---
        snapshot.missed_count = len(missed)
        missed_winners = [m for m in missed if m.get("first_hit") == "TP"]
        snapshot.missed_would_have_won = len(missed_winners)
        snapshot.missed_potential_pnl = round(
            sum(
                float(m.get("outcome_pnl_24h") or 0.0)
                for m in missed
                if float(m.get("outcome_pnl_24h") or 0.0) > 0
            ),
            4,
        )

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

        # --- HEARTBEAT / OPERATIONS ---
        self._apply_heartbeat_metrics(snapshot, heartbeats)

        # --- ROLLING PERFORMANCE ---
        self._apply_rolling_performance(snapshot)
        snapshot.sharpe_ratio = snapshot.sharpe_rolling_30d
        snapshot.sharpe = snapshot.sharpe_ratio
        snapshot.max_drawdown = snapshot.max_drawdown_pct
        snapshot.drawdown = snapshot.max_drawdown_pct

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
            "avg_pnl": round(sum(pnls) / len(completed), 4) if completed else 0.0,
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

    def _load_heartbeats(self, date_str: str) -> list:
        return self._load_jsonl("heartbeats", "heartbeat", date_str)

    def _apply_heartbeat_metrics(self, snapshot: DailySnapshot, heartbeats: list[dict]) -> None:
        if not heartbeats:
            return

        parsed: list[dict] = []
        for hb in heartbeats:
            ts_raw = hb.get("timestamp")
            if not isinstance(ts_raw, str):
                continue
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                continue
            parsed.append({"ts": ts, "payload": hb})

        if not parsed:
            return

        parsed.sort(key=lambda item: item["ts"])
        snapshot.heartbeat_count = len(parsed)

        expected_interval = max(self._heartbeat_interval_seconds, 1)
        gap_threshold = expected_interval * max(self._heartbeat_gap_multiplier, 1.0)
        gap_count = 0
        for previous, current in zip(parsed, parsed[1:]):
            delta = (current["ts"] - previous["ts"]).total_seconds()
            if delta > gap_threshold:
                gap_count += 1

        span_seconds = max((parsed[-1]["ts"] - parsed[0]["ts"]).total_seconds(), 0.0)
        expected_count = int(span_seconds // expected_interval) + 1 if span_seconds > 0 else len(parsed)
        if expected_count > 0:
            snapshot.uptime_pct = round(min(100.0, (len(parsed) / expected_count) * 100.0), 2)
        snapshot.data_gaps = gap_count
        snapshot.heartbeat_gap_count = gap_count

        max_exposure = 0.0
        max_exposure_pct = 0.0
        allocated_nav = None
        for item in parsed:
            exposure = item["payload"].get("portfolio_exposure", {})
            if not isinstance(exposure, dict):
                continue
            gross_notional = float(exposure.get("gross_notional", 0.0) or 0.0)
            exposure_pct = float(exposure.get("exposure_pct", 0.0) or 0.0)
            max_exposure = max(max_exposure, gross_notional)
            max_exposure_pct = max(max_exposure_pct, exposure_pct)
            nav_value = exposure.get("allocated_nav")
            if nav_value is not None:
                try:
                    allocated_nav = float(nav_value)
                except (TypeError, ValueError):
                    pass

        snapshot.max_exposure = round(max_exposure, 4)
        snapshot.exposure_pct = round(max_exposure_pct, 4)
        snapshot.allocated_nav = allocated_nav

    def _apply_rolling_performance(self, snapshot: DailySnapshot) -> None:
        returns: list[float] = []
        for record in self._load_daily_history(snapshot.date, limit=29):
            daily_return = self._extract_daily_return(record)
            if daily_return is not None:
                returns.append(daily_return)

        current_return = self._extract_daily_return(snapshot.to_dict())
        if current_return is not None:
            returns.append(current_return)

        if len(returns) < 2:
            return

        mean_return = sum(returns) / len(returns)
        variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        if std_dev > 0:
            snapshot.sharpe_rolling_30d = round((mean_return / std_dev) * math.sqrt(252), 4)

        downside = [min(0.0, value) for value in returns]
        downside_variance = sum(value ** 2 for value in downside) / len(downside) if downside else 0.0
        downside_dev = math.sqrt(downside_variance) if downside_variance > 0 else 0.0
        if downside_dev > 0:
            snapshot.sortino_rolling_30d = round((mean_return / downside_dev) * math.sqrt(252), 4)

        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for daily_return in returns:
            equity *= (1.0 + daily_return)
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)

        snapshot.max_drawdown_pct = round(max_drawdown * 100.0, 4)
        if max_drawdown > 0:
            snapshot.calmar_rolling_30d = round(((mean_return * 252.0) / max_drawdown), 4)

    def _load_daily_history(self, current_date: str, limit: int) -> list[dict]:
        daily_dir = self.data_dir / "daily"
        if not daily_dir.exists():
            return []

        history: list[tuple[str, dict]] = []
        for path in daily_dir.glob("daily_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            date_value = data.get("date")
            if not isinstance(date_value, str) or date_value >= current_date:
                continue
            history.append((date_value, data))

        history.sort(key=lambda item: item[0])
        return [item[1] for item in history[-limit:]]

    @staticmethod
    def _extract_daily_return(record: dict) -> float | None:
        nav = record.get("allocated_nav")
        pnl = record.get("net_pnl")
        try:
            nav_value = float(nav)
            pnl_value = float(pnl)
        except (TypeError, ValueError):
            return None
        if nav_value == 0:
            return None
        return pnl_value / nav_value
