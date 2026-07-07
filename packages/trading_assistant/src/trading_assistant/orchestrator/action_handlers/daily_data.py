"""Daily curated-data rebuild and raw-event loading support."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

DAILY_RAW_EVENT_TAXONOMY: dict[str, tuple[str, ...]] = {
    "trade": ("trade",),
    "missed_opportunity": ("missed_opportunity",),
    "filter_decision": ("filter_decision",),
    "indicator_snapshot": ("indicator_snapshot",),
    "orderbook_context": ("orderbook_context",),
    "order": ("order",),
    "fill": ("fill", "inferred_fill"),
    "post_exit": ("post_exit",),
    "pipeline_funnel": ("pipeline_funnel", "pipeline_funnels"),
    "portfolio_rule": ("portfolio_rule_check", "portfolio_rule", "risk_decision", "risk_decisions"),
}

DAILY_CURATED_EVENT_FILES: dict[str, tuple[str, ...]] = {
    "trade": ("trades.jsonl",),
    "missed_opportunity": ("missed.jsonl",),
    "filter_decision": ("filter_decisions.json",),
    "orderbook_context": ("orderbook_stats.json",),
    "order": ("order_lifecycle.json",),
    "fill": ("order_lifecycle.json",),
    "post_exit": ("exit_efficiency.json",),
    "pipeline_funnel": ("funnel_snapshots.jsonl", "funnel_analysis.json"),
    "portfolio_rule": ("rule_blocks_summary.json",),
}


def iter_daily_event_input_paths(
    *,
    curated_dir: Path,
    raw_data_dir: Path,
    bot_id: str,
    day,
):
    """Yield event-class inputs used by daily curation and sufficiency coverage."""

    day_key = day.isoformat()
    bot_curated = Path(curated_dir) / day_key / bot_id
    for event_type, filenames in DAILY_CURATED_EVENT_FILES.items():
        for filename in filenames:
            if event_type == "portfolio_rule":
                yield event_type, Path(curated_dir) / day_key / "portfolio" / filename
            else:
                yield event_type, bot_curated / filename

    bot_raw = Path(raw_data_dir) / day_key / bot_id
    for event_type, raw_names in DAILY_RAW_EVENT_TAXONOMY.items():
        for raw_name in raw_names:
            for suffix in (".jsonl", ".json"):
                yield event_type, bot_raw / f"{raw_name}{suffix}"


class DailyDataSupport:
    """Daily curated-data rebuild and raw-event loading support."""

    @staticmethod
    def _merge_daily_snapshots(snapshots: list[dict]) -> dict:
        """Merge per-strategy daily snapshots into one combined snapshot."""
        if not snapshots:
            return {}
        if len(snapshots) == 1:
            return snapshots[0]

        merged = dict(snapshots[-1])  # non-additive account-level fields from latest

        # Additive integers
        for key in ("total_trades", "win_count", "loss_count",
                    "missed_count", "missed_would_have_won", "error_count"):
            merged[key] = sum(int(s.get(key, 0) or 0) for s in snapshots)

        # Additive floats
        for key in ("gross_pnl", "net_pnl"):
            merged[key] = sum(float(s.get(key, 0.0) or 0.0) for s in snapshots)

        # Count-weighted averages
        total_wins = merged["win_count"]
        total_losses = merged["loss_count"]
        total_trades = merged["total_trades"]

        if total_wins > 0:
            merged["avg_win"] = sum(
                int(s.get("win_count", 0) or 0) * float(s.get("avg_win", 0.0) or 0.0)
                for s in snapshots
            ) / total_wins
        if total_losses > 0:
            merged["avg_loss"] = sum(
                int(s.get("loss_count", 0) or 0) * float(s.get("avg_loss", 0.0) or 0.0)
                for s in snapshots
            ) / total_losses
        if total_trades > 0:
            merged["avg_process_quality"] = sum(
                int(s.get("total_trades", 0) or 0) * float(s.get("avg_process_quality", 0.0) or 0.0)
                for s in snapshots
            ) / total_trades
            merged["win_rate"] = merged["win_count"] / total_trades * 100
        else:
            merged["win_rate"] = 0.0

        # Union per_strategy_summary dicts
        combined_pss: dict = {}
        for s in snapshots:
            pss = s.get("per_strategy_summary", {})
            if isinstance(pss, dict):
                combined_pss.update(pss)
        if combined_pss:
            merged["per_strategy_summary"] = combined_pss

        # Sum root_cause_distribution counts
        combined_rc: dict = {}
        for s in snapshots:
            rc = s.get("root_cause_distribution", {})
            if isinstance(rc, dict):
                for cause, count in rc.items():
                    combined_rc[cause] = combined_rc.get(cause, 0) + int(count or 0)
        if combined_rc:
            merged["root_cause_distribution"] = combined_rc

        return merged

    @staticmethod
    def _load_json_file(path: Path) -> dict | list | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _rebuild_daily_curated_from_raw(self, date: str, bots: list[str] | None = None) -> None:
        """Build canonical daily curated files from assistant-owned raw JSONL."""
        if not self._raw_data_dir.exists():
            return

        from trading_assistant.schemas.events import (
            HealthReportSnapshot,
            MissedOpportunityEvent,
            PipelineFunnelSnapshot,
            TradeEvent,
        )
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        findings_dir = self._memory_dir / "findings"
        target_bots = bots or self._bots
        for bot_id in target_bots:
            bot_raw = self._raw_data_dir / date / bot_id
            if not bot_raw.exists():
                continue

            trade_records = self._load_raw_json_records(bot_raw, "trade")
            missed_records = self._load_raw_json_records(bot_raw, "missed_opportunity")
            trades = self._validate_raw_models(TradeEvent, trade_records, "trade", bot_id, date)
            missed = self._validate_raw_models(
                MissedOpportunityEvent, missed_records, "missed_opportunity", bot_id, date,
            )

            kwargs: dict = {}
            for event_type, param_name in {
                "filter_decision": "filter_decision_events",
                "indicator_snapshot": "indicator_snapshot_events",
                "orderbook_context": "orderbook_context_events",
                "parameter_change": "parameter_change_events",
                "order": "order_events",
                "fill": "order_events",
                "inferred_fill": "order_events",
                "process_quality": "process_quality_events",
                "stop_adjustment": "stop_adjustment_events",
                "post_exit": "post_exit_events",
            }.items():
                events = self._load_raw_json_records(bot_raw, event_type)
                if events:
                    kwargs[param_name] = kwargs.get(param_name, []) + events

            daily_snapshots = self._load_raw_json_records(bot_raw, "daily_snapshot")
            if daily_snapshots:
                kwargs["daily_snapshot"] = self._merge_daily_snapshots(daily_snapshots)

            coordinator_events = self._load_raw_json_records(bot_raw, "coordinator_action")
            if coordinator_events:
                kwargs["coordination_events"] = coordinator_events

            funnel_records = (
                self._load_raw_json_records(bot_raw, "pipeline_funnel")
                + self._load_raw_json_records(bot_raw, "pipeline_funnels")
            )
            if funnel_records:
                kwargs["funnel_snapshots"] = self._validate_raw_models(
                    PipelineFunnelSnapshot, funnel_records, "pipeline_funnel", bot_id, date,
                )

            health_records = (
                self._load_raw_json_records(bot_raw, "health_report")
                + self._load_raw_json_records(bot_raw, "health_reports")
            )
            if health_records:
                kwargs["health_snapshots"] = self._validate_raw_models(
                    HealthReportSnapshot, health_records, "health_report", bot_id, date,
                )

            if not trades and not missed and not kwargs:
                continue

            try:
                bot_timezone = "UTC"
                if self._bot_configs and bot_id in self._bot_configs:
                    bot_timezone = getattr(self._bot_configs[bot_id], "timezone", "UTC")
                builder = DailyMetricsBuilder(date=date, bot_id=bot_id, bot_timezone=bot_timezone)
                builder.write_curated(
                    trades=trades,
                    missed=missed,
                    base_dir=self._curated_dir,
                    findings_dir=findings_dir,
                    **kwargs,
                )
            except Exception:
                logger.warning(
                    "Failed to rebuild curated data for %s/%s from raw events",
                    date, bot_id,
                    exc_info=True,
                )

        # ?? Portfolio-level curated files ??????????????????????????????
        try:
            from trading_assistant.schemas.daily_metrics import BotDailySummary
            from trading_assistant.skills.build_daily_metrics import (
                build_concurrent_position_analysis,
                build_family_snapshots,
                build_macro_regime_analysis,
                build_portfolio_rules_summary,
                build_sector_exposure,
            )
            from trading_assistant.skills.compute_portfolio_risk import PortfolioRiskComputer
            from trading_assistant.skills.portfolio_metrics_tracker import PortfolioMetricsTracker

            # Collect all trade records, portfolio_rule events, and daily snapshots across bots
            all_trade_records: list[dict] = []
            all_rule_events: list[dict] = []
            all_daily_snapshots: list[dict] = []
            for bot_id in target_bots:
                bot_raw = self._raw_data_dir / date / bot_id
                if not bot_raw.exists():
                    continue
                all_trade_records.extend(self._load_raw_json_records(bot_raw, "trade"))
                all_rule_events.extend(self._load_raw_json_records(bot_raw, "portfolio_rule_check"))
                all_rule_events.extend(self._load_raw_json_records(bot_raw, "portfolio_rule"))
                all_rule_events.extend(self._load_raw_json_records(bot_raw, "risk_decision"))
                all_rule_events.extend(self._load_raw_json_records(bot_raw, "risk_decisions"))
                all_daily_snapshots.extend(self._load_raw_json_records(bot_raw, "daily_snapshot"))

            # Load BotDailySummary from just-written per-bot summary.json files
            bot_summaries: list[BotDailySummary] = []
            for bot_id in target_bots:
                summary_path = self._curated_dir / date / bot_id / "summary.json"
                if summary_path.exists():
                    try:
                        raw = json.loads(summary_path.read_text(encoding="utf-8"))
                        bot_summaries.append(BotDailySummary.model_validate(raw))
                    except Exception:
                        logger.warning("Failed to load summary for %s/%s", date, bot_id)

            if all_trade_records or all_rule_events or bot_summaries:
                portfolio_dir = self._curated_dir / date / "portfolio"
                portfolio_dir.mkdir(parents=True, exist_ok=True)

                # 1. rule_blocks_summary.json
                rules_summary = build_portfolio_rules_summary(all_rule_events)
                (portfolio_dir / "rule_blocks_summary.json").write_text(
                    json.dumps(rules_summary, indent=2, default=str), encoding="utf-8",
                )

                # 2. family_snapshots.json (must be BEFORE portfolio_rolling_metrics)
                family_snaps = build_family_snapshots(bot_summaries, self._strategy_registry)
                (portfolio_dir / "family_snapshots.json").write_text(
                    json.dumps(family_snaps, indent=2, default=str), encoding="utf-8",
                )

                # 3. concurrent_position_analysis.json
                concurrent = build_concurrent_position_analysis(all_trade_records)
                (portfolio_dir / "concurrent_position_analysis.json").write_text(
                    json.dumps(concurrent, indent=2, default=str), encoding="utf-8",
                )

                # 4. sector_exposure.json
                sector_exp = build_sector_exposure(all_trade_records)
                (portfolio_dir / "sector_exposure.json").write_text(
                    json.dumps(sector_exp, indent=2, default=str), encoding="utf-8",
                )

                # 4b. macro_regime_analysis.json
                # Unwrap payload if snapshots have event wrapper
                unwrapped_snapshots = [
                    s.get("payload", s) for s in all_daily_snapshots
                ]
                macro_regime = build_macro_regime_analysis(unwrapped_snapshots, date)
                if macro_regime:
                    (portfolio_dir / "macro_regime_analysis.json").write_text(
                        json.dumps(macro_regime, indent=2, default=str), encoding="utf-8",
                    )

                # 5. portfolio_rolling_metrics.json (reads family_snapshots.json)
                try:
                    tracker = PortfolioMetricsTracker(self._curated_dir)
                    rolling = tracker.compute(date)
                    (portfolio_dir / "portfolio_rolling_metrics.json").write_text(
                        json.dumps(rolling.model_dump(mode="json"), indent=2, default=str),
                        encoding="utf-8",
                    )
                except Exception:
                    logger.warning("Failed to compute portfolio rolling metrics for %s", date, exc_info=True)

                # 6. portfolio_risk_card.json
                try:
                    position_details: dict[str, list[dict]] = {}
                    for evt in all_trade_records:
                        payload = evt.get("payload", evt)
                        bid = payload.get("bot_id", "")
                        if bid:
                            position_details.setdefault(bid, []).append({
                                "symbol": payload.get("pair", ""),
                                "direction": payload.get("side", "LONG"),
                                "exposure_pct": payload.get("exposure_pct", 0.0),
                            })

                    historical_pnl: dict[str, list[float]] = {}
                    for hbot_id in target_bots:
                        pnls: list[float] = []
                        for d in range(20):
                            past = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=d)).strftime("%Y-%m-%d")
                            sp = self._curated_dir / past / hbot_id / "summary.json"
                            if sp.exists():
                                try:
                                    pnls.append(json.loads(sp.read_text(encoding="utf-8")).get("net_pnl", 0.0))
                                except (json.JSONDecodeError, OSError):
                                    pass
                        if pnls:
                            pnls.reverse()
                            historical_pnl[hbot_id] = pnls

                    sector_map: dict[str, str] = {}
                    for evt in all_trade_records:
                        payload = evt.get("payload", evt)
                        sym = payload.get("pair", "")
                        sec = payload.get("sector", "")
                        if sym and sec:
                            sector_map[sym] = sec

                    computer = PortfolioRiskComputer(
                        date=date,
                        bot_summaries=bot_summaries,
                        position_details=position_details,
                        historical_pnl=historical_pnl,
                        sector_map=sector_map,
                    )
                    risk_card = computer.compute()
                    (self._curated_dir / date / "portfolio_risk_card.json").write_text(
                        json.dumps(risk_card.model_dump(mode="json"), indent=2, default=str),
                        encoding="utf-8",
                    )
                except Exception:
                    logger.warning("Failed to compute portfolio risk card for %s", date, exc_info=True)
        except Exception:
            logger.warning("Failed to rebuild portfolio curated files for %s", date, exc_info=True)

    def _build_enriched_curated(self, date: str, bots: list[str] | None = None) -> None:
        """Compatibility wrapper for the old enriched-curated rebuild entrypoint."""
        self._rebuild_daily_curated_from_raw(date, bots)

    def _load_raw_json_records(self, bot_raw: Path, event_type: str) -> list[dict]:
        """Load JSON/JSONL raw event records, skipping malformed lines."""
        records: list[dict] = []
        candidate_paths = [
            bot_raw / f"{event_type}.jsonl",
            bot_raw / f"{event_type}.json",
        ]
        for path in candidate_paths:
            if not path.exists():
                continue
            if path.suffix == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    records.append(data)
                elif isinstance(data, list):
                    records.extend(d for d in data if isinstance(d, dict))
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    records.append(data)
        return records

    def _validate_raw_models(
        self,
        model_cls: type[BaseModel],
        records: list[dict],
        event_type: str,
        bot_id: str,
        date: str,
    ) -> list[BaseModel]:
        validated: list[BaseModel] = []
        for record in records:
            try:
                validated.append(model_cls.model_validate(record))
            except Exception:
                logger.warning(
                    "Skipping malformed raw %s event for %s on %s",
                    event_type, bot_id, date,
                )
        return validated

    def _write_run_report(
        self,
        run_id: str,
        report_name: str,
        content: str,
        mirror_response: bool = False,
    ) -> None:
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / report_name).write_text(content, encoding="utf-8")
        response_path = run_dir / "response.md"
        if mirror_response or not response_path.exists():
            response_path.write_text(content, encoding="utf-8")

    def _get_latest_heartbeat_time(self, bot_id: str) -> datetime | None:
        """Get the latest heartbeat timestamp for a bot from heartbeat files."""
        hb_file = self._heartbeat_dir / f"{bot_id}.heartbeat"
        if hb_file.exists():
            try:
                raw = hb_file.read_text(encoding="utf-8").strip()
                if raw:
                    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except (ValueError, OSError):
                pass

        hb_file = self._heartbeat_dir / f"{bot_id}.json"
        if not hb_file.exists():
            return None
        try:
            data = json.loads(hb_file.read_text(encoding="utf-8"))
            ts = data.get("last_seen") or data.get("timestamp")
            if ts:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (json.JSONDecodeError, ValueError, OSError):
            pass
        return None

    def _count_daily_trades(self, date: str) -> int:
        """Count total trades across all bots for a given date."""
        count = 0
        for bot_id in self._bots:
            trades_file = self._curated_dir / date / bot_id / "trades.jsonl"
            if trades_file.exists():
                try:
                    count += sum(
                        1 for line in trades_file.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                except OSError:
                    pass
        return count
