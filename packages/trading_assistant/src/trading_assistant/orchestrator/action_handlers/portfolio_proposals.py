"""Portfolio detector and proposal recording support."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


class PortfolioProposalActions:
    """Portfolio detector and proposal recording support."""

    def _run_portfolio_detectors(
        self, engine, week_start: str, week_end: str, portfolio_summary,
    ) -> list:
        """Run portfolio-level detectors from the strategy engine.

        Returns list of StrategySuggestion objects with tier=PORTFOLIO.
        """
        results: list = []

        # Load family snapshots for the week
        family_snapshots: dict = {}
        family_allocations: dict = {}
        correlation_matrix: dict = {}

        try:
            weekly_dir = self._curated_dir / "weekly" / week_start
            snap_path = weekly_dir / "allocation_analysis.json"
            if snap_path.exists():
                alloc_data = json.loads(snap_path.read_text(encoding="utf-8"))
                family_allocations = alloc_data.get("current_allocations", {})

            # Load family daily snapshots aggregated over the week
            from datetime import timedelta as _td_pf
            start_dt = datetime.strptime(week_start, "%Y-%m-%d")
            for i in range(7):
                date_str = (start_dt + _td_pf(days=i)).strftime("%Y-%m-%d")
                snap_path = self._curated_dir / date_str / "portfolio" / "family_snapshots.json"
                if snap_path.exists():
                    daily_snaps = json.loads(snap_path.read_text(encoding="utf-8"))
                    if isinstance(daily_snaps, list):
                        for snap in daily_snaps:
                            fam = snap.get("family", "")
                            if fam:
                                family_snapshots.setdefault(fam, []).append(snap)

            # Load correlation matrix from latest risk card
            for i in range(6, -1, -1):
                date_str = (start_dt + _td_pf(days=i)).strftime("%Y-%m-%d")
                risk_path = self._curated_dir / date_str / "portfolio_risk_card.json"
                if risk_path.exists():
                    risk_data = json.loads(risk_path.read_text(encoding="utf-8"))
                    correlation_matrix = risk_data.get("correlation_matrix", {})
                    if correlation_matrix:
                        break
        except Exception:
            logger.warning("Failed to load portfolio detector inputs", exc_info=True)

        # Aggregate daily snapshots into per-family summary dicts
        aggregated_families: dict = {}
        for fam, daily_list in family_snapshots.items():
            total_pnl = sum(s.get("total_net_pnl", 0.0) for s in daily_list)
            total_trades = sum(s.get("trade_count", 0) for s in daily_list)
            total_wins = sum(s.get("win_count", 0) for s in daily_list)
            aggregated_families[fam] = {
                "total_net_pnl": total_pnl,
                "trade_count": total_trades,
                "win_count": total_wins,
                "days": len(daily_list),
            }

        # Run each detector
        if aggregated_families and family_allocations:
            try:
                results.extend(engine.detect_family_imbalance(
                    aggregated_families, family_allocations,
                ))
            except Exception:
                logger.warning("detect_family_imbalance failed", exc_info=True)

        if correlation_matrix and family_allocations:
            try:
                results.extend(engine.detect_correlation_concentration(
                    correlation_matrix, family_allocations,
                ))
            except Exception:
                logger.warning("detect_correlation_concentration failed", exc_info=True)

        # Drawdown tier miscalibration
        if self._strategy_registry and hasattr(self._strategy_registry, "portfolio"):
            tiers = getattr(self._strategy_registry.portfolio, "drawdown_tiers", [])
            if tiers:
                try:
                    # Collect historical drawdowns from risk cards across the week
                    hist_drawdowns: list[float] = []
                    for i in range(7):
                        date_str = (start_dt + _td_pf(days=i)).strftime("%Y-%m-%d")
                        risk_path = self._curated_dir / date_str / "portfolio_risk_card.json"
                        if risk_path.exists():
                            rdata = json.loads(risk_path.read_text(encoding="utf-8"))
                            dd = rdata.get("max_drawdown_pct", 0.0)
                            if isinstance(dd, (int, float)):
                                hist_drawdowns.append(float(dd))
                    if hist_drawdowns:
                        results.extend(engine.detect_drawdown_tier_miscalibration(
                            hist_drawdowns, tiers,
                        ))
                except Exception:
                    logger.warning("detect_drawdown_tier_miscalibration failed", exc_info=True)

        # Coordination gaps
        try:
            concurrent_path = self._curated_dir / "weekly" / week_start / "concurrent_position_analysis.json"
            if concurrent_path.exists():
                concurrent_data = json.loads(concurrent_path.read_text(encoding="utf-8"))
                coord_config = None
                if self._strategy_registry and hasattr(self._strategy_registry, "coordination"):
                    coord_config = self._strategy_registry.coordination.model_dump(mode="json")
                results.extend(engine.detect_coordination_gaps(
                    concurrent_data, coord_config,
                ))
        except Exception:
            logger.warning("detect_coordination_gaps failed", exc_info=True)

        # Heat cap utilization
        if self._strategy_registry and hasattr(self._strategy_registry, "portfolio"):
            heat_cap = self._strategy_registry.portfolio.heat_cap_R
            if heat_cap > 0:
                try:
                    daily_heat: list[float] = []
                    for i in range(7):
                        date_str = (start_dt + _td_pf(days=i)).strftime("%Y-%m-%d")
                        risk_path = self._curated_dir / date_str / "portfolio_risk_card.json"
                        if risk_path.exists():
                            rdata = json.loads(risk_path.read_text(encoding="utf-8"))
                            heat = rdata.get("total_heat_R") or rdata.get("total_exposure", 0)
                            if isinstance(heat, (int, float)):
                                daily_heat.append(float(heat))
                    if daily_heat:
                        results.extend(engine.detect_heat_cap_utilization(
                            daily_heat, heat_cap,
                        ))
                except Exception:
                    logger.warning("detect_heat_cap_utilization failed", exc_info=True)

        # Drawdown correlation across families
        if family_snapshots and len(family_snapshots) >= 2:
            try:
                from itertools import accumulate
                from trading_assistant.skills.compute_portfolio_risk import PortfolioRiskComputer

                family_equity: dict[str, list[float]] = {}
                for fam, daily_list in family_snapshots.items():
                    # Convert daily PnL to cumulative equity curve - # _drawdown_series() computes drawdown from peak and
                    # expects an equity curve, not individual daily values.
                    family_equity[fam] = list(accumulate(
                        s.get("total_net_pnl", 0.0) for s in daily_list
                    ))
                dd_corr = PortfolioRiskComputer.compute_drawdown_correlation(family_equity)
                weekly_dir = self._curated_dir / "weekly" / week_start
                weekly_dir.mkdir(parents=True, exist_ok=True)
                (weekly_dir / "drawdown_correlation.json").write_text(
                    json.dumps(dd_corr, indent=2, default=str), encoding="utf-8"
                )
            except Exception:
                logger.warning("compute_drawdown_correlation failed", exc_info=True)

        return results

    def _record_portfolio_proposals(self, validation, run_id: str) -> None:
        """Record approved portfolio proposals as SuggestionRecords.

        Enforces cadence gate and concurrent deployment limit.
        """
        if not self._suggestion_tracker or validation is None:
            return
        proposals = getattr(validation, "approved_portfolio_proposals", [])
        if not proposals:
            return

        import hashlib
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        # Concurrent deployment check: max 1 DEPLOYED portfolio change
        try:
            deployed_count = self._suggestion_tracker.get_deployed_portfolio_count()
            if deployed_count >= 1:
                logger.info(
                    "Skipping portfolio proposal recording: %d already deployed", deployed_count,
                )
                return
        except Exception:
            logger.warning("Portfolio deployment count check failed - proceeding cautiously")

        for idx, proposal in enumerate(proposals):
            ptype = getattr(proposal, "proposal_type", "unknown")
            ptype_str = ptype.value if hasattr(ptype, "value") else str(ptype)

            # Cadence gate
            if not self._check_portfolio_cadence(ptype_str):
                logger.info("Cadence gate blocked portfolio proposal: %s", ptype_str)
                continue

            # Run what-if analysis for allocation proposals
            what_if_result = None
            if ptype_str == "allocation_rebalance":
                try:
                    from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

                    proposed = getattr(proposal, "proposed_config", {}) or {}
                    current = getattr(proposal, "current_config", {}) or {}
                    # Build family PnL series from curated snapshots
                    family_pnl = self._load_family_pnl_for_what_if()
                    # Try trade-level loading for enriched metrics
                    family_trades = None
                    if self._strategy_registry:
                        try:
                            family_trades = self._load_family_trades_for_what_if()
                        except Exception:
                            logger.warning("Trade-level loading failed, using daily aggregates")
                    if (family_pnl or family_trades) and current:
                        what_if = PortfolioWhatIf(
                            family_daily_pnl=family_pnl or {},
                            current_weights=current,
                            family_trades=family_trades,
                        )
                        what_if_result = what_if.evaluate(proposed)
                    if what_if_result and what_if_result.get("calmar_delta", 0) < 0:
                        logger.info(
                            "What-if shows negative Calmar delta for allocation proposal - skipping",
                        )
                        continue
                except Exception:
                    logger.warning("Portfolio what-if failed - recording without it")

            raw = f"{run_id}:portfolio:{idx}:{ptype_str}"
            suggestion_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

            detection_ctx: dict = {}
            if what_if_result:
                detection_ctx["what_if_result"] = what_if_result
            detection_ctx["current_config"] = getattr(proposal, "current_config", {})
            detection_ctx["proposed_config"] = getattr(proposal, "proposed_config", {})

            ledger_proposal_id = self._ledger_write_candidate(
                source="portfolio",
                kind_hint="portfolio_change",
                bot_id="PORTFOLIO",
                title=f"Portfolio: {ptype_str}",
                description=getattr(proposal, "evidence_summary", "") or "",
                suggestion_id=suggestion_id,
                run_id=run_id,
                evaluation_method="approval",
            )
            record = SuggestionRecord(
                suggestion_id=suggestion_id,
                bot_id="PORTFOLIO",
                title=f"Portfolio: {ptype_str}",
                tier="portfolio",
                category=f"portfolio_{ptype_str}" if not ptype_str.startswith("portfolio_") else ptype_str,
                source_report_id=run_id,
                description=getattr(proposal, "evidence_summary", "") or "",
                confidence=float(getattr(proposal, "confidence", 0.5) or 0.5),
                detection_context=detection_ctx if detection_ctx else None,
                proposal_id=ledger_proposal_id or None,
            )
            recorded = self._suggestion_tracker.record(record)
            if recorded is not False:
                logger.info("Recorded portfolio proposal %s: %s", suggestion_id, ptype_str)
                self._event_stream.broadcast("portfolio_proposal_recorded", {
                    "run_id": run_id, "proposal_type": ptype_str,
                    "suggestion_id": suggestion_id,
                })

    def _check_portfolio_cadence(self, proposal_type: str) -> bool:
        """Check if enough time has passed since the last portfolio proposal of this type.

        Returns True if the cadence gate allows a new proposal.
        """
        if not self._suggestion_tracker:
            return True

        try:
            # Allocation: 30 days; risk/drawdown: 90 days
            if "allocation" in proposal_type or "coordination" in proposal_type:
                min_days = 30
            else:
                min_days = 90

            last_date = self._suggestion_tracker.get_last_portfolio_proposal_date(
                proposal_type=proposal_type,
            )
            if not last_date:
                return True  # No history - allow

            last_dt = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last_dt).days
            if elapsed < min_days:
                logger.info(
                    "Portfolio cadence gate: %s last proposed %d days ago (min %d)",
                    proposal_type, elapsed, min_days,
                )
                return False
            return True
        except Exception:
            logger.warning("Portfolio cadence check failed - allowing proposal")
            return True

    def _load_family_pnl_for_what_if(self, lookback_days: int = 60) -> dict[str, list[float]]:
        """Load family daily PnL series for what-if analysis."""
        from datetime import timedelta as _td_wif

        family_pnl: dict[str, list[float]] = {}
        end = datetime.now(timezone.utc)
        for d in range(lookback_days):
            date_str = (end - _td_wif(days=d)).strftime("%Y-%m-%d")
            snap_path = self._curated_dir / date_str / "portfolio" / "family_snapshots.json"
            if not snap_path.exists():
                continue
            try:
                snaps = json.loads(snap_path.read_text(encoding="utf-8"))
                for snap in snaps:
                    fam = snap.get("family", "")
                    if fam:
                        family_pnl.setdefault(fam, []).append(
                            snap.get("total_net_pnl", 0.0),
                        )
            except (json.JSONDecodeError, OSError):
                continue
        # Reverse so oldest first
        for fam in family_pnl:
            family_pnl[fam].reverse()
        return family_pnl

    def _load_family_trades_for_what_if(
        self, lookback_days: int = 60,
    ) -> dict[str, list]:
        """Load per-family trade-level data for enriched what-if analysis.

        Scans curated directories for trades.jsonl files, groups trades by
        family using strategy_registry bot_id - family mapping.
        Returns family_name - list[TradeEvent].
        """
        from datetime import timedelta as _td_trades
        from trading_assistant.schemas.events import TradeEvent

        if not self._strategy_registry:
            return {}

        # Build bot_id - family mapping
        bot_to_family: dict[str, str] = {}
        for _sid, profile in self._strategy_registry.strategies.items():
            if profile.bot_id and profile.family:
                bot_to_family[profile.bot_id] = profile.family

        if not bot_to_family:
            return {}

        family_trades: dict[str, list] = {}
        end = datetime.now(timezone.utc)

        for d in range(lookback_days):
            date_str = (end - _td_trades(days=d)).strftime("%Y-%m-%d")
            date_dir = self._curated_dir / date_str
            if not date_dir.is_dir():
                continue

            for bot_id, family in bot_to_family.items():
                trades_file = date_dir / bot_id / "trades.jsonl"
                if not trades_file.exists():
                    continue
                for line in trades_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            trade = TradeEvent(**json.loads(line))
                            family_trades.setdefault(family, []).append(trade)
                        except Exception:
                            logger.warning("Bad trade record in %s", trades_file)

        logger.info(
            "Loaded %d families with %d total trades for what-if",
            len(family_trades),
            sum(len(ts) for ts in family_trades.values()),
        )
        return family_trades
