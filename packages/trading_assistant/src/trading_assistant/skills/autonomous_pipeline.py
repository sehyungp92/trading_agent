"""Autonomous pipeline for suggestion-to-approval automation."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from trading_assistant.schemas.approval import ApprovalRequest, RepoRiskTier
from trading_assistant.schemas.parameter_definition import ParameterDefinition
from trading_assistant.schemas.repo_changes import ChangeKind, FileChange
from trading_assistant.schemas.experiments import ExperimentType
from trading_assistant.schemas.parameter_search import SearchRouting
from trading_assistant.skills.approval_tracker import ApprovalTracker
from trading_assistant.skills.config_registry import ConfigRegistry
from trading_assistant.skills.repo_change_guard import RepoChangeGuard
from trading_assistant.skills.suggestion_backtester import SuggestionBacktester

logger = logging.getLogger(__name__)

_ACTIONABLE_TIERS = {"parameter", "filter", "hypothesis"}
_MIN_CONFIDENCE = 0.5
_MAX_ACTIVE_EXPERIMENTS = 2


class AutonomousPipeline:
    """Processes recorded suggestions into approval requests."""

    def __init__(
        self,
        config_registry: ConfigRegistry,
        backtester: SuggestionBacktester,
        approval_tracker: ApprovalTracker,
        suggestion_tracker: Any,
        telegram_bot: Any | None = None,
        telegram_renderer: Any | None = None,
        event_stream: Any | None = None,
        repo_change_guard: RepoChangeGuard | None = None,
        parameter_searcher: Any | None = None,
        experiment_config_generator: Any | None = None,
        experiment_tracker: Any | None = None,
        experiment_manager: Any | None = None,
        calibration_tracker: Any | None = None,
        proposal_ledger: Any | None = None,
        search_log_dir: Path | None = None,
        curated_dir: Path | None = None,
    ) -> None:
        self._registry = config_registry
        self._backtester = backtester
        self._approval_tracker = approval_tracker
        self._suggestion_tracker = suggestion_tracker
        self._telegram_bot = telegram_bot
        self._telegram_renderer = telegram_renderer
        self._event_stream = event_stream
        self._repo_change_guard = repo_change_guard or RepoChangeGuard()
        self._parameter_searcher = parameter_searcher
        self._experiment_config_generator = experiment_config_generator
        self._experiment_tracker = experiment_tracker
        self._experiment_manager = experiment_manager
        self._proposal_ledger = proposal_ledger
        self._calibration_tracker = calibration_tracker
        self._search_log_dir = search_log_dir
        self._curated_dir = curated_dir

    async def process_new_suggestions(
        self,
        suggestion_ids: list[str],
        run_id: str | None = None,
    ) -> list[ApprovalRequest]:
        """Process new suggestions into approval requests."""
        results: list[ApprovalRequest] = []
        all_suggestions = self._load_all_suggestions()

        for suggestion_id in suggestion_ids:
            try:
                result = await self._process_one(suggestion_id, run_id, all_suggestions)
                if result is not None:
                    results.append(result)
            except Exception:
                logger.exception("Pipeline error processing suggestion %s", suggestion_id)

        if results and self._event_stream:
            self._event_stream.broadcast("autonomous_pipeline_complete", {
                "run_id": run_id,
                "approval_requests_created": len(results),
            })
        return results

    async def _process_one(
        self,
        suggestion_id: str,
        run_id: str | None,
        all_suggestions: dict[str, dict],
    ) -> ApprovalRequest | None:
        suggestion = all_suggestions.get(suggestion_id)
        if suggestion is None:
            logger.debug("Suggestion %s not found in tracker", suggestion_id)
            return None
        if not self._is_actionable(suggestion):
            logger.debug("Suggestion %s not actionable", suggestion_id)
            return None

        tier = suggestion.get("tier", "")
        if tier in {"parameter", "filter"}:
            request = await self._build_parameter_request(suggestion_id, suggestion)
        elif tier == "hypothesis":
            request = self._build_structural_request(suggestion_id, suggestion)
        else:
            request = None

        if request is None:
            return None

        self._approval_tracker.create_request(request)
        logger.info("Created approval request %s for suggestion %s", request.request_id, suggestion_id)
        await self._send_telegram_notification(request)
        return request

    async def _build_parameter_request(
        self,
        suggestion_id: str,
        suggestion: dict,
    ) -> ApprovalRequest | None:
        params = self._registry.resolve_suggestion_to_params(suggestion)
        if not params:
            logger.debug("No matching parameters for suggestion %s", suggestion_id)
            return None

        param = params[0]
        proposed_value = self._extract_proposed_value(suggestion, param)
        if proposed_value is None:
            logger.debug("Could not extract proposed value for suggestion %s", suggestion_id)
            return None

        valid, message = self._registry.validate_value(param, proposed_value)
        if not valid:
            logger.info("Proposed value invalid for %s: %s", suggestion_id, message)
            return None

        bot_id = suggestion.get("bot_id", "")

        backtest_summary = None
        search_summary_json = ""
        if self._parameter_searcher:
            # NEW PATH: neighborhood search
            trades, missed = self._load_trade_data(bot_id)
            report = self._parameter_searcher.search_with_regime_analysis(
                suggestion_id, bot_id, param, proposed_value, trades, missed,
            )
            self._persist_search_report(report)

            if report.routing == SearchRouting.DISCARD:
                self._record_search_signal(bot_id, param.category, positive=False)
                self._record_parameter_search_evaluation(suggestion, report)
                logger.info(
                    "Search DISCARD for %s: %s", suggestion_id, report.discard_reason,
                )
                return None

            # Calibration override: if backtest is unreliable, force EXPERIMENT
            if (
                self._calibration_tracker
                and report.routing == SearchRouting.APPROVE
            ):
                modifier = self._calibration_tracker.get_approval_modifier(
                    bot_id, param.category,
                )
                if modifier == "require_experiment":
                    report.routing = SearchRouting.EXPERIMENT
                    logger.info(
                        "Overriding APPROVE→EXPERIMENT: backtest unreliable for %s/%s",
                        bot_id, param.category,
                    )
                elif modifier == "fast_track":
                    logger.info(
                        "Fast-track: high-reliability backtest for %s/%s — proceeding with confidence",
                        bot_id, param.category,
                    )

            self._record_parameter_search_evaluation(suggestion, report)

            # Record calibration prediction for BOTH APPROVE and EXPERIMENT paths
            if self._calibration_tracker and report.baseline_composite > 0:
                self._calibration_tracker.record_prediction(
                    suggestion_id, bot_id, param.category,
                    predicted_improvement=report.best_composite / report.baseline_composite,
                    predicted_routing=report.routing,
                )

            if report.routing == SearchRouting.EXPERIMENT:
                return self._route_to_experiment(
                    suggestion_id, suggestion, param, report,
                )

            # APPROVE path — use best_value (may differ from Claude's proposed)
            best_value = report.best_value
            self._record_search_signal(bot_id, param.category, positive=True)

            # Build lightweight backtest summary from search report
            search_summary_json = json.dumps({
                "baseline_composite": round(report.baseline_composite, 4),
                "best_composite": round(report.best_composite, 4),
                "robustness_score": round(report.best_robustness_score, 4),
                "exploration_summary": report.exploration_summary or "",
                "search_routing": report.routing.value,
            })
        else:
            # LEGACY PATH: existing single-value backtest (unchanged)
            comparison = await self._backtester.backtest_suggestion(
                suggestion_id=suggestion_id,
                bot_id=bot_id,
                param_name=param.param_name,
                current_value=param.current_value,
                proposed_value=proposed_value,
            )
            if not comparison.passes_safety:
                logger.info("Backtest failed safety for %s: %s", suggestion_id, comparison.safety_notes)
                return None
            best_value = proposed_value
            backtest_summary = comparison

        profile = self._registry.get_profile(bot_id)
        if profile and self._repo_change_guard.blocked_paths(profile, [param.file_path]):
            logger.info(
                "Blocked parameter suggestion %s outside allowed_edit_paths: %s",
                suggestion_id,
                param.file_path,
            )
            return None
        risk_tier = self._risk_tier_for_files(profile, [param.file_path]) if profile else RepoRiskTier.REQUIRES_APPROVAL
        request_id = hashlib.sha256(
            f"{suggestion_id}:{param.param_name}:{best_value}".encode(),
        ).hexdigest()[:12]

        return ApprovalRequest(
            request_id=request_id,
            suggestion_id=suggestion_id,
            bot_id=bot_id,
            change_kind=ChangeKind.PARAMETER_CHANGE,
            title=f"Update {param.param_name}",
            summary=suggestion.get("title", ""),
            param_changes=[{
                "param_name": param.param_name,
                "current": param.current_value,
                "proposed": best_value,
            }],
            planned_files=[param.file_path],
            verification_commands=profile.verification_commands if profile else [],
            risk_tier=risk_tier,
            backtest_summary=backtest_summary,
            implementation_notes=search_summary_json,
        )

    def _build_structural_request(
        self,
        suggestion_id: str,
        suggestion: dict,
    ) -> ApprovalRequest | None:
        context = suggestion.get("implementation_context") or {}
        file_changes = self._parse_file_changes(context.get("file_changes", []))
        implementation_notes = context.get("notes", "") or suggestion.get("description", "")
        if not file_changes and not implementation_notes:
            logger.debug("No implementation payload for structural suggestion %s", suggestion_id)
            return None

        bot_id = suggestion.get("bot_id", "")

        # Structural validation gate: check category track record if available
        category = suggestion.get("category", "structural")
        calibration_escalate = False
        if self._calibration_tracker:
            modifier = self._calibration_tracker.get_approval_modifier(
                bot_id, category,
            )
            if modifier == "require_experiment":
                calibration_escalate = True
                logger.info(
                    "Structural suggestion %s in low-reliability category %s/%s — "
                    "escalating to double approval",
                    suggestion_id, bot_id, category,
                )

        profile = self._registry.get_profile(bot_id)
        planned_files = list(context.get("planned_files", []) or [])
        if not planned_files:
            planned_files = [file_change.file_path for file_change in file_changes]
        if profile and self._repo_change_guard.blocked_paths(profile, planned_files):
            logger.info(
                "Blocked structural suggestion %s outside allowed_edit_paths: %s",
                suggestion_id,
                planned_files,
            )
            return None
        risk_tier = (
            self._risk_tier_for_files(profile, planned_files)
            if profile and planned_files
            else RepoRiskTier.REQUIRES_APPROVAL
        )
        # Escalate risk tier for low-reliability categories
        if calibration_escalate and risk_tier != RepoRiskTier.REQUIRES_DOUBLE_APPROVAL:
            risk_tier = RepoRiskTier.REQUIRES_DOUBLE_APPROVAL
        request_id = hashlib.sha256(
            f"{suggestion_id}:structural:{'|'.join(planned_files)}:{implementation_notes}".encode(),
        ).hexdigest()[:12]

        return ApprovalRequest(
            request_id=request_id,
            suggestion_id=suggestion_id,
            bot_id=bot_id,
            change_kind=ChangeKind.STRUCTURAL_CHANGE,
            title=suggestion.get("title", "Structural change"),
            summary=suggestion.get("description", ""),
            file_changes=file_changes,
            planned_files=planned_files,
            verification_commands=context.get("verification_commands", []) or (
                profile.verification_commands if profile else []
            ),
            risk_tier=risk_tier,
            draft_pr=risk_tier == RepoRiskTier.AUTO and bool(file_changes),
            implementation_notes=implementation_notes,
            hypothesis_id=suggestion.get("hypothesis_id"),
        )

    def _load_all_suggestions(self) -> dict[str, dict]:
        suggestions: dict[str, dict] = {}
        if not self._suggestion_tracker:
            return suggestions
        for suggestion in self._suggestion_tracker.load_all():
            suggestion_id = suggestion.get("suggestion_id", "")
            if suggestion_id:
                suggestions[suggestion_id] = suggestion
        return suggestions

    def _is_actionable(self, suggestion: dict) -> bool:
        tier = suggestion.get("tier", "")
        if tier not in _ACTIONABLE_TIERS:
            return False
        if suggestion.get("confidence", 0.0) < _MIN_CONFIDENCE:
            return False
        suggestion_id = suggestion.get("suggestion_id", "")
        return self._approval_tracker.find_latest_for_suggestion(suggestion_id) is None

    def _extract_proposed_value(
        self,
        suggestion: dict,
        param: ParameterDefinition,
    ) -> Any | None:
        structured_value = suggestion.get("proposed_value")
        if structured_value is not None:
            return self._cast_value(structured_value, param.value_type)

        text = f"{suggestion.get('title', '')} {suggestion.get('description', '')}"
        param_variants = [param.param_name, param.param_name.replace("_", " ")]
        for variant in param_variants:
            pattern = rf"(?:increase|decrease|set|adjust)\s+{re.escape(variant)}\s+to\s+([0-9.]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._cast_value(match.group(1), param.value_type)

            pattern = rf"change\s+{re.escape(variant)}\s+from\s+[0-9.]+\s+to\s+([0-9.]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._cast_value(match.group(1), param.value_type)

        match = re.search(r"\bto\s+([0-9]+\.?[0-9]*)\b", text)
        if match:
            return self._cast_value(match.group(1), param.value_type)
        return None

    @staticmethod
    def _cast_value(value: Any, value_type: str) -> Any:
        try:
            if value_type == "int":
                return int(float(value))
            if value_type == "float":
                return float(value)
            if value_type == "bool":
                return str(value).lower() in ("true", "1", "yes")
            return value
        except (TypeError, ValueError):
            return None

    def _risk_tier_for_files(
        self,
        profile,
        file_paths: list[str],
    ) -> RepoRiskTier:
        if profile is None:
            return RepoRiskTier.REQUIRES_APPROVAL
        result = self._repo_change_guard.check_paths(profile, file_paths)
        if result.tier.value >= 2:
            return RepoRiskTier.REQUIRES_DOUBLE_APPROVAL
        if result.tier.value == 0:
            return RepoRiskTier.AUTO
        return RepoRiskTier.REQUIRES_APPROVAL

    @staticmethod
    def _parse_file_changes(raw_changes: list[dict]) -> list[FileChange]:
        parsed: list[FileChange] = []
        for raw_change in raw_changes:
            try:
                parsed.append(FileChange(**raw_change))
            except Exception:
                logger.warning("Skipping malformed file change payload", exc_info=True)
        return parsed

    def _load_trade_data(
        self,
        bot_id: str,
    ) -> tuple[list, list]:
        """Load TradeEvent and MissedOpportunityEvent from curated data."""
        from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent

        trades: list[TradeEvent] = []
        missed: list[MissedOpportunityEvent] = []
        if not self._curated_dir or not self._curated_dir.exists():
            return trades, missed
        for date_dir in sorted(self._curated_dir.iterdir(), reverse=True)[:30]:
            bot_dir = date_dir / bot_id
            trades_file = bot_dir / "trades.jsonl"
            if trades_file.exists():
                for line in trades_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            trades.append(TradeEvent(**json.loads(line)))
                        except Exception:
                            pass
            missed_file = bot_dir / "missed.jsonl"
            if missed_file.exists():
                for line in missed_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            missed.append(MissedOpportunityEvent(**json.loads(line)))
                        except Exception:
                            pass
        return trades, missed

    def _persist_search_report(self, report) -> None:
        """Persist search report to JSONL."""
        if not self._search_log_dir:
            return
        try:
            self._search_log_dir.mkdir(parents=True, exist_ok=True)
            path = self._search_log_dir / "search_reports.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(report.model_dump_json() + "\n")
        except Exception:
            logger.exception("Failed to persist search report")

    def _record_search_signal(
        self,
        bot_id: str,
        category: str,
        positive: bool,
    ) -> None:
        """Append search signal to JSONL for weekly synthesis."""
        if not self._search_log_dir:
            return
        try:
            self._search_log_dir.mkdir(parents=True, exist_ok=True)
            path = self._search_log_dir / "search_signals.jsonl"
            from datetime import datetime, timezone
            record = json.dumps({
                "bot_id": bot_id,
                "category": category,
                "positive": positive,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            with path.open("a", encoding="utf-8") as f:
                f.write(record + "\n")
        except Exception:
            logger.exception("Failed to record search signal")

    def _record_parameter_search_evaluation(self, suggestion: dict, report) -> None:
        """Append ParameterSearcher routing as a ProposalLedger evaluation."""
        ledger = getattr(self, "_proposal_ledger", None)
        proposal_id = suggestion.get("proposal_id") if isinstance(suggestion, dict) else ""
        if not ledger or not proposal_id:
            return
        try:
            from trading_assistant.schemas.proposal_ledger import ProposalEvaluation
            decision = {
                SearchRouting.APPROVE: "approve",
                SearchRouting.EXPERIMENT: "experiment",
                SearchRouting.DISCARD: "reject",
            }.get(report.routing, "defer")
            ledger.record_evaluation(
                proposal_id,
                ProposalEvaluation(
                    proposal_id=proposal_id,
                    method="parameter_search",
                    decision=decision,
                    decision_reason=getattr(report, "discard_reason", "") or "",
                    objective_score=float(getattr(report, "best_composite", 0.0) or 0.0),
                    confidence=min(
                        1.0,
                        max(0.0, float(getattr(report, "best_robustness_score", 0.0) or 0.0) / 100.0),
                    ),
                    summary=getattr(report, "exploration_summary", "") or "",
                ),
            )
        except Exception:
            logger.warning("Failed to record parameter-search proposal evaluation", exc_info=True)

    def _route_to_experiment(
        self,
        suggestion_id: str,
        suggestion: dict,
        param: ParameterDefinition,
        report,
    ) -> ApprovalRequest | None:
        """Route marginal result to A/B experiment."""
        if not self._experiment_config_generator:
            logger.debug("No experiment_config_generator; skipping experiment route for %s", suggestion_id)
            return None

        bot_id = suggestion.get("bot_id", "")

        # Check experiment cap
        if self._experiment_manager:
            active = self._experiment_manager.get_active()
            bot_active = [e for e in active if getattr(e, "bot_id", "") == bot_id]
            if len(bot_active) >= _MAX_ACTIVE_EXPERIMENTS:
                logger.info(
                    "Experiment cap reached for %s (%d active); skipping %s",
                    bot_id, len(bot_active), suggestion_id,
                )
                return None
        # Determine experiment type: ABLATION for boolean params, PARAMETER_AB for others
        is_ablation = (
            param.value_type == "bool"
            or getattr(param, "category", "") == "ablation"
        )
        experiment_type = ExperimentType.ABLATION if is_ablation else ExperimentType.PARAMETER_AB

        config = self._experiment_config_generator.generate_from_suggestion(
            suggestion_id=suggestion_id,
            bot_id=bot_id,
            param_name=param.param_name,
            current_value=param.current_value,
            proposed_value=report.best_value,
            title=suggestion.get("title", ""),
        )
        config.experiment_type = experiment_type

        # Persist DRAFT experiment via ExperimentManager so it survives
        # beyond the ApprovalRequest string reference
        if self._experiment_manager is not None:
            try:
                self._experiment_manager.create_experiment(config)
                logger.info(
                    "Persisted DRAFT experiment %s for suggestion %s",
                    config.experiment_id, suggestion_id,
                )
            except Exception:
                logger.warning("Failed to persist experiment config for %s", suggestion_id)

        profile = self._registry.get_profile(bot_id)
        risk_tier = self._risk_tier_for_files(profile, [param.file_path]) if profile else RepoRiskTier.REQUIRES_APPROVAL
        request_id = hashlib.sha256(
            f"{suggestion_id}:experiment:{param.param_name}".encode(),
        ).hexdigest()[:12]

        return ApprovalRequest(
            request_id=request_id,
            suggestion_id=suggestion_id,
            bot_id=bot_id,
            change_kind=ChangeKind.STRUCTURAL_CHANGE,
            title=f"A/B experiment: {param.param_name}",
            summary=f"Marginal search result for {param.param_name}; proposing A/B experiment. {report.exploration_summary}",
            planned_files=[param.file_path],
            verification_commands=profile.verification_commands if profile else [],
            risk_tier=risk_tier,
            implementation_notes=f"Experiment config: {config.experiment_id}",
        )

    async def _send_telegram_notification(self, request: ApprovalRequest) -> None:
        if self._telegram_renderer is None or self._telegram_bot is None:
            return
        try:
            text, keyboard = self._telegram_renderer.render_approval_request(request)
            message_id = await self._telegram_bot.send_message(text, keyboard=keyboard)
            if message_id is not None:
                request.message_id = message_id
                self._approval_tracker.set_message_id(request.request_id, message_id)
        except Exception:
            logger.exception("Failed to send Telegram notification for %s", request.request_id)
