"""Bug triage action orchestration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.orchestrator.orchestrator_brain import Action
from trading_assistant.schemas.notifications import NotificationPriority

logger = logging.getLogger(__name__)


class BugTriageActions:
    """Bug triage action orchestration."""

    async def handle_triage(self, action: Action) -> None:
        """Run the triage pipeline: classify -> context -> assemble -> invoke -> notify."""
        details = action.details or {}
        bot_id = details.get("bot_id", action.bot_id)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"triage-{bot_id}-{timestamp}"

        try:
            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "started", "handler": "triage",
            })

            from trading_assistant.analysis.triage_response_parser import parse_triage_response
            from trading_assistant.schemas.bug_triage import ErrorEvent, TriageOutcome
            from trading_assistant.skills.run_bug_triage import TriageRunner
            from trading_assistant.skills.triage_context_builder import TriageContextBuilder
            from trading_assistant.analysis.triage_prompt_assembler import TriagePromptAssembler

            event = ErrorEvent(
                bot_id=bot_id,
                error_type=details.get("error_type", "Unknown"),
                message=details.get("message", ""),
                stack_trace=details.get("stack_trace", ""),
                source_file=details.get("source_file", ""),
                source_line=details.get("source_line", 0),
                context=details.get("context", {}),
            )

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "classification", "handler": "triage",
            })

            triage_runner = TriageRunner(
                source_root=self._source_root,
                failure_log_path=self._failure_log_path,
            )
            triage_result = triage_runner.triage(event)

            self._event_stream.broadcast("triage_classified", {
                "bot_id": bot_id,
                "severity": triage_result.severity.value,
                "complexity": triage_result.complexity.value,
                "outcome": triage_result.outcome.value,
            })

            # Record recurrence against open reliability interventions
            if self._reliability_tracker is not None:
                try:
                    bug_class = self._map_error_to_bug_class(
                        triage_result.error_event.category.value
                        if triage_result.error_event.category else "unknown"
                    )
                    matched = self._reliability_tracker.record_recurrence(
                        bot_id, bug_class, event.error_type,
                    )
                    if matched:
                        logger.info(
                            "Recurrence matched intervention %s for %s/%s",
                            matched, bot_id, bug_class.value,
                        )
                except Exception:
                    logger.warning("Failed to check reliability recurrence for %s", bot_id)

            agent_response = ""
            repair_proposal = None
            if triage_result.outcome in (TriageOutcome.KNOWN_FIX, TriageOutcome.NEEDS_INVESTIGATION):
                self._event_stream.broadcast("handler_progress", {
                    "run_id": run_id, "stage": "context_build", "handler": "triage",
                })
                ctx_builder = TriageContextBuilder(source_root=self._source_root)
                from trading_assistant.skills.failure_log import FailureLog
                failure_log = FailureLog(self._failure_log_path)
                past_rejections = failure_log.get_past_rejections(
                    error_type=event.error_type, limit=5,
                )
                from trading_assistant.schemas.bug_triage import ErrorCategory

                category = triage_result.error_event.category or ErrorCategory.UNKNOWN
                context = ctx_builder.build(
                    event,
                    triage_result.severity,
                    category,
                    past_rejections,
                )

                assembler = TriagePromptAssembler(memory_dir=self._memory_dir)
                package = assembler.assemble(
                    context, triage_result.severity, triage_result.complexity,
                    session_store=self._agent_runner.session_store,
                    bot_id=bot_id,
                )

                result = await self._agent_runner.invoke(
                    agent_type="triage",
                    prompt_package=package,
                    run_id=run_id,
                    allowed_tools=["Read", "Bash", "Grep", "Glob"],
                )

                if result.success:
                    agent_response = result.response
                    repair_proposal = parse_triage_response(result.response)

                    # Record reliability intervention for successful triage fix
                    if self._reliability_tracker is not None and repair_proposal:
                        try:
                            from trading_assistant.schemas.reliability_learning import (
                                ReliabilityIntervention,
                            )
                            _bug_class = self._map_error_to_bug_class(
                                triage_result.error_event.category.value
                                if triage_result.error_event.category else "unknown"
                            )
                            from trading_assistant.skills.reliability_tracker import ReliabilityTracker
                            intervention = ReliabilityIntervention(
                                intervention_id=ReliabilityTracker.generate_id(
                                    bot_id, _bug_class.value,
                                    datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                ),
                                bot_id=bot_id,
                                bug_class=_bug_class,
                                error_category=event.error_type,
                                triage_run_id=run_id,
                                fix_description=(
                                    getattr(repair_proposal, "fix_plan", "")
                                    or getattr(repair_proposal, "issue_title", "")
                                )[:200],
                            )
                            self._reliability_tracker.record_intervention(intervention)
                        except Exception:
                            logger.warning("Failed to record reliability intervention for %s", run_id)

            if triage_result.outcome == TriageOutcome.KNOWN_FIX:
                handled = await self._handle_known_fix_triage(
                    triage_result,
                    repair_proposal,
                    agent_response,
                )
                if not handled:
                    await self._notify(
                        notification_type="triage_result",
                        priority=self._severity_to_priority(triage_result.severity),
                        title=f"Triage [{triage_result.severity.value.upper()}] {bot_id}",
                        body=(agent_response or triage_result.suggested_fix or event.message)[:2000],
                    )
            elif triage_result.outcome == TriageOutcome.NEEDS_INVESTIGATION:
                handled = await self._handle_investigation_triage(
                    triage_result,
                    repair_proposal,
                    agent_response,
                )
                if not handled:
                    await self._notify(
                        notification_type="triage_result",
                        priority=self._severity_to_priority(triage_result.severity),
                        title=f"Triage [{triage_result.severity.value.upper()}] {bot_id}",
                        body=(agent_response or event.message)[:2000],
                    )
            elif triage_result.outcome == TriageOutcome.NEEDS_HUMAN:
                await self._notify(
                    notification_type="triage_needs_human",
                    priority=NotificationPriority.HIGH,
                    title=f"Triage [{triage_result.severity.value.upper()}] {bot_id} — needs human",
                    body=f"{event.error_type}: {event.message}",
                )

        except Exception as exc:
            logger.exception("Triage handler failed for %s", run_id)
            self._event_stream.broadcast("triage_error", {
                "bot_id": bot_id,
                "error": str(exc),
            })

    async def _handle_known_fix_triage(
        self,
        triage_result,
        repair_proposal,
        agent_response: str,
    ) -> bool:
        if (
            repair_proposal is None
            or self._approval_tracker is None
            or self._config_registry is None
        ):
            return False

        from trading_assistant.schemas.approval import ApprovalRequest, RepoRiskTier
        from trading_assistant.schemas.repo_changes import ChangeKind
        from trading_assistant.skills.repo_change_guard import RepoChangeGuard

        profile = self._config_registry.get_profile(triage_result.error_event.bot_id)
        if profile is None:
            return False

        file_changes = list(repair_proposal.file_changes)
        planned_files = repair_proposal.candidate_files or [
            file_change.file_path for file_change in file_changes
        ]
        if not planned_files and not (repair_proposal.fix_plan or agent_response):
            return False

        guard = RepoChangeGuard()
        blocked_paths = guard.blocked_paths(profile, planned_files) if planned_files else []
        if blocked_paths:
            await self._notify(
                notification_type="triage_result",
                priority=self._severity_to_priority(triage_result.severity),
                title=f"Triage [{triage_result.severity.value.upper()}] {triage_result.error_event.bot_id}",
                body=(
                    "Blocked bug-fix proposal outside allowed_edit_paths: "
                    f"{', '.join(blocked_paths)}"
                )[:2000],
            )
            return True
        risk_tier = RepoRiskTier.REQUIRES_APPROVAL
        if planned_files:
            permission_result = guard.check_paths(profile, planned_files)
            risk_tier = self._permission_tier_to_risk(permission_result.tier)

        import hashlib

        request_id = hashlib.sha256(
            (
                f"{triage_result.error_event.bot_id}:{triage_result.error_event.error_type}:"
                f"{triage_result.error_event.message}:{'|'.join(planned_files)}"
            ).encode(),
        ).hexdigest()[:12]

        request = self._approval_tracker.get_by_id(request_id)
        if request is None:
            request = ApprovalRequest(
                request_id=request_id,
                suggestion_id=request_id,
                bot_id=triage_result.error_event.bot_id,
                change_kind=ChangeKind.BUG_FIX,
                title=repair_proposal.issue_title or f"Fix {triage_result.error_event.error_type}",
                summary=repair_proposal.fix_plan or triage_result.error_event.message,
                file_changes=file_changes,
                planned_files=planned_files,
                verification_commands=profile.verification_commands,
                risk_tier=risk_tier,
                draft_pr=risk_tier == RepoRiskTier.AUTO,
                implementation_notes=repair_proposal.risk_notes,
            )
            self._approval_tracker.create_request(request)
        elif request.pr_url:
            triage_result.pr_url = request.pr_url
            self._record_triage_followup(triage_result)
            await self._notify(
                notification_type="triage_result",
                priority=self._severity_to_priority(triage_result.severity),
                title=f"Triage [{triage_result.severity.value.upper()}] {request.bot_id}",
                body=(
                    f"Existing bug-fix PR: {request.pr_url}\n\n"
                    f"{(repair_proposal.fix_plan or agent_response or triage_result.error_event.message)[:1700]}"
                ),
            )
            return True

        self._event_stream.broadcast("triage_bug_fix_request_created", {
            "request_id": request.request_id,
            "bot_id": request.bot_id,
            "risk_tier": request.risk_tier.value,
            "planned_files": request.planned_files,
        })

        if request.risk_tier == RepoRiskTier.AUTO and self._approval_handler is not None:
            approval_message = await self._approval_handler.handle_approve(request.request_id)
            updated = self._approval_tracker.get_by_id(request.request_id)
            if updated and updated.pr_url:
                triage_result.pr_url = updated.pr_url
                self._record_triage_followup(triage_result)
            auto_body = approval_message
            if updated and updated.pr_url:
                auto_body = repair_proposal.fix_plan or agent_response or approval_message
            await self._notify(
                notification_type="triage_result",
                priority=self._severity_to_priority(triage_result.severity),
                title=f"Triage [{triage_result.severity.value.upper()}] {request.bot_id}",
                body=auto_body[:2000],
            )
            return True

        await self._notify(
            notification_type="triage_result",
            priority=self._severity_to_priority(triage_result.severity),
            title=f"Triage [{triage_result.severity.value.upper()}] {request.bot_id}",
            body=(
                f"Created {request.risk_tier.value} bug-fix request `{request.request_id}` "
                f"for {', '.join(planned_files)}.\n\n"
                f"{(repair_proposal.fix_plan or agent_response)[:1600]}"
            ),
        )
        return True

    async def _handle_investigation_triage(
        self,
        triage_result,
        repair_proposal,
        agent_response: str,
    ) -> bool:
        if repair_proposal is None or self._pr_builder is None or self._config_registry is None:
            return False

        from trading_assistant.schemas.repo_changes import GitHubIssueRequest

        profile = self._config_registry.get_profile(triage_result.error_event.bot_id)
        if profile is None:
            return False

        repo_task = None
        repo_dir = None
        try:
            if self._repo_workspace_manager is not None:
                repo_task = self._repo_workspace_manager.prepare_workspace(
                    profile,
                    f"triage-issue-{triage_result.error_event.bot_id}-{triage_result.timestamp.strftime('%Y%m%d%H%M%S')}",
                )
                repo_dir = Path(repo_task.worktree_dir)
            elif profile.repo_dir:
                repo_dir = Path(profile.repo_dir)
            if repo_dir is None:
                return False

            category = triage_result.error_event.category.value if triage_result.error_event.category else "unknown"
            dedupe_key = (
                f"{triage_result.error_event.bot_id}:{triage_result.error_event.error_type}:"
                f"{triage_result.error_event.message[:80]}"
            )
            issue_request = GitHubIssueRequest(
                bot_id=triage_result.error_event.bot_id,
                title=repair_proposal.issue_title or f"Investigate {triage_result.error_event.error_type}",
                body=repair_proposal.issue_body or agent_response or triage_result.error_event.message,
                repo_dir=str(repo_dir),
                labels=[
                    "trading-assistant",
                    f"severity/{triage_result.severity.value}",
                    f"category/{category}",
                ],
                dedupe_key=dedupe_key,
                repo_task=repo_task,
            )
            result = await self._pr_builder.create_issue(issue_request)
        finally:
            if repo_task and self._repo_workspace_manager is not None:
                try:
                    self._repo_workspace_manager.cleanup(repo_task)
                except Exception:
                    logger.warning("Failed to clean up triage issue workspace %s", repo_task.task_id)

        issue_url = result.issue_url or result.existing_issue_url
        if not result.success or not issue_url:
            return False

        triage_result.github_issue_url = issue_url
        self._record_triage_followup(triage_result)
        self._event_stream.broadcast("triage_issue_created", {
            "bot_id": triage_result.error_event.bot_id,
            "issue_url": issue_url,
            "existing_issue_url": result.existing_issue_url or "",
        })
        await self._notify(
            notification_type="triage_result",
            priority=self._severity_to_priority(triage_result.severity),
            title=f"Triage [{triage_result.severity.value.upper()}] {triage_result.error_event.bot_id}",
            body=(
                f"Investigation issue: {issue_url}\n\n"
                f"{(repair_proposal.fix_plan or agent_response or triage_result.error_event.message)[:1700]}"
            ),
        )
        return True

    def _record_triage_followup(self, triage_result) -> None:
        try:
            from trading_assistant.skills.failure_log import FailureLog

            FailureLog(self._failure_log_path).record_triage(triage_result)
        except Exception:
            logger.warning("Failed to append triage follow-up", exc_info=True)

    @staticmethod
    def _permission_tier_to_risk(permission_tier) -> str:
        from trading_assistant.schemas.approval import RepoRiskTier
        from trading_assistant.schemas.permissions import PermissionTier

        if permission_tier == PermissionTier.AUTO:
            return RepoRiskTier.AUTO
        if permission_tier == PermissionTier.REQUIRES_DOUBLE_APPROVAL:
            return RepoRiskTier.REQUIRES_DOUBLE_APPROVAL
        return RepoRiskTier.REQUIRES_APPROVAL
