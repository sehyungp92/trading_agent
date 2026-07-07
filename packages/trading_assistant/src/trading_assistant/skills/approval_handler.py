"""Approval handling for parameter, bug-fix, and structural repo changes."""
from __future__ import annotations

import difflib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_assistant.schemas.approval import ApprovalRequest, RepoRiskTier
from trading_assistant.schemas.repo_changes import ChangeKind, FileChange, FileChangeMode, PRRequest
from trading_assistant.schemas.permissions import PermissionTier
from trading_assistant.schemas.prompt_package import PromptPackage
from trading_assistant.skills.approval_tracker import ApprovalTracker
from trading_assistant.skills.config_registry import ConfigRegistry
from trading_assistant.skills.file_change_generator import FileChangeGenerator
from trading_assistant.skills.github_pr import PRBuilder
from trading_assistant.skills.repo_change_guard import RepoChangeGuard
from trading_assistant.skills.repo_workspace import RepoWorkspaceManager

logger = logging.getLogger(__name__)
_REPO_TASK_ALLOWED_TOOLS = ["Read", "Edit", "Bash", "Grep", "Glob"]


class ApprovalHandler:
    """Handles approval lifecycle and executes approved repo mutations."""

    def __init__(
        self,
        approval_tracker: ApprovalTracker,
        suggestion_tracker: Any,
        file_change_generator: FileChangeGenerator,
        pr_builder: PRBuilder,
        config_registry: ConfigRegistry,
        event_stream: Any | None = None,
        telegram_bot: Any | None = None,
        repo_workspace_manager: RepoWorkspaceManager | None = None,
        repo_change_guard: RepoChangeGuard | None = None,
        repo_task_runner: Any | None = None,
        structural_experiment_tracker: Any | None = None,
        hypothesis_library: Any | None = None,
    ) -> None:
        self._approval_tracker = approval_tracker
        self._suggestion_tracker = suggestion_tracker
        self._file_change_gen = file_change_generator
        self._pr_builder = pr_builder
        self._config_registry = config_registry
        self._event_stream = event_stream
        self._telegram_bot = telegram_bot
        self._repo_workspace_manager = repo_workspace_manager
        self._repo_change_guard = repo_change_guard or RepoChangeGuard()
        self._repo_task_runner = repo_task_runner
        self._structural_experiment_tracker = structural_experiment_tracker
        self._hypothesis_library = hypothesis_library

    async def handle_approve(self, request_id: str) -> str:
        """Approve a pending request, generate a PR, and record the outcome."""
        request = self._approval_tracker.get_by_id(request_id)
        if request is None:
            return f"Request {request_id} not found"

        try:
            approved = self._approval_tracker.approve(request_id)
        except ValueError as exc:
            return str(exc)

        if (
            approved.risk_tier == RepoRiskTier.REQUIRES_DOUBLE_APPROVAL
            and approved.status == request.status
            and approved.approval_count == 1
        ):
            await self._edit_approval_card(
                approved,
                "[APPROVAL 1/2] Awaiting second approval",
            )
            return (
                f"First approval recorded for request {request_id}. "
                "A second approval is required before repo changes are made."
            )

        repo_task = None
        try:
            profile = self._config_registry.get_profile(approved.bot_id)
            if profile is None:
                self._approval_tracker.revert_to_pending(request_id)
                return f"No config profile for bot {approved.bot_id}. Request reverted to PENDING."

            repo_task = self._prepare_repo_task(profile, approved)
            repo_dir = Path(repo_task.worktree_dir if repo_task else profile.repo_dir)

            file_changes = await self._resolve_repo_changes(
                request=approved,
                profile=profile,
                repo_dir=repo_dir,
                repo_task=repo_task,
            )
            if not file_changes:
                self._approval_tracker.revert_to_pending(request_id)
                return (
                    f"No applicable file changes for request {request_id}. "
                    "Request reverted to PENDING."
                )

            guard_result = self._repo_change_guard.check_paths(
                profile,
                [file_change.file_path for file_change in file_changes],
            )
            expected_tier = self._tier_for_request(approved.risk_tier)
            if guard_result.tier > expected_tier:
                self._approval_tracker.revert_to_pending(request_id)
                return (
                    f"Repo guard blocked request {request_id}: {guard_result.reason}. "
                    "Request reverted to PENDING."
                )

            branch_name = self._build_branch_name(approved)
            title = approved.title or self._default_title(approved)
            verification_commands = approved.verification_commands or profile.verification_commands
            pr_request = PRRequest(
                approval_request_id=request_id,
                suggestion_id=approved.suggestion_id,
                bot_id=approved.bot_id,
                repo_dir=str(repo_dir),
                branch_name=branch_name,
                title=title,
                change_kind=approved.change_kind,
                draft=approved.draft_pr,
                verification_commands=verification_commands,
                repo_task=repo_task,
                file_changes=file_changes,
            )
            pr_request.body = self._pr_builder.format_pr_body(
                pr_request,
                approved.backtest_summary,
            )
            result = await self._pr_builder.create_pr(pr_request)
        except Exception as exc:
            self._approval_tracker.revert_to_pending(request_id)
            logger.exception("Approval workflow failed for %s", request_id)
            return f"PR creation failed: {exc}. Request reverted to PENDING."
        finally:
            if repo_task and self._repo_workspace_manager:
                try:
                    self._repo_workspace_manager.cleanup(repo_task)
                except Exception:
                    logger.warning("Failed to clean up repo task %s", repo_task.task_id)

        if result.existing_pr_url:
            self._approval_tracker.set_pr_result(
                request_id,
                pr_url=result.existing_pr_url,
                diff_summary=result.diff_summary,
            )
            self._mark_suggestion_accepted(approved)
            await self._edit_approval_card(
                approved,
                f"[APPROVED] Existing PR: {result.existing_pr_url}",
            )
            return f"Existing PR: {result.existing_pr_url}"

        if not result.success:
            self._approval_tracker.revert_to_pending(request_id)
            preflight_detail = ""
            if result.preflight and not result.preflight.passed:
                preflight_detail = f" (preflight: {'; '.join(result.preflight.reasons)})"
            logger.error(
                "PR creation failed for %s: %s%s - reverting to PENDING",
                request_id,
                result.error,
                preflight_detail,
            )
            return (
                f"PR creation failed: {result.error}{preflight_detail}. "
                "Request reverted to PENDING."
            )

        self._approval_tracker.set_pr_result(
            request_id,
            pr_url=result.pr_url or "",
            diff_summary=result.diff_summary,
        )
        self._mark_suggestion_accepted(approved)

        if self._event_stream:
            self._event_stream.broadcast("suggestion_pr_created", {
                "request_id": request_id,
                "suggestion_id": approved.suggestion_id,
                "change_kind": approved.change_kind.value,
                "pr_url": result.pr_url,
                "task_id": repo_task.task_id if repo_task else "",
                "diff_summary": result.diff_summary,
            })

        await self._edit_approval_card(
            approved,
            f"[APPROVED] PR: {result.pr_url or 'created'}",
        )
        return f"PR created: {result.pr_url}"

    async def handle_reject(self, request_id: str, reason: str = "") -> str:
        """Reject a pending request."""
        try:
            rejected = self._approval_tracker.reject(request_id, reason)
        except ValueError as exc:
            return str(exc)

        if self._suggestion_tracker:
            try:
                self._suggestion_tracker.reject(
                    rejected.suggestion_id,
                    reason=reason or "rejected via Telegram",
                )
            except Exception:
                logger.warning("Failed to reject suggestion %s", rejected.suggestion_id)

        # Record hypothesis rejection for structural changes
        if (
            rejected.change_kind == ChangeKind.STRUCTURAL_CHANGE
            and self._hypothesis_library
            and rejected.hypothesis_id
        ):
            try:
                self._hypothesis_library.record_rejection(rejected.hypothesis_id)
            except Exception:
                logger.warning("Failed to record hypothesis rejection for %s", rejected.hypothesis_id)

        rejected_req = self._approval_tracker.get_by_id(request_id)
        if rejected_req:
            await self._edit_approval_card(
                rejected_req,
                f"[REJECTED] {reason or 'rejected via Telegram'}",
            )

        return f"Rejected request {request_id}"

    async def handle_detail(self, request_id: str) -> str:
        """Return the stored details for a request."""
        request = self._approval_tracker.get_by_id(request_id)
        if request is None:
            return f"Request {request_id} not found"

        lines = [f"Details for request {request_id}:"]
        lines.append(f"Bot: {request.bot_id}")
        lines.append(f"Change kind: {request.change_kind.value}")
        lines.append(f"Risk tier: {request.risk_tier.value}")
        lines.append(f"Status: {request.status.value}")
        if request.risk_tier == RepoRiskTier.REQUIRES_DOUBLE_APPROVAL:
            lines.append(f"Approvals: {request.approval_count}/2")

        if request.param_changes:
            lines.append("\nParameter Changes:")
            for change in request.param_changes:
                lines.append(
                    f"  {change.get('param_name', '?')}: "
                    f"{change.get('current', '?')} -> {change.get('proposed', '?')}"
                )
        elif request.file_changes:
            lines.append("\nPlanned Files:")
            for file_change in request.file_changes:
                lines.append(f"  - {file_change.file_path}")

        if request.verification_commands:
            lines.append("\nVerification:")
            for command in request.verification_commands:
                lines.append(f"  - {command}")

        backtest = request.backtest_summary
        if backtest:
            lines.append(
                f"\nBacktest ({backtest.context.trade_count} trades, {backtest.context.data_days} days):"
            )
            lines.append(
                f"  Sharpe: {backtest.baseline.sharpe_ratio:.2f} -> "
                f"{backtest.proposed.sharpe_ratio:.2f} ({backtest.sharpe_change_pct:+.1f}%)"
            )
            lines.append(
                f"  MaxDD: {backtest.baseline.max_drawdown_pct:.1f}% -> "
                f"{backtest.proposed.max_drawdown_pct:.1f}% ({backtest.max_dd_change_pct:+.1f}%)"
            )
            lines.append(
                f"  PF: {backtest.baseline.profit_factor:.2f} -> "
                f"{backtest.proposed.profit_factor:.2f} ({backtest.profit_factor_change_pct:+.1f}%)"
            )
            lines.append(
                f"  WR: {backtest.baseline.win_rate:.1%} -> "
                f"{backtest.proposed.win_rate:.1%} ({backtest.win_rate_change_pct:+.1f}%)"
            )
            lines.append(f"  Safety: {'PASS' if backtest.passes_safety else 'FAIL'}")

        if request.issue_url:
            lines.append(f"\nIssue: {request.issue_url}")
        if request.pr_url:
            lines.append(f"PR: {request.pr_url}")

        return "\n".join(lines)

    async def _resolve_repo_changes(
        self,
        request: ApprovalRequest,
        profile,
        repo_dir: Path,
        repo_task,
    ) -> list[FileChange]:
        if self._should_run_repo_task(request, repo_task):
            try:
                repo_changes = await self._run_repo_task(
                    request=request,
                    profile=profile,
                    repo_dir=repo_dir,
                    repo_task=repo_task,
                )
                repo_changes = self._filter_material_file_changes(repo_changes)
                if repo_changes:
                    return repo_changes
            except Exception:
                if not request.file_changes:
                    raise
                logger.warning(
                    "Repo task runner failed for %s; falling back to structured file changes",
                    request.request_id,
                    exc_info=True,
                )

        return self._filter_material_file_changes(
            self._resolve_file_changes(request, repo_dir),
        )

    def _should_run_repo_task(self, request: ApprovalRequest, repo_task) -> bool:
        if self._repo_task_runner is None or repo_task is None:
            return False
        return request.change_kind in {
            ChangeKind.BUG_FIX,
            ChangeKind.STRUCTURAL_CHANGE,
        }

    async def _run_repo_task(
        self,
        request: ApprovalRequest,
        profile,
        repo_dir: Path,
        repo_task,
    ) -> list[FileChange]:
        context_files = []
        if self._repo_workspace_manager is not None:
            context_files = self._repo_workspace_manager.collect_context_files(
                profile,
                repo_dir,
                preferred_paths=request.planned_files,
            )

        prompt = self._build_repo_task_prompt(
            request=request,
            profile=profile,
            repo_task=repo_task,
            context_files=context_files,
        )
        response_path = Path(repo_task.artifact_dir) / "repo_task_response.md"
        context_path = Path(repo_task.artifact_dir) / "repo_context.json"
        context_path.write_text(
            json.dumps({
                "context_files": context_files,
                "planned_files": request.planned_files,
                "allowed_edit_paths": profile.allowed_edit_paths,
                "structural_context_paths": profile.structural_context_paths,
            }, indent=2),
            encoding="utf-8",
        )

        agent_type = (
            "triage"
            if request.change_kind == ChangeKind.BUG_FIX
            else "weekly_analysis"
        )
        result = await self._repo_task_runner.invoke(
            agent_type=agent_type,
            prompt_package=prompt,
            run_id=f"repo-task-{request.request_id}",
            allowed_tools=_REPO_TASK_ALLOWED_TOOLS,
        )
        if not result.success and self._should_retry_repo_task_without_tool_allowlist(result.error):
            logger.warning(
                "Retrying repo task %s without tool allowlist after CLI rejection: %s",
                request.request_id,
                result.error,
            )
            result = await self._repo_task_runner.invoke(
                agent_type=agent_type,
                prompt_package=prompt,
                run_id=f"repo-task-{request.request_id}",
                allowed_tools=None,
            )
        response_path.write_text(result.response or "", encoding="utf-8")
        if not result.success:
            raise RuntimeError(result.error or "Repo task runner failed")

        file_changes = self._collect_worktree_file_changes(repo_dir)
        (Path(repo_task.artifact_dir) / "repo_task_changes.json").write_text(
            json.dumps([change.model_dump(mode="json") for change in file_changes], indent=2),
            encoding="utf-8",
        )
        return file_changes

    def _build_repo_task_prompt(
        self,
        request: ApprovalRequest,
        profile,
        repo_task,
        context_files: list[str],
    ) -> PromptPackage:
        task_prompt = (
            f"Implement the already-approved {request.change_kind.value} in the bot repo.\n"
            f"Worktree: {repo_task.worktree_dir}\n"
            "Before editing, read `request.json` and `repo_context.json` from your current run directory.\n"
            "Make the smallest safe change that satisfies the approved request.\n"
            "Only modify files inside the worktree, stay within allowed_edit_paths, and do not create commits or branches.\n"
            "Do not delete files. Leave the repository edited but uncommitted for the orchestrator to verify and open a PR.\n"
            "After editing, briefly summarize what you changed."
        )
        instructions = (
            "Use the repo context manifest to inspect the most relevant files first. "
            "If structured file changes are provided, treat them as approved intent and refine them against the real codebase. "
            "Prefer minimal diffs, preserve existing style, and avoid touching unrelated files."
        )
        return PromptPackage(
            task_prompt=task_prompt,
            instructions=instructions,
            data={
                "request": {
                    "request_id": request.request_id,
                    "bot_id": request.bot_id,
                    "change_kind": request.change_kind.value,
                    "title": request.title,
                    "summary": request.summary,
                    "implementation_notes": request.implementation_notes,
                    "planned_files": request.planned_files,
                    "verification_commands": request.verification_commands or profile.verification_commands,
                    "allowed_edit_paths": profile.allowed_edit_paths,
                    "structural_context_paths": profile.structural_context_paths,
                    "proposed_file_changes": [
                        change.model_dump(mode="json")
                        for change in request.file_changes
                    ],
                },
                "repo_context": {
                    "worktree_dir": repo_task.worktree_dir,
                    "context_files": context_files,
                },
            },
        )

    def _collect_worktree_file_changes(self, repo_dir: Path) -> list[FileChange]:
        changed_paths = self._git_lines(repo_dir, ["diff", "--name-only", "--diff-filter=AM", "--relative"])
        untracked_paths = self._git_lines(
            repo_dir,
            ["ls-files", "--others", "--exclude-standard"],
        )
        ordered_paths = []
        seen: set[str] = set()
        for path in [*changed_paths, *untracked_paths]:
            if path and path not in seen:
                seen.add(path)
                ordered_paths.append(path)

        file_changes: list[FileChange] = []
        for relative_path in ordered_paths:
            file_path = repo_dir / relative_path
            if not file_path.exists() or not file_path.is_file():
                continue
            original = self._git_show_file(repo_dir, relative_path)
            new_content = file_path.read_text(encoding="utf-8")
            diff_preview = "\n".join(difflib.unified_diff(
                original.splitlines(),
                new_content.splitlines(),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
                lineterm="",
            ))
            file_changes.append(FileChange(
                file_path=relative_path,
                original_content=original,
                new_content=new_content,
                diff_preview=diff_preview,
            ))
        return file_changes

    @staticmethod
    def _git_lines(repo_dir: Path, args: list[str]) -> list[str]:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            return []
        return [
            line.strip().replace("\\", "/")
            for line in result.stdout.splitlines()
            if line.strip()
        ]

    @staticmethod
    def _git_show_file(repo_dir: Path, relative_path: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "show", f"HEAD:{relative_path}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout

    @staticmethod
    def _should_retry_repo_task_without_tool_allowlist(error: str) -> bool:
        lowered = (error or "").lower()
        if not lowered:
            return False
        return any(
            phrase in lowered
            for phrase in (
                "allowed-tools",
                "allowed tools",
                "unknown tool",
                "invalid tool",
                "unsupported tool",
            )
        )

    def _resolve_file_changes(
        self,
        request: ApprovalRequest,
        repo_dir: Path,
    ) -> list[FileChange]:
        if request.file_changes:
            materialized: list[FileChange] = []
            for file_change in request.file_changes:
                materialized.append(self._materialize_file_change(file_change, repo_dir))
            return materialized
        return self._build_param_file_changes(
            request.bot_id,
            request.param_changes,
            repo_dir,
        )

    def _materialize_file_change(
        self,
        file_change: FileChange,
        repo_dir: Path,
    ) -> FileChange:
        if file_change.change_mode == FileChangeMode.UNIFIED_DIFF:
            return self._file_change_gen.generate_patch_change(
                file_change.file_path,
                file_change.patch,
                repo_dir,
            )

        if file_change.new_content:
            full_path = repo_dir / file_change.file_path
            original = file_change.original_content
            if not original and full_path.exists():
                original = full_path.read_text(encoding="utf-8")
            diff_preview = file_change.diff_preview or "\n".join(difflib.unified_diff(
                original.splitlines(),
                file_change.new_content.splitlines(),
                fromfile=f"a/{file_change.file_path}",
                tofile=f"b/{file_change.file_path}",
                lineterm="",
            ))
            return FileChange(
                file_path=file_change.file_path,
                original_content=original,
                new_content=file_change.new_content,
                change_mode=file_change.change_mode,
                metadata=file_change.metadata,
                patch=file_change.patch,
                diff_preview=diff_preview,
            )

        raise ValueError(f"File change for {file_change.file_path} has no applicable content")

    def _build_param_file_changes(
        self,
        bot_id: str,
        param_changes: list[dict],
        repo_dir: Path,
    ) -> list[FileChange]:
        file_changes: list[FileChange] = []
        for param_change in param_changes:
            param = self._config_registry.get_parameter(
                bot_id,
                param_change.get("param_name", ""),
            )
            if param is None:
                continue
            try:
                change = self._file_change_gen.generate_change(
                    param,
                    param_change.get("proposed"),
                    repo_dir,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to generate change for %s: %s",
                    param_change.get("param_name"),
                    exc,
                )
                continue
            file_changes.append(change)
        return file_changes

    @staticmethod
    def _filter_material_file_changes(file_changes: list[FileChange]) -> list[FileChange]:
        return [
            file_change for file_change in file_changes
            if file_change.new_content != file_change.original_content
        ]

    def _prepare_repo_task(self, profile, request: ApprovalRequest):
        if request.repo_task:
            return request.repo_task
        if self._repo_workspace_manager is None:
            return None
        return self._repo_workspace_manager.prepare_workspace(profile, request.request_id)

    def _mark_suggestion_accepted(self, request: ApprovalRequest) -> None:
        if not self._suggestion_tracker:
            return
        try:
            self._suggestion_tracker.accept(
                request.suggestion_id,
                approval_request_id=request.request_id,
            )
        except Exception:
            logger.warning("Failed to mark suggestion %s as ACCEPTED", request.suggestion_id)

        # Record hypothesis acceptance for structural changes
        if (
            request.change_kind == ChangeKind.STRUCTURAL_CHANGE
            and self._hypothesis_library
            and request.hypothesis_id
        ):
            try:
                self._hypothesis_library.record_acceptance(request.hypothesis_id)
            except Exception:
                logger.warning("Failed to record hypothesis acceptance for %s", request.hypothesis_id)

        # Activate structural experiment if this is a structural change
        if (
            request.change_kind == ChangeKind.STRUCTURAL_CHANGE
            and self._structural_experiment_tracker
        ):
            try:
                exp = self._structural_experiment_tracker.find_by_suggestion_id(
                    request.suggestion_id,
                )
                if exp is not None:
                    self._structural_experiment_tracker.activate(exp.experiment_id)
                    logger.info(
                        "Activated structural experiment %s for suggestion %s",
                        exp.experiment_id, request.suggestion_id,
                    )
            except Exception:
                logger.warning(
                    "Failed to activate structural experiment for suggestion %s",
                    request.suggestion_id,
                )

    async def _edit_approval_card(self, request: ApprovalRequest, status_line: str) -> None:
        if self._telegram_bot is None or not request.message_id:
            return
        try:
            text = (
                f"Suggestion {request.request_id}\n"
                f"Bot: {request.bot_id}\n"
                f"Kind: {request.change_kind.value}\n"
                f"{status_line}"
            )
            await self._telegram_bot.edit_message(request.message_id, text)
        except Exception:
            logger.warning("Failed to edit approval card for %s", request.request_id)

    @staticmethod
    def _tier_for_request(risk_tier: RepoRiskTier) -> PermissionTier:
        if risk_tier == RepoRiskTier.AUTO:
            return PermissionTier.AUTO
        if risk_tier == RepoRiskTier.REQUIRES_DOUBLE_APPROVAL:
            return PermissionTier.REQUIRES_DOUBLE_APPROVAL
        return PermissionTier.REQUIRES_APPROVAL

    @staticmethod
    def _build_branch_name(request: ApprovalRequest) -> str:
        prefix = {
            ChangeKind.PARAMETER_CHANGE: "suggestion",
            ChangeKind.BUG_FIX: "bugfix",
            ChangeKind.STRUCTURAL_CHANGE: "structural",
            ChangeKind.ROLLBACK: "rollback",
        }[request.change_kind]
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"codex/{prefix}-{request.suggestion_id[:8]}-{date_str}"

    @staticmethod
    def _default_title(request: ApprovalRequest) -> str:
        if request.change_kind == ChangeKind.PARAMETER_CHANGE and request.param_changes:
            changed = ", ".join(change.get("param_name", "?") for change in request.param_changes)
            return f"Update {changed}"
        return request.summary or request.change_kind.value.replace("_", " ").title()
