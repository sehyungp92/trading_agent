"""GitHub PR and issue utilities for bot repositories."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from trading_assistant.schemas.approval import BacktestComparison
from trading_assistant.schemas.repo_changes import (
    GitHubIssueRequest,
    GitHubIssueResult,
    PRRequest,
    PRResult,
    PRReviewStatus,
    PreflightResult,
    ReviewState,
)

logger = logging.getLogger(__name__)


class PRBuilder:
    """Creates pull requests and issues against bot repositories using CLI tools."""

    def __init__(self, dry_run: bool = False, github_token: str = "") -> None:
        self._dry_run = dry_run
        self._github_token = github_token.strip()

    async def preflight_check(self, repo_dir: Path) -> PreflightResult:
        """Run pre-flight checks before PR creation."""
        checks: list[dict] = []
        reasons: list[str] = []
        passed = True

        code, stdout, _ = await self._run_git(["status", "--porcelain"], repo_dir)
        clean = code == 0 and not stdout.strip()
        checks.append({
            "name": "clean_tree",
            "passed": clean,
            "detail": "clean" if clean else "uncommitted changes present",
        })
        if not clean:
            logger.warning("Pre-flight: working tree is dirty in %s", repo_dir)

        code, _, stderr = await self._run_git(
            ["ls-remote", "--exit-code", "origin", "HEAD"],
            repo_dir,
        )
        remote_ok = code == 0
        checks.append({
            "name": "remote_reachable",
            "passed": remote_ok,
            "detail": "ok" if remote_ok else f"unreachable: {stderr.strip()}",
        })
        if not remote_ok:
            passed = False
            reasons.append(f"Remote unreachable: {stderr.strip()}")

        return PreflightResult(passed=passed, checks=checks, reasons=reasons)

    async def check_existing_pr(
        self,
        branch_name: str,
        repo_dir: Path,
    ) -> tuple[bool, str | None]:
        """Check whether a branch already has an open PR."""
        code, stdout, _ = await self._run_git(
            ["ls-remote", "--heads", "origin", branch_name],
            repo_dir,
        )
        if code != 0 or not stdout.strip():
            return False, None

        code, stdout, _ = await self._run_cmd(
            [
                "gh", "pr", "list",
                "--head", branch_name,
                "--state", "open",
                "--json", "number,url",
            ],
            repo_dir,
        )
        if code == 0 and stdout.strip():
            try:
                prs = json.loads(stdout)
                if prs:
                    return True, prs[0].get("url", "")
            except Exception:
                logger.warning("Failed to parse gh pr list output", exc_info=True)
        return True, None

    async def create_pr(self, request: PRRequest) -> PRResult:
        """Create a PR from a prepared worktree or regular repo checkout."""
        repo_dir = Path(request.repo_dir)
        material_changes = [
            file_change for file_change in request.file_changes
            if file_change.new_content != file_change.original_content
        ]
        diff_summary = [fc.file_path for fc in material_changes]

        if self._dry_run:
            return PRResult(
                success=True,
                pr_url=None,
                branch_name=request.branch_name,
                error="dry-run: skipped git commands",
                diff_summary=diff_summary,
            )

        if not material_changes:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error="No material file changes to commit",
                diff_summary=diff_summary,
            )

        preflight = await self.preflight_check(repo_dir)
        if not preflight.passed:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error=f"Pre-flight failed: {'; '.join(preflight.reasons)}",
                preflight=preflight,
                diff_summary=diff_summary,
            )

        exists, existing_url = await self.check_existing_pr(request.branch_name, repo_dir)
        if exists and existing_url:
            return PRResult(
                success=True,
                branch_name=request.branch_name,
                existing_pr_url=existing_url,
                diff_summary=diff_summary,
            )
        if exists:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error=(
                    f"Branch {request.branch_name} exists on remote but has no open PR "
                    "(stale state)"
                ),
                diff_summary=diff_summary,
            )

        if request.repo_task is None:
            code, _, stderr = await self._run_git(["pull", "--ff-only"], repo_dir)
            if code != 0:
                return PRResult(
                    success=False,
                    branch_name=request.branch_name,
                    error=f"git pull failed: {stderr}",
                    diff_summary=diff_summary,
                )

        code, _, stderr = await self._run_git(
            ["checkout", "-b", request.branch_name],
            repo_dir,
        )
        if code != 0:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error=f"Branch creation failed: {stderr}",
                diff_summary=diff_summary,
            )

        for file_change in material_changes:
            file_path = repo_dir / file_change.file_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file_change.new_content, encoding="utf-8")

        verification_error = await self._run_verifications(
            repo_dir,
            request.verification_commands,
        )
        if verification_error:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error=verification_error,
                diff_summary=diff_summary,
            )

        staged_paths = [fc.file_path for fc in material_changes]
        if not staged_paths:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error="No file changes to stage",
                diff_summary=diff_summary,
            )

        code, _, stderr = await self._run_git(["add", "--", *staged_paths], repo_dir)
        if code != 0:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error=f"git add failed: {stderr}",
                diff_summary=diff_summary,
            )

        commit_msg = f"trading-assistant: {request.title} (#{request.suggestion_id})"
        code, _, stderr = await self._run_git(["commit", "-m", commit_msg], repo_dir)
        if code != 0:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error=f"git commit failed: {stderr}",
                diff_summary=diff_summary,
            )

        code, _, stderr = await self._run_git(
            ["push", "-u", "origin", request.branch_name],
            repo_dir,
        )
        if code != 0:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error=f"git push failed: {stderr}",
                diff_summary=diff_summary,
            )

        gh_args = [
            "gh", "pr", "create",
            "--title", f"[trading-assistant] {request.title}",
            "--body", request.body,
        ]
        if request.draft:
            gh_args.append("--draft")
        code, stdout, stderr = await self._run_cmd(gh_args, repo_dir)
        if code != 0:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error=f"gh pr create failed: {stderr}",
                diff_summary=diff_summary,
            )

        return PRResult(
            success=True,
            pr_url=stdout.strip(),
            branch_name=request.branch_name,
            diff_summary=diff_summary,
        )

    async def create_issue(self, request: GitHubIssueRequest) -> GitHubIssueResult:
        """Create or deduplicate an issue for a repo task."""
        repo_dir = Path(request.repo_dir)
        if self._dry_run:
            return GitHubIssueResult(success=True, error="dry-run: skipped gh issue create")

        existing_url = await self.check_existing_issue(request, repo_dir)
        if existing_url:
            return GitHubIssueResult(
                success=True,
                existing_issue_url=existing_url,
                issue_url=existing_url,
            )

        args = [
            "gh", "issue", "create",
            "--title", request.title,
            "--body", self._render_issue_body(request),
        ]
        for label in request.labels:
            args.extend(["--label", label])
        code, stdout, stderr = await self._run_cmd(args, repo_dir)
        if code != 0:
            return GitHubIssueResult(success=False, error=f"gh issue create failed: {stderr}")

        issue_url = stdout.strip()
        return GitHubIssueResult(
            success=True,
            issue_url=issue_url,
            issue_number=self._extract_issue_number(issue_url),
        )

    async def check_existing_issue(
        self,
        request: GitHubIssueRequest,
        repo_dir: Path,
    ) -> str | None:
        """Check for an open issue matching the request title or dedupe key."""
        search_term = request.dedupe_key or request.title
        code, stdout, _ = await self._run_cmd(
            [
                "gh", "issue", "list",
                "--state", "open",
                "--limit", "20",
                "--json", "number,title,url",
                "--search", search_term,
            ],
            repo_dir,
        )
        if code != 0 or not stdout.strip():
            return None

        try:
            issues = json.loads(stdout)
        except Exception:
            return None

        for issue in issues:
            title = issue.get("title", "")
            if title == request.title:
                return issue.get("url")
            if request.dedupe_key:
                return issue.get("url")
        return None

    def _render_issue_body(self, request: GitHubIssueRequest) -> str:
        if not request.dedupe_key:
            return request.body
        marker = f"<!-- trading-assistant-dedupe: {request.dedupe_key} -->"
        if marker in request.body:
            return request.body
        body = request.body.rstrip()
        if not body:
            return marker
        return f"{body}\n\n{marker}"

    async def _run_verifications(
        self,
        repo_dir: Path,
        commands: list[str],
    ) -> str | None:
        for command in commands:
            code, _, stderr = await self._run_shell(command, repo_dir)
            if code != 0:
                return f"Verification failed for `{command}`: {stderr.strip()}"
        return None

    async def _run_git(self, args: list[str], cwd: Path) -> tuple[int, str, str]:
        return await self._run_cmd(["git", *args], cwd)

    async def run_gh_command(
        self,
        args: list[str],
        cwd: Path,
        timeout_seconds: int = 60,
    ) -> tuple[int, str, str]:
        return await self._run_cmd(args, cwd, timeout_seconds=timeout_seconds)

    async def _run_cmd(
        self,
        args: list[str],
        cwd: Path,
        timeout_seconds: int = 60,
    ) -> tuple[int, str, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._command_env(),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return 1, "", f"Command timed out after {timeout_seconds}s"
            return (
                proc.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except Exception as exc:
            return 1, "", str(exc)

    async def _run_shell(
        self,
        command: str,
        cwd: Path,
        timeout_seconds: int = 300,
    ) -> tuple[int, str, str]:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._command_env(),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return 1, "", f"Command timed out after {timeout_seconds}s"
            return (
                proc.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except Exception as exc:
            return 1, "", str(exc)

    async def check_pr_reviews(
        self,
        pr_url: str,
        repo_dir: Path,
    ) -> PRReviewStatus | None:
        """Check review status of a PR."""
        pr_number = self._extract_pr_number(pr_url)
        if pr_number is None:
            return None

        code, stdout, _ = await self._run_cmd(
            ["gh", "pr", "view", str(pr_number), "--json", "reviews,comments,state"],
            repo_dir,
        )
        if code != 0:
            return None

        try:
            data = json.loads(stdout)
        except Exception:
            return None

        reviews = data.get("reviews", []) or []
        comments = data.get("comments", []) or []
        states_seen: set[str] = set()
        reviewers: list[str] = []
        for review in reviews:
            state = review.get("state", "")
            states_seen.add(state)
            author = review.get("author", {}).get("login", "")
            if author and author not in reviewers:
                reviewers.append(author)

        if "CHANGES_REQUESTED" in states_seen:
            review_state = ReviewState.CHANGES_REQUESTED
        elif "APPROVED" in states_seen:
            review_state = ReviewState.APPROVED
        elif "COMMENTED" in states_seen:
            review_state = ReviewState.COMMENTED
        else:
            review_state = ReviewState.PENDING

        actionable = []
        for comment in comments[:5]:
            body = comment.get("body", "").strip()
            if body:
                actionable.append(body[:200])

        return PRReviewStatus(
            pr_number=pr_number,
            pr_url=pr_url,
            review_state=review_state,
            reviewers=reviewers,
            actionable_comments=actionable,
            needs_attention=review_state == ReviewState.CHANGES_REQUESTED or bool(actionable),
        )

    @staticmethod
    def _extract_pr_number(pr_url: str) -> int | None:
        import re

        match = re.search(r"/pull/(\d+)", pr_url)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _extract_issue_number(issue_url: str) -> int | None:
        import re

        match = re.search(r"/issues/(\d+)", issue_url)
        if match:
            return int(match.group(1))
        return None

    def format_pr_body(
        self,
        request: PRRequest,
        comparison: BacktestComparison | None = None,
    ) -> str:
        """Format a PR body with context, rollback notes, and diff previews."""
        lines: list[str] = []
        lines.append("## Change Type")
        lines.append("")
        lines.append(f"- `{request.change_kind.value}`")

        if comparison:
            ctx = comparison.context
            lines.extend([
                "",
                "## Parameter Changes",
                "",
                "| Parameter | Current | Proposed |",
                "|-----------|---------|----------|",
                f"| {ctx.param_name} | {ctx.current_value} | {ctx.proposed_value} |",
                "",
                "## Backtest Results",
                "",
                "| Metric | Baseline | Proposed | Change |",
                "|--------|----------|----------|--------|",
                (
                    f"| Sharpe | {comparison.baseline.sharpe_ratio:.2f} "
                    f"| {comparison.proposed.sharpe_ratio:.2f} "
                    f"| {comparison.sharpe_change_pct:+.1f}% |"
                ),
                (
                    f"| MaxDD | {comparison.baseline.max_drawdown_pct:.1f}% "
                    f"| {comparison.proposed.max_drawdown_pct:.1f}% "
                    f"| {comparison.max_dd_change_pct:+.1f}% |"
                ),
                (
                    f"| ProfitFactor | {comparison.baseline.profit_factor:.2f} "
                    f"| {comparison.proposed.profit_factor:.2f} "
                    f"| {comparison.profit_factor_change_pct:+.1f}% |"
                ),
                (
                    f"| WinRate | {comparison.baseline.win_rate:.1%} "
                    f"| {comparison.proposed.win_rate:.1%} "
                    f"| {comparison.win_rate_change_pct:+.1f}% |"
                ),
                "",
                f"**Safety Check:** {'PASS' if comparison.passes_safety else 'FAIL'}",
            ])
            for note in comparison.safety_notes:
                lines.append(f"- {note}")
        elif request.file_changes:
            lines.extend([
                "",
                "## Files",
                "",
                *[f"- `{file_change.file_path}`" for file_change in request.file_changes],
            ])

        if request.verification_commands:
            lines.extend([
                "",
                "## Verification",
                "",
                *[f"- `{command}`" for command in request.verification_commands],
            ])

        lines.extend([
            "",
            "## Rollback",
            "",
            "```bash",
            "git revert <commit-sha>",
            "```",
        ])

        if request.file_changes:
            lines.extend(["", "## File Changes", ""])
            for file_change in request.file_changes:
                lines.append(f"### `{file_change.file_path}`")
                if file_change.diff_preview:
                    lines.extend(["", "```diff", file_change.diff_preview, "```", ""])

        lines.extend(["", f"Suggestion ID: `{request.suggestion_id}`"])
        if request.repo_task:
            lines.append(f"Repo task ID: `{request.repo_task.task_id}`")

        return "\n".join(lines)

    def _command_env(self) -> dict[str, str] | None:
        if not self._github_token:
            return None
        import os

        env = os.environ.copy()
        env["GH_TOKEN"] = self._github_token
        env["GITHUB_TOKEN"] = self._github_token
        return env
