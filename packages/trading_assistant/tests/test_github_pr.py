# tests/test_github_pr.py
"""Tests for PRBuilder."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from trading_assistant.schemas.approval import (
    BacktestComparison,
    BacktestContext,
)
from trading_assistant.schemas.repo_changes import (
    FileChange,
    GitHubIssueRequest,
    PRRequest,
    ReviewState,
)
from trading_assistant.schemas.simulation_metrics import SimulationMetrics
from trading_assistant.skills.github_pr import PRBuilder


def _make_request(**kwargs) -> PRRequest:
    return PRRequest(
        approval_request_id=kwargs.get("approval_request_id", "ar1"),
        suggestion_id=kwargs.get("suggestion_id", "s1"),
        bot_id=kwargs.get("bot_id", "bot1"),
        repo_dir=kwargs.get("repo_dir", "/repos/bot1"),
        branch_name=kwargs.get("branch_name", "codex/suggestion-s1-2026-03-07"),
        title=kwargs.get("title", "Increase quality threshold"),
        body=kwargs.get("body", "Backtest results look good"),
        file_changes=kwargs.get("file_changes", [
            FileChange(
                file_path="config.yaml",
                original_content="quality: 0.6",
                new_content="quality: 0.7",
                diff_preview="-quality: 0.6\n+quality: 0.7",
            )
        ]),
    )


def _temp_repo_dir(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(exist_ok=True)
    return repo_dir


class TestPRBuilder:
    @pytest.mark.asyncio
    async def test_dry_run_skips_git(self):
        builder = PRBuilder(dry_run=True)
        result = await builder.create_pr(_make_request())
        assert result.success is True
        assert "dry-run" in result.error
        assert result.branch_name == "codex/suggestion-s1-2026-03-07"

    @pytest.mark.asyncio
    async def test_successful_pr_creation(self, tmp_path: Path):
        builder = PRBuilder()
        with patch.object(builder, "_run_git", new_callable=AsyncMock) as mock_git, \
             patch.object(builder, "_run_cmd", new_callable=AsyncMock) as mock_cmd:
            mock_git.return_value = (0, "", "")
            mock_cmd.return_value = (0, "https://github.com/user/repo/pull/42\n", "")
            result = await builder.create_pr(_make_request(repo_dir=str(_temp_repo_dir(tmp_path))))
            assert result.success is True
            assert result.pr_url == "https://github.com/user/repo/pull/42"

    @pytest.mark.asyncio
    async def test_branch_name_format(self):
        req = _make_request(branch_name="codex/suggestion-abc12345-2026-03-07")
        assert req.branch_name.startswith("codex/suggestion-")

    @pytest.mark.asyncio
    async def test_commit_message_format(self, tmp_path: Path):
        builder = PRBuilder()
        calls = []

        async def mock_git(args, cwd):
            calls.append(args)
            return (0, "", "")

        with patch.object(builder, "_run_git", side_effect=mock_git), \
             patch.object(builder, "_run_cmd", new_callable=AsyncMock, return_value=(0, "url", "")):
            await builder.create_pr(_make_request(repo_dir=str(_temp_repo_dir(tmp_path))))

        commit_calls = [c for c in calls if c[0] == "commit"]
        assert len(commit_calls) == 1
        assert "trading-assistant:" in commit_calls[0][2]

    @pytest.mark.asyncio
    async def test_git_pull_failure(self, tmp_path: Path):
        builder = PRBuilder()
        call_count = [0]

        async def mock_git(args, cwd):
            call_count[0] += 1
            if args[0] == "status":
                return (0, "", "")        # clean tree
            if args[0] == "ls-remote" and "--exit-code" in args:
                return (0, "abc HEAD", "")  # remote ok
            if args[0] == "ls-remote" and "--heads" in args:
                return (0, "", "")          # no existing branch
            if args[0] == "pull":
                return (1, "", "conflict")  # pull fails
            return (0, "", "")

        with patch.object(builder, "_run_git", side_effect=mock_git):
            result = await builder.create_pr(_make_request(repo_dir=str(_temp_repo_dir(tmp_path))))
            assert result.success is False
            assert "git pull failed" in result.error

    @pytest.mark.asyncio
    async def test_gh_pr_create_failure(self, tmp_path: Path):
        builder = PRBuilder()
        call_count = [0]

        async def mock_git(args, cwd):
            return (0, "", "")

        async def mock_cmd(args, cwd):
            call_count[0] += 1
            return (1, "", "auth required")

        with patch.object(builder, "_run_git", side_effect=mock_git), \
             patch.object(builder, "_run_cmd", side_effect=mock_cmd):
            result = await builder.create_pr(_make_request(repo_dir=str(_temp_repo_dir(tmp_path))))
            assert result.success is False
            assert "gh pr create failed" in result.error

    def test_pr_body_contains_backtest(self):
        builder = PRBuilder()
        comparison = BacktestComparison(
            context=BacktestContext(
                suggestion_id="s1", bot_id="bot1",
                param_name="quality_min", current_value=0.6, proposed_value=0.7,
            ),
            baseline=SimulationMetrics(
                sharpe_ratio=1.0, max_drawdown_pct=-10.0,
                profit_factor=1.5, total_trades=20, win_count=12,
            ),
            proposed=SimulationMetrics(
                sharpe_ratio=1.2, max_drawdown_pct=-8.0,
                profit_factor=1.8, total_trades=20, win_count=14,
            ),
            passes_safety=True,
        )
        body = builder.format_pr_body(_make_request(), comparison)
        assert "Sharpe" in body
        assert "MaxDD" in body
        assert "PASS" in body
        assert "Rollback" in body

    def test_pr_body_contains_rollback(self):
        builder = PRBuilder()
        body = builder.format_pr_body(_make_request())
        assert "git revert" in body
        assert "Suggestion ID" in body

    @pytest.mark.asyncio
    async def test_multiple_file_changes(self):
        builder = PRBuilder(dry_run=True)
        changes = [
            FileChange(file_path="a.yaml", original_content="x: 1", new_content="x: 2"),
            FileChange(file_path="b.py", original_content="Y = 1", new_content="Y = 2"),
        ]
        result = await builder.create_pr(_make_request(file_changes=changes))
        assert result.success is True

    @pytest.mark.asyncio
    async def test_create_pr_stages_only_changed_files(self, tmp_path: Path):
        builder = PRBuilder()
        calls = []

        async def mock_git(args, cwd):
            calls.append(args)
            if args[0] == "ls-remote" and "--heads" in args:
                return (0, "", "")
            return (0, "", "")

        with patch.object(builder, "_run_git", side_effect=mock_git), \
             patch.object(builder, "_run_cmd", new_callable=AsyncMock, return_value=(0, "https://github.com/x/y/pull/1\n", "")):
            await builder.create_pr(_make_request(
                repo_dir=str(_temp_repo_dir(tmp_path)),
                file_changes=[
                    FileChange(file_path="a.yaml", original_content="x: 1", new_content="x: 2"),
                    FileChange(file_path="b.py", original_content="Y = 1", new_content="Y = 2"),
                ],
            ))

        add_calls = [call for call in calls if call[0] == "add"]
        assert add_calls == [["add", "--", "a.yaml", "b.py"]]

    @pytest.mark.asyncio
    async def test_create_pr_rejects_no_material_changes(self, tmp_path: Path):
        builder = PRBuilder()
        result = await builder.create_pr(_make_request(
            repo_dir=str(_temp_repo_dir(tmp_path)),
            file_changes=[
                FileChange(file_path="config.yaml", original_content="x: 1", new_content="x: 1"),
            ],
        ))
        assert result.success is False
        assert "No material file changes" in result.error

    def test_pr_title_format(self):
        req = _make_request(title="Increase quality threshold")
        expected_title = f"[trading-assistant] {req.title}"
        assert expected_title.startswith("[trading-assistant]")


class TestPreflightCheck:
    @pytest.mark.asyncio
    async def test_clean_tree_passes(self):
        builder = PRBuilder()
        with patch.object(builder, "_run_git", new_callable=AsyncMock) as mock_git:
            mock_git.side_effect = [
                (0, "", ""),          # status --porcelain (clean)
                (0, "abc HEAD", ""),  # ls-remote (reachable)
            ]
            result = await builder.preflight_check(Path("/tmp/repo"))
        assert result.passed is True
        assert len(result.checks) == 2
        assert all(c["passed"] for c in result.checks)

    @pytest.mark.asyncio
    async def test_dirty_tree_warns_but_passes(self):
        builder = PRBuilder()
        with patch.object(builder, "_run_git", new_callable=AsyncMock) as mock_git:
            mock_git.side_effect = [
                (0, "M config.yaml\n", ""),  # dirty
                (0, "abc HEAD", ""),          # remote ok
            ]
            result = await builder.preflight_check(Path("/tmp/repo"))
        assert result.passed is True
        assert result.checks[0]["passed"] is False  # dirty tree warning
        assert result.checks[1]["passed"] is True

    @pytest.mark.asyncio
    async def test_remote_unreachable_fails(self):
        builder = PRBuilder()
        with patch.object(builder, "_run_git", new_callable=AsyncMock) as mock_git:
            mock_git.side_effect = [
                (0, "", ""),                  # clean tree
                (128, "", "fatal: remote"),   # remote unreachable
            ]
            result = await builder.preflight_check(Path("/tmp/repo"))
        assert result.passed is False
        assert "unreachable" in result.reasons[0].lower()

    @pytest.mark.asyncio
    async def test_dry_run_skips_preflight(self):
        builder = PRBuilder(dry_run=True)
        result = await builder.create_pr(_make_request())
        assert result.success is True
        assert result.preflight is None

    @pytest.mark.asyncio
    async def test_create_pr_blocks_on_preflight_failure(self, tmp_path: Path):
        builder = PRBuilder()
        with patch.object(builder, "_run_git", new_callable=AsyncMock) as mock_git:
            mock_git.side_effect = [
                (0, "", ""),                # clean
                (128, "", "unreachable"),    # remote fail
            ]
            result = await builder.create_pr(_make_request(repo_dir=str(_temp_repo_dir(tmp_path))))
        assert result.success is False
        assert result.preflight is not None
        assert not result.preflight.passed

    @pytest.mark.asyncio
    async def test_create_pr_proceeds_with_dirty_tree(self, tmp_path: Path):
        builder = PRBuilder()
        with patch.object(builder, "_run_git", new_callable=AsyncMock) as mock_git, \
             patch.object(builder, "_run_cmd", new_callable=AsyncMock) as mock_cmd:
            mock_git.side_effect = [
                (0, "M file.txt\n", ""),    # dirty tree (warning)
                (0, "abc HEAD", ""),         # remote ok
                # check_existing_pr: ls-remote --heads
                (0, "", ""),                 # no existing branch
                # create_pr steps
                (0, "", ""),                 # pull
                (0, "", ""),                 # checkout -b
                (0, "", ""),                 # add
                (0, "", ""),                 # commit
                (0, "", ""),                 # push
            ]
            mock_cmd.return_value = (0, "https://github.com/x/y/pull/1\n", "")
            result = await builder.create_pr(_make_request(repo_dir=str(_temp_repo_dir(tmp_path))))
        assert result.success is True


class TestBranchPRDedup:
    @pytest.mark.asyncio
    async def test_no_existing_branch(self):
        builder = PRBuilder()
        with patch.object(builder, "_run_git", new_callable=AsyncMock, return_value=(0, "", "")):
            exists, url = await builder.check_existing_pr("codex/test", Path("/tmp"))
        assert exists is False
        assert url is None

    @pytest.mark.asyncio
    async def test_branch_with_open_pr(self):
        builder = PRBuilder()
        with patch.object(builder, "_run_git", new_callable=AsyncMock, return_value=(0, "abc refs/heads/codex/test", "")), \
             patch.object(builder, "_run_cmd", new_callable=AsyncMock, return_value=(0, '[{"number":42,"url":"https://github.com/x/y/pull/42"}]', "")):
            exists, url = await builder.check_existing_pr("codex/test", Path("/tmp"))
        assert exists is True
        assert url == "https://github.com/x/y/pull/42"

    @pytest.mark.asyncio
    async def test_branch_no_open_pr(self):
        builder = PRBuilder()
        with patch.object(builder, "_run_git", new_callable=AsyncMock, return_value=(0, "abc refs/heads/codex/test", "")), \
             patch.object(builder, "_run_cmd", new_callable=AsyncMock, return_value=(0, "[]", "")):
            exists, url = await builder.check_existing_pr("codex/test", Path("/tmp"))
        assert exists is True
        assert url is None

    @pytest.mark.asyncio
    async def test_dry_run_skips_dedup(self):
        builder = PRBuilder(dry_run=True)
        result = await builder.create_pr(_make_request())
        assert result.existing_pr_url is None

    @pytest.mark.asyncio
    async def test_create_pr_returns_existing(self, tmp_path: Path):
        builder = PRBuilder()
        with patch.object(builder, "preflight_check", new_callable=AsyncMock) as mock_pf, \
             patch.object(builder, "check_existing_pr", new_callable=AsyncMock) as mock_dedup:
            from trading_assistant.schemas.repo_changes import PreflightResult
            mock_pf.return_value = PreflightResult(passed=True)
            mock_dedup.return_value = (True, "https://github.com/x/y/pull/42")
            result = await builder.create_pr(_make_request(repo_dir=str(_temp_repo_dir(tmp_path))))
        assert result.success is True
        assert result.existing_pr_url == "https://github.com/x/y/pull/42"


class TestIssueCreation:
    @pytest.mark.asyncio
    async def test_create_issue_returns_existing(self):
        builder = PRBuilder()
        request = GitHubIssueRequest(
            bot_id="bot1",
            title="Investigate crash",
            body="details",
            repo_dir=str(Path.cwd()),
            dedupe_key="bot1:ImportError",
        )
        with patch.object(builder, "_run_cmd", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = (0, '[{"number": 4, "title": "Investigate crash", "url": "https://github.com/x/y/issues/4"}]', "")
            result = await builder.create_issue(request)
        assert result.success is True
        assert result.existing_issue_url == "https://github.com/x/y/issues/4"

    @pytest.mark.asyncio
    async def test_create_issue_success(self):
        builder = PRBuilder()
        request = GitHubIssueRequest(
            bot_id="bot1",
            title="Investigate crash",
            body="details",
            repo_dir=str(Path.cwd()),
            labels=["trading-assistant"],
        )

        async def mock_cmd(args, cwd, timeout_seconds=60):
            if args[:3] == ["gh", "issue", "list"]:
                return (0, "[]", "")
            return (0, "https://github.com/x/y/issues/7\n", "")

        with patch.object(builder, "_run_cmd", side_effect=mock_cmd):
            result = await builder.create_issue(request)
        assert result.success is True
        assert result.issue_url == "https://github.com/x/y/issues/7"

    @pytest.mark.asyncio
    async def test_create_issue_dedupes_by_dedupe_key_search(self):
        builder = PRBuilder()
        request = GitHubIssueRequest(
            bot_id="bot1",
            title="Investigate crash",
            body="details",
            repo_dir=str(Path.cwd()),
            dedupe_key="bot1:ImportError:No module named foo",
        )
        with patch.object(builder, "_run_cmd", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = (
                0,
                '[{"number": 9, "title": "Some other issue title", "url": "https://github.com/x/y/issues/9"}]',
                "",
            )
            result = await builder.create_issue(request)
        assert result.success is True
        assert result.existing_issue_url == "https://github.com/x/y/issues/9"


class TestDiffPreview:
    def test_body_includes_diff(self):
        builder = PRBuilder()
        req = _make_request(file_changes=[
            FileChange(
                file_path="config.yaml",
                diff_preview="-quality: 0.6\n+quality: 0.7",
            ),
        ])
        body = builder.format_pr_body(req)
        assert "## File Changes" in body
        assert "```diff" in body
        assert "+quality: 0.7" in body

    def test_body_multiple_files(self):
        builder = PRBuilder()
        req = _make_request(file_changes=[
            FileChange(file_path="a.yaml", diff_preview="-x: 1\n+x: 2"),
            FileChange(file_path="b.py", diff_preview="-Y = 1\n+Y = 2"),
        ])
        body = builder.format_pr_body(req)
        assert "`a.yaml`" in body
        assert "`b.py`" in body
        assert body.count("```diff") == 2

    def test_body_no_file_changes(self):
        builder = PRBuilder()
        req = _make_request(file_changes=[])
        body = builder.format_pr_body(req)
        assert "## File Changes" not in body


class TestPRReviewMonitoring:
    @pytest.mark.asyncio
    async def test_approved_review(self):
        builder = PRBuilder()
        with patch.object(builder, "_run_cmd", new_callable=AsyncMock) as mock:
            import json
            mock.return_value = (0, json.dumps({
                "reviews": [{"state": "APPROVED", "author": {"login": "reviewer1"}}],
                "comments": [],
                "state": "OPEN",
            }), "")
            result = await builder.check_pr_reviews("https://github.com/x/y/pull/42", Path("/tmp"))
        assert result is not None
        assert result.review_state == ReviewState.APPROVED
        assert "reviewer1" in result.reviewers
        assert result.needs_attention is False

    @pytest.mark.asyncio
    async def test_changes_requested(self):
        builder = PRBuilder()
        with patch.object(builder, "_run_cmd", new_callable=AsyncMock) as mock:
            import json
            mock.return_value = (0, json.dumps({
                "reviews": [
                    {"state": "APPROVED", "author": {"login": "r1"}},
                    {"state": "CHANGES_REQUESTED", "author": {"login": "r2"}},
                ],
                "comments": [],
                "state": "OPEN",
            }), "")
            result = await builder.check_pr_reviews("https://github.com/x/y/pull/42", Path("/tmp"))
        assert result.review_state == ReviewState.CHANGES_REQUESTED
        assert result.needs_attention is True

    @pytest.mark.asyncio
    async def test_with_comments(self):
        builder = PRBuilder()
        with patch.object(builder, "_run_cmd", new_callable=AsyncMock) as mock:
            import json
            mock.return_value = (0, json.dumps({
                "reviews": [],
                "comments": [{"body": "Please fix this"}],
                "state": "OPEN",
            }), "")
            result = await builder.check_pr_reviews("https://github.com/x/y/pull/42", Path("/tmp"))
        assert result.review_state == ReviewState.PENDING
        assert len(result.actionable_comments) == 1
        assert result.needs_attention is True

    @pytest.mark.asyncio
    async def test_pending_no_reviews(self):
        builder = PRBuilder()
        with patch.object(builder, "_run_cmd", new_callable=AsyncMock) as mock:
            import json
            mock.return_value = (0, json.dumps({
                "reviews": [],
                "comments": [],
                "state": "OPEN",
            }), "")
            result = await builder.check_pr_reviews("https://github.com/x/y/pull/42", Path("/tmp"))
        assert result.review_state == ReviewState.PENDING
        assert result.needs_attention is False

    @pytest.mark.asyncio
    async def test_invalid_url_returns_none(self):
        builder = PRBuilder()
        result = await builder.check_pr_reviews("not-a-url", Path("/tmp"))
        assert result is None

    def test_extract_pr_number(self):
        assert PRBuilder._extract_pr_number("https://github.com/user/repo/pull/42") == 42
        assert PRBuilder._extract_pr_number("not-a-url") is None
        assert PRBuilder._extract_pr_number("https://github.com/user/repo/pull/0") == 0
