# skills/deployment_monitor.py
"""Deployment monitor -- tracks post-merge deployments and detects regressions.

Lifecycle: PR created -> merge detected -> bot heartbeat confirms deploy ->
monitor window (24h) -> regression check -> success or rollback PR.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.schemas.repo_changes import FileChange, PRRequest, PRResult
from trading_assistant.schemas.deployment_monitoring import (
    DeploymentMetricsSnapshot,
    DeploymentRecord,
    DeploymentStatus,
)

logger = logging.getLogger(__name__)


class DeploymentMonitor:
    """Monitors post-merge deployments for regressions."""

    def __init__(
        self,
        findings_dir: Path,
        curated_dir: Path,
        pr_builder=None,  # PRBuilder (optional to avoid circular imports)
        config_registry=None,  # ConfigRegistry
        event_stream=None,  # EventStream
        file_change_generator=None,  # FileChangeGenerator
    ) -> None:
        self._path = findings_dir / "deployments.jsonl"
        self._curated_dir = curated_dir
        self._pr_builder = pr_builder
        self._config_registry = config_registry
        self._event_stream = event_stream
        self._file_change_generator = file_change_generator
        self._lock = threading.Lock()

    def create_deployment(
        self,
        deployment_id: str,
        approval_request_id: str,
        pr_url: str,
        bot_id: str,
        param_changes: list[dict],
        pr_number: int = 0,
        suggestion_id: str | None = None,
    ) -> DeploymentRecord:
        """Create a deployment record when a PR is created."""
        existing = self.get_by_id(deployment_id)
        if existing is not None:
            return existing

        record = DeploymentRecord(
            deployment_id=deployment_id,
            approval_request_id=approval_request_id,
            suggestion_id=suggestion_id,
            pr_url=pr_url,
            pr_number=pr_number,
            bot_id=bot_id,
            param_changes=param_changes,
            status=DeploymentStatus.PENDING_MERGE,
        )
        self._append_record(record)
        return record

    async def check_merge_status(self, deployment_id: str) -> bool:
        """Check if the PR has been merged via ``gh pr view``.

        Returns True if merge was detected (status changed).
        """
        record = self.get_by_id(deployment_id)
        if record is None or record.status != DeploymentStatus.PENDING_MERGE:
            return False

        if self._pr_builder is None:
            return False

        try:
            returncode, stdout, stderr = await self._pr_builder.run_gh_command(
                ["gh", "pr", "view", record.pr_url, "--json", "state,mergedAt"],
                cwd=Path("."), timeout_seconds=30,
            )
            if returncode != 0:
                logger.warning("Failed to check PR status: %s", stderr)
                return False

            data = json.loads(stdout)
            if data.get("state") == "MERGED":
                record.status = DeploymentStatus.MERGED
                merged_at = data.get("mergedAt")
                if isinstance(merged_at, str) and merged_at:
                    record.merge_time = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
                else:
                    record.merge_time = datetime.now(timezone.utc)
                self._update_record(record)
                if self._event_stream:
                    self._event_stream.broadcast(
                        "deployment_pr_merged",
                        {"deployment_id": deployment_id, "bot_id": record.bot_id},
                    )
                return True
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Error checking merge status: %s", exc)

        return False

    def record_pre_deploy_metrics(
        self,
        deployment_id: str,
        snapshot: DeploymentMetricsSnapshot,
    ) -> None:
        """Record pre-deployment metrics for regression baseline."""
        record = self.get_by_id(deployment_id)
        if record is None:
            return
        record.pre_deploy_metrics = snapshot
        self._update_record(record)

    def record_post_deploy_metrics(
        self,
        deployment_id: str,
        snapshot: DeploymentMetricsSnapshot,
    ) -> None:
        """Accumulate post-deployment metric snapshots (max 48 = 24h at 30min)."""
        record = self.get_by_id(deployment_id)
        if record is None:
            return
        record.post_deploy_snapshots.append(snapshot)
        if len(record.post_deploy_snapshots) > 48:
            record.post_deploy_snapshots = record.post_deploy_snapshots[-48:]
        self._update_record(record)

    def is_heartbeat_confirmed(
        self, deployment_id: str, latest_heartbeat_time: datetime | None,
    ) -> bool:
        """Check if a heartbeat arrived after merge time, confirming deployment."""
        record = self.get_by_id(deployment_id)
        if record is None or record.merge_time is None:
            return False
        if latest_heartbeat_time is None:
            return False
        return latest_heartbeat_time > record.merge_time

    def check_merged_timeout(self, deployment_id: str, timeout_hours: int = 6) -> bool:
        """Check if MERGED state has timed out (no heartbeat confirmation).

        Returns True if timed out -> transitions to STALE.
        """
        record = self.get_by_id(deployment_id)
        if record is None or record.status != DeploymentStatus.MERGED:
            return False
        if record.merge_time is None:
            return False
        elapsed = datetime.now(timezone.utc) - record.merge_time
        if elapsed > timedelta(hours=timeout_hours):
            record.status = DeploymentStatus.STALE
            self._update_record(record)
            return True
        return False

    def mark_deployed(self, deployment_id: str, detected_at: datetime | None = None) -> None:
        """Mark as deployed when bot heartbeat confirms new version."""
        record = self.get_by_id(deployment_id)
        if record is None:
            return
        detected = detected_at or datetime.now(timezone.utc)
        record.status = DeploymentStatus.DEPLOYED
        record.deploy_detected_time = detected
        record.monitoring_end_time = detected + timedelta(hours=record.monitoring_window_hours)
        self._update_record(record)

    def check_regression(self, deployment_id: str) -> bool:
        """Compare pre-deploy vs worst post-deploy snapshot.

        Returns True if regression detected.
        """
        record = self.get_by_id(deployment_id)
        if record is None:
            return False
        if record.pre_deploy_metrics is None:
            logger.warning(
                "Cannot detect regression for %s — no baseline metrics",
                deployment_id,
            )
            return False
        if not record.post_deploy_snapshots:
            return False

        pre = record.pre_deploy_metrics
        # Use worst snapshot: min avg_pnl, min win_rate, max drawdown
        post = min(record.post_deploy_snapshots, key=lambda s: s.avg_pnl)
        regression_reasons: list[str] = []

        # PnL regression: post avg_pnl significantly below pre
        if pre.avg_pnl != 0:
            pnl_change_pct = (post.avg_pnl - pre.avg_pnl) / abs(pre.avg_pnl) * 100
            if pnl_change_pct < -50:  # >50% decline
                regression_reasons.append(
                    f"Avg PnL declined {pnl_change_pct:.1f}% "
                    f"(${pre.avg_pnl:.2f} -> ${post.avg_pnl:.2f})"
                )

        # Win rate decline > 15 percentage points
        wr_delta = post.win_rate - pre.win_rate
        if wr_delta < -0.15:
            regression_reasons.append(
                f"Win rate declined {wr_delta:.1%} "
                f"({pre.win_rate:.1%} -> {post.win_rate:.1%})"
            )

        # Max drawdown >50% worse
        if pre.max_drawdown_pct > 0:
            dd_change = (
                (post.max_drawdown_pct - pre.max_drawdown_pct) / pre.max_drawdown_pct
            )
            if dd_change > 0.5:
                regression_reasons.append(
                    f"Max drawdown worsened {dd_change:.0%} "
                    f"({pre.max_drawdown_pct:.1f}% -> {post.max_drawdown_pct:.1f}%)"
                )

        if regression_reasons:
            record.regression_detected = True
            record.regression_details = "; ".join(regression_reasons)
            record.status = DeploymentStatus.REGRESSION_DETECTED
            self._update_record(record)
            return True

        return False

    def check_monitoring_window_expired(self, deployment_id: str) -> bool:
        """Check if monitoring window has expired without regression.

        If expired, transitions to MONITORING_COMPLETE.
        """
        record = self.get_by_id(deployment_id)
        if record is None or record.status != DeploymentStatus.DEPLOYED:
            return False
        if record.monitoring_end_time is None:
            return False
        if datetime.now(timezone.utc) >= record.monitoring_end_time:
            record.status = DeploymentStatus.MONITORING_COMPLETE
            self._update_record(record)
            return True
        return False

    def check_stale_pending(self, deployment_id: str, max_days: int = 7) -> bool:
        """Mark PENDING_MERGE as STALE if older than max_days.

        Returns True if transitioned to STALE.
        """
        record = self.get_by_id(deployment_id)
        if record is None or record.status != DeploymentStatus.PENDING_MERGE:
            return False
        elapsed = datetime.now(timezone.utc) - record.created_at
        if elapsed > timedelta(days=max_days):
            record.status = DeploymentStatus.STALE
            self._update_record(record)
            return True
        return False

    async def create_rollback_pr(self, deployment_id: str) -> PRResult | None:
        """Create a PR that reverts the parameter change."""
        record = self.get_by_id(deployment_id)
        if record is None:
            return None
        if self._pr_builder is None:
            logger.warning(
                "Cannot create rollback PR for %s — pr_builder not available "
                "(autonomous_enabled may be False)", deployment_id,
            )
            return PRResult(
                success=False,
                branch_name="",
                error="pr_builder not available (autonomous_enabled=False)",
            )

        repo_dir = self._get_repo_dir(record.bot_id)
        if repo_dir is None:
            logger.warning(
                "Cannot create rollback PR for %s — repo_dir not resolvable for %s",
                deployment_id, record.bot_id,
            )
            return PRResult(
                success=False,
                branch_name="",
                error=f"repo_dir not resolvable for {record.bot_id}",
            )

        # Build reverse changes using FileChangeGenerator when available
        file_changes = []
        for change in record.param_changes:
            param_name = change.get("param_name", "unknown")
            old_value = change.get("old_value")
            new_value = change.get("new_value")

            if self._file_change_generator and self._config_registry:
                # Use FileChangeGenerator for proper file content reversal
                param_def = self._config_registry.get_parameter(record.bot_id, param_name)
                if param_def is not None:
                    try:
                        fc = self._file_change_generator.generate_change(
                            param_def, old_value, repo_dir,
                        )
                        file_changes.append(fc)
                        continue
                    except Exception:
                        logger.warning("FileChangeGenerator failed for %s, using fallback", param_name)

            # Fallback: create a descriptive FileChange (won't produce valid diff
            # but is better than overwriting entire file with a raw value)
            file_changes.append(
                FileChange(
                    file_path=change.get("file_path", param_name),
                    original_content="",
                    new_content="",
                    diff_preview=f"Revert {param_name}: {new_value} -> {old_value}",
                )
            )

        pr_request = PRRequest(
            approval_request_id=record.approval_request_id,
            suggestion_id=deployment_id,
            bot_id=record.bot_id,
            repo_dir=str(repo_dir),
            branch_name=f"codex/rollback-{deployment_id[:8]}",
            title=f"[trading-assistant] ROLLBACK: revert changes on {record.bot_id}",
            body=(
                f"## Automated Rollback\n\n"
                f"Regression detected after deployment:\n"
                f"{record.regression_details}\n\n"
                f"### Changes Reverted\n"
                + "\n".join(
                    f"- `{c.get('param_name')}`: {c.get('new_value')} -> {c.get('old_value')}"
                    for c in record.param_changes
                )
                + f"\n\nOriginal PR: {record.pr_url}"
            ),
            file_changes=file_changes,
        )

        result = await self._pr_builder.create_pr(pr_request)
        if result.success and result.pr_url:
            record.rollback_pr_url = result.pr_url
            record.status = DeploymentStatus.ROLLED_BACK
            self._update_record(record)
            if self._event_stream:
                self._event_stream.broadcast(
                    "deployment_rolled_back",
                    {"deployment_id": deployment_id, "rollback_pr": result.pr_url},
                )
        return result

    def get_monitoring(self) -> list[DeploymentRecord]:
        """Return deployments that need active monitoring (excludes terminal states)."""
        _TERMINAL = {
            DeploymentStatus.MONITORING_COMPLETE,
            DeploymentStatus.STALE,
            DeploymentStatus.REGRESSION_DETECTED,
            DeploymentStatus.ROLLED_BACK,
        }
        return [d for d in self._load_all() if d.status not in _TERMINAL]

    def get_by_id(self, deployment_id: str) -> DeploymentRecord | None:
        """Retrieve deployment by ID."""
        for d in self._load_all():
            if d.deployment_id == deployment_id:
                return d
        return None

    def collect_metrics_snapshot(
        self, bot_id: str
    ) -> DeploymentMetricsSnapshot | None:
        """Collect current bot metrics from most recent curated data."""
        if not self._curated_dir.exists():
            return None

        summaries = self._load_recent_summaries(bot_id, limit=7)
        if not summaries:
            return None

        total_trades = 0
        weighted_wins = 0.0
        total_net_pnl = 0.0
        drawdowns: list[float] = []
        daily_pnls: list[float] = []

        for data in summaries:
            trades = int(data.get("total_trades", data.get("trade_count", 0)) or 0)
            total_trades += trades
            if "win_count" in data:
                weighted_wins += float(data.get("win_count", 0) or 0)
            else:
                weighted_wins += float(data.get("win_rate", 0.0) or 0.0) * trades
            if "net_pnl" in data:
                net_pnl = float(data.get("net_pnl", 0.0) or 0.0)
            else:
                # Legacy fixtures stored avg_pnl directly; convert back to daily total.
                net_pnl = float(data.get("avg_pnl", 0.0) or 0.0) * trades
            total_net_pnl += net_pnl
            daily_pnls.append(net_pnl)
            drawdowns.append(float(data.get("max_drawdown_pct", 0.0) or 0.0))

        return DeploymentMetricsSnapshot(
            bot_id=bot_id,
            total_trades=total_trades,
            win_rate=(weighted_wins / total_trades) if total_trades > 0 else 0.0,
            avg_pnl=(total_net_pnl / total_trades) if total_trades > 0 else 0.0,
            sharpe_rolling_7d=self._compute_daily_sharpe(daily_pnls),
            max_drawdown_pct=max(drawdowns) if drawdowns else 0.0,
        )

    def _load_recent_summaries(self, bot_id: str, limit: int = 7) -> list[dict]:
        date_dirs = sorted(
            [
                d for d in self._curated_dir.iterdir()
                if d.is_dir() and d.name != "weekly"
            ],
            reverse=True,
        )
        summaries: list[dict] = []
        for date_dir in date_dirs:
            summary_path = date_dir / bot_id / "summary.json"
            if not summary_path.exists():
                continue
            try:
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
            if len(summaries) >= limit:
                break
        return summaries

    @staticmethod
    def _compute_daily_sharpe(pnls: list[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        try:
            std = statistics.stdev(pnls)
        except statistics.StatisticsError:
            return 0.0
        if std <= 0:
            return 0.0
        return statistics.mean(pnls) / std * math.sqrt(252)

    def _get_repo_dir(self, bot_id: str) -> Path | None:
        """Get repo directory for a bot. Returns None if not resolvable."""
        if self._config_registry:
            profile = self._config_registry.get_profile(bot_id)
            if profile:
                return Path(profile.repo_dir)
        return None

    def _load_all(self) -> list[DeploymentRecord]:
        """Load all deployments from JSONL."""
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text().strip().splitlines():
            try:
                records.append(DeploymentRecord.model_validate(json.loads(line)))
            except (json.JSONDecodeError, Exception):
                continue
        return records

    def _append_record(self, record: DeploymentRecord) -> None:
        """Append a single record to JSONL."""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as f:
                f.write(json.dumps(record.model_dump(), default=str) + "\n")

    def _update_record(self, record: DeploymentRecord) -> None:
        """Update an existing record in JSONL (full rewrite)."""
        with self._lock:
            records = self._load_all()
            updated = []
            for r in records:
                if r.deployment_id == record.deployment_id:
                    updated.append(record)
                else:
                    updated.append(r)
            self._save_all_unlocked(updated)

    def _save_all(self, records: list[DeploymentRecord]) -> None:
        """Overwrite all records to JSONL (acquires lock)."""
        with self._lock:
            self._save_all_unlocked(records)

    def _save_all_unlocked(self, records: list[DeploymentRecord]) -> None:
        """Overwrite all records to JSONL (caller must hold lock)."""
        from trading_assistant.skills._atomic_write import atomic_rewrite_jsonl

        atomic_rewrite_jsonl(self._path, records)
