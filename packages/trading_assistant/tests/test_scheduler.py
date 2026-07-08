from unittest.mock import AsyncMock


from trading_assistant.orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs


class TestSchedulerConfig:
    def test_default_config(self):
        config = SchedulerConfig()
        assert config.monitoring_interval_minutes == 60
        assert config.worker_interval_seconds == 60
        assert config.relay_poll_interval_seconds == 60

    def test_custom_config(self):
        config = SchedulerConfig(
            monitoring_interval_minutes=5,
            worker_interval_seconds=30,
        )
        assert config.monitoring_interval_minutes == 5
        assert config.worker_interval_seconds == 30


class TestCreateSchedulerJobs:
    def test_creates_expected_jobs(self):
        config = SchedulerConfig()
        worker_fn = AsyncMock()
        monitoring_fn = AsyncMock()
        relay_fn = AsyncMock()

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=worker_fn,
            monitoring_fn=monitoring_fn,
            relay_fn=relay_fn,
        )

        assert len(jobs) == 3
        job_names = {j["name"] for j in jobs}
        assert "worker" in job_names
        assert "monitoring" in job_names
        assert "relay_poll" in job_names

    def test_per_tz_morning_scan_jobs(self):
        config = SchedulerConfig()
        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            morning_scan_fns=[
                {"fn": AsyncMock(), "hour": 7, "minute": 0, "name_suffix": "0600"},
                {"fn": AsyncMock(), "hour": 22, "minute": 0, "name_suffix": "2200"},
            ],
        )
        morning_jobs = [j for j in jobs if j["name"].startswith("morning_scan_")]
        assert len(morning_jobs) == 2
        assert morning_jobs[0]["hour"] == 7
        assert morning_jobs[1]["hour"] == 22

    def test_per_tz_evening_report_jobs(self):
        config = SchedulerConfig()
        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            evening_report_fns=[
                {"fn": AsyncMock(), "hour": 17, "minute": 0, "name_suffix": "0600"},
                {"fn": AsyncMock(), "hour": 8, "minute": 0, "name_suffix": "2200"},
            ],
        )
        evening_jobs = [j for j in jobs if j["name"].startswith("evening_report_")]
        assert len(evening_jobs) == 2

    def test_per_tz_scan_overrides_global(self):
        """Per-tz morning/evening fns should prevent global fallback."""
        config = SchedulerConfig()
        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            morning_scan_fn=AsyncMock(),
            morning_scan_fns=[
                {"fn": AsyncMock(), "hour": 7, "minute": 0, "name_suffix": "utc"},
            ],
            evening_report_fn=AsyncMock(),
            evening_report_fns=[
                {"fn": AsyncMock(), "hour": 22, "minute": 0, "name_suffix": "utc"},
            ],
        )
        job_names = [j["name"] for j in jobs]
        # Per-tz versions should exist, global fallbacks should not
        assert "morning_scan_utc" in job_names
        assert "morning_scan" not in job_names
        assert "evening_report_utc" in job_names
        assert "evening_report" not in job_names
