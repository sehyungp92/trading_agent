"""Tests for report checklist schema."""
from trading_assistant.schemas.report_checklist import ReportChecklist, CheckResult


class TestCheckResult:
    def test_passing_check(self):
        cr = CheckResult(name="all_bots_reported", passed=True, detail="5/5 bots")
        assert cr.passed is True

    def test_failing_check(self):
        cr = CheckResult(name="open_risks_section", passed=False, detail="CRITICAL error unaddressed")
        assert cr.passed is False


class TestReportChecklist:
    def test_all_pass(self):
        checklist = ReportChecklist(
            report_id="daily-2026-03-01",
            checks=[
                CheckResult(name="all_bots_reported", passed=True, detail="5/5"),
                CheckResult(name="anomaly_detection_ran", passed=True, detail="ok"),
            ],
        )
        assert checklist.overall == "PASS"
        assert checklist.blocking_issues == []

    def test_one_fails(self):
        checklist = ReportChecklist(
            report_id="daily-2026-03-01",
            checks=[
                CheckResult(name="all_bots_reported", passed=True, detail="5/5"),
                CheckResult(name="open_risks_section", passed=False, detail="CRITICAL error"),
            ],
        )
        assert checklist.overall == "FAIL"
        assert len(checklist.blocking_issues) == 1
        assert "open_risks_section" in checklist.blocking_issues[0]

    def test_empty_checks(self):
        checklist = ReportChecklist(report_id="daily-2026-03-01", checks=[])
        assert checklist.overall == "PASS"
