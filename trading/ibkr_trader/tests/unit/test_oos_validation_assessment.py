from backtests.shared.validation.oos_validation import OOSResult, WindowMetrics, _assess, format_report


def test_assessment_uses_reproduced_is_pf_not_reference_baseline():
    is_metrics = WindowMetrics(total_trades=412, profit_factor=5.72, avg_r=0.844)
    oos_metrics = WindowMetrics(total_trades=21, profit_factor=7.10, avg_r=1.192)

    assessment, action = _assess("nq_regime", is_metrics, oos_metrics)

    assert assessment == "YELLOW"
    assert action == "Positive expectancy with marginal sample -- monitor"


def test_report_labels_reproduced_and_reference_pf_separately():
    result = OOSResult(
        strategy="nq_regime",
        family="momentum",
        is_metrics=WindowMetrics(total_trades=412, profit_factor=5.72),
        oos_metrics=WindowMetrics(total_trades=21, profit_factor=7.10, avg_r=1.192),
        assessment="YELLOW",
        action="Positive expectancy with marginal sample -- monitor",
    )

    report = format_report([result], "2026-05-01")

    assert "Repro IS PF" in report
    assert "Ref PF" in report
    assert "Assessment basis: reproduced IS metrics from the same frozen replay" in report
