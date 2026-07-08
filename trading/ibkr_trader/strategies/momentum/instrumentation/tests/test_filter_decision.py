import pytest
from strategies.momentum.instrumentation.src.filter_decision import FilterDecision, build_filter_decisions


def test_filter_decision_pass():
    fd = FilterDecision("heat_cap", threshold=3.0, actual_value=2.1, passed=True)
    d = fd.to_dict()
    assert d["filter_name"] == "heat_cap"
    assert d["threshold"] == 3.0
    assert d["actual_value"] == 2.1
    assert d["passed"] is True
    assert d["margin_pct"] == pytest.approx(-30.0, abs=0.1)


def test_filter_decision_fail():
    fd = FilterDecision("spread", threshold=0.50, actual_value=0.75, passed=False)
    d = fd.to_dict()
    assert d["passed"] is False
    assert d["margin_pct"] == pytest.approx(50.0, abs=0.1)


def test_filter_decision_zero_threshold():
    fd = FilterDecision("news_blocked", threshold=0.0, actual_value=1.0, passed=False)
    d = fd.to_dict()
    assert d["margin_pct"] is None


def test_build_filter_decisions_returns_list_of_dicts():
    decisions = [
        FilterDecision("heat_cap", 3.0, 2.1, True),
        FilterDecision("spread", 0.50, 0.30, True),
    ]
    result = build_filter_decisions(decisions)
    assert len(result) == 2
    assert all(isinstance(d, dict) for d in result)
