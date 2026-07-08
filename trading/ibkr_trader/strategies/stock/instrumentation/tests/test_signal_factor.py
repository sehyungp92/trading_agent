from strategies.stock.instrumentation.src.signal_factor import SignalFactor, build_signal_factors


def test_signal_factor_to_dict():
    sf = SignalFactor("alignment_score", factor_value=2.0, threshold=1.0, contribution=0.667)
    d = sf.to_dict()
    assert d["factor_name"] == "alignment_score"
    assert d["factor_value"] == 2.0
    assert d["threshold"] == 1.0
    assert d["contribution"] == 0.667


def test_build_signal_factors():
    factors = [
        SignalFactor("alignment_score", 2.0, 1.0, 0.667),
        SignalFactor("trend_strength", 0.85, 0.5, 0.283),
    ]
    result = build_signal_factors(factors)
    assert len(result) == 2
    assert result[0]["factor_name"] == "alignment_score"
