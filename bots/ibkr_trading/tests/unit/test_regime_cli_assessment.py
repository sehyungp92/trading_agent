from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from backtests.regime.analysis import assessment_validation, scanner_validation
from backtests.regime.analysis.diagnostics import generate_regime_diagnostics_report
from backtests.regime.analysis.metrics import PortfolioMetrics
from backtests.regime.auto.scoring import CompositeScore
from backtests.regime import cli


def _dummy_cached_data():
    idx = pd.date_range("2021-01-01", periods=10, freq="W-FRI")
    macro_df = pd.DataFrame({"GROWTH": 0.0, "INFLATION": 0.0}, index=idx)
    market_df = pd.DataFrame({"VIX": 20.0, "SPREAD": 1.0}, index=idx)
    strat_ret_df = pd.DataFrame(
        {"SPY": 0.0, "EFA": 0.0, "TLT": 0.0, "GLD": 0.0, "CASH": 0.0},
        index=idx,
    )
    return macro_df, market_df, strat_ret_df


def test_calibration_sweep_command_reports_winner(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "backtests.regime.data.downloader.load_cached_data",
        lambda data_dir: _dummy_cached_data(),
    )

    def fake_run_signal_engine(*args, **kwargs):
        cfg = kwargs["cfg"]
        key = (round(cfg.posterior_temperature, 2), round(cfg.posterior_ema_alpha, 2))
        summary_map = {
            (1.0, 0.8): {
                "avg_p_dom": 1.0,
                "spy_allocation_range_bp": 270.0,
                "sharpe": 1.44,
                "max_drawdown_pct": 0.087,
                "regime_2022": "S",
            },
            (1.2, 0.8): {
                "avg_p_dom": 0.78,
                "spy_allocation_range_bp": 215.0,
                "sharpe": 1.46,
                "max_drawdown_pct": 0.090,
                "regime_2022": "S",
            },
            (1.5, 0.7): {
                "avg_p_dom": 0.82,
                "spy_allocation_range_bp": 205.0,
                "sharpe": 1.43,
                "max_drawdown_pct": 0.095,
                "regime_2022": "D",
            },
            (1.5, 0.8): {
                "avg_p_dom": 0.68,
                "spy_allocation_range_bp": 180.0,
                "sharpe": 1.38,
                "max_drawdown_pct": 0.11,
                "regime_2022": "R",
            },
        }
        df = pd.DataFrame(
            {"P_G": [1.0], "P_R": [0.0], "P_S": [0.0], "P_D": [0.0], "pi_SPY": [0.02]},
            index=[pd.Timestamp("2022-01-07")],
        )
        df.attrs["summary"] = summary_map[key]
        return df

    monkeypatch.setattr("regime.engine.run_signal_engine", fake_run_signal_engine)
    monkeypatch.setattr(
        "backtests.regime.engine.portfolio_sim.simulate_portfolio",
        lambda *args, **kwargs: SimpleNamespace(metrics=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "backtests.regime.analysis.assessment_validation.summarize_calibration_candidate",
        lambda signals, result: dict(signals.attrs["summary"]),
    )

    mutations_path = tmp_path / "base.json"
    mutations_path.write_text(json.dumps({"posterior_ema_alpha": 0.8}))
    args = SimpleNamespace(
        data_dir=str(tmp_path),
        equity=100_000.0,
        cost_bps=0.0,
        mutations_json=str(mutations_path),
    )

    cli.cmd_calibration_sweep(args)
    captured = capsys.readouterr().out

    assert "Recommendation: Adopt temp=1.2, ema=0.8" in captured
    assert "temp=1.2, ema=0.8" in captured
    assert Path("backtests/regime/auto/output/calibration_sweep.json").exists()


def test_calibration_sweep_mutations_json_overrides_preset_and_records_source(
    monkeypatch, tmp_path, capsys,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "backtests.regime.data.downloader.load_cached_data",
        lambda data_dir: _dummy_cached_data(),
    )

    def fake_run_signal_engine(*args, **kwargs):
        cfg = kwargs["cfg"]
        key = (round(cfg.posterior_temperature, 2), round(cfg.posterior_ema_alpha, 2))
        summary_map = {
            (1.0, 0.7): {
                "avg_p_dom": 0.95,
                "spy_allocation_range_bp": 240.0,
                "sharpe": 1.40,
                "max_drawdown_pct": 0.095,
                "regime_2022": "S",
            },
            (1.2, 0.8): {
                "avg_p_dom": 0.78,
                "spy_allocation_range_bp": 215.0,
                "sharpe": 1.46,
                "max_drawdown_pct": 0.090,
                "regime_2022": "S",
            },
            (1.5, 0.7): {
                "avg_p_dom": 0.82,
                "spy_allocation_range_bp": 205.0,
                "sharpe": 1.43,
                "max_drawdown_pct": 0.095,
                "regime_2022": "D",
            },
            (1.5, 0.8): {
                "avg_p_dom": 0.68,
                "spy_allocation_range_bp": 180.0,
                "sharpe": 1.38,
                "max_drawdown_pct": 0.11,
                "regime_2022": "R",
            },
        }
        df = pd.DataFrame(
            {"P_G": [1.0], "P_R": [0.0], "P_S": [0.0], "P_D": [0.0], "pi_SPY": [0.02]},
            index=[pd.Timestamp("2022-01-07")],
        )
        df.attrs["summary"] = summary_map[key]
        return df

    monkeypatch.setattr("regime.engine.run_signal_engine", fake_run_signal_engine)
    monkeypatch.setattr(
        "backtests.regime.engine.portfolio_sim.simulate_portfolio",
        lambda *args, **kwargs: SimpleNamespace(metrics=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "backtests.regime.analysis.assessment_validation.summarize_calibration_candidate",
        lambda signals, result: dict(signals.attrs["summary"]),
    )

    mutations_path = tmp_path / "base.json"
    mutations_path.write_text(json.dumps({"posterior_ema_alpha": 0.7}))
    args = SimpleNamespace(
        data_dir=str(tmp_path),
        equity=100_000.0,
        cost_bps=0.0,
        mutations_json=str(mutations_path),
        preset="recommended_full_stack",
    )

    cli.cmd_calibration_sweep(args)
    captured = capsys.readouterr().out
    output = json.loads(
        Path("backtests/regime/auto/output/calibration_sweep.json").read_text()
    )

    assert "overrides preset 'recommended_full_stack'" in captured
    assert output["baseline_source"]["type"] == "json"
    assert output["baseline_source"]["label"] == str(mutations_path)


def test_validate_2022_command_reports_deltas_and_pass(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "backtests.regime.data.downloader.load_cached_data",
        lambda data_dir: _dummy_cached_data(),
    )

    scenario_summaries = {
        "full_stack": {
            "first_jan_alert": "2022-01-14",
            "p_crisis_feb25": 0.60,
            "p_crisis_peak_feb": 0.75,
            "return_2022": 0.02,
            "max_dd_2022": 0.08,
            "max_dd_full_window": 0.09,
        },
        "scanner_off": {
            "first_jan_alert": None,
            "p_crisis_feb25": 0.40,
            "p_crisis_peak_feb": 0.45,
            "return_2022": 0.00,
            "max_dd_2022": 0.09,
            "max_dd_full_window": 0.11,
        },
        "crisis_off": {
            "first_jan_alert": "2022-01-14",
            "p_crisis_feb25": 0.55,
            "p_crisis_peak_feb": 0.70,
            "return_2022": 0.01,
            "max_dd_2022": 0.085,
            "max_dd_full_window": 0.10,
        },
        "r3_reference": {
            "first_jan_alert": None,
            "p_crisis_feb25": 0.35,
            "p_crisis_peak_feb": 0.40,
            "return_2022": -0.05,
            "max_dd_2022": 0.12,
            "max_dd_full_window": 0.12,
        },
    }

    def fake_run_signal_engine(*args, **kwargs):
        cfg = kwargs["cfg"]
        if round(cfg.posterior_temperature, 1) == 0.9:
            scenario = "r3_reference"
        elif not cfg.scanner_enabled:
            scenario = "scanner_off"
        elif not cfg.crisis_leverage_enabled:
            scenario = "crisis_off"
        else:
            scenario = "full_stack"
        df = pd.DataFrame(
            {"P_G": [1.0], "P_R": [0.0], "P_S": [0.0], "P_D": [0.0]},
            index=[pd.Timestamp("2022-01-07")],
        )
        df.attrs["scenario"] = scenario
        return df

    monkeypatch.setattr("regime.engine.run_signal_engine", fake_run_signal_engine)
    monkeypatch.setattr(
        "backtests.regime.engine.portfolio_sim.simulate_portfolio",
        lambda *args, **kwargs: SimpleNamespace(metrics=SimpleNamespace(max_drawdown_pct=0.1)),
    )
    monkeypatch.setattr(
        "backtests.regime.analysis.assessment_validation.summarize_2022_validation",
        lambda signals, result, scanner_threshold=0.5: dict(
            scenario_summaries[signals.attrs["scenario"]]
        ),
    )

    full_path = tmp_path / "full.json"
    full_path.write_text(json.dumps({"posterior_temperature": 1.8}))
    r3_path = tmp_path / "r3.json"
    r3_path.write_text(json.dumps({"posterior_temperature": 0.9}))
    args = SimpleNamespace(
        data_dir=str(tmp_path),
        equity=100_000.0,
        cost_bps=0.0,
        mutations_json=str(full_path),
        r3_mutations_json=str(r3_path),
    )

    cli.cmd_validate_2022(args)
    captured = capsys.readouterr().out

    assert "PASS: True" in captured
    assert "vs scanner_off" in captured
    assert "vs r3_reference" in captured
    assert Path("backtests/regime/auto/output/validate_2022.json").exists()


def test_validate_2022_defaults_to_presets_and_records_sources(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "backtests.regime.data.downloader.load_cached_data",
        lambda data_dir: _dummy_cached_data(),
    )

    scenario_summaries = {
        "full_stack": {
            "first_jan_alert": "2022-01-14",
            "p_crisis_feb25": 0.60,
            "p_crisis_peak_feb": 0.75,
            "return_2022": 0.02,
            "max_dd_2022": 0.08,
            "max_dd_full_window": 0.09,
        },
        "scanner_off": {
            "first_jan_alert": None,
            "p_crisis_feb25": 0.40,
            "p_crisis_peak_feb": 0.45,
            "return_2022": 0.00,
            "max_dd_2022": 0.09,
            "max_dd_full_window": 0.11,
        },
        "crisis_off": {
            "first_jan_alert": "2022-01-14",
            "p_crisis_feb25": 0.55,
            "p_crisis_peak_feb": 0.70,
            "return_2022": 0.01,
            "max_dd_2022": 0.085,
            "max_dd_full_window": 0.10,
        },
        "r3_reference": {
            "first_jan_alert": None,
            "p_crisis_feb25": 0.35,
            "p_crisis_peak_feb": 0.40,
            "return_2022": -0.05,
            "max_dd_2022": 0.12,
            "max_dd_full_window": 0.12,
        },
    }

    def fake_run_signal_engine(*args, **kwargs):
        cfg = kwargs["cfg"]
        if cfg.n_ensemble_models == 1:
            assert cfg.scanner_enabled is False
            assert tuple(cfg.crisis_weights) == (0.3, 0.6, 0.1)
            scenario = "r3_reference"
        elif not cfg.scanner_enabled:
            assert cfg.n_ensemble_models == 5
            assert tuple(cfg.crisis_weights) == (0.3, 0.6, 0.1, 0.1)
            scenario = "scanner_off"
        elif not cfg.crisis_leverage_enabled:
            assert cfg.n_ensemble_models == 5
            scenario = "crisis_off"
        else:
            assert cfg.scanner_enabled is True
            assert cfg.n_ensemble_models == 5
            assert cfg.crisis_z_window == 21
            assert tuple(cfg.crisis_weights) == (0.3, 0.6, 0.1, 0.1)
            scenario = "full_stack"
        df = pd.DataFrame(
            {"P_G": [1.0], "P_R": [0.0], "P_S": [0.0], "P_D": [0.0]},
            index=[pd.Timestamp("2022-01-07")],
        )
        df.attrs["scenario"] = scenario
        return df

    monkeypatch.setattr("regime.engine.run_signal_engine", fake_run_signal_engine)
    monkeypatch.setattr(
        "backtests.regime.engine.portfolio_sim.simulate_portfolio",
        lambda *args, **kwargs: SimpleNamespace(metrics=SimpleNamespace(max_drawdown_pct=0.1)),
    )
    monkeypatch.setattr(
        "backtests.regime.analysis.assessment_validation.summarize_2022_validation",
        lambda signals, result, scanner_threshold=0.5: dict(
            scenario_summaries[signals.attrs["scenario"]]
        ),
    )

    args = SimpleNamespace(
        data_dir=str(tmp_path),
        equity=100_000.0,
        cost_bps=0.0,
        mutations_json=None,
        r3_mutations_json=None,
        preset=None,
        r3_preset=None,
    )

    cli.cmd_validate_2022(args)
    captured = capsys.readouterr().out
    output = json.loads(
        Path("backtests/regime/auto/output/validate_2022.json").read_text()
    )

    assert "preset 'recommended_full_stack'" in captured
    assert "preset 'r3_reference'" in captured
    assert output["sources"]["full_stack"]["label"] == "recommended_full_stack"
    assert output["sources"]["r3_reference"]["label"] == "r3_reference"


def test_scanner_validate_defaults_to_recommended_preset(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "backtests.regime.data.downloader.load_cached_data",
        lambda data_dir: _dummy_cached_data(),
    )

    def fake_run_signal_engine(*args, **kwargs):
        cfg = kwargs["cfg"]
        assert cfg.scanner_enabled is True
        assert cfg.n_ensemble_models == 5
        assert cfg.crisis_z_window == 21
        assert tuple(cfg.crisis_weights) == (0.3, 0.6, 0.1, 0.1)
        assert cfg.posterior_ema_alpha == 0.8
        assert cfg.posterior_temperature == 1.0
        return pd.DataFrame(
            {
                "P_G": [1.0],
                "P_R": [0.0],
                "P_S": [0.0],
                "P_D": [0.0],
                "shift_prob": [0.7],
                "shift_dir": ["risk_off"],
            },
            index=[pd.Timestamp("2022-01-07")],
        )

    monkeypatch.setattr("regime.engine.run_signal_engine", fake_run_signal_engine)
    monkeypatch.setattr(
        "backtests.regime.analysis.scanner_validation.validate_scanner",
        lambda signals, threshold=0.5: {
            "transitions_analyzed": 1,
            "total_risk_off_alerts": 1,
            "jan_2022_first_week": "2022-01-07",
            "lead_times": [2.0],
            "verdict": "PASS",
            "criteria": {
                "lead_time": {"passed": True, "actual": "2.0 weeks", "target": "2-4 weeks"},
                "fpr": {"passed": True, "actual": "0.0%", "target": "<20%"},
                "jan_2022": {"passed": True, "actual": True, "target": True},
            },
        },
    )

    args = SimpleNamespace(
        data_dir=str(tmp_path),
        equity=100_000.0,
        cost_bps=0.0,
        mutations_json=None,
        preset=None,
        diagnostics=False,
    )

    cli.cmd_scanner_validate(args)
    captured = capsys.readouterr().out

    assert "preset 'recommended_full_stack'" in captured
    assert "VERDICT: PASS" in captured


def test_step9_optimize_command_uses_isolated_profile(monkeypatch, tmp_path, capsys):
    captured_args = {}

    def fake_phase_auto(args):
        captured_args["preset"] = args.preset
        captured_args["candidate_profile"] = args.candidate_profile
        captured_args["output_dir"] = args.output_dir
        captured_args["max_workers"] = args.max_workers
        captured_args["max_retries"] = args.max_retries
        captured_args["allow_extra_experiments"] = args.allow_extra_experiments
        captured_args["phase_sequence"] = args.phase_sequence
        captured_args["phase_max_rounds"] = args.phase_max_rounds

    monkeypatch.setattr(cli, "cmd_phase_auto", fake_phase_auto)

    args = SimpleNamespace(
        data_dir=str(tmp_path),
        equity=100_000.0,
        cost_bps=0.0,
        max_rounds=20,
        max_workers=3,
        min_delta=0.001,
        prune_threshold=-0.10,
        candidate_timeout=600.0,
        max_retries=0,
        output_dir=None,
    )

    cli.cmd_step9_optimize(args)
    captured = capsys.readouterr().out

    assert "Launching assessment Step 9 R6 optimization" in captured
    assert captured_args["preset"] == "step9_r6"
    assert captured_args["candidate_profile"] == "step9_r6"
    assert captured_args["max_workers"] == 3
    assert captured_args["max_retries"] == 0
    assert captured_args["allow_extra_experiments"] is False
    assert captured_args["phase_sequence"] == (1, 2, 3)
    assert captured_args["phase_max_rounds"] == {1: 20, 2: 1, 3: 1}
    assert captured_args["output_dir"].replace("\\", "/").endswith("backtests/regime/auto/output/step9_r6")


def test_phase_auto_strict_mode_ignores_suggested_experiments_and_skips_phase4(
    monkeypatch, tmp_path, capsys,
):
    calls = []

    def fake_run_phase_core(args, phase, extra_candidates=None):
        calls.append({"phase": phase, "extra_candidates": extra_candidates})
        analysis = SimpleNamespace(
            recommendation="expand_and_proceed",
            suggested_experiments=[("off_profile", {"stability_weight": 0.6})],
            recommendation_reason="test",
        )
        state = SimpleNamespace(
            completed_phases=list(range(1, phase + 1)),
            cumulative_mutations={"last_phase": phase},
        )
        result = SimpleNamespace()
        return result, analysis, state

    monkeypatch.setattr(cli, "_run_phase_core", fake_run_phase_core)

    args = SimpleNamespace(
        data_dir=str(tmp_path),
        equity=100_000.0,
        cost_bps=0.0,
        max_rounds=20,
        max_workers=3,
        min_delta=0.001,
        prune_threshold=-0.10,
        candidate_timeout=600.0,
        max_retries=0,
        mutations_json=None,
        preset="step9_r6",
        output_dir=str(tmp_path / "step9_strict"),
        allow_extra_experiments=False,
        phase_sequence=(1, 2, 3),
        phase_max_rounds={1: 20, 2: 1, 3: 1},
    )

    cli.cmd_phase_auto(args)
    captured = capsys.readouterr().out

    assert "Strict mode: analyzer suggested experiments are disabled" in captured
    assert [call["phase"] for call in calls] == [1, 2, 3]
    assert all(call["extra_candidates"] is None for call in calls)


def test_diagnostics_report_includes_new_assessment_sections():
    weekly_idx = pd.date_range("2022-01-07", periods=4, freq="W-FRI")
    signals = pd.DataFrame(
        {
            "P_G": [0.7, 0.2, 0.1, 0.6],
            "P_R": [0.1, 0.1, 0.1, 0.1],
            "P_S": [0.1, 0.3, 0.6, 0.2],
            "P_D": [0.1, 0.4, 0.2, 0.1],
            "Conf": [0.8, 0.6, 0.55, 0.75],
            "p_crisis": [0.2, 0.45, 0.7, 0.3],
            "crisis_severity": ["none", "elevated", "acute", "none"],
            "crisis_vix_component": [0.1, 0.2, 0.3, 0.1],
            "crisis_spread_component": [0.05, 0.06, 0.07, 0.04],
            "crisis_vol_component": [0.02, 0.03, 0.05, 0.02],
            "crisis_corr_component": [0.0, 0.05, 0.08, 0.01],
            "crisis_legacy_corr_component": [0.0, 0.0, 0.0, 0.0],
            "crisis_corr_trigger": [0.0, 0.2, 0.3, 0.1],
            "crisis_composite": [0.17, 0.34, 0.50, 0.17],
            "hmm_leverage": [1.0, 1.0, 1.0, 1.0],
            "crisis_leverage_adj": [1.0, 0.9, 0.7, 1.0],
            "scanner_leverage_adj": [1.0, 0.95, 0.85, 1.0],
            "final_leverage": [1.0, 0.855, 0.595, 1.0],
            "L": [1.0, 0.855, 0.595, 1.0],
            "consensus_ratio": [1.0, 0.8, 0.5, 0.9],
            "consensus_trend_4w": [0.0, -0.2, -0.3, 0.1],
            "avg_disagreement": [0.0, 0.05, 0.10, 0.02],
            "uncertainty_level": ["low", "moderate", "high", "low"],
            "minority_regime": ["G", "D", "D", "S"],
            "disagree_std_G": [0.0, 0.05, 0.10, 0.02],
            "disagree_std_R": [0.0, 0.01, 0.02, 0.01],
            "disagree_std_S": [0.0, 0.03, 0.04, 0.01],
            "disagree_std_D": [0.0, 0.04, 0.11, 0.02],
            "shift_prob": [0.2, 0.7, 0.8, 0.4],
            "shift_dir": ["neutral", "risk_off", "risk_off", "neutral"],
            "dominant_indicator": ["vix_momentum", "cross_asset_corr", "cross_asset_corr", "credit_spread_mom"],
            "pi_SPY": [0.30, 0.12, 0.05, 0.25],
            "pi_TLT": [0.20, 0.25, 0.20, 0.18],
            "pi_GLD": [0.10, 0.15, 0.18, 0.12],
            "pi_CASH": [0.40, 0.48, 0.57, 0.45],
            "w_SPY": [0.30, 0.14, 0.08, 0.25],
            "w_TLT": [0.20, 0.29, 0.34, 0.18],
            "w_GLD": [0.10, 0.18, 0.30, 0.12],
            "w_CASH": [0.40, 0.39, 0.28, 0.45],
        },
        index=weekly_idx,
    )
    daily_idx = pd.date_range("2022-01-03", periods=30, freq="B")
    equity = pd.Series(range(100, 130), index=daily_idx, dtype=float)
    daily_returns = pd.Series(0.001, index=daily_idx)
    metrics = PortfolioMetrics(
        total_return=0.29,
        cagr=0.20,
        sharpe=1.5,
        sortino=2.0,
        calmar=1.8,
        max_drawdown_pct=0.08,
        max_drawdown_duration=15,
        avg_annual_turnover=4.0,
        n_rebalances=4,
    )
    result = SimpleNamespace(metrics=metrics, equity_curve=equity, daily_returns=daily_returns)
    benchmark = SimpleNamespace(metrics=metrics, equity_curve=equity, daily_returns=daily_returns)
    score = CompositeScore(0.8, 0.7, 0.6, 0.5, 0.7, total=0.71)
    sim_cfg = SimpleNamespace(initial_equity=100_000)

    report = generate_regime_diagnostics_report(
        signals=signals,
        result=result,
        benchmark=benchmark,
        score=score,
        sim_cfg=sim_cfg,
    )

    assert "CRISIS CHANNEL DECOMPOSITION" in report
    assert "ENSEMBLE DISAGREEMENT" in report


def test_validate_scanner_uses_alert_episodes_and_explicit_threshold():
    idx = pd.date_range("2021-12-17", periods=9, freq="W-FRI")
    signals = pd.DataFrame(
        {
            "P_G": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.1],
            "P_R": [0.1] * 9,
            "P_S": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7],
            "P_D": [0.05] * 9,
            "shift_prob": [0.20, 0.30, 0.40, 0.50, 0.62, 0.68, 0.40, 0.30, 0.20],
            "shift_dir": [
                "neutral",
                "neutral",
                "neutral",
                "neutral",
                "risk_off",
                "risk_off",
                "neutral",
                "neutral",
                "neutral",
            ],
        },
        index=idx,
    )

    results = scanner_validation.validate_scanner(signals, threshold=0.6)

    assert results["total_risk_off_alerts"] == 1
    assert results["jan_2022_detected"] is True
    assert results["jan_2022_first_week"] == "2022-01-14"
    assert results["lead_time_median_weeks"] == 4.0


def test_scanner_lead_time_uses_first_alert_episode_not_last():
    """With multiple alert episodes before a transition, lead time should measure
    from the FIRST episode start (earliest warning), not the last."""
    idx = pd.date_range("2021-12-03", periods=10, freq="W-FRI")
    # idx[0]=Dec 3, idx[1]=Dec 10, ..., idx[3]=Dec 24, idx[4]=Dec 31,
    # idx[5]=Jan 7, idx[6]=Jan 14, idx[7]=Jan 21, idx[8]=Jan 28, idx[9]=Feb 4
    signals = pd.DataFrame(
        {
            "P_G": [0.8] * 8 + [0.1, 0.1],
            "P_R": [0.1] * 10,
            "P_S": [0.05] * 8 + [0.7, 0.75],
            "P_D": [0.05] * 10,
            # Two alert episodes: early (weeks 1-2) and late (weeks 5-6)
            "shift_prob": [0.20, 0.65, 0.70, 0.30, 0.20, 0.62, 0.68, 0.30, 0.20, 0.20],
            "shift_dir": [
                "neutral", "risk_off", "risk_off",  # early episode at idx[1]=Dec 10
                "neutral", "neutral",
                "risk_off", "risk_off",              # late episode at idx[5]=Jan 7
                "neutral", "neutral", "neutral",
            ],
        },
        index=idx,
    )

    results = scanner_validation.validate_scanner(signals, threshold=0.6)

    # Transition G->S at idx[8]=Jan 28. Both episodes within 8-week lookahead.
    #   episode 1 starts at idx[1]=Dec 10: lead = (Jan 28 - Dec 10) / 7 = 7 weeks
    #   episode 2 starts at idx[5]=Jan 7:  lead = (Jan 28 - Jan 7) / 7  = 3 weeks
    # With [0] (first), lead = 7 weeks. With [-1] (last), lead = 3 weeks.
    assert results["transitions_analyzed"] >= 1
    assert len(results["lead_times"]) >= 1
    assert results["lead_times"][0] > 5.0, "Should use first episode, giving >5 weeks lead time"


def test_step7_validation_requires_better_2022_and_full_window_drawdown():
    verdict = assessment_validation.validate_step7_outcome(
        full_stack={
            "first_jan_alert": "2022-01-14",
            "p_crisis_feb25": 0.6,
            "return_2022": 0.02,
            "max_dd_2022": 0.09,
            "max_dd_full_window": 0.08,
        },
        scanner_off={
            "first_jan_alert": None,
            "p_crisis_feb25": 0.4,
            "return_2022": 0.0,
            "max_dd_2022": 0.08,
            "max_dd_full_window": 0.10,
        },
        crisis_off={
            "first_jan_alert": "2022-01-14",
            "p_crisis_feb25": 0.5,
            "return_2022": 0.01,
            "max_dd_2022": 0.10,
            "max_dd_full_window": 0.09,
        },
        r3_reference={
            "first_jan_alert": None,
            "p_crisis_feb25": 0.35,
            "return_2022": -0.05,
            "max_dd_2022": 0.12,
            "max_dd_full_window": 0.12,
        },
    )

    assert verdict["dd_2022_ok"] is False
    assert verdict["dd_full_window_ok"] is True
    assert verdict["dd_ok"] is False
    assert verdict["passed"] is False
