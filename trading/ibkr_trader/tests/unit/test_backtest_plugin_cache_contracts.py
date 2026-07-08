from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.plugin_utils import mutation_signature
from backtests.shared.auto.types import ScoredCandidate
from backtests.stock.auto.alcb import worker as alcb_worker_mod
from backtests.stock.auto.iaric import worker as iaric_worker_mod
from backtests.swing.auto.atrss import plugin as atrss_plugin_mod
from backtests.swing.auto.helix import plugin as helix_plugin_mod
from backtests.momentum.auto.downturn import plugin as downturn_plugin_mod
from backtests.momentum.auto.downturn import worker as downturn_worker_mod
from backtests.momentum.auto.nqdtc import plugin as nqdtc_plugin_mod
from backtests.momentum.auto.nqdtc import worker as nqdtc_worker_mod
from backtests.momentum.auto.vdubus import plugin as vdub_plugin_mod
from backtests.momentum.auto.vdubus import worker as vdub_worker_mod
from backtests.stock.auto.alcb import plugin as alcb_plugin_mod
from backtests.stock.auto.iaric import plugin as iaric_plugin_mod


class _DummyBatchEvaluator:
    def __call__(self, candidates, current_mutations):
        del candidates, current_mutations
        return []

    def close(self) -> None:
        return None


def test_alcb_phase_batch_seeds_metrics_cache_with_raw_mutation_signature(monkeypatch, tmp_path: Path) -> None:
    plugin = alcb_plugin_mod.ALCBP16Plugin(tmp_path, max_workers=1)
    mutations = {"param_overrides.example": 1.0}
    metrics = {
        "net_profit": 1.0,
        "expectancy_dollar": 1.0,
        "expected_total_r": 1.0,
        "trades_per_month": 1.0,
        "profit_factor": 1.0,
        "max_drawdown_pct": 0.01,
    }

    monkeypatch.setattr(alcb_plugin_mod, "_LocalBatchEvaluator", lambda *args, **kwargs: _DummyBatchEvaluator())
    monkeypatch.setattr(plugin, "_run_config", lambda muts, **kwargs: {"metrics": dict(metrics)})
    monkeypatch.setattr(plugin, "_resolve_phase_hard_rejects", lambda phase, base_metrics, hard_rejects: dict(hard_rejects))
    monkeypatch.setattr(
        plugin,
        "_seed_result_for_metrics",
        lambda name, phase, base_metrics, hard_rejects, scoring_weights: ScoredCandidate(
            name=name,
            score=1.0,
            metrics=dict(base_metrics),
        ),
    )
    monkeypatch.setattr(plugin, "_replay_data_fingerprint", lambda: "fingerprint")

    plugin.create_evaluate_batch(1, mutations)

    assert plugin._metrics_cache == {mutation_signature(mutations): metrics}


def test_iaric_phase_batch_seeds_metrics_cache_with_raw_mutation_signature(monkeypatch, tmp_path: Path) -> None:
    plugin = iaric_plugin_mod.IARICPullbackPlugin(tmp_path, max_workers=1)
    mutations = {"param_overrides.example": 1.0}
    metrics = {
        "total_trades": 12.0,
        "avg_r": 0.2,
        "profit_factor": 2.0,
        "max_drawdown_pct": 0.01,
        "sharpe": 1.0,
    }

    monkeypatch.setattr(iaric_plugin_mod, "_LocalBatchEvaluator", lambda *args, **kwargs: _DummyBatchEvaluator())
    monkeypatch.setattr(plugin, "_run_config", lambda muts, **kwargs: {"metrics": dict(metrics)})
    monkeypatch.setattr(plugin, "_replay_data_fingerprint", lambda: "fingerprint")

    plugin.create_evaluate_batch(1, mutations)

    assert plugin._metrics_cache == {mutation_signature(mutations): metrics}


def test_atrss_phase_batch_namespaces_candidate_cache_by_source_fingerprint(monkeypatch, tmp_path: Path) -> None:
    plugin = atrss_plugin_mod.ATRSSPlugin(tmp_path, max_workers=1)
    monkeypatch.setattr(
        plugin,
        "_ensure_bundle",
        lambda: type("Bundle", (), {"cache_source_fingerprint": "atrss-fp"})(),
    )

    evaluator = plugin.create_evaluate_batch(3, {})

    assert evaluator._signature_prefix == build_cache_key(
        "swing.atrss.evaluation",
        source_fingerprint="atrss-fp",
        extra={"phase": 3, "scoring_profile": "r1_independent", "scoring_weights": {}, "hard_rejects": {}},
    )


def test_helix_phase_batch_namespaces_candidate_cache_by_source_fingerprint(monkeypatch, tmp_path: Path) -> None:
    plugin = helix_plugin_mod.HelixPlugin(tmp_path, max_workers=1)
    base_metrics = {
        "total_trades": 300,
        "profit_factor": 1.5,
        "net_return_pct": 50.0,
        "max_r_dd": 10.0,
        "exit_efficiency": 0.20,
        "waste_ratio": 0.60,
        "tail_pct": 0.50,
        "min_regime_pf": 1.0,
    }
    monkeypatch.setattr(
        plugin,
        "_replay_bundle",
        lambda: type("Bundle", (), {"cache_source_fingerprint": "helix-fp"})(),
    )
    monkeypatch.setattr(plugin, "compute_final_metrics", lambda muts: dict(base_metrics))

    evaluator = plugin.create_evaluate_batch(1, {})
    resolved_hard_rejects = plugin._resolve_phase_hard_rejects(
        1,
        base_metrics,
        helix_plugin_mod.PHASE_HARD_REJECTS[1],
    )

    assert evaluator._signature_prefix == build_cache_key(
        "swing.helix.evaluation",
        source_fingerprint="helix-fp",
        extra={
            "phase": 1,
            "scoring_weights": {},
            "hard_rejects": resolved_hard_rejects,
            "initial_equity": plugin.initial_equity,
            "start_date": plugin.start_date,
            "end_date": plugin.end_date,
        },
    )


def test_helix_cached_metrics_refresh_diagnostic_context(monkeypatch, tmp_path: Path) -> None:
    from backtests.swing.auto.helix import scoring as helix_scoring_mod
    from backtests.swing.engine import helix_portfolio_engine as helix_engine_mod

    plugin = helix_plugin_mod.HelixPlugin(tmp_path, max_workers=1)
    mutations = {"param_overrides.EXAMPLE": 1.0}
    sig = mutation_signature(mutations)
    cached_metrics = {
        "total_trades": 1,
        "profit_factor": 1.5,
        "net_return_pct": 10.0,
        "max_r_dd": 1.0,
        "exit_efficiency": 0.2,
        "waste_ratio": 0.6,
        "tail_pct": 0.5,
        "min_regime_pf": 1.0,
    }
    plugin._metrics_cache[sig] = dict(cached_metrics)
    plugin._last_context = {"mutation_signature": "stale", "all_trades": ["old_trade"]}

    fresh_trade = object()
    fresh_result = SimpleNamespace(
        symbol_results={"QQQ": SimpleNamespace(trades=[fresh_trade])},
    )
    calls = {"run": 0}

    def fake_run(data, config):
        del data, config
        calls["run"] += 1
        return fresh_result

    monkeypatch.setattr(
        plugin,
        "_replay_bundle",
        lambda: SimpleNamespace(data={"QQQ": object()}),
    )
    monkeypatch.setattr(helix_engine_mod, "run_helix_independent", fake_run)
    monkeypatch.setattr(
        helix_scoring_mod,
        "extract_helix_metrics",
        lambda result, equity: helix_scoring_mod.HelixMetrics(**cached_metrics),
    )

    metrics = plugin.compute_final_metrics(mutations)

    assert {key: metrics[key] for key in cached_metrics} == cached_metrics
    assert calls["run"] == 1
    assert plugin._last_context["mutation_signature"] == sig
    assert plugin._last_context["all_trades"] == [fresh_trade]

    plugin.compute_final_metrics(mutations)

    assert calls["run"] == 1


def test_vdub_phase_batch_namespaces_candidate_cache_by_source_fingerprint(monkeypatch, tmp_path: Path) -> None:
    plugin = vdub_plugin_mod.VdubusPlugin(tmp_path, max_workers=1)
    monkeypatch.setattr(
        plugin,
        "_replay_bundle",
        lambda: type("Bundle", (), {"cache_source_fingerprint": "vdub-fp"})(),
    )

    evaluator = plugin.create_evaluate_batch(4, {})

    assert evaluator._signature_prefix == build_cache_key(
        "momentum.vdub.evaluation",
        source_fingerprint="vdub-fp",
        extra={"phase": 4, "scoring_weights": {}, "hard_rejects": {}},
    )


def test_downturn_phase_batch_namespaces_candidate_cache_by_source_fingerprint(monkeypatch, tmp_path: Path) -> None:
    plugin = downturn_plugin_mod.DownturnPlugin(tmp_path, max_workers=1)
    monkeypatch.setattr(
        plugin,
        "_replay_bundle",
        lambda: type("Bundle", (), {"cache_source_fingerprint": "downturn-fp"})(),
    )

    evaluator = plugin.create_evaluate_batch(2, {})

    assert evaluator._signature_prefix == build_cache_key(
        "downturn.evaluation",
        source_fingerprint="downturn-fp",
        extra={"phase": 2, "scoring_weights": {}, "hard_rejects": {}},
    )


def test_nqdtc_phase_batch_namespaces_candidate_cache_by_source_fingerprint(monkeypatch, tmp_path: Path) -> None:
    plugin = nqdtc_plugin_mod.NQDTCPlugin(tmp_path, max_workers=1)
    monkeypatch.setattr(
        plugin,
        "_replay_bundle",
        lambda: type("Bundle", (), {"cache_source_fingerprint": "nqdtc-fp"})(),
    )

    evaluator = plugin.create_evaluate_batch(3, {})

    assert evaluator._signature_prefix == build_cache_key(
        "nqdtc.evaluation",
        source_fingerprint="nqdtc-fp",
        extra={"phase": 3, "scoring_weights": {}, "hard_rejects": {}},
    )


def test_alcb_worker_init_uses_shared_stock_replay_bundle(monkeypatch, tmp_path: Path) -> None:
    from backtests.stock import config_alcb as config_mod
    from backtests.stock.data import replay_cache as replay_cache_mod

    replay = object()

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        replay_cache_mod,
        "load_research_replay_bundle",
        lambda data_dir: type("Bundle", (), {"data": replay})(),
    )
    monkeypatch.setattr(config_mod, "ALCBBacktestConfig", DummyConfig)

    alcb_worker_mod.init_worker(
        str(tmp_path),
        "2024-01-01",
        "2024-12-31",
        10_000.0,
    )

    assert alcb_worker_mod._worker_replay is replay


def test_iaric_worker_init_uses_shared_stock_replay_bundle(monkeypatch, tmp_path: Path) -> None:
    from backtests.stock import config_iaric as config_mod
    from backtests.stock.data import replay_cache as replay_cache_mod

    replay = object()

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        replay_cache_mod,
        "load_research_replay_bundle",
        lambda data_dir: type("Bundle", (), {"data": replay})(),
    )
    monkeypatch.setattr(config_mod, "IARICBacktestConfig", DummyConfig)

    iaric_worker_mod.init_worker(
        str(tmp_path),
        "2024-01-01",
        "2024-12-31",
        10_000.0,
    )

    assert iaric_worker_mod._worker_replay is replay


def test_vdub_plugin_refresh_clears_cached_metrics_and_pool(monkeypatch, tmp_path: Path) -> None:
    from backtests.momentum.data import replay_cache as replay_cache_mod

    plugin = vdub_plugin_mod.VdubusPlugin(tmp_path, max_workers=1)
    plugin._metrics_cache["sig"] = {"score": 1.0}
    plugin._evaluation_cache["candidate"] = object()
    plugin._last_context = {"trades": [1]}
    plugin._last_metrics_sig = "sig"
    plugin._last_metrics_result = {"score": 1.0}
    plugin._pool = object()

    bundles = iter(
        [
            type("Bundle", (), {"cache_source_fingerprint": "fp-a", "data": {"root": "a"}})(),
            type("Bundle", (), {"cache_source_fingerprint": "fp-b", "data": {"root": "b"}})(),
        ]
    )
    close_calls = {"count": 0}

    monkeypatch.setattr(replay_cache_mod, "load_vdub_replay_bundle", lambda *args, **kwargs: next(bundles))
    monkeypatch.setattr(plugin, "close_pool", lambda: close_calls.__setitem__("count", close_calls["count"] + 1))

    first = plugin._replay_bundle()
    second = plugin._replay_bundle()

    assert first.cache_source_fingerprint == "fp-a"
    assert second.cache_source_fingerprint == "fp-b"
    assert plugin._metrics_cache == {}
    assert plugin._evaluation_cache == {}
    assert plugin._last_context == {}
    assert plugin._last_metrics_sig == ""
    assert plugin._last_metrics_result is None
    assert close_calls["count"] == 2


def test_downturn_plugin_refresh_clears_cached_metrics_and_pool(monkeypatch, tmp_path: Path) -> None:
    from backtests.momentum.auto.downturn import worker as worker_mod

    plugin = downturn_plugin_mod.DownturnPlugin(tmp_path, max_workers=1)
    plugin._metrics_cache["sig"] = {"score": 1.0}
    plugin._evaluation_cache["candidate"] = object()
    plugin._final_metrics_cache["final"] = {"metrics": {"score": 1.0}}
    plugin._last_context = {"trades": [1]}
    plugin._pool = object()

    bundles = iter(
        [
            type("Bundle", (), {"cache_source_fingerprint": "fp-a", "data": {"root": "a"}})(),
            type("Bundle", (), {"cache_source_fingerprint": "fp-b", "data": {"root": "b"}})(),
        ]
    )
    close_calls = {"count": 0}

    monkeypatch.setattr(worker_mod, "load_worker_data", lambda *args, **kwargs: next(bundles))
    monkeypatch.setattr(plugin, "close_pool", lambda: close_calls.__setitem__("count", close_calls["count"] + 1))

    first = plugin._replay_bundle()
    second = plugin._replay_bundle()

    assert first.cache_source_fingerprint == "fp-a"
    assert second.cache_source_fingerprint == "fp-b"
    assert plugin._metrics_cache == {}
    assert plugin._evaluation_cache == {}
    assert plugin._final_metrics_cache == {}
    assert plugin._last_context == {}
    assert close_calls["count"] == 2


def test_nqdtc_plugin_refresh_clears_cached_metrics_and_pool(monkeypatch, tmp_path: Path) -> None:
    from backtests.momentum.auto.nqdtc import worker as worker_mod

    plugin = nqdtc_plugin_mod.NQDTCPlugin(tmp_path, max_workers=1)
    plugin._metrics_cache["sig"] = {"score": 1.0}
    plugin._evaluation_cache["candidate"] = object()
    plugin._final_metrics_cache["final"] = {"metrics": {"score": 1.0}}
    plugin._last_context = {"trades": [1]}
    plugin._pool = object()

    bundles = iter(
        [
            type("Bundle", (), {"cache_source_fingerprint": "fp-a", "data": {"root": "a"}})(),
            type("Bundle", (), {"cache_source_fingerprint": "fp-b", "data": {"root": "b"}})(),
        ]
    )
    close_calls = {"count": 0}

    monkeypatch.setattr(worker_mod, "load_worker_data", lambda *args, **kwargs: next(bundles))
    monkeypatch.setattr(plugin, "close_pool", lambda: close_calls.__setitem__("count", close_calls["count"] + 1))

    first = plugin._replay_bundle()
    second = plugin._replay_bundle()

    assert first.cache_source_fingerprint == "fp-a"
    assert second.cache_source_fingerprint == "fp-b"
    assert plugin._metrics_cache == {}
    assert plugin._evaluation_cache == {}
    assert plugin._final_metrics_cache == {}
    assert plugin._last_context == {}
    assert close_calls["count"] == 2


def test_downturn_worker_reloads_replay_bundle_when_data_dir_changes(monkeypatch, tmp_path: Path) -> None:
    from backtests.momentum import config_downturn as config_mod

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    seen_paths: list[Path] = []

    def fake_load_worker_data(symbol: str, data_dir: Path):
        seen_paths.append(Path(data_dir))
        return {"symbol": symbol, "root": str(data_dir)}

    monkeypatch.setattr(config_mod, "DownturnBacktestConfig", DummyConfig)
    monkeypatch.setattr(downturn_worker_mod, "load_worker_data", fake_load_worker_data)
    monkeypatch.setattr(downturn_worker_mod, "_worker_data", None)
    monkeypatch.setattr(downturn_worker_mod, "_worker_config", None)
    monkeypatch.setattr(downturn_worker_mod, "_worker_data_dir_key", None)

    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()

    downturn_worker_mod.init_worker(str(first_dir), 100_000.0)
    downturn_worker_mod.init_worker(str(first_dir), 100_000.0)
    downturn_worker_mod.init_worker(str(second_dir), 100_000.0)

    assert seen_paths == [first_dir, second_dir]


def test_nqdtc_worker_reloads_replay_bundle_when_data_dir_changes(monkeypatch, tmp_path: Path) -> None:
    from backtests.momentum import config_nqdtc as config_mod

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    seen_paths: list[Path] = []

    def fake_load_worker_data(symbol: str, data_dir: Path):
        seen_paths.append(Path(data_dir))
        return type("Bundle", (), {"data": {"symbol": symbol, "root": str(data_dir)}})()

    monkeypatch.setattr(config_mod, "NQDTCBacktestConfig", DummyConfig)
    monkeypatch.setattr(nqdtc_worker_mod, "load_worker_data", fake_load_worker_data)
    monkeypatch.setattr(nqdtc_worker_mod, "_worker_data", None)
    monkeypatch.setattr(nqdtc_worker_mod, "_worker_config", None)
    monkeypatch.setattr(nqdtc_worker_mod, "_worker_data_dir_key", None)

    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()

    nqdtc_worker_mod.init_worker(str(first_dir), 10_000.0)
    nqdtc_worker_mod.init_worker(str(first_dir), 10_000.0)
    nqdtc_worker_mod.init_worker(str(second_dir), 10_000.0)

    assert seen_paths == [first_dir, second_dir]


def test_vdub_plugin_replay_bundle_includes_five_min_surface(monkeypatch, tmp_path: Path) -> None:
    from backtests.momentum.data import replay_cache as replay_cache_mod

    plugin = vdub_plugin_mod.VdubusPlugin(tmp_path, max_workers=1)
    seen: dict[str, object] = {}

    def fake_load(symbol, data_dir, *, include_5m=False):
        seen["symbol"] = symbol
        seen["data_dir"] = Path(data_dir)
        seen["include_5m"] = include_5m
        return type("Bundle", (), {"cache_source_fingerprint": "vdub-fp", "data": {"bars_5m": object()}})()

    monkeypatch.setattr(replay_cache_mod, "load_vdub_replay_bundle", fake_load)

    bundle = plugin._replay_bundle()

    assert bundle.cache_source_fingerprint == "vdub-fp"
    assert seen == {
        "symbol": "NQ",
        "data_dir": tmp_path,
        "include_5m": True,
    }


def test_vdub_worker_init_uses_shared_replay_bundle_with_five_min_surface(monkeypatch, tmp_path: Path) -> None:
    from backtests.momentum import config_vdubus as config_mod

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    seen: dict[str, object] = {}

    def fake_load_worker_data(symbol: str, data_dir: Path):
        seen["symbol"] = symbol
        seen["data_dir"] = Path(data_dir)
        return type(
            "Bundle",
            (),
            {
                "data": {"bars_15m": object(), "bars_5m": object(), "five_to_15_idx_map": object()},
                "cache_key": "vdub-key",
                "cache_source_fingerprint": "vdub-fp",
            },
        )()

    monkeypatch.setattr(config_mod, "VdubusBacktestConfig", DummyConfig)
    monkeypatch.setattr(vdub_worker_mod, "load_worker_data", fake_load_worker_data)
    monkeypatch.setattr(vdub_worker_mod, "_worker_data", None)
    monkeypatch.setattr(vdub_worker_mod, "_worker_config", None)
    monkeypatch.setattr(vdub_worker_mod, "_worker_data_dir_key", None)

    vdub_worker_mod.init_worker(str(tmp_path), 10_000.0)

    assert seen == {
        "symbol": "NQ",
        "data_dir": tmp_path,
    }


def test_vdub_worker_reloads_replay_bundle_when_data_dir_changes(monkeypatch, tmp_path: Path) -> None:
    from backtests.momentum import config_vdubus as config_mod

    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    seen_paths: list[Path] = []

    def fake_load_worker_data(symbol: str, data_dir: Path):
        seen_paths.append(Path(data_dir))
        return type(
            "Bundle",
            (),
            {
                "data": {"symbol": symbol, "root": str(data_dir), "bars_5m": object(), "five_to_15_idx_map": object()},
                "cache_key": "vdub-key",
                "cache_source_fingerprint": str(data_dir),
            },
        )()

    monkeypatch.setattr(config_mod, "VdubusBacktestConfig", DummyConfig)
    monkeypatch.setattr(vdub_worker_mod, "load_worker_data", fake_load_worker_data)
    monkeypatch.setattr(vdub_worker_mod, "_worker_data", None)
    monkeypatch.setattr(vdub_worker_mod, "_worker_config", None)
    monkeypatch.setattr(vdub_worker_mod, "_worker_data_dir_key", None)

    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()

    vdub_worker_mod.init_worker(str(first_dir), 10_000.0)
    vdub_worker_mod.init_worker(str(first_dir), 10_000.0)
    vdub_worker_mod.init_worker(str(second_dir), 10_000.0)

    assert seen_paths == [first_dir, second_dir]


def test_alcb_plugin_refresh_clears_context_and_shared_pool(monkeypatch, tmp_path: Path) -> None:
    from backtests.stock.data import replay_cache as replay_cache_mod

    plugin = alcb_plugin_mod.ALCBP16Plugin(tmp_path, max_workers=1)
    plugin._metrics_cache["sig"] = {"score": 1.0}
    plugin._config_cache[("key",)] = {"metrics": {}}
    plugin._evaluation_cache["candidate"] = ScoredCandidate(name="x", score=1.0)
    plugin._last_context = {"metrics": {"net_profit": 1.0}}
    plugin._phase_runtime_context[1] = {"base_metrics": {"net_profit": 1.0}}
    plugin._shared_pool = object()

    bundles = iter(
        [
            type("Bundle", (), {"cache_source_fingerprint": "fp-a", "data": object()})(),
            type("Bundle", (), {"cache_source_fingerprint": "fp-b", "data": object()})(),
        ]
    )
    close_calls = {"count": 0}

    monkeypatch.setattr(replay_cache_mod, "load_research_replay_bundle", lambda *args, **kwargs: next(bundles))
    monkeypatch.setattr(plugin, "close_pool", lambda: close_calls.__setitem__("count", close_calls["count"] + 1))

    first = plugin._replay_bundle()
    second = plugin._replay_bundle()

    assert first.cache_source_fingerprint == "fp-a"
    assert second.cache_source_fingerprint == "fp-b"
    assert plugin._metrics_cache == {}
    assert plugin._config_cache == {}
    assert plugin._evaluation_cache == {}
    assert plugin._last_context == {}
    assert plugin._phase_runtime_context == {}
    assert close_calls["count"] == 2
