from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtests.momentum.data.replay_cache import (
    load_replay_bundle,
    load_vdub_replay_bundle,
    replay_engine_kwargs,
)
from backtests.shared.auto.replay_bundle import ReplayBundle
from backtests.shared.auto.cache_keys import (
    build_cache_key,
    fingerprint_paths,
    mutation_subset,
    stable_signature,
)
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.data.replay_cache import (
    load_atrss_replay_bundle,
    load_unified_portfolio_replay_bundle,
)
from backtests.stock.data.replay_cache import load_research_replay_bundle
from backtests.stock.engine.research_replay import ResearchReplayEngine

UTC = timezone.utc


def test_stable_signature_is_order_insensitive_for_json_like_payloads() -> None:
    left = {"path": Path("foo/bar"), "when": datetime(2026, 4, 25, 12, 0, tzinfo=UTC), "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "when": datetime(2026, 4, 25, 12, 0, tzinfo=UTC), "path": Path("foo/bar")}

    assert stable_signature(left) == stable_signature(right)


def test_mutation_subset_extracts_exact_keys_and_prefixes() -> None:
    mutations = {
        "param_overrides.tp1": 1.5,
        "param_overrides.tp2": 2.0,
        "flags.use_filter": True,
        "risk.base_pct": 0.01,
    }

    subset = mutation_subset(
        mutations,
        exact_keys=("flags.use_filter",),
        prefixes=("param_overrides",),
    )

    assert subset == {
        "flags.use_filter": True,
        "param_overrides.tp1": 1.5,
        "param_overrides.tp2": 2.0,
    }


def test_build_cache_key_changes_when_source_fingerprint_changes() -> None:
    mutations = {"param_overrides.tp1": 1.5}

    key_a = build_cache_key("demo", source_fingerprint="abc", mutations=mutations)
    key_b = build_cache_key("demo", source_fingerprint="def", mutations=mutations)

    assert key_a != key_b


def test_load_replay_bundle_reuses_cached_preprocessing(monkeypatch, tmp_path) -> None:
    from backtests.momentum.data import cache as cache_mod
    from backtests.momentum.data import preprocessing as prep_mod

    five_min_path = tmp_path / "NQ_5m.parquet"
    daily_path = tmp_path / "NQ_1d.parquet"
    es_path = tmp_path / "ES_1d.parquet"
    for path in (five_min_path, daily_path, es_path):
        path.write_text("stub", encoding="utf-8")

    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100.0, 200.0],
        },
        index=pd.DatetimeIndex(
            [
                datetime(2026, 4, 25, 9, 30, tzinfo=UTC),
                datetime(2026, 4, 25, 9, 35, tzinfo=UTC),
            ]
        ),
    )

    calls = {"load_bars": 0, "build_numpy_arrays": 0}

    def fake_load_bars(_path: Path) -> pd.DataFrame:
        calls["load_bars"] += 1
        return frame.copy()

    def fake_build_numpy_arrays(df: pd.DataFrame) -> dict[str, int]:
        calls["build_numpy_arrays"] += 1
        return {"rows": len(df)}

    monkeypatch.setattr(cache_mod, "load_bars", fake_load_bars)
    monkeypatch.setattr(prep_mod, "normalize_timezone", lambda df: df)
    monkeypatch.setattr(prep_mod, "filter_eth", lambda df: df)
    monkeypatch.setattr(prep_mod, "resample_5m_to_15m", lambda df: df.iloc[:1])
    monkeypatch.setattr(prep_mod, "resample_5m_to_30m", lambda df: df.iloc[:1])
    monkeypatch.setattr(prep_mod, "resample_5m_to_1h", lambda df: df.iloc[:1])
    monkeypatch.setattr(prep_mod, "resample_5m_to_4h", lambda df: df.iloc[:1])
    monkeypatch.setattr(prep_mod, "resample_5m_to_daily", lambda df: df.iloc[:1])
    monkeypatch.setattr(prep_mod, "build_numpy_arrays", fake_build_numpy_arrays)
    monkeypatch.setattr(prep_mod, "align_higher_tf_to_5m", lambda *_args: ("htf",))
    monkeypatch.setattr(prep_mod, "align_daily_to_5m", lambda *_args: ("daily",))

    first = load_replay_bundle("NQ", tmp_path, include_fifteen_min=True)
    second = load_replay_bundle("NQ", tmp_path, include_fifteen_min=True)
    third = load_replay_bundle("NQ", tmp_path, include_fifteen_min=False)

    assert first is second
    assert third is not first
    assert first.cache_source_fingerprint == fingerprint_paths(
        [five_min_path, daily_path, es_path],
        root=tmp_path,
    )
    assert "cache_key" not in first.data
    assert "cache_source_fingerprint" not in first.data
    assert calls["load_bars"] == 6
    assert calls["build_numpy_arrays"] == 13


def test_replay_engine_kwargs_strips_cache_metadata() -> None:
    bundle = {
        "five_min": {"rows": 10},
        "daily": {"rows": 5},
        "cache_key": "abc",
        "cache_source_fingerprint": "fp",
    }

    assert replay_engine_kwargs(bundle) == {
        "five_min": {"rows": 10},
        "daily": {"rows": 5},
    }

    typed_bundle = ReplayBundle(
        data={"five_min": {"rows": 10}, "daily": {"rows": 5}},
        cache_key="abc",
        cache_source_fingerprint="fp",
    )

    assert replay_engine_kwargs(typed_bundle) == {
        "five_min": {"rows": 10},
        "daily": {"rows": 5},
    }


def test_load_vdub_replay_bundle_reuses_cached_preprocessing(monkeypatch, tmp_path) -> None:
    from backtests.momentum import cli as momentum_cli

    fifteen_min_path = tmp_path / "NQ_15m.parquet"
    es_daily_path = tmp_path / "ES_1d.parquet"
    for path in (fifteen_min_path, es_daily_path):
        path.write_text("stub", encoding="utf-8")

    calls = {"load": 0}

    def fake_load(_symbol: str, _data_dir: Path, include_5m: bool = False) -> dict[str, object]:
        calls["load"] += 1
        return {
            "bars_15m": {"rows": 10},
            "hourly": {"rows": 5},
            "daily_es": {"rows": 2},
            "hourly_idx_map": (0, 1),
            "daily_es_idx_map": (0, 1),
            "bars_5m": {"rows": 0} if include_5m else None,
            "five_to_15_idx_map": None,
        }

    monkeypatch.setattr(momentum_cli, "_load_vdubus_data", fake_load)

    first = load_vdub_replay_bundle("NQ", tmp_path)
    second = load_vdub_replay_bundle("NQ", tmp_path)

    assert first is second
    assert first.cache_source_fingerprint == fingerprint_paths([fifteen_min_path, es_daily_path], root=tmp_path)
    assert calls["load"] == 1


def test_load_vdub_replay_bundle_does_not_alias_identical_roots(monkeypatch, tmp_path) -> None:
    from backtests.momentum import cli as momentum_cli

    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for base_dir in (left, right):
        (base_dir / "NQ_15m.parquet").write_text("stub", encoding="utf-8")
        (base_dir / "ES_1d.parquet").write_text("stub", encoding="utf-8")

    def fake_load(_symbol: str, data_dir: Path, include_5m: bool = False) -> dict[str, object]:
        return {"root": str(data_dir.resolve()), "include_5m": include_5m}

    monkeypatch.setattr(momentum_cli, "_load_vdubus_data", fake_load)

    first = load_vdub_replay_bundle("NQ", left)
    second = load_vdub_replay_bundle("NQ", right)

    assert first is not second
    assert first.data["root"] != second.data["root"]


def test_load_unified_portfolio_replay_bundle_reuses_source_fingerprinted_cache(monkeypatch, tmp_path) -> None:
    from backtests.swing.engine import unified_portfolio_engine as unified_mod

    for symbol in ("QQQ", "GLD"):
        (tmp_path / f"{symbol}_1h.parquet").write_text("hourly", encoding="utf-8")
        (tmp_path / f"{symbol}_1d.parquet").write_text("daily", encoding="utf-8")
        (tmp_path / f"{symbol}_15m.parquet").write_text("15m", encoding="utf-8")

    calls = {"load": 0}

    def fake_load(_config: UnifiedBacktestConfig) -> dict[str, object]:
        calls["load"] += 1
        return {"unified": True}

    monkeypatch.setattr(unified_mod, "load_unified_data", fake_load)

    config = UnifiedBacktestConfig(initial_equity=25_000.0, data_dir=tmp_path)
    first = load_unified_portfolio_replay_bundle(config)
    second = load_unified_portfolio_replay_bundle(config)

    assert first is second
    # source_paths includes 1h+1d for all symbols + 15m for TPC symbols
    assert first.cache_source_fingerprint == fingerprint_paths(
        [
            tmp_path / f"{symbol}_{timeframe}.parquet"
            for symbol in ("GLD", "QQQ")
            for timeframe in ("1h", "1d")
        ] + [
            tmp_path / f"{symbol}_15m.parquet"
            for symbol in ("GLD", "QQQ")
        ],
        root=tmp_path,
    )
    assert calls["load"] == 1


def test_load_atrss_replay_bundle_reuses_source_fingerprinted_cache(monkeypatch, tmp_path) -> None:
    from backtests.swing.data import cache as cache_mod
    from backtests.swing.data import preprocessing as prep_mod

    for symbol in ("QQQ", "GLD"):
        (tmp_path / f"{symbol}_1h.parquet").write_text("hourly", encoding="utf-8")
        (tmp_path / f"{symbol}_1d.parquet").write_text("daily", encoding="utf-8")

    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100.0, 200.0],
        },
        index=pd.DatetimeIndex(
            [
                datetime(2026, 4, 25, 9, 30, tzinfo=UTC),
                datetime(2026, 4, 25, 10, 30, tzinfo=UTC),
            ]
        ),
    )

    calls = {"load_bars": 0, "build_numpy_arrays": 0}

    def fake_load_bars(_path: Path) -> pd.DataFrame:
        calls["load_bars"] += 1
        return frame.copy()

    def fake_build_numpy_arrays(df: pd.DataFrame) -> dict[str, int]:
        calls["build_numpy_arrays"] += 1
        return {"rows": len(df)}

    monkeypatch.setattr(cache_mod, "load_bars", fake_load_bars)
    monkeypatch.setattr(prep_mod, "normalize_timezone", lambda df: df)
    monkeypatch.setattr(prep_mod, "filter_rth", lambda df: df)
    monkeypatch.setattr(prep_mod, "build_numpy_arrays", fake_build_numpy_arrays)
    monkeypatch.setattr(prep_mod, "align_daily_to_hourly", lambda *_args: (0, 1))

    first = load_atrss_replay_bundle(tmp_path)
    second = load_atrss_replay_bundle(tmp_path)

    assert first is second
    assert calls["load_bars"] == 4
    assert calls["build_numpy_arrays"] == 4


def test_load_stock_research_replay_bundle_reuses_source_fingerprinted_cache(monkeypatch, tmp_path) -> None:
    from backtests.stock.engine import research_replay as replay_mod

    daily_path = tmp_path / "AAA_1d.parquet"
    daily_path.write_text("daily", encoding="utf-8")

    calls = {"load_all_data": 0}

    class FakeResearchReplayEngine:
        def __init__(self, data_dir):
            self.data_dir = Path(data_dir)

        def load_all_data(self) -> None:
            calls["load_all_data"] += 1

    monkeypatch.setattr(replay_mod, "ResearchReplayEngine", FakeResearchReplayEngine)

    first = load_research_replay_bundle(tmp_path)
    second = load_research_replay_bundle(tmp_path)

    assert first is second
    assert first.cache_source_fingerprint == fingerprint_paths([daily_path], root=tmp_path)
    assert calls["load_all_data"] == 1


def test_load_stock_research_replay_bundle_does_not_alias_identical_roots(monkeypatch, tmp_path) -> None:
    from backtests.stock.engine import research_replay as replay_mod

    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for base_dir in (left, right):
        (base_dir / "AAA_1d.parquet").write_text("daily", encoding="utf-8")

    class FakeResearchReplayEngine:
        def __init__(self, data_dir):
            self.data_dir = Path(data_dir)

        def load_all_data(self) -> None:
            return None

    monkeypatch.setattr(replay_mod, "ResearchReplayEngine", FakeResearchReplayEngine)

    first = load_research_replay_bundle(left)
    second = load_research_replay_bundle(right)

    assert first is not second
    assert first.data.data_dir != second.data.data_dir


def test_research_replay_data_fingerprint_tracks_source_files(tmp_path) -> None:
    first = tmp_path / "AAA_1d.parquet"
    first.write_text("one", encoding="utf-8")

    replay = ResearchReplayEngine(data_dir=tmp_path)
    initial = replay.data_fingerprint()

    second = tmp_path / "BBB_5m.parquet"
    second.write_text("two", encoding="utf-8")
    replay._data_fingerprint = None

    assert replay.data_fingerprint() != initial
