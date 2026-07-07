from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from backtests.shared.auto.cache_keys import build_cache_key, fingerprint_tree
from backtests.shared.auto.replay_bundle import ReplayBundle

if TYPE_CHECKING:
    from backtests.stock.engine.research_replay import ResearchReplayEngine


_REPLAY_CACHE: dict[str, ReplayBundle[ResearchReplayEngine]] = {}


def load_research_replay_bundle(data_dir: Path) -> ReplayBundle[ResearchReplayEngine]:
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    base_dir = Path(data_dir)
    source_fingerprint = fingerprint_tree(base_dir, patterns=("*.parquet",))
    cache_key = build_cache_key(
        "stock.research_replay_bundle",
        source_fingerprint=source_fingerprint,
        extra={"data_dir": str(base_dir.resolve())},
    )
    cached = _REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    replay = ResearchReplayEngine(data_dir=base_dir)
    replay.load_all_data()
    bundle = ReplayBundle(
        data=replay,
        cache_key=cache_key,
        cache_source_fingerprint=source_fingerprint,
    )
    _REPLAY_CACHE[cache_key] = bundle
    return bundle
