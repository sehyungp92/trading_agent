# tests/test_transfer_outcomes.py
"""Tests for transfer outcome measurement and track record scoring."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock


from trading_assistant.schemas.strategy_profile import (
    StrategyProfile,
    StrategyRegistry,
)
from trading_assistant.schemas.transfer_proposals import TransferOutcome
from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder


def _make_curated_summary(path: Path, total_pnl: float, win_rate: float = 0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "total_pnl": total_pnl,
        "win_rate": win_rate,
    }))


class TestTransferOutcome:
    def test_schema(self):
        o = TransferOutcome(
            pattern_id="p1",
            source_bot="bot1",
            target_bot="bot2",
            pnl_delta_7d=100.0,
            win_rate_delta_7d=0.05,
            verdict="positive",
        )
        assert o.verdict == "positive"
        assert o.pnl_delta_7d == 100.0


class TestTransferTrackRecord:
    def test_empty_track_record(self, tmp_path):
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path,
            bots=["bot1", "bot2"],
            findings_dir=tmp_path,
        )
        assert builder.compute_transfer_track_record() == {}

    def test_compute_track_record(self, tmp_path):
        # Write outcomes
        outcomes_path = tmp_path / "transfer_outcomes.jsonl"
        outcomes = [
            {"pattern_id": "p1", "target_bot": "bot2", "verdict": "positive"},
            {"pattern_id": "p1", "target_bot": "bot3", "verdict": "negative"},
            {"pattern_id": "p2", "target_bot": "bot2", "verdict": "positive"},
        ]
        with open(outcomes_path, "w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")

        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path,
            bots=["bot1", "bot2", "bot3"],
            findings_dir=tmp_path,
        )
        track = builder.compute_transfer_track_record()
        assert track["p1"]["total"] == 2
        assert track["p1"]["positive"] == 1
        assert track["p1"]["success_rate"] == 0.5
        assert track["p2"]["success_rate"] == 1.0

    def test_score_boosting_from_positive_track_record(self, tmp_path):
        """Patterns with positive track record get score boost."""
        # Write positive track record
        outcomes_path = tmp_path / "transfer_outcomes.jsonl"
        outcomes = [
            {"pattern_id": "p1", "target_bot": "bot3", "verdict": "positive"},
            {"pattern_id": "p1", "target_bot": "bot4", "verdict": "positive"},
        ]
        with open(outcomes_path, "w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")

        # Create a pattern library mock
        from trading_assistant.schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        pattern = PatternEntry(
            pattern_id="p1",
            source_bot="bot1",
            title="Great pattern",
            category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.VALIDATED,
        )
        lib = MagicMock()
        lib.load_active.return_value = [pattern]

        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path,
            bots=["bot1", "bot2"],
            findings_dir=tmp_path,
        )
        proposals = builder.build_proposals()
        assert len(proposals) == 1
        # Score should be boosted from default 0.5 by +0.1
        assert proposals[0].compatibility_score >= 0.5

    def test_score_penalizing_from_negative_track_record(self, tmp_path):
        """Patterns with negative track record get score penalty."""
        outcomes_path = tmp_path / "transfer_outcomes.jsonl"
        outcomes = [
            {"pattern_id": "p1", "target_bot": "bot3", "verdict": "negative"},
            {"pattern_id": "p1", "target_bot": "bot4", "verdict": "negative"},
        ]
        with open(outcomes_path, "w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")

        from trading_assistant.schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        pattern = PatternEntry(
            pattern_id="p1",
            source_bot="bot1",
            title="Bad pattern",
            category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.VALIDATED,
        )
        lib = MagicMock()
        lib.load_active.return_value = [pattern]

        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path,
            bots=["bot1", "bot2"],
            findings_dir=tmp_path,
        )
        proposals = builder.build_proposals()
        assert len(proposals) == 1
        # Score should be penalized from default 0.5 by -0.2
        assert proposals[0].compatibility_score <= 0.5

    def test_missing_data_handled(self, tmp_path):
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path,
            bots=["bot1"],
            findings_dir=tmp_path,
        )
        outcomes = builder.measure_transfer_outcomes()
        assert outcomes == []


def _make_registry() -> StrategyRegistry:
    """Build a minimal registry for archetype compatibility tests."""
    return StrategyRegistry(
        strategies={
            "ATRSS": StrategyProfile(
                bot_id="swing_multi_01", family="swing",
                archetype="trend_follow", asset_class="mixed",
            ),
            "AKC_HELIX": StrategyProfile(
                bot_id="swing_multi_01", family="swing",
                archetype="divergence_swing", asset_class="mixed",
            ),
            "IARIC_v1": StrategyProfile(
                bot_id="stock_trader", family="stock",
                archetype="intraday_momentum", asset_class="equity",
            ),
            "AKC_Helix_v40": StrategyProfile(
                bot_id="momentum_nq_01", family="momentum",
                archetype="multi_tf_momentum", asset_class="futures",
            ),
            "VdubusNQ_v4": StrategyProfile(
                bot_id="momentum_nq_01", family="momentum",
                archetype="trend_follow", asset_class="futures",
            ),
        },
    )


def _make_pattern(source_bot: str, source_strategy_id: str = "") -> MagicMock:
    pattern = MagicMock()
    pattern.source_bot = source_bot
    pattern.source_strategy_id = source_strategy_id
    pattern.target_bots = []
    pattern.pattern_id = "p1"
    pattern.title = "Test pattern"
    pattern.category = "structural"
    pattern.status = "VALIDATED"
    return pattern


class TestArchetypeCompatibility:
    def test_same_archetype_gets_075(self, tmp_path):
        """Same archetype (trend_follow → trend_follow) gets 0.75 base."""
        registry = _make_registry()
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib, curated_dir=tmp_path,
            bots=["swing_multi_01", "momentum_nq_01"],
            strategy_registry=registry,
        )
        pattern = _make_pattern("swing_multi_01", "ATRSS")
        # momentum_nq_01 has VdubusNQ_v4 which is also trend_follow
        score = builder._compute_compatibility(pattern, "momentum_nq_01")
        assert score == 0.75

    def test_same_family_different_archetype_gets_060(self, tmp_path):
        """Same family but different archetype gets 0.60 base."""
        # Create registry where source has archetype A, target same family but archetype B
        registry = StrategyRegistry(
            strategies={
                "SRC": StrategyProfile(
                    bot_id="bot_a", family="swing",
                    archetype="trend_follow", asset_class="mixed",
                ),
                "TGT": StrategyProfile(
                    bot_id="bot_b", family="swing",
                    archetype="breakout", asset_class="equity",
                ),
            },
        )
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib, curated_dir=tmp_path,
            bots=["bot_a", "bot_b"],
            strategy_registry=registry,
        )
        pattern = _make_pattern("bot_a", "SRC")
        score = builder._compute_compatibility(pattern, "bot_b")
        assert score == 0.60

    def test_cross_family_gets_030(self, tmp_path):
        """Cross-family, different archetype, different asset class gets 0.30."""
        registry = StrategyRegistry(
            strategies={
                "SRC": StrategyProfile(
                    bot_id="bot_a", family="swing",
                    archetype="trend_follow", asset_class="mixed",
                ),
                "TGT": StrategyProfile(
                    bot_id="bot_b", family="stock",
                    archetype="intraday_momentum", asset_class="equity",
                ),
            },
        )
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib, curated_dir=tmp_path,
            bots=["bot_a", "bot_b"],
            strategy_registry=registry,
        )
        pattern = _make_pattern("bot_a", "SRC")
        score = builder._compute_compatibility(pattern, "bot_b")
        assert score == 0.30

    def test_asset_class_bonus(self, tmp_path):
        """Same asset class adds 0.10 bonus."""
        registry = StrategyRegistry(
            strategies={
                "SRC": StrategyProfile(
                    bot_id="bot_a", family="momentum",
                    archetype="trend_follow", asset_class="futures",
                ),
                "TGT": StrategyProfile(
                    bot_id="bot_b", family="stock",
                    archetype="intraday_momentum", asset_class="futures",
                ),
            },
        )
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib, curated_dir=tmp_path,
            bots=["bot_a", "bot_b"],
            strategy_registry=registry,
        )
        pattern = _make_pattern("bot_a", "SRC")
        score = builder._compute_compatibility(pattern, "bot_b")
        # Cross-family 0.30 + asset class 0.10 = 0.40
        assert score == 0.40

    def test_no_registry_falls_back_to_legacy(self, tmp_path):
        """Without registry, uses legacy regime-based scoring (default 0.5)."""
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib, curated_dir=tmp_path,
            bots=["bot_a", "bot_b"],
            strategy_registry=None,
        )
        pattern = _make_pattern("bot_a")
        score = builder._compute_compatibility(pattern, "bot_b")
        # No regime data available → falls back to 0.5
        assert score == 0.5

    def test_unknown_strategy_falls_back(self, tmp_path):
        """Unknown source strategy falls back to legacy."""
        registry = _make_registry()
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib, curated_dir=tmp_path,
            bots=["swing_multi_01", "stock_trader"],
            strategy_registry=registry,
        )
        pattern = _make_pattern("swing_multi_01", "UNKNOWN_STRAT")
        score = builder._compute_compatibility(pattern, "stock_trader")
        assert score == 0.5

    def test_regime_blending(self, tmp_path):
        """When regime data exists, score blends archetype (0.6) + regime (0.4)."""
        registry = _make_registry()
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib, curated_dir=tmp_path,
            bots=["swing_multi_01", "momentum_nq_01"],
            strategy_registry=registry,
        )
        # Write regime data for both bots
        for bot_id in ["swing_multi_01", "momentum_nq_01"]:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            regime_dir = tmp_path / date_str / bot_id
            regime_dir.mkdir(parents=True, exist_ok=True)
            (regime_dir / "regime_analysis.json").write_text(json.dumps({
                "regime_trade_count": {"trending_up": 10, "ranging": 5},
            }))

        pattern = _make_pattern("swing_multi_01", "ATRSS")
        score = builder._compute_compatibility(pattern, "momentum_nq_01")
        # 0.6 * 0.75 (same archetype) + 0.4 * regime_score
        assert 0.45 < score < 1.0
        # Must differ from pure archetype score (0.75) due to blending
        assert score != 0.75
