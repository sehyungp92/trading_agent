"""Tests for InstrumentationScorer — confirms newly added curated artifacts are scored."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.skills.instrumentation_scorer import InstrumentationScorer


def _seed_curated(curated_dir: Path, bot_id: str, days: int, files: dict[str, dict]) -> None:
    """Seed `days` days of summary.json + the named optional files."""
    end = datetime.now(timezone.utc)
    for d in range(days):
        date_str = (end - timedelta(days=d)).strftime("%Y-%m-%d")
        bot_dir = curated_dir / date_str / bot_id
        bot_dir.mkdir(parents=True, exist_ok=True)
        # minimal summary.json so days_with_data is counted
        (bot_dir / "summary.json").write_text(
            json.dumps({"net_pnl": 0.0, "total_trades": 1, "winning_trades": 0}),
            encoding="utf-8",
        )
        for fname, payload in files.items():
            (bot_dir / fname).write_text(json.dumps(payload), encoding="utf-8")


def test_scorer_credits_new_curated_files(tmp_path: Path) -> None:
    """When orderbook_stats / sizing_analysis / portfolio_context / fill_quality
    are present, the scorer should report higher coverage on the matching
    capability rows than when they are absent."""
    # Without files
    bare_dir = tmp_path / "bare"
    _seed_curated(bare_dir, "bot_a", days=14, files={})
    scorer_a = InstrumentationScorer(curated_dir=bare_dir, lookback_days=14)
    report_bare = scorer_a.score_bot("bot_a")

    # With files
    full_dir = tmp_path / "full"
    _seed_curated(full_dir, "bot_a", days=14, files={
        "orderbook_stats.json": {"by_context": {"bull": {"count": 5}}},
        "sizing_analysis.json": {"coverage": 1.0},
        "portfolio_context.json": {"correlation": []},
        "fill_quality.json": {"score": 0.9},
        "signal_health.json": {"signal_strength": 0.8},
    })
    scorer_b = InstrumentationScorer(curated_dir=full_dir, lookback_days=14)
    report_full = scorer_b.score_bot("bot_a")

    new_caps = {"order_book_analysis", "sizing_analysis", "portfolio_context", "fill_quality"}
    bare_scores = {c.capability: c.score for c in report_bare.capabilities}
    full_scores = {c.capability: c.score for c in report_full.capabilities}

    for cap in new_caps:
        assert cap in bare_scores, f"capability '{cap}' should be present in scorer output"
        assert cap in full_scores
        assert full_scores[cap] > bare_scores[cap], (
            f"'{cap}' coverage should improve when curated file is present "
            f"(bare={bare_scores[cap]}, full={full_scores[cap]})"
        )


def test_scorer_lists_new_optional_files(tmp_path: Path) -> None:
    """All four new optional files should appear in field_coverage entries."""
    _seed_curated(tmp_path, "bot_a", days=7, files={
        "orderbook_stats.json": {},
        "sizing_analysis.json": {},
        "portfolio_context.json": {},
        "fill_quality.json": {},
    })
    scorer = InstrumentationScorer(curated_dir=tmp_path, lookback_days=7)
    report = scorer.score_bot("bot_a")
    field_names = {f.field_name for f in report.field_coverage}
    for fname in (
        "file:orderbook_stats.json",
        "file:sizing_analysis.json",
        "file:portfolio_context.json",
        "file:fill_quality.json",
    ):
        assert fname in field_names, f"expected {fname} in field_coverage"
