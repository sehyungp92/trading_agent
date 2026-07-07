"""Portfolio, market, and curated-artifact prompt context loaders."""

from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.analysis.context_sources.common import (
    apply_temporal_window as _apply_temporal_window,
    safe_jsonl as _safe_jsonl,
)



class MarketContextMixin:
    def load_portfolio_outcomes(self) -> list[dict]:
        """Load portfolio-level suggestion outcomes from findings/portfolio_outcomes.jsonl.

        Returns recent portfolio change outcomes with verdicts and composite deltas.
        """
        path = self._memory_dir / "findings" / "portfolio_outcomes.jsonl"
        if not path.exists():
            return []
        try:
            entries = _safe_jsonl(path)
            return _apply_temporal_window(entries, max_entries=20)
        except Exception:
            return []

    def load_portfolio_metrics(self) -> dict:
        """Load latest portfolio rolling metrics from curated data.

        Returns the most recent portfolio_rolling_metrics.json if available.
        """
        if not self._curated_dir:
            return {}
        try:
            # Find most recent date directory with portfolio metrics
            portfolio_dirs = sorted(
                self._curated_dir.glob("*/portfolio/portfolio_rolling_metrics.json"),
                reverse=True,
            )
            if portfolio_dirs:
                return json.loads(portfolio_dirs[0].read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def load_consolidated_patterns(self) -> str:
        """Load patterns_consolidated.md if it exists."""
        path = self._memory_dir / "findings" / "patterns_consolidated.md"
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def load_search_reports(self, bot_id: str = "", lookback_n: int = 5) -> list[dict]:
        """Historical parameter-search reports used as read-only context."""
        path = self._memory_dir / "findings" / "search_reports.jsonl"
        if not path.exists():
            return []
        try:
            reports: list[dict] = []
            for entry in _safe_jsonl(path):
                if bot_id and entry.get("bot_id") != bot_id:
                    continue
                # Strip large candidate arrays, keep summary fields
                reports.append({
                    "suggestion_id": entry.get("suggestion_id"),
                    "bot_id": entry.get("bot_id"),
                    "param_name": entry.get("param_name"),
                    "routing": entry.get("routing"),
                    "best_value": entry.get("best_value"),
                    "discard_reason": entry.get("discard_reason", ""),
                    "exploration_summary": entry.get("exploration_summary", ""),
                    "searched_at": entry.get("searched_at", ""),
                    "context_role": "historical_read_only",
                })
            return reports[-lookback_n:]
        except Exception:
            return []

    def load_regime_parameter_analysis(self, bot_id: str = "") -> list[dict]:
        """Extract regime-conditional parameter analyses from search reports.

        Reads the same search_reports.jsonl as load_search_reports() but
        extracts the regime_analysis field where regime_sensitivity > 0.3.
        """
        path = self._memory_dir / "findings" / "search_reports.jsonl"
        if not path.exists():
            return []
        try:
            results: list[dict] = []
            for entry in _safe_jsonl(path):
                if bot_id and entry.get("bot_id") != bot_id:
                    continue
                regime = entry.get("regime_analysis")
                if not regime or regime.get("regime_sensitivity", 0) <= 0.3:
                    continue
                results.append(regime)
            return results[-10:]  # Last 10 significant analyses
        except Exception:
            return []

    def load_backtest_reliability(self, bot_id: str = "") -> dict[str, float]:
        """Historical per-category backtest reliability ratios."""
        path = self._memory_dir / "findings" / "backtest_calibration.jsonl"
        if not path.exists():
            return {}
        try:
            from collections import defaultdict
            correct: dict[str, int] = defaultdict(int)
            total: dict[str, int] = defaultdict(int)
            for entry in _safe_jsonl(path):
                if bot_id and entry.get("bot_id") != bot_id:
                    continue
                cat = entry.get("param_category", "")
                if entry.get("prediction_correct") is not None:
                    total[cat] += 1
                    if entry.get("prediction_correct"):
                        correct[cat] += 1
            return {
                cat: round(correct[cat] / total[cat], 2)
                for cat in total
                if total[cat] >= 3
            }
        except Exception:
            return {}

    def load_engine_decomposition(self, bot_id: str = "") -> dict:
        """Load engine-level metrics decomposition from curated data.

        Finds the most recent engine_decomposition.json for the given bot_id.
        """
        if not self._curated_dir:
            return {}
        try:
            curated = Path(self._curated_dir)
            date_dirs = sorted(
                [d for d in curated.iterdir() if d.is_dir() and not d.name.startswith(".")],
                reverse=True,
            )
            for date_dir in date_dirs[:7]:
                if bot_id:
                    candidate = date_dir / bot_id / "engine_decomposition.json"
                    if candidate.exists():
                        return json.loads(candidate.read_text(encoding="utf-8"))
                else:
                    for bot_dir in date_dir.iterdir():
                        if bot_dir.is_dir():
                            candidate = bot_dir / "engine_decomposition.json"
                            if candidate.exists():
                                return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def load_ablation_analysis(self, bot_id: str = "") -> dict:
        """Load ablation flag analysis from curated data.

        Finds the most recent ablation_analysis.json for the given bot_id.
        """
        if not self._curated_dir:
            return {}
        try:
            curated = Path(self._curated_dir)
            date_dirs = sorted(
                [d for d in curated.iterdir() if d.is_dir() and not d.name.startswith(".")],
                reverse=True,
            )
            for date_dir in date_dirs[:7]:
                if bot_id:
                    candidate = date_dir / bot_id / "ablation_analysis.json"
                    if candidate.exists():
                        return json.loads(candidate.read_text(encoding="utf-8"))
                else:
                    for bot_dir in date_dir.iterdir():
                        if bot_dir.is_dir():
                            candidate = bot_dir / "ablation_analysis.json"
                            if candidate.exists():
                                return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def load_exit_tier_analysis(self, bot_id: str = "") -> dict:
        """Load exit tier hit-rate analysis from curated data.

        Finds the most recent exit_tier_analysis.json for the given bot_id.
        """
        if not self._curated_dir:
            return {}
        try:
            curated = Path(self._curated_dir)
            date_dirs = sorted(
                [d for d in curated.iterdir() if d.is_dir() and not d.name.startswith(".")],
                reverse=True,
            )
            for date_dir in date_dirs[:7]:
                if bot_id:
                    candidate = date_dir / bot_id / "exit_tier_analysis.json"
                    if candidate.exists():
                        return json.loads(candidate.read_text(encoding="utf-8"))
                else:
                    for bot_dir in date_dir.iterdir():
                        if bot_dir.is_dir():
                            candidate = bot_dir / "exit_tier_analysis.json"
                            if candidate.exists():
                                return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def load_macro_regime_context(self) -> dict:
        """Load latest macro regime state from curated portfolio data.

        Looks for macro_regime_analysis.json in the most recent curated portfolio dir.
        """
        if not self._curated_dir:
            return {}
        try:
            # Find most recent date dir with portfolio data
            curated = Path(self._curated_dir)
            if not curated.exists():
                return {}
            date_dirs = sorted(
                [d for d in curated.iterdir() if d.is_dir() and not d.name.startswith(".")],
                reverse=True,
            )
            for date_dir in date_dirs[:7]:  # check last 7 days
                regime_file = date_dir / "portfolio" / "macro_regime_analysis.json"
                if regime_file.exists():
                    data = json.loads(regime_file.read_text(encoding="utf-8"))
                    if data:
                        return data
        except Exception:
            pass
        return {}

    def load_regime_config_history(self) -> list[dict]:
        """Load rolling regime config from recent curated bot dirs.

        Collects applied_regime_config.json from the last 30 days of curated data.
        """
        if not self._curated_dir:
            return []
        try:
            curated = Path(self._curated_dir)
            if not curated.exists():
                return []
            date_dirs = sorted(
                [d for d in curated.iterdir() if d.is_dir() and not d.name.startswith(".")],
                reverse=True,
            )
            history: list[dict] = []
            for date_dir in date_dirs[:30]:
                for bot_dir in date_dir.iterdir():
                    if not bot_dir.is_dir() or bot_dir.name == "portfolio":
                        continue
                    config_file = bot_dir / "applied_regime_config.json"
                    if config_file.exists():
                        data = json.loads(config_file.read_text(encoding="utf-8"))
                        if data:
                            history.append({
                                "date": date_dir.name,
                                "bot_id": bot_dir.name,
                                **data,
                            })
            return history
        except Exception:
            return []
