"""Integration tests for the Strategy-Aware Intelligence Upgrade (Phases 4-6).

Tests cross-cutting integration of:
- Response parser: new AgentSuggestion fields (engine, ablation_flag, regime_condition)
- Response validator: new gates (safety-critical ablation, engine existence)
- Prompt assembler: _CURATED_FILES includes engine/ablation/exit-tier files
- Context builder: load_engine_decomposition, load_ablation_analysis, load_exit_tier_analysis
- Schema fields: AgentSuggestion + StrategySuggestion new fields
"""
from __future__ import annotations

import json
from pathlib import Path


from trading_assistant.schemas.agent_response import AgentSuggestion, ParsedAnalysis
from trading_assistant.schemas.strategy_suggestions import StrategySuggestion, SuggestionTier


# ---------------------------------------------------------------------------
# 1. Schema field tests
# ---------------------------------------------------------------------------

class TestAgentSuggestionNewFields:
    def test_engine_field_default_empty(self):
        s = AgentSuggestion(bot_id="b", title="t")
        assert s.engine == ""

    def test_engine_field_set(self):
        s = AgentSuggestion(bot_id="b", title="t", engine="REVERSAL")
        assert s.engine == "REVERSAL"

    def test_ablation_flag_field_default_empty(self):
        s = AgentSuggestion(bot_id="b", title="t")
        assert s.ablation_flag == ""

    def test_ablation_flag_field_set(self):
        s = AgentSuggestion(
            bot_id="b", title="t", ablation_flag="fade_oscillation_gate",
        )
        assert s.ablation_flag == "fade_oscillation_gate"

    def test_regime_condition_field_default_empty(self):
        s = AgentSuggestion(bot_id="b", title="t")
        assert s.regime_condition == ""

    def test_regime_condition_field_set(self):
        s = AgentSuggestion(bot_id="b", title="t", regime_condition="volatile")
        assert s.regime_condition == "volatile"

    def test_all_new_fields_together(self):
        s = AgentSuggestion(
            bot_id="b",
            title="Tighten REVERSAL stop in volatile",
            engine="REVERSAL",
            ablation_flag="fade_oscillation_gate",
            regime_condition="volatile",
            category="stop_loss",
            confidence=0.7,
        )
        assert s.engine == "REVERSAL"
        assert s.ablation_flag == "fade_oscillation_gate"
        assert s.regime_condition == "volatile"
        assert s.category == "stop_loss"

    def test_serialization_round_trip(self):
        s = AgentSuggestion(
            bot_id="b", title="t",
            engine="MOMENTUM", ablation_flag="flag_x", regime_condition="trending",
        )
        data = s.model_dump(mode="json")
        assert data["engine"] == "MOMENTUM"
        assert data["ablation_flag"] == "flag_x"
        assert data["regime_condition"] == "trending"
        restored = AgentSuggestion(**data)
        assert restored.engine == "MOMENTUM"
        assert restored.ablation_flag == "flag_x"
        assert restored.regime_condition == "trending"


class TestStrategySuggestionNewFields:
    def test_engine_field_default(self):
        s = StrategySuggestion(
            tier=SuggestionTier.PARAMETER, title="t", description="d",
        )
        assert s.engine == ""

    def test_engine_field_set(self):
        s = StrategySuggestion(
            tier=SuggestionTier.PARAMETER, title="t", description="d",
            engine="REVERSAL",
        )
        assert s.engine == "REVERSAL"

    def test_regime_condition_field_default(self):
        s = StrategySuggestion(
            tier=SuggestionTier.FILTER, title="t", description="d",
        )
        assert s.regime_condition == ""

    def test_regime_condition_field_set(self):
        s = StrategySuggestion(
            tier=SuggestionTier.FILTER, title="t", description="d",
            regime_condition="volatile",
        )
        assert s.regime_condition == "volatile"


# ---------------------------------------------------------------------------
# 2. Response parser: new fields parsed from structured output
# ---------------------------------------------------------------------------

class TestResponseParserNewFields:
    def test_parse_suggestion_with_engine_ablation_regime(self):
        """Structured output with new fields should parse into AgentSuggestion."""
        from trading_assistant.analysis.response_parser import parse_response

        structured = {
            "predictions": [],
            "suggestions": [
                {
                    "suggestion_id": "s1",
                    "bot_id": "test_bot",
                    "category": "stop_loss",
                    "title": "Tighten REVERSAL stop in volatile regime",
                    "expected_impact": "+5% win rate",
                    "confidence": 0.75,
                    "evidence_summary": "Backtested on 60 trades",
                    "engine": "REVERSAL",
                    "ablation_flag": "fade_oscillation_gate",
                    "regime_condition": "volatile",
                },
            ],
            "structural_proposals": [],
        }
        response = (
            "Some analysis text.\n"
            "<!-- STRUCTURED_OUTPUT\n"
            + json.dumps(structured)
            + "\n-->"
        )
        result = parse_response(response)
        assert result.parse_success is True
        assert len(result.suggestions) == 1

        s = result.suggestions[0]
        assert s.engine == "REVERSAL"
        assert s.ablation_flag == "fade_oscillation_gate"
        assert s.regime_condition == "volatile"
        assert s.suggestion_id == "s1"
        assert s.confidence == 0.75

    def test_parse_suggestion_without_new_fields_defaults(self):
        """Legacy suggestions without new fields should still parse with defaults."""
        from trading_assistant.analysis.response_parser import parse_response

        structured = {
            "predictions": [],
            "suggestions": [
                {
                    "bot_id": "test_bot",
                    "category": "exit_timing",
                    "title": "Adjust exit timing",
                    "confidence": 0.6,
                },
            ],
            "structural_proposals": [],
        }
        response = (
            "<!-- STRUCTURED_OUTPUT\n"
            + json.dumps(structured)
            + "\n-->"
        )
        result = parse_response(response)
        assert result.parse_success is True
        assert len(result.suggestions) == 1
        s = result.suggestions[0]
        assert s.engine == ""
        assert s.ablation_flag == ""
        assert s.regime_condition == ""

    def test_parse_multiple_suggestions_mixed_fields(self):
        """Mix of suggestions with and without new fields."""
        from trading_assistant.analysis.response_parser import parse_response

        structured = {
            "predictions": [],
            "suggestions": [
                {
                    "bot_id": "bot_a",
                    "title": "Engine-specific change",
                    "engine": "MOMENTUM",
                    "regime_condition": "trending",
                },
                {
                    "bot_id": "bot_b",
                    "title": "Generic change",
                },
            ],
            "structural_proposals": [],
        }
        response = (
            "<!-- STRUCTURED_OUTPUT\n"
            + json.dumps(structured)
            + "\n-->"
        )
        result = parse_response(response)
        assert len(result.suggestions) == 2
        assert result.suggestions[0].engine == "MOMENTUM"
        assert result.suggestions[0].regime_condition == "trending"
        assert result.suggestions[1].engine == ""
        assert result.suggestions[1].regime_condition == ""


# ---------------------------------------------------------------------------
# 3. Response validator: new gates
# ---------------------------------------------------------------------------

class TestResponseValidatorNewGates:
    def _make_parsed(self, suggestions: list[AgentSuggestion]) -> ParsedAnalysis:
        return ParsedAnalysis(
            suggestions=suggestions,
            predictions=[],
            structural_proposals=[],
        )

    def test_safety_critical_ablation_blocked(self):
        """Suggestion targeting a safety-critical ablation flag is blocked."""
        from trading_assistant.analysis.response_validator import ResponseValidator

        suggestion = AgentSuggestion(
            bot_id="test_bot",
            title="Disable stop loss gate",
            ablation_flag="stop_loss_gate",
            confidence=0.8,
        )
        validator = ResponseValidator(
            safety_critical_params={"stop_loss_gate", "max_risk_gate"},
        )
        parsed = self._make_parsed([suggestion])
        result = validator.validate(parsed)
        assert len(result.blocked_suggestions) == 1
        assert len(result.approved_suggestions) == 0
        assert "safety-critical" in result.blocked_suggestions[0].reason.lower()

    def test_non_safety_critical_ablation_not_blocked(self):
        """Ablation flag NOT in safety_critical_params is allowed through gate 1b."""
        from trading_assistant.analysis.response_validator import ResponseValidator

        suggestion = AgentSuggestion(
            bot_id="test_bot",
            title="Disable fade gate",
            ablation_flag="fade_oscillation_gate",
            confidence=0.8,
        )
        validator = ResponseValidator(
            safety_critical_params={"stop_loss_gate"},
        )
        parsed = self._make_parsed([suggestion])
        result = validator.validate(parsed)
        # Should pass gate 1b (may be blocked by other gates, but not ablation safety)
        ablation_blocked = [
            b for b in result.blocked_suggestions
            if "safety-critical" in b.reason.lower()
        ]
        assert len(ablation_blocked) == 0

    def test_invalid_engine_blocked(self):
        """Engine not in any strategy sub_engines for the bot -> blocked."""
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.strategy_profile import (
            StrategyProfile,
            StrategyRegistry,
        )

        profile = StrategyProfile(
            display_name="TestStrat",
            bot_id="test_bot",
            sub_engines=["MOMENTUM", "BREAKOUT"],
        )
        registry = StrategyRegistry(strategies={"s1": profile})

        suggestion = AgentSuggestion(
            bot_id="test_bot",
            title="Adjust REVERSAL engine params",
            engine="REVERSAL",
            confidence=0.8,
        )
        validator = ResponseValidator(strategy_registry=registry)
        parsed = self._make_parsed([suggestion])
        result = validator.validate(parsed)

        engine_blocked = [
            b for b in result.blocked_suggestions
            if "engine" in b.reason.lower() and "not found" in b.reason.lower()
        ]
        assert len(engine_blocked) == 1

    def test_valid_engine_approved(self):
        """Engine present in sub_engines -> passes gate 1c."""
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.strategy_profile import (
            StrategyProfile,
            StrategyRegistry,
        )

        profile = StrategyProfile(
            display_name="TestStrat",
            bot_id="test_bot",
            sub_engines=["MOMENTUM", "BREAKOUT"],
        )
        registry = StrategyRegistry(strategies={"s1": profile})

        suggestion = AgentSuggestion(
            bot_id="test_bot",
            title="Adjust MOMENTUM params",
            engine="MOMENTUM",
            confidence=0.8,
        )
        validator = ResponseValidator(strategy_registry=registry)
        parsed = self._make_parsed([suggestion])
        result = validator.validate(parsed)

        engine_blocked = [
            b for b in result.blocked_suggestions
            if "engine" in b.reason.lower() and "not found" in b.reason.lower()
        ]
        assert len(engine_blocked) == 0

    def test_engine_case_insensitive(self):
        """Engine check should be case-insensitive."""
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.strategy_profile import (
            StrategyProfile,
            StrategyRegistry,
        )

        profile = StrategyProfile(
            display_name="TestStrat",
            bot_id="test_bot",
            sub_engines=["MOMENTUM"],
        )
        registry = StrategyRegistry(strategies={"s1": profile})

        suggestion = AgentSuggestion(
            bot_id="test_bot",
            title="momentum tweak",
            engine="momentum",  # lowercase
            confidence=0.8,
        )
        validator = ResponseValidator(strategy_registry=registry)
        parsed = self._make_parsed([suggestion])
        result = validator.validate(parsed)

        engine_blocked = [
            b for b in result.blocked_suggestions
            if "engine" in b.reason.lower() and "not found" in b.reason.lower()
        ]
        assert len(engine_blocked) == 0

    def test_no_strategy_registry_engine_check_skipped(self):
        """Without strategy_registry, engine check is skipped (approved)."""
        from trading_assistant.analysis.response_validator import ResponseValidator

        suggestion = AgentSuggestion(
            bot_id="test_bot",
            title="Tweak REVERSAL",
            engine="REVERSAL",
            confidence=0.8,
        )
        validator = ResponseValidator(strategy_registry=None)
        parsed = self._make_parsed([suggestion])
        result = validator.validate(parsed)

        engine_blocked = [
            b for b in result.blocked_suggestions
            if "engine" in b.reason.lower() and "not found" in b.reason.lower()
        ]
        assert len(engine_blocked) == 0

    def test_no_engine_field_passes_gate(self):
        """Suggestion without engine field passes gate 1c unconditionally."""
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.strategy_profile import (
            StrategyProfile,
            StrategyRegistry,
        )

        profile = StrategyProfile(
            display_name="TestStrat",
            bot_id="test_bot",
            sub_engines=["MOMENTUM"],
        )
        registry = StrategyRegistry(strategies={"s1": profile})

        suggestion = AgentSuggestion(
            bot_id="test_bot",
            title="Generic change",
            engine="",
            confidence=0.8,
        )
        validator = ResponseValidator(strategy_registry=registry)
        parsed = self._make_parsed([suggestion])
        result = validator.validate(parsed)

        engine_blocked = [
            b for b in result.blocked_suggestions
            if "engine" in b.reason.lower() and "not found" in b.reason.lower()
        ]
        assert len(engine_blocked) == 0

    def test_empty_safety_critical_params_ablation_passes(self):
        """No safety_critical_params -> any ablation_flag passes gate 1b."""
        from trading_assistant.analysis.response_validator import ResponseValidator

        suggestion = AgentSuggestion(
            bot_id="test_bot",
            title="Toggle gate",
            ablation_flag="some_gate",
            confidence=0.8,
        )
        validator = ResponseValidator(safety_critical_params=set())
        parsed = self._make_parsed([suggestion])
        result = validator.validate(parsed)

        ablation_blocked = [
            b for b in result.blocked_suggestions
            if "safety-critical" in b.reason.lower()
        ]
        assert len(ablation_blocked) == 0


# ---------------------------------------------------------------------------
# 4. Prompt assembler: _CURATED_FILES
# ---------------------------------------------------------------------------

class TestPromptAssemblerCuratedFiles:
    def test_curated_files_contains_engine_decomposition(self):
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "engine_decomposition.json" in _CURATED_FILES

    def test_curated_files_contains_ablation_analysis(self):
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "ablation_analysis.json" in _CURATED_FILES

    def test_curated_files_contains_exit_tier_analysis(self):
        from trading_assistant.analysis.prompt_assembler import _CURATED_FILES
        assert "exit_tier_analysis.json" in _CURATED_FILES


# ---------------------------------------------------------------------------
# 5. Context builder: new load methods
# ---------------------------------------------------------------------------

class TestContextBuilderNewLoaders:
    def test_load_engine_decomposition_from_curated(self, tmp_path: Path):
        """Loads engine_decomposition.json from curated data dir."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        curated = tmp_path / "curated" / "2026-03-01" / "test_bot"
        curated.mkdir(parents=True)
        data = {"engines": {"MOMENTUM": {"pnl": 150}}}
        (curated / "engine_decomposition.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        cb = ContextBuilder(
            memory_dir=tmp_path / "memory",
            curated_dir=tmp_path / "curated",
        )
        result = cb.load_engine_decomposition(bot_id="test_bot")
        assert result == data

    def test_load_engine_decomposition_missing(self, tmp_path: Path):
        """Returns empty dict when no engine_decomposition.json exists."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        curated = tmp_path / "curated" / "2026-03-01" / "test_bot"
        curated.mkdir(parents=True)
        cb = ContextBuilder(
            memory_dir=tmp_path / "memory",
            curated_dir=tmp_path / "curated",
        )
        result = cb.load_engine_decomposition(bot_id="test_bot")
        assert result == {}

    def test_load_engine_decomposition_no_curated_dir(self, tmp_path: Path):
        """Returns empty dict when curated_dir is None."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        cb = ContextBuilder(memory_dir=tmp_path / "memory", curated_dir=None)
        result = cb.load_engine_decomposition(bot_id="test_bot")
        assert result == {}

    def test_load_ablation_analysis_from_curated(self, tmp_path: Path):
        """Loads ablation_analysis.json from curated data dir."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        curated = tmp_path / "curated" / "2026-03-01" / "test_bot"
        curated.mkdir(parents=True)
        data = {"flags": [{"flag_name": "fade_gate", "pnl_delta": 50.0}]}
        (curated / "ablation_analysis.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        cb = ContextBuilder(
            memory_dir=tmp_path / "memory",
            curated_dir=tmp_path / "curated",
        )
        result = cb.load_ablation_analysis(bot_id="test_bot")
        assert result == data

    def test_load_ablation_analysis_missing(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        curated = tmp_path / "curated" / "2026-03-01" / "test_bot"
        curated.mkdir(parents=True)
        cb = ContextBuilder(
            memory_dir=tmp_path / "memory",
            curated_dir=tmp_path / "curated",
        )
        result = cb.load_ablation_analysis(bot_id="test_bot")
        assert result == {}

    def test_load_ablation_analysis_no_curated_dir(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        cb = ContextBuilder(memory_dir=tmp_path / "memory", curated_dir=None)
        result = cb.load_ablation_analysis()
        assert result == {}

    def test_load_exit_tier_analysis_from_curated(self, tmp_path: Path):
        """Loads exit_tier_analysis.json from curated data dir."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        curated = tmp_path / "curated" / "2026-03-01" / "test_bot"
        curated.mkdir(parents=True)
        data = {"tiers": [{"tier_name": "TP1", "hit_rate": 0.75}]}
        (curated / "exit_tier_analysis.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        cb = ContextBuilder(
            memory_dir=tmp_path / "memory",
            curated_dir=tmp_path / "curated",
        )
        result = cb.load_exit_tier_analysis(bot_id="test_bot")
        assert result == data

    def test_load_exit_tier_analysis_missing(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        curated = tmp_path / "curated" / "2026-03-01" / "test_bot"
        curated.mkdir(parents=True)
        cb = ContextBuilder(
            memory_dir=tmp_path / "memory",
            curated_dir=tmp_path / "curated",
        )
        result = cb.load_exit_tier_analysis(bot_id="test_bot")
        assert result == {}

    def test_load_exit_tier_analysis_no_curated_dir(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        cb = ContextBuilder(memory_dir=tmp_path / "memory", curated_dir=None)
        result = cb.load_exit_tier_analysis()
        assert result == {}

    def test_load_engine_decomposition_picks_most_recent(self, tmp_path: Path):
        """When multiple dates exist, picks the most recent one."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        for date, value in [("2026-02-28", 100), ("2026-03-01", 200)]:
            d = tmp_path / "curated" / date / "test_bot"
            d.mkdir(parents=True)
            (d / "engine_decomposition.json").write_text(
                json.dumps({"value": value}), encoding="utf-8",
            )
        cb = ContextBuilder(
            memory_dir=tmp_path / "memory",
            curated_dir=tmp_path / "curated",
        )
        result = cb.load_engine_decomposition(bot_id="test_bot")
        assert result["value"] == 200  # most recent date

    def test_load_without_bot_id_scans_all(self, tmp_path: Path):
        """Without bot_id, scans all bot dirs for the file."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        d = tmp_path / "curated" / "2026-03-01" / "bot_a"
        d.mkdir(parents=True)
        (d / "engine_decomposition.json").write_text(
            json.dumps({"bot": "a"}), encoding="utf-8",
        )
        cb = ContextBuilder(
            memory_dir=tmp_path / "memory",
            curated_dir=tmp_path / "curated",
        )
        result = cb.load_engine_decomposition(bot_id="")
        assert result == {"bot": "a"}


# ---------------------------------------------------------------------------
# 6. Context builder base_package includes new data
# ---------------------------------------------------------------------------

class TestBasePackageIntegration:
    def test_base_package_includes_engine_decomposition(self, tmp_path: Path):
        """engine_decomposition data injected into base_package when present."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory = tmp_path / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        (memory / "findings").mkdir(parents=True)

        curated = tmp_path / "curated" / "2026-03-01" / "test_bot"
        curated.mkdir(parents=True)
        (curated / "engine_decomposition.json").write_text(
            json.dumps({"engines": {"M": {"pnl": 100}}}), encoding="utf-8",
        )
        cb = ContextBuilder(memory_dir=memory, curated_dir=tmp_path / "curated")
        pkg = cb.base_package(bot_id="test_bot")
        assert "engine_decomposition" in pkg.data

    def test_base_package_includes_ablation_analysis(self, tmp_path: Path):
        """ablation_analysis data injected into base_package when present."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory = tmp_path / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        (memory / "findings").mkdir(parents=True)

        curated = tmp_path / "curated" / "2026-03-01" / "test_bot"
        curated.mkdir(parents=True)
        (curated / "ablation_analysis.json").write_text(
            json.dumps({"flags": []}), encoding="utf-8",
        )
        cb = ContextBuilder(memory_dir=memory, curated_dir=tmp_path / "curated")
        pkg = cb.base_package(bot_id="test_bot")
        assert "ablation_analysis" in pkg.data

    def test_base_package_includes_exit_tier_analysis(self, tmp_path: Path):
        """exit_tier_analysis data injected into base_package when present."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory = tmp_path / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        (memory / "findings").mkdir(parents=True)

        curated = tmp_path / "curated" / "2026-03-01" / "test_bot"
        curated.mkdir(parents=True)
        (curated / "exit_tier_analysis.json").write_text(
            json.dumps({"tiers": [{"name": "TP1"}]}), encoding="utf-8",
        )
        cb = ContextBuilder(memory_dir=memory, curated_dir=tmp_path / "curated")
        pkg = cb.base_package(bot_id="test_bot")
        assert "exit_tier_analysis" in pkg.data

    def test_context_priority_includes_new_keys(self):
        """_CONTEXT_PRIORITY should include engine/ablation/exit tier keys."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        priority = ContextBuilder._CONTEXT_PRIORITY
        assert "engine_decomposition" in priority
        assert "ablation_analysis" in priority
        assert "exit_tier_analysis" in priority


# ---------------------------------------------------------------------------
# 7. Regime-conditional parameter search integration
# ---------------------------------------------------------------------------

class TestRegimeParameterSearchIntegration:
    def test_regime_analysis_serializes_in_report(self):
        """ParameterSearchReport with regime_analysis round-trips through JSON."""
        from trading_assistant.schemas.parameter_search import ParameterSearchReport, SearchRouting
        from trading_assistant.schemas.regime_conditional import (
            RegimeParameterAnalysis,
            RegimeParameterStats,
        )

        regime = RegimeParameterAnalysis(
            param_name="signal_strength_min",
            bot_id="bot1",
            regimes_analyzed=["trending", "ranging"],
            optimal_per_regime={"trending": 0.3, "ranging": 0.7},
            current_value=0.5,
            regime_sensitivity=0.45,
            regime_stats=[
                RegimeParameterStats(
                    regime="trending", trade_count=20,
                    optimal_value=0.3, win_rate=0.65, avg_pnl=15.0,
                    profit_factor=2.1,
                ),
                RegimeParameterStats(
                    regime="ranging", trade_count=18,
                    optimal_value=0.7, win_rate=0.55, avg_pnl=8.0,
                    profit_factor=1.6,
                ),
            ],
            recommendations=["In trending: consider signal_strength_min=0.3"],
        )
        report = ParameterSearchReport(
            suggestion_id="s1",
            bot_id="bot1",
            param_name="signal_strength_min",
            original_proposed=0.7,
            current_value=0.5,
            routing=SearchRouting.EXPERIMENT,
            regime_analysis=regime,
        )
        # Round-trip through JSON
        json_str = report.model_dump_json()
        restored = ParameterSearchReport.model_validate_json(json_str)
        assert restored.regime_analysis is not None
        assert restored.regime_analysis.regime_sensitivity == 0.45
        assert len(restored.regime_analysis.regime_stats) == 2
        assert restored.regime_analysis.optimal_per_regime["trending"] == 0.3

    def test_context_builder_load_regime_parameter_analysis(self, tmp_path: Path):
        """Writes JSONL with regime_analysis, verifies extraction."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory = tmp_path / "memory"
        (memory / "findings").mkdir(parents=True)
        (memory / "policies" / "v1").mkdir(parents=True)

        # Write search reports with regime_analysis
        reports_path = memory / "findings" / "search_reports.jsonl"
        entries = [
            # High sensitivity — should be extracted
            json.dumps({
                "suggestion_id": "s1",
                "bot_id": "bot1",
                "param_name": "signal_strength_min",
                "routing": "experiment",
                "regime_analysis": {
                    "param_name": "signal_strength_min",
                    "bot_id": "bot1",
                    "regime_sensitivity": 0.45,
                    "regimes_analyzed": ["trending", "ranging"],
                    "optimal_per_regime": {"trending": 0.3, "ranging": 0.7},
                },
            }),
            # Low sensitivity — should be filtered out
            json.dumps({
                "suggestion_id": "s2",
                "bot_id": "bot1",
                "param_name": "lookback_period",
                "routing": "discard",
                "regime_analysis": {
                    "param_name": "lookback_period",
                    "bot_id": "bot1",
                    "regime_sensitivity": 0.1,
                },
            }),
            # No regime_analysis — should be filtered out
            json.dumps({
                "suggestion_id": "s3",
                "bot_id": "bot1",
                "param_name": "stop_loss_atr",
                "routing": "approve",
            }),
        ]
        reports_path.write_text("\n".join(entries) + "\n", encoding="utf-8")

        cb = ContextBuilder(memory_dir=memory)
        results = cb.load_regime_parameter_analysis(bot_id="bot1")
        assert len(results) == 1
        assert results[0]["param_name"] == "signal_strength_min"
        assert results[0]["regime_sensitivity"] == 0.45

    def test_workflow_priorities_include_regime_parameter_analysis(self):
        """regime_parameter_analysis is in default and weekly_analysis priority lists."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        default = ContextBuilder._CONTEXT_PRIORITY
        weekly = ContextBuilder._WORKFLOW_PRIORITIES["weekly_analysis"]
        assert "regime_parameter_analysis" in default
        assert "regime_parameter_analysis" in weekly


# ---------------------------------------------------------------------------
# Analyzer → curated-file → ContextBuilder.load_* round-trip
#
# These tests exercise the production data path that the unit-mocked analyzer
# tests skip: actually run the analyzer, serialize via model_dump, drop the
# JSON into the curated directory layout the loaders expect, and confirm the
# loaders read it back into a usable shape.
# ---------------------------------------------------------------------------


class TestAnalyzerCuratedRoundTrip:
    def _curated_path(self, tmp_path: Path, bot_id: str, date: str, name: str) -> Path:
        target = tmp_path / "curated" / date / bot_id
        target.mkdir(parents=True, exist_ok=True)
        return target / name

    def _memory_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "memory"
        (d / "policies" / "v1").mkdir(parents=True, exist_ok=True)
        (d / "findings").mkdir(parents=True, exist_ok=True)
        return d

    def test_engine_decomposer_round_trip(self, tmp_path: Path):
        """EngineDecomposer output → file → ContextBuilder.load_engine_decomposition."""
        from trading_assistant.analysis.context_builder import ContextBuilder
        from trading_assistant.schemas.strategy_profile import StrategyProfile, StrategyRegistry
        from trading_assistant.skills.engine_decomposer import EngineDecomposer
        from tests.factories import make_trade

        registry = StrategyRegistry(
            strategies={
                "Strat1": StrategyProfile(
                    display_name="Strat1",
                    bot_id="bot_x",
                    sub_engines=["reversal", "breakdown"],
                ),
            },
        )
        trades = [
            make_trade(
                trade_id=f"t{i}",
                bot_id="bot_x",
                strategy_id="Strat1",
                entry_signal="reversal_short" if i % 2 == 0 else "breakdown_long",
                pnl=100.0 if i % 2 == 0 else -50.0,
                market_regime="trending",
            )
            for i in range(10)
        ]
        decomp = EngineDecomposer(registry).decompose(trades, "bot_x", period="2026-04-01")
        assert decomp.engines, "decomposer produced no engines from valid input"

        out = self._curated_path(tmp_path, "bot_x", "2026-04-01", "engine_decomposition.json")
        out.write_text(json.dumps(decomp.model_dump(mode="json"), default=str), encoding="utf-8")

        cb = ContextBuilder(
            memory_dir=self._memory_dir(tmp_path),
            curated_dir=tmp_path / "curated",
        )
        loaded = cb.load_engine_decomposition(bot_id="bot_x")
        assert loaded, "load_engine_decomposition returned empty for round-trip data"

    def test_ablation_analyzer_round_trip(self, tmp_path: Path):
        """AblationAnalyzer output → file → ContextBuilder.load_ablation_analysis."""
        from trading_assistant.analysis.context_builder import ContextBuilder
        from trading_assistant.skills.ablation_analyzer import AblationAnalyzer
        from tests.factories import make_trade

        flag = "use_oscillation_gate"
        trades = []
        for i in range(15):
            trades.append(make_trade(
                trade_id=f"on_{i}", bot_id="bot_x", pnl=-20.0 + i * 0.1,
                market_regime="trending",
                strategy_params_at_entry={flag: True},
            ))
        for i in range(15):
            trades.append(make_trade(
                trade_id=f"off_{i}", bot_id="bot_x", pnl=50.0 + i * 0.1,
                market_regime="trending",
                strategy_params_at_entry={flag: False},
            ))
        analysis = AblationAnalyzer().analyze(trades, "bot_x", period="2026-04-01")
        assert analysis.flags, "ablation analyzer produced no flags from valid input"

        out = self._curated_path(tmp_path, "bot_x", "2026-04-01", "ablation_analysis.json")
        out.write_text(json.dumps(analysis.model_dump(mode="json"), default=str), encoding="utf-8")

        cb = ContextBuilder(
            memory_dir=self._memory_dir(tmp_path),
            curated_dir=tmp_path / "curated",
        )
        loaded = cb.load_ablation_analysis(bot_id="bot_x")
        assert loaded, "load_ablation_analysis returned empty for round-trip data"

    def test_exit_tier_analyzer_round_trip(self, tmp_path: Path):
        """ExitTierAnalyzer output → file → ContextBuilder.load_exit_tier_analysis."""
        from datetime import datetime, timezone
        from trading_assistant.analysis.context_builder import ContextBuilder
        from trading_assistant.schemas.events import TradeEvent
        from trading_assistant.schemas.strategy_profile import (
            ExitProfile, ExitTier, StrategyProfile, StrategyRegistry,
        )
        from trading_assistant.skills.exit_tier_analyzer import ExitTierAnalyzer

        registry = StrategyRegistry(
            strategies={
                "test_strat": StrategyProfile(
                    display_name="TestStrategy",
                    bot_id="bot_x",
                    sub_engines=["MOMENTUM"],
                    exit_profile=ExitProfile(
                        tiers=[
                            ExitTier(tier_name="TP1", tier_type="take_profit",
                                     r_target=1.0, partial_pct=0.5),
                            ExitTier(tier_name="TP2", tier_type="take_profit",
                                     r_target=2.0, partial_pct=0.5),
                        ],
                    ),
                ),
            },
        )
        trades = [
            TradeEvent(
                trade_id=f"t{i}",
                bot_id="bot_x",
                pair="BTC/USDT",
                side="LONG",
                entry_time=datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
                exit_time=datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc),
                entry_price=50000.0,
                exit_price=50500.0 if i % 2 == 0 else 49500.0,
                position_size=1.0,
                pnl=500.0 if i % 2 == 0 else -500.0,
                pnl_pct=1.0 if i % 2 == 0 else -1.0,
                mfe_r=2.5 if i % 2 == 0 else 0.5,
                mae_r=-0.5 if i % 2 == 0 else -1.5,
                atr_at_entry=500.0,
            )
            for i in range(10)
        ]
        analysis = ExitTierAnalyzer(registry).analyze(trades, "bot_x", period="2026-04-01")
        assert analysis and analysis.get("tiers"), (
            "exit tier analyzer produced no tiers from valid input"
        )

        out = self._curated_path(tmp_path, "bot_x", "2026-04-01", "exit_tier_analysis.json")
        out.write_text(json.dumps(analysis, default=str), encoding="utf-8")

        cb = ContextBuilder(
            memory_dir=self._memory_dir(tmp_path),
            curated_dir=tmp_path / "curated",
        )
        loaded = cb.load_exit_tier_analysis(bot_id="bot_x")
        assert loaded, "load_exit_tier_analysis returned empty for round-trip data"
