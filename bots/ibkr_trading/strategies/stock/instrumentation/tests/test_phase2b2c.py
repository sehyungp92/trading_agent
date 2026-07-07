"""Tests for Phase 2B/2C instrumentation changes."""
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from strategies.stock.instrumentation.src.indicator_logger import IndicatorLogger, IndicatorSnapshot
from strategies.stock.instrumentation.src.filter_event_logger import FilterEventLogger
from strategies.stock.instrumentation.src.filter_decision import FilterDecision
from strategies.stock.instrumentation.src.orderbook_logger import OrderBookLogger, OrderBookContext
from strategies.stock.instrumentation.src.config_watcher import ConfigWatcher
from strategies.stock.instrumentation.src.experiment import ExperimentMetadata, ExperimentRegistry
from strategies.stock.instrumentation.src.sidecar import _DIR_TO_EVENT_TYPE, _EVENT_PRIORITY


# =========================================================================
# 2B-1: IndicatorSnapshot Tests
# =========================================================================

class TestIndicatorSnapshot:
    def test_helix_indicators_captured(self):
        """Helix-specific indicators (MACD, volume percentile) present."""
        snap = IndicatorSnapshot(
            bot_id="test", pair="NQ", timestamp="2026-03-15T10:30:00Z",
            indicators={
                "ema_9": 21250.50, "ema_21": 21180.25, "ema_50": 21050.0,
                "atr_14": 85.5, "macd": 70.25, "macd_signal": 55.10,
                "macd_histogram": 15.15, "volume_percentile": 72.0,
                "trend_strength": 0.82,
            },
            signal_name="helix_class_M", signal_strength=0.75,
            decision="enter", strategy_type="helix",
        )
        assert snap.indicators["macd"] == 70.25
        assert snap.indicators["volume_percentile"] == 72.0

    def test_context_includes_session_and_contract(self):
        """Context has RTH/ETH and contract_month populated."""
        snap = IndicatorSnapshot(
            bot_id="test", pair="NQ", timestamp="t",
            indicators={"ema_9": 1.0}, signal_name="x",
            signal_strength=0.5, decision="skip", strategy_type="helix",
            context={"session": "RTH", "contract_month": "2026-06"},
        )
        assert snap.context["session"] == "RTH"
        assert snap.context["contract_month"] == "2026-06"

    def test_signal_class_in_context(self):
        """Signal class M/F/T correctly reported in context."""
        snap = IndicatorSnapshot(
            bot_id="test", pair="NQ", timestamp="t",
            indicators={}, signal_name="helix_class_T",
            signal_strength=0.9, decision="enter", strategy_type="helix",
            context={"signal_class": "T"},
        )
        assert snap.context["signal_class"] == "T"

    def test_jsonl_roundtrip(self):
        """Write and read preserves all fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lgr = IndicatorLogger(data_dir=tmpdir, bot_id="test_bot")
            ts = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)
            result = lgr.log_snapshot(
                pair="NQ",
                indicators={"ema_9": 100.5, "atr_14": 20.0},
                signal_name="helix_class_M",
                signal_strength=0.6,
                decision="enter",
                strategy_type="helix",
                exchange_timestamp=ts,
                bar_id="2026-03-15T14:00Z_1h",
                context={"session": "RTH", "concurrent_positions": 1},
            )
            assert result.event_id != ""
            assert result.bot_id == "test_bot"

            # Read back from JSONL
            files = list(Path(tmpdir, "indicators").glob("*.jsonl"))
            assert len(files) == 1
            data = json.loads(files[0].read_text().strip())
            assert data["pair"] == "NQ"
            assert data["indicators"]["ema_9"] == 100.5
            assert data["context"]["session"] == "RTH"
            assert data["bar_id"] == "2026-03-15T14:00Z_1h"


# =========================================================================
# 2B-2: FilterDecisionEvent Tests
# =========================================================================

class TestFilterEventLogger:
    def test_margin_pct_matches_filter_decision(self):
        """margin_pct in event matches FilterDecision.margin_pct()."""
        fd = FilterDecision("pullback_range", 1.6, 1.2, True)
        expected_margin = fd.margin_pct()

        with tempfile.TemporaryDirectory() as tmpdir:
            lgr = FilterEventLogger(data_dir=tmpdir, bot_id="test")
            lgr.log_decision(fd, pair="NQ", strategy_type="helix")
            files = list(Path(tmpdir, "filter_decisions").glob("*.jsonl"))
            data = json.loads(files[0].read_text().strip())
            assert data["margin_pct"] == expected_margin

    def test_all_signal_class_filters_emitted(self):
        """Multiple filter decisions emitted from a single evaluation."""
        fds = [
            FilterDecision("pullback_range", 1.6, 1.2, True),
            FilterDecision("trend_strength", 0.80, 0.85, True),
            FilterDecision("extension_block", 2.5, 1.8, True),
            FilterDecision("stop_distance", 0.50, 0.45, True),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            lgr = FilterEventLogger(data_dir=tmpdir, bot_id="test")
            lgr.log_decisions(
                fds, pair="NQ", signal_name="helix_class_M",
                signal_strength=0.75, strategy_type="helix",
            )
            files = list(Path(tmpdir, "filter_decisions").glob("*.jsonl"))
            lines = files[0].read_text().strip().split("\n")
            assert len(lines) == 4
            names = {json.loads(l)["filter_name"] for l in lines}
            assert names == {"pullback_range", "trend_strength", "extension_block", "stop_distance"}

    def test_dict_and_object_accepted_via_facade(self):
        """Facade handles both FilterDecision objects and dicts."""
        from strategies.stock.instrumentation.src.facade import InstrumentationKit

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = MagicMock()
            mgr._config = {"data_dir": tmpdir, "bot_id": "test"}
            kit = InstrumentationKit(mgr, strategy_type="helix")

            # Mix of dict and FilterDecision
            fds = [
                FilterDecision("pullback_range", 1.6, 1.2, True),
                {"filter_name": "trend_strength", "threshold": 0.80,
                 "actual_value": 0.85, "passed": True},
            ]
            kit.on_filter_decisions(
                filter_decisions=fds, pair="NQ",
                signal_name="helix_class_M", strategy_type="helix",
            )
            files = list(Path(tmpdir, "filter_decisions").glob("*.jsonl"))
            lines = files[0].read_text().strip().split("\n")
            assert len(lines) == 2


# =========================================================================
# 2B-3: OrderBookContext Tests
# =========================================================================

class TestOrderBookContext:
    def test_nq_depth_data_included(self):
        """bid_levels/ask_levels with NQ tick size 0.25 prices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lgr = OrderBookLogger(data_dir=tmpdir, bot_id="test")
            ctx = lgr.log_context(
                pair="NQ", best_bid=21248.50, best_ask=21249.00,
                bid_levels=[
                    {"price": 21248.50, "size": 150},
                    {"price": 21248.25, "size": 120},
                ],
                ask_levels=[
                    {"price": 21249.00, "size": 130},
                    {"price": 21249.25, "size": 100},
                ],
                trade_context="signal_eval",
            )
            assert len(ctx.bid_levels) == 2
            assert len(ctx.ask_levels) == 2
            assert ctx.bid_levels[0]["price"] == 21248.50

    def test_imbalance_ratio_computed(self):
        """bid_depth / ask_depth gives imbalance_ratio."""
        ctx = OrderBookContext(
            bot_id="test", pair="NQ", timestamp="t",
            best_bid=21248.50, best_ask=21249.00,
            bid_depth_10bps=450.0, ask_depth_10bps=380.0,
        )
        assert ctx.imbalance_ratio == round(450.0 / 380.0, 4)

    def test_imbalance_zero_ask(self):
        ctx = OrderBookContext(
            bot_id="test", pair="NQ", timestamp="t",
            best_bid=100, best_ask=101, ask_depth_10bps=0.0,
        )
        assert ctx.imbalance_ratio == 0.0

    def test_spread_calculated_correctly(self):
        """0.50 / 21248.75 * 10000 ≈ 0.24 bps for NQ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lgr = OrderBookLogger(data_dir=tmpdir, bot_id="test")
            ctx = lgr.log_context(pair="NQ", best_bid=21248.50, best_ask=21249.00)
            mid = (21248.50 + 21249.00) / 2
            expected_bps = round((21249.00 - 21248.50) / mid * 10000, 2)
            assert ctx.spread_bps == expected_bps

    def test_jsonl_includes_imbalance(self):
        """to_dict includes computed imbalance_ratio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lgr = OrderBookLogger(data_dir=tmpdir, bot_id="test")
            lgr.log_context(
                pair="NQ", best_bid=100, best_ask=101,
                bid_depth_10bps=200, ask_depth_10bps=100,
            )
            files = list(Path(tmpdir, "orderbook").glob("*.jsonl"))
            data = json.loads(files[0].read_text().strip())
            assert data["imbalance_ratio"] == 2.0

    def test_event_id_changes_with_trade_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lgr = OrderBookLogger(data_dir=tmpdir, bot_id="test")
            ts = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)
            entry = lgr.log_context(
                pair="AAPL",
                best_bid=100.0,
                best_ask=100.1,
                trade_context="entry",
                exchange_timestamp=ts,
            )
            exit_ctx = lgr.log_context(
                pair="AAPL",
                best_bid=100.0,
                best_ask=100.1,
                trade_context="exit",
                exchange_timestamp=ts,
            )
            assert entry.event_id != exit_ctx.event_id


# =========================================================================
# 2B-4: ParameterChangeEvent Tests
# =========================================================================

class TestConfigWatcher:
    def test_uses_config_snapshot_infrastructure(self):
        """ConfigWatcher reuses snapshot_config_module."""
        # Create a temporary module
        import types
        mod = types.ModuleType("_test_config_mod")
        mod.BASE_RISK_PCT = 0.0125
        mod.MACD_FAST = 8
        import sys
        sys.modules["_test_config_mod"] = mod

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                watcher = ConfigWatcher(
                    bot_id="test", config_modules=["_test_config_mod"], data_dir=tmpdir,
                )
                # No changes yet
                changes = watcher.check()
                assert changes == []
        finally:
            del sys.modules["_test_config_mod"]

    def test_change_detected_after_reload(self):
        """Modify constant, re-snapshot → change event emitted."""
        import types
        import sys
        mod = types.ModuleType("_test_config_mod2")
        mod.STOP_MULT = 0.50
        sys.modules["_test_config_mod2"] = mod

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                watcher = ConfigWatcher(
                    bot_id="test", config_modules=["_test_config_mod2"], data_dir=tmpdir,
                )
                # Modify the constant
                mod.STOP_MULT = 0.40
                changes = watcher.check()
                assert len(changes) == 1
                assert changes[0]["param_name"] == "STOP_MULT"
                assert changes[0]["old_value"] == 0.50
                assert changes[0]["new_value"] == 0.40
                assert changes[0]["event_type"] == "parameter_change"

                # Verify JSONL written
                files = list(Path(tmpdir, "config_changes").glob("*.jsonl"))
                assert len(files) == 1
        finally:
            del sys.modules["_test_config_mod2"]


# =========================================================================
# 2B-5: Sidecar Mapping Tests
# =========================================================================

class TestSidecarMappings:
    def test_four_new_directories_mapped(self):
        """indicators/, filter_decisions/, orderbook/, config_changes/ all mapped."""
        assert _DIR_TO_EVENT_TYPE["indicators"] == "indicator_snapshot"
        assert _DIR_TO_EVENT_TYPE["filter_decisions"] == "filter_decision"
        assert _DIR_TO_EVENT_TYPE["orderbook"] == "orderbook_context"
        assert _DIR_TO_EVENT_TYPE["config_changes"] == "parameter_change"

    def test_new_event_types_have_priority(self):
        """New event types have priority entries."""
        assert "indicator_snapshot" in _EVENT_PRIORITY
        assert "filter_decision" in _EVENT_PRIORITY
        assert "orderbook_context" in _EVENT_PRIORITY
        assert "parameter_change" in _EVENT_PRIORITY


# =========================================================================
# 2C-1: ExperimentRegistry Orchestrator Format Tests
# =========================================================================

class TestExperimentRegistryOrchestrator:
    def test_legacy_yaml_still_works(self):
        """Existing experiments.yaml loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiments.yaml"
            path.write_text(yaml.dump({"experiments": {
                "exp_001": {
                    "hypothesis": "Tighter trail",
                    "variants": ["control", "tight"],
                    "start_date": "2026-03-01",
                    "strategy_type": "helix",
                },
            }}))
            reg = ExperimentRegistry(config_path=path)
            exp = reg.get("exp_001")
            assert exp is not None
            assert exp.variants == ["control", "tight"]
            assert exp.hypothesis == "Tighter trail"

    def test_orchestrator_yaml_format(self):
        """List of dicts with variant params/allocations loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiments.yaml"
            data = {
                "experiments": [
                    {
                        "experiment_id": "abc123",
                        "title": "Test tighter stop",
                        "status": "active",
                        "hypothesis": "Tighter stop improves Sharpe",
                        "start_date": "2026-03-01",
                        "variants": {
                            "control": {"params": {"CLASS_M_STOP_ATR_MULT": 0.50}, "allocation_pct": 50},
                            "treatment": {"params": {"CLASS_M_STOP_ATR_MULT": 0.40}, "allocation_pct": 50},
                        },
                        "allocation_method": "hash",
                        "success_metric": "pnl",
                        "max_duration_days": 14,
                    },
                ],
            }
            path.write_text(yaml.dump(data))
            reg = ExperimentRegistry(config_path=path)
            exp = reg.get("abc123")
            assert exp is not None
            assert exp.variants == ["control", "treatment"]
            assert exp.variant_params["control"] == {"CLASS_M_STOP_ATR_MULT": 0.50}
            assert exp.variant_params["treatment"] == {"CLASS_M_STOP_ATR_MULT": 0.40}
            assert exp.variant_allocations["control"] == 50
            assert exp.title == "Test tighter stop"
            assert exp.allocation_method == "hash"
            assert exp.max_duration_days == 14

    def test_orchestrator_format_with_list_variants(self):
        """Orchestrator format where variants is a plain list (no params)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiments.yaml"
            data = {
                "experiments": [
                    {
                        "experiment_id": "simple_exp",
                        "variants": ["control", "treatment"],
                        "start_date": "2026-01-01",
                    },
                ],
            }
            path.write_text(yaml.dump(data))
            reg = ExperimentRegistry(config_path=path)
            exp = reg.get("simple_exp")
            assert exp.variants == ["control", "treatment"]
            assert exp.variant_params == {}


# =========================================================================
# 2C-2: Deterministic Variant Assignment Tests
# =========================================================================

class TestVariantAssignment:
    def _make_registry(self, tmpdir, experiments):
        path = Path(tmpdir) / "experiments.yaml"
        path.write_text(yaml.dump({"experiments": experiments}))
        return ExperimentRegistry(config_path=path)

    def test_deterministic_same_inputs(self):
        """Same inputs give same output across calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir, [
                {
                    "experiment_id": "exp1",
                    "variants": {"control": {"allocation_pct": 50}, "treatment": {"allocation_pct": 50}},
                    "start_date": "2026-01-01",
                },
            ])
            v1 = reg.assign_variant("exp1", "trade_abc")
            v2 = reg.assign_variant("exp1", "trade_abc")
            assert v1 == v2
            assert v1 in ("control", "treatment")

    def test_distribution_approximates_allocation(self):
        """10k hashes approximate 50/50 allocation (within 5%)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir, [
                {
                    "experiment_id": "exp1",
                    "variants": {"control": {"allocation_pct": 50}, "treatment": {"allocation_pct": 50}},
                    "start_date": "2026-01-01",
                },
            ])
            counts = {"control": 0, "treatment": 0}
            for i in range(10000):
                v = reg.assign_variant("exp1", f"trade_{i}")
                counts[v] += 1
            # Within 5% of 50/50
            assert abs(counts["control"] / 10000 - 0.50) < 0.05
            assert abs(counts["treatment"] / 10000 - 0.50) < 0.05

    def test_unequal_allocation(self):
        """70/30 allocation is approximately reflected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir, [
                {
                    "experiment_id": "exp1",
                    "variants": {"control": {"allocation_pct": 70}, "treatment": {"allocation_pct": 30}},
                    "start_date": "2026-01-01",
                },
            ])
            counts = {"control": 0, "treatment": 0}
            for i in range(10000):
                v = reg.assign_variant("exp1", f"trade_{i}")
                counts[v] += 1
            assert abs(counts["control"] / 10000 - 0.70) < 0.05

    def test_equal_allocation_no_explicit_alloc(self):
        """Without variant_allocations, uses equal allocation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir, {
                "exp1": {
                    "hypothesis": "test",
                    "variants": ["a", "b", "c"],
                    "start_date": "2026-01-01",
                    "strategy_type": "helix",
                },
            })
            counts = {"a": 0, "b": 0, "c": 0}
            for i in range(9000):
                v = reg.assign_variant("exp1", f"trade_{i}")
                counts[v] += 1
            # Each ~33%
            for vname in counts:
                assert abs(counts[vname] / 9000 - 1/3) < 0.05

    def test_missing_experiment_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir, [])
            assert reg.assign_variant("nonexistent", "trade_1") == ""

    def test_get_variant_params_returns_overrides(self):
        """Correct params dict for each variant."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir, [
                {
                    "experiment_id": "exp1",
                    "variants": {
                        "control": {"params": {"STOP_ATR_MULT": 0.50}, "allocation_pct": 50},
                        "treatment": {"params": {"STOP_ATR_MULT": 0.40}, "allocation_pct": 50},
                    },
                    "start_date": "2026-01-01",
                },
            ])
            assert reg.get_variant_params("exp1", "control") == {"STOP_ATR_MULT": 0.50}
            assert reg.get_variant_params("exp1", "treatment") == {"STOP_ATR_MULT": 0.40}
            assert reg.get_variant_params("exp1", "nonexistent") == {}
            assert reg.get_variant_params("no_exp", "control") == {}


# =========================================================================
# 2C-4: Export Active with New Fields
# =========================================================================

class TestExportActiveEnhanced:
    def test_export_includes_variant_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiments.yaml"
            data = {
                "experiments": [
                    {
                        "experiment_id": "exp1",
                        "title": "Test stop",
                        "start_date": "2026-01-01",
                        "variants": {
                            "control": {"params": {"X": 1}, "allocation_pct": 50},
                            "treatment": {"params": {"X": 2}, "allocation_pct": 50},
                        },
                        "allocation_method": "hash",
                    },
                ],
            }
            path.write_text(yaml.dump(data))
            reg = ExperimentRegistry(config_path=path)
            export = reg.export_active(as_of="2026-06-01")
            assert "exp1" in export
            entry = export["exp1"]
            assert entry["variant_params"]["control"] == {"X": 1}
            assert entry["variant_allocations"]["control"] == 50
            assert entry["title"] == "Test stop"
            assert entry["allocation_method"] == "hash"


# =========================================================================
# Facade Integration Tests
# =========================================================================

class TestFacadePhase2B:
    def test_indicator_snapshot_via_facade(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = MagicMock()
            mgr._config = {"data_dir": tmpdir, "bot_id": "test"}
            kit = InstrumentationKit(mgr, strategy_type="helix")

            kit.on_indicator_snapshot(
                pair="NQ",
                indicators={"ema_9": 100.0, "atr_14": 10.0},
                signal_name="helix_class_M",
                signal_strength=0.8,
                decision="enter",
                strategy_type="helix",
                context={"session": "RTH"},
            )
            files = list(Path(tmpdir, "indicators").glob("*.jsonl"))
            assert len(files) == 1

    def test_orderbook_context_via_facade(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = MagicMock()
            mgr._config = {"data_dir": tmpdir, "bot_id": "test"}
            kit = InstrumentationKit(mgr, strategy_type="helix")

            kit.on_orderbook_context(
                pair="NQ", best_bid=21248.50, best_ask=21249.00,
                trade_context="entry",
            )
            files = list(Path(tmpdir, "orderbook").glob("*.jsonl"))
            assert len(files) == 1

    def test_inactive_facade_no_errors(self):
        """InstrumentationKit with None manager doesn't error on new methods."""
        kit = InstrumentationKit(None)
        # These should all be no-ops
        kit.on_indicator_snapshot(
            pair="NQ", indicators={}, signal_name="x",
            signal_strength=0, decision="skip", strategy_type="helix",
        )
        kit.on_filter_decisions([], pair="NQ")
        kit.on_orderbook_context(pair="NQ", best_bid=0, best_ask=0)


# Import at module level for the facade tests
from strategies.stock.instrumentation.src.facade import InstrumentationKit
