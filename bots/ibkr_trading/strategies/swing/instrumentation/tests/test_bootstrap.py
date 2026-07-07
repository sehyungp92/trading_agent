"""Tests for bootstrap_instrumentation function."""
from unittest.mock import patch, MagicMock
from strategies.swing.instrumentation.src.bootstrap import bootstrap_instrumentation


class TestBootstrapInstrumentation:
    def test_bootstrap_with_strategy_id_overrides_bot_id(self):
        """Verify that passing strategy_id overrides bot_id in config."""
        ctx = bootstrap_instrumentation(strategy_id="ATRSS")
        assert ctx.trade_logger.bot_id == "ATRSS"

    def test_bootstrap_without_strategy_id_uses_config_default(self):
        """Verify that without strategy_id, bot_id defaults to 'swing_multi_01'."""
        ctx = bootstrap_instrumentation()
        assert ctx.trade_logger.bot_id == "swing_multi_01"


def test_bootstrap_kit_returns_kit_instance():
    """Verify that bootstrap_kit returns an InstrumentationKit instance."""
    from strategies.swing.instrumentation.src.bootstrap import bootstrap_kit
    from strategies.swing.instrumentation.src.kit import InstrumentationKit

    kit = bootstrap_kit(strategy_id="ATRSS", symbols=["QQQ"])
    assert isinstance(kit, InstrumentationKit)
    assert kit.strategy_id == "ATRSS"
