"""Unit tests for libs.config.registry — register() decorator and STRATEGY_FACTORIES."""
from __future__ import annotations

import pytest

from libs.config.registry import STRATEGY_FACTORIES, get_factory, register


# ---------------------------------------------------------------------------
# Fixture — restore STRATEGY_FACTORIES after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_factories():
    saved = dict(STRATEGY_FACTORIES)
    yield
    STRATEGY_FACTORIES.clear()
    STRATEGY_FACTORIES.update(saved)


# ---------------------------------------------------------------------------
# Tests — successful registration
# ---------------------------------------------------------------------------

class TestRegister:
    """Verify @register places the factory in STRATEGY_FACTORIES."""

    def test_register_adds_factory(self) -> None:
        @register("TEST_1")
        async def my_factory(**kwargs):  # noqa: ARG001
            ...

        assert "TEST_1" in STRATEGY_FACTORIES
        assert STRATEGY_FACTORIES["TEST_1"] is my_factory

    def test_get_factory_returns_registered(self) -> None:
        @register("TEST_GET")
        async def factory_a(**kwargs):  # noqa: ARG001
            ...

        result = get_factory("TEST_GET")
        assert result is factory_a

    def test_register_returns_original_function(self) -> None:
        @register("TEST_IDENTITY")
        async def original(**kwargs):  # noqa: ARG001
            ...

        assert original.__name__ == "original"


# ---------------------------------------------------------------------------
# Tests — duplicate registration error
# ---------------------------------------------------------------------------

class TestDuplicateRegistration:
    """Registering two factories under the same ID must raise ValueError."""

    def test_duplicate_raises_value_error(self) -> None:
        @register("TEST_2")
        async def first_factory(**kwargs):  # noqa: ARG001
            ...

        with pytest.raises(ValueError, match="Duplicate factory"):
            @register("TEST_2")
            async def second_factory(**kwargs):  # noqa: ARG001
                ...

    def test_error_mentions_both_function_names(self) -> None:
        @register("TEST_DUP")
        async def alpha(**kwargs):  # noqa: ARG001
            ...

        with pytest.raises(ValueError) as exc_info:
            @register("TEST_DUP")
            async def beta(**kwargs):  # noqa: ARG001
                ...

        message = str(exc_info.value)
        assert "alpha" in message
        assert "beta" in message

    def test_original_factory_survives_duplicate_attempt(self) -> None:
        @register("TEST_SURVIVE")
        async def keeper(**kwargs):  # noqa: ARG001
            ...

        with pytest.raises(ValueError):
            @register("TEST_SURVIVE")
            async def intruder(**kwargs):  # noqa: ARG001
                ...

        assert get_factory("TEST_SURVIVE") is keeper


# ---------------------------------------------------------------------------
# Tests — get_factory for unknown ID
# ---------------------------------------------------------------------------

class TestGetFactoryUnknown:
    """get_factory should return None for unregistered strategy IDs."""

    def test_returns_none(self) -> None:
        assert get_factory("UNKNOWN_XYZ") is None

    def test_returns_none_empty_string(self) -> None:
        assert get_factory("") is None
