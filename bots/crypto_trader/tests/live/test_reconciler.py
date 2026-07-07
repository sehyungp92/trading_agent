"""Tests for position reconciler."""

import pytest

from crypto_trader.core.models import Position, Side
from crypto_trader.live.reconciler import PositionReconciler


class TestPositionReconciler:
    def test_all_reconciled(self):
        r = PositionReconciler()
        expected = {
            "BTC": Position("BTC", Side.LONG, 0.1, 50000.0),
        }
        actual = [
            Position("BTC", Side.LONG, 0.1, 50000.0),
        ]
        discrepancies = r.reconcile(expected, actual)
        assert len(discrepancies) == 0

    def test_missing_position(self):
        r = PositionReconciler()
        expected = {
            "BTC": Position("BTC", Side.LONG, 0.1, 50000.0),
        }
        actual = []
        discrepancies = r.reconcile(expected, actual)
        assert len(discrepancies) == 1
        assert discrepancies[0].kind == "missing"
        assert discrepancies[0].symbol == "BTC"

    def test_phantom_position(self):
        r = PositionReconciler()
        expected = {}
        actual = [
            Position("ETH", Side.SHORT, 1.0, 3000.0),
        ]
        discrepancies = r.reconcile(expected, actual)
        assert len(discrepancies) == 1
        assert discrepancies[0].kind == "phantom"
        assert discrepancies[0].symbol == "ETH"

    def test_qty_mismatch(self):
        r = PositionReconciler()
        expected = {
            "BTC": Position("BTC", Side.LONG, 0.1, 50000.0),
        }
        actual = [
            Position("BTC", Side.LONG, 0.15, 50000.0),
        ]
        discrepancies = r.reconcile(expected, actual)
        assert len(discrepancies) == 1
        assert discrepancies[0].kind == "qty_mismatch"

    def test_direction_mismatch(self):
        r = PositionReconciler()
        expected = {
            "BTC": Position("BTC", Side.LONG, 0.1, 50000.0),
        }
        actual = [
            Position("BTC", Side.SHORT, 0.1, 50000.0),
        ]
        discrepancies = r.reconcile(expected, actual)
        assert len(discrepancies) == 1
        assert discrepancies[0].kind == "direction_mismatch"

    def test_expected_none_is_clean_when_actual_flat(self):
        r = PositionReconciler()
        expected = {"BTC": None, "ETH": Position("ETH", Side.LONG, 1.0, 3000.0)}
        actual = [Position("ETH", Side.LONG, 1.0, 3000.0)]
        discrepancies = r.reconcile(expected, actual)
        assert len(discrepancies) == 0

    def test_expected_none_flags_configured_symbol_phantom(self):
        r = PositionReconciler()
        expected = {"BTC": None}
        actual = [Position("BTC", Side.LONG, 0.1, 50000.0)]

        discrepancies = r.reconcile(expected, actual)

        assert len(discrepancies) == 1
        assert discrepancies[0].kind == "phantom"
        assert discrepancies[0].symbol == "BTC"

    def test_expected_unknown_qty_checks_direction_without_qty_mismatch(self):
        r = PositionReconciler()
        expected = {
            "BTC": Position(
                "BTC",
                Side.LONG,
                0.0,
                0.0,
                metadata={"qty_known": False},
            ),
        }
        actual = [Position("BTC", Side.LONG, 0.1, 50000.0)]

        assert r.reconcile(expected, actual) == []

    def test_expected_unknown_qty_still_requires_nonzero_actual_position(self):
        r = PositionReconciler()
        expected = {
            "BTC": Position(
                "BTC",
                Side.LONG,
                0.0,
                0.0,
                metadata={"qty_known": False},
            ),
        }
        actual = [Position("BTC", Side.LONG, 0.0, 50000.0)]

        discrepancies = r.reconcile(expected, actual)

        assert len(discrepancies) == 1
        assert discrepancies[0].kind == "missing"

    def test_flat_unexpected_exchange_rows_are_ignored(self):
        r = PositionReconciler()

        discrepancies = r.reconcile({}, [Position("BTC", Side.LONG, 0.0, 50000.0)])

        assert discrepancies == []

    def test_multiple_discrepancies(self):
        r = PositionReconciler()
        expected = {
            "BTC": Position("BTC", Side.LONG, 0.1, 50000.0),
            "ETH": Position("ETH", Side.SHORT, 2.0, 3000.0),
        }
        actual = [
            Position("BTC", Side.LONG, 0.2, 50000.0),  # qty mismatch
            Position("SOL", Side.LONG, 10.0, 150.0),     # phantom
        ]
        discrepancies = r.reconcile(expected, actual)
        assert len(discrepancies) == 3  # qty_mismatch + missing ETH + phantom SOL
