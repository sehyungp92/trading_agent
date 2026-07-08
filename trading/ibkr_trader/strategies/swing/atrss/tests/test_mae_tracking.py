"""Tests for MAE (Maximum Adverse Excursion) tracking in ATRSS PositionBook."""
from strategies.swing.atrss.models import PositionBook


def test_position_book_has_mae_fields():
    pb = PositionBook(symbol="SPY")
    assert hasattr(pb, "mae")
    assert hasattr(pb, "mae_price")
    assert pb.mae == 0.0
    assert pb.mae_price == 0.0


def test_mae_defaults_alongside_mfe():
    """MAE fields should default to 0.0 just like MFE fields."""
    pb = PositionBook(symbol="ES")
    assert pb.mfe == 0.0
    assert pb.mfe_price == 0.0
    assert pb.mae == 0.0
    assert pb.mae_price == 0.0
