"""Tests for MAE (Maximum Adverse Excursion) tracking in AKC-Helix ActiveSetup."""

from strategies.swing.akc_helix.models import SetupInstance


class TestMAETracking:
    """Verify mae_r_trough field exists and behaves correctly on SetupInstance."""

    def test_mae_r_trough_field_exists(self):
        setup = SetupInstance()
        assert hasattr(setup, "mae_r_trough"), "SetupInstance must have mae_r_trough field"

    def test_mae_r_trough_default_zero(self):
        setup = SetupInstance()
        assert setup.mae_r_trough == 0.0

    def test_mae_r_trough_tracks_minimum(self):
        """Simulate what the engine does: update mae_r_trough when r_now drops."""
        setup = SetupInstance()
        r_values = [0.5, -0.3, 0.2, -1.1, -0.5, 0.8]
        for r_now in r_values:
            if r_now < setup.mae_r_trough:
                setup.mae_r_trough = r_now
        assert setup.mae_r_trough == -1.1
