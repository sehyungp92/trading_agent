import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from strategies.stock.instrumentation.src.market_snapshot import MarketSnapshotService, MarketSnapshot


class MockDataProvider:
    """Mock IBKR data provider for testing."""
    def get_bid_ask(self, symbol):
        return (20500.0, 20500.50)

    def get_last_price(self, symbol):
        return 20500.25

    def get_atr(self, symbol):
        return 85.0


class TestMarketSnapshot:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "market_snapshots": {"interval_seconds": 60, "symbols": ["NQ"]},
        }
        self.service = MarketSnapshotService(self.config, MockDataProvider())

    def test_capture_now_returns_snapshot(self):
        snap = self.service.capture_now("NQ")
        assert snap.symbol == "NQ"
        assert snap.bid == 20500.0
        assert snap.ask == 20500.50
        assert snap.last_trade_price == 20500.25
        assert snap.atr_14 == 85.0

    def test_mid_price_computed(self):
        snap = self.service.capture_now("NQ")
        expected_mid = (20500.0 + 20500.50) / 2
        assert abs(snap.mid - expected_mid) < 0.01

    def test_spread_bps_positive(self):
        snap = self.service.capture_now("NQ")
        assert snap.spread_bps > 0

    def test_snapshot_id_is_set(self):
        snap = self.service.capture_now("NQ")
        assert snap.snapshot_id
        assert len(snap.snapshot_id) == 12

    def test_capture_writes_to_file(self):
        self.service.capture_now("NQ")
        files = list(Path(self.tmpdir).joinpath("snapshots").glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text().strip()
        data = json.loads(content)
        assert data["symbol"] == "NQ"
        assert data["bid"] == 20500.0

    def test_degraded_snapshot_on_failure(self):
        """Snapshot service must never crash, even with bad data."""
        bad_provider = MagicMock()
        bad_provider.get_bid_ask.side_effect = Exception("connection lost")
        bad_provider.get_last_price.side_effect = Exception("connection lost")
        bad_provider.get_atr.side_effect = Exception("connection lost")
        service = MarketSnapshotService(self.config, bad_provider)
        snap = service.capture_now("NQ")
        assert snap.symbol == "NQ"
        assert snap.bid == 0  # degraded

    def test_degraded_snapshot_on_none_provider(self):
        """No data provider should produce degraded snapshot, not crash."""
        service = MarketSnapshotService(self.config, data_provider=None)
        snap = service.capture_now("NQ")
        assert snap.symbol == "NQ"
        assert snap.bid == 0
        assert snap.ask == 0

    def test_cache_stores_latest(self):
        self.service.capture_now("NQ")
        cached = self.service.get_latest("NQ")
        assert cached is not None
        assert cached.symbol == "NQ"
        assert cached.bid == 20500.0

    def test_cache_miss_returns_none(self):
        assert self.service.get_latest("ES") is None

    def test_run_periodic_captures_all_symbols(self):
        config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "market_snapshots": {"interval_seconds": 60, "symbols": ["NQ", "MNQ"]},
        }
        service = MarketSnapshotService(config, MockDataProvider())
        service.run_periodic()
        assert service.get_latest("NQ") is not None
        assert service.get_latest("MNQ") is not None

    def test_to_dict(self):
        snap = self.service.capture_now("NQ")
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert d["symbol"] == "NQ"
        assert "bid" in d
        assert "ask" in d
        assert "mid" in d
