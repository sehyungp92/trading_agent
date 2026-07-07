"""Tests for MarketSnapshotService."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from strategies.swing.instrumentation.src.market_snapshot import MarketSnapshotService, MarketSnapshot


class MockDataProvider:
    """Mock exchange data for testing."""
    def get_ticker(self, symbol):
        return {"bid": 50000.0, "ask": 50010.0, "last": 50005.0, "quoteVolume": 1000000}

    def get_ohlcv(self, symbol, timeframe="1h", limit=15):
        base = 50000
        return [[i * 3600000, base, base + 100, base - 100, base + 50, 1000]
                for i in range(limit)]


class TestMarketSnapshot:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "market_snapshots": {"interval_seconds": 60, "symbols": ["BTC/USDT"]},
        }
        self.service = MarketSnapshotService(self.config, MockDataProvider())

    def test_capture_now_returns_snapshot(self):
        snap = self.service.capture_now("BTC/USDT")
        assert snap.symbol == "BTC/USDT"
        assert snap.bid == 50000.0
        assert snap.ask == 50010.0
        assert snap.mid == 50005.0
        assert snap.spread_bps > 0

    def test_capture_writes_to_file(self):
        self.service.capture_now("BTC/USDT")
        files = list(Path(self.tmpdir).joinpath("snapshots").glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text().strip()
        data = json.loads(content)
        assert data["symbol"] == "BTC/USDT"

    def test_degraded_snapshot_on_failure(self):
        bad_provider = MagicMock()
        bad_provider.get_ticker.side_effect = Exception("connection lost")
        service = MarketSnapshotService(self.config, bad_provider)
        snap = service.capture_now("BTC/USDT")
        assert snap.symbol == "BTC/USDT"
        assert snap.bid == 0

    def test_get_latest_returns_cached(self):
        self.service.capture_now("BTC/USDT")
        latest = self.service.get_latest("BTC/USDT")
        assert latest is not None
        assert latest.symbol == "BTC/USDT"

    def test_get_latest_returns_none_for_unknown(self):
        assert self.service.get_latest("UNKNOWN") is None

    def test_snapshot_has_snapshot_id(self):
        snap = self.service.capture_now("BTC/USDT")
        assert snap.snapshot_id
        assert len(snap.snapshot_id) == 12

    def test_snapshot_to_dict(self):
        snap = self.service.capture_now("BTC/USDT")
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert "symbol" in d
        assert "bid" in d

    def test_run_periodic(self):
        self.service.symbols = ["BTC/USDT"]
        self.service.run_periodic()
        assert self.service.get_latest("BTC/USDT") is not None

    def test_dict_provider(self):
        """Test with dict-based data provider (as strategies pass cached data)."""
        config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "market_snapshots": {"interval_seconds": 60, "symbols": []},
        }
        provider = {"QQQ": {"last_price": 500.0, "bid": 499.9, "ask": 500.1}}
        service = MarketSnapshotService(config, provider)
        snap = service.capture_now("QQQ")
        assert snap.last_trade_price == 500.0
        assert snap.bid == 499.9
