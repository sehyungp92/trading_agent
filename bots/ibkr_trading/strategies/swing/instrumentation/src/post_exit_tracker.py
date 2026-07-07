"""Post-exit price tracker — backfills 1h/4h price movement after trade exit.

Run periodically (e.g. every 30 min) to enrich completed trades with
post-exit price data for exit timing analysis.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("instrumentation.post_exit_tracker")

_MIN_AGE_HOURS = 4  # Wait at least 4h after exit before backfilling


class PostExitTracker:
    """Backfills post-exit price movement on completed trades."""

    def __init__(self, data_dir: str, data_provider):
        self._trades_dir = Path(data_dir) / "trades"
        self._results_dir = Path(data_dir) / "post_exit"
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._data_provider = data_provider

    def run_backfill(self) -> list[dict]:
        """Scan recent trade files and backfill post-exit prices.

        Merges results back onto the trade JSONL records so the sidecar
        forwards enriched events with post_exit_1h_price/4h_price fields.
        Also writes to a separate post_exit/ file for direct consumption.

        Returns list of backfilled result dicts.
        """
        results = []
        now = datetime.now(timezone.utc)

        if not self._trades_dir.exists():
            return results

        # Collect results per source file for efficient in-place amendment
        amendments_by_file: dict[Path, list[dict]] = {}

        for filepath in sorted(self._trades_dir.glob("trades_*.jsonl")):
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        trade = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if trade.get("stage") != "exit":
                        continue
                    if trade.get("post_exit_1h_pct") is not None:
                        continue

                    exit_time_str = trade.get("exit_time")
                    if not exit_time_str:
                        continue

                    exit_time = datetime.fromisoformat(exit_time_str)
                    if (now - exit_time).total_seconds() < _MIN_AGE_HOURS * 3600:
                        continue

                    result = self._backfill_trade(trade, exit_time)
                    if result:
                        results.append(result)
                        amendments_by_file.setdefault(filepath, []).append(result)

        if results:
            self._write_results(results)
            # Merge results back onto trade JSONL records
            for filepath, file_results in amendments_by_file.items():
                self._amend_trade_file(filepath, file_results)

        return results

    def _backfill_trade(self, trade: dict, exit_time: datetime) -> Optional[dict]:
        """Compute post-exit price movement for a single trade."""
        try:
            symbol = trade["pair"]
            exit_price = trade["exit_price"]
            side = trade["side"]

            price_1h = self._data_provider.get_price_at(symbol, exit_time + timedelta(hours=1))
            price_4h = self._data_provider.get_price_at(symbol, exit_time + timedelta(hours=4))

            if price_1h is None or price_4h is None:
                return None

            # Compute % move from exit price
            move_1h = (price_1h - exit_price) / exit_price * 100
            move_4h = (price_4h - exit_price) / exit_price * 100

            # For SHORT trades, favorable = price going down
            if side == "SHORT":
                move_1h = -move_1h
                move_4h = -move_4h

            return {
                "trade_id": trade["trade_id"],
                "pair": symbol,
                "side": side,
                "exit_price": exit_price,
                "exit_time": trade["exit_time"],
                "post_exit_1h_pct": round(move_1h, 4),
                "post_exit_4h_pct": round(move_4h, 4),
                "post_exit_1h_price": price_1h,
                "post_exit_4h_price": price_4h,
            }
        except Exception as e:
            logger.debug("Post-exit backfill failed for %s: %s", trade.get("trade_id"), e)
            return None

    def _amend_trade_file(self, filepath: Path, results: list[dict]) -> None:
        """Merge post-exit data back onto trade JSONL records by trade_id."""
        try:
            lookup = {r["trade_id"]: r for r in results}
            lines = filepath.read_text(encoding="utf-8").rstrip("\n").split("\n")
            modified = False

            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = event.get("trade_id")
                if tid in lookup:
                    r = lookup[tid]
                    event["post_exit_1h_pct"] = r["post_exit_1h_pct"]
                    event["post_exit_4h_pct"] = r["post_exit_4h_pct"]
                    event["post_exit_1h_price"] = r["post_exit_1h_price"]
                    event["post_exit_4h_price"] = r["post_exit_4h_price"]
                    lines[i] = json.dumps(event, default=str)
                    modified = True

            if modified:
                filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to amend trade file %s: %s", filepath, e)

    def _write_results(self, results: list[dict]) -> None:
        """Append results to daily JSONL file."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self._results_dir / f"post_exit_{today}.jsonl"
        with open(filepath, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
