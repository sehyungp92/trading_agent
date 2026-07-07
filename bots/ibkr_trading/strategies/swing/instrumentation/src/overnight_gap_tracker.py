class OvernightGapTracker:
    def __init__(self):
        self._prev_closes: dict[str, float] = {}

    def record_close(self, symbol: str, close_price: float) -> None:
        self._prev_closes[symbol] = close_price

    def compute_gap(self, symbol: str, current_open: float) -> dict:
        prev = self._prev_closes.get(symbol)
        if prev is None:
            return {"overnight_gap_pct": None, "prev_close_price": None}
        gap_pct = 100.0 * (current_open - prev) / prev
        return {
            "overnight_gap_pct": round(gap_pct, 4),
            "prev_close_price": prev,
        }
