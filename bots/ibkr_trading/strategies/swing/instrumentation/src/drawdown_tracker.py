from dataclasses import dataclass

TIER_THRESHOLDS = [
    (15.0, "HALT", 0.0),
    (10.0, "DANGER", 0.25),
    (5.0, "CAUTION", 0.5),
    (0.0, "NORMAL", 1.0),
]


@dataclass
class DrawdownTracker:
    initial_equity: float
    peak_equity: float = 0.0
    current_equity: float = 0.0
    drawdown_pct: float = 0.0
    current_tier: str = "NORMAL"
    position_size_multiplier: float = 1.0

    def __post_init__(self):
        self.peak_equity = self.initial_equity
        self.current_equity = self.initial_equity

    def update_equity(self, equity: float) -> None:
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.drawdown_pct = 100.0 * (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        for threshold, tier, multiplier in TIER_THRESHOLDS:
            if self.drawdown_pct >= threshold:
                self.current_tier = tier
                self.position_size_multiplier = multiplier
                break

    def get_entry_context(self) -> dict:
        return {
            "drawdown_pct_at_entry": round(self.drawdown_pct, 4),
            "drawdown_tier_at_entry": self.current_tier,
            "position_size_multiplier": self.position_size_multiplier,
        }
