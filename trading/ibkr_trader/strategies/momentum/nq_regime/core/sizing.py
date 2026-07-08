from __future__ import annotations

from strategies.momentum.nq_regime import config
from strategies.momentum.nq_regime.config import Grade, ModuleId
from strategies.momentum.nq_regime.core.scoring import risk_pct_for_grade
from strategies.scalp._shared.nq_contract import compute_contracts, spec_for


def compute_position_size(
    *,
    equity: float,
    grade: Grade,
    stop_distance: float,
    trade_symbol: str = "MNQ",
    max_contracts: int | None = None,
    post_news: bool = False,
    after_loss: bool = False,
    module: ModuleId = ModuleId.NONE,
) -> tuple[int, float]:
    risk_pct = risk_pct_for_grade(grade, post_news=post_news, after_loss=after_loss)
    spec = spec_for(trade_symbol)
    qty = compute_contracts(
        equity=equity,
        risk_pct=risk_pct,
        stop_distance=stop_distance,
        point_value=spec.point_value,
        max_contracts=max_contracts,
    )
    if (
        qty <= 0
        and module is ModuleId.STRUCTURAL_EXPANSION
        and config.STRUCTURAL_ALLOW_MIN_MICRO_SIZE
        and trade_symbol.upper() == "MNQ"
    ):
        per_contract_risk = stop_distance * spec.point_value
        max_risk = equity * config.STRUCTURAL_MIN_MICRO_MAX_RISK_PCT
        if 0 < per_contract_risk <= max_risk:
            return 1, per_contract_risk / equity
    return qty, risk_pct
