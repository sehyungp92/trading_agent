import sys
import time
from pathlib import Path

BOT_ROOT = Path(__file__).resolve().parents[4]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from backtests.stock.engine.iaric_pullback_engine import IARICPullbackDailyEngine
from backtests.stock.engine.iaric_pullback_intraday_hybrid_engine import IARICPullbackIntradayHybridEngine
from backtests.stock.engine.research_replay import ResearchReplay
from strategies.stock.iaric.config import StrategySettings

V2_COMMON = dict(
    pb_v2_enabled=True,
    pb_v2_signal_floor=75.0,
    pb_v2_flow_grace_days=2,
    pb_entry_rank_max=999,
    pb_entry_rank_pct_max=100.0,
    pb_v2_depth_thresh=1.5,
    pb_open_scored_enabled=True,
)

replay = ResearchReplay(
    data_dir=str(BOT_ROOT / "backtests" / "stock" / "data" / "raw"),
    start_date="2023-06-01",
    end_date="2025-03-31",
)
replay.load_all_data()

print(f"{'Floor':>5}  {'Mode':>8}  {'Trades':>6}  {'Net':>10}  {'PF':>6}  {'AvgR':>8}  {'DD':>6}  {'Sharpe':>7}  {'Time':>5}")
print("-" * 80)

for floor in [75, 85, 95]:
    for mode in ["daily", "intraday_hybrid"]:
        muts = {**V2_COMMON, "pb_v2_signal_floor": float(floor)}
        if mode == "intraday_hybrid":
            muts["pb_execution_mode"] = "intraday_hybrid"
        
        settings = StrategySettings(**muts)
        
        t0 = time.time()
        if mode == "daily":
            engine = IARICPullbackDailyEngine(replay, settings)
        else:
            engine = IARICPullbackIntradayHybridEngine(replay, settings)
        
        result = engine.run()
        elapsed = time.time() - t0
        
        p = result.metrics
        label = "hybrid" if mode == "intraday_hybrid" else "daily"
        print(f"F{floor:>3}  {label:>8}  {p.total_trades:>6}  {p.net_pnl:>10,.0f}  {p.profit_factor:>6.2f}  {p.avg_r:>8.3f}  {p.max_drawdown_pct:>5.1f}%  {p.sharpe:>7.2f}  {elapsed:>4.0f}s")

print("\nDone.")
