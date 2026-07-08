# ALCB T2 Greedy Optimization - Speedup Analysis

## Current Evaluation Flow

Each candidate evaluation in `run_greedy` follows this path:

1. **Config mutation** (~0ms): `_make_config` → `mutate_alcb_config` (dataclass `replace`, trivial)
2. **Engine construction** (~0ms): `ALCBIntradayEngine(config, replay)` — stores refs, applies `param_overrides`
3. **Engine.run()** (~24s): The bottleneck. Iterates 541 trading dates:
   - **Selection** (`alcb_selection_for_date`): Builds snapshot + runs selection pipeline.
     **CACHED by date** in `replay._alcb_selection_cache` — costs ~0s after first candidate.
   - **Compression box detection** (`detect_compression_box`): Called per tradable symbol per date
     (~20 syms × 541 dates = 10,820 calls). **NOT cached** — recomputed every run.
     Internally calls `squeeze_history` which is O(lookback × 50) = expensive.
   - **30m bar replay**: 541 dates × ~23 symbols × 13 bars = ~162K bar iterations.
     Most bars early-exit (no position, no campaign, outside entry window).
   - **Indicator computation per bar** (when entry/exit path is active):
     - `compute_campaign_avwap`: O(n) scan through bars since anchor (~100-200 bars)
     - `atr_from_bars`: O(50) per call, called 2-3× per entry check
     - `_compute_intraday_score`: calls AVWAP + RVOL + `classify_4h_regime` (EMA+ADX) + `ttm_squeeze_direction_bonus` (EMA)
   - **deepcopy of Campaign objects**: 10,820 calls per run
4. **Scoring** (~0ms): `extract_metrics` + `composite_score` over trade list

## Where the 600s Is Spent

- 27 candidates × 24s/candidate = **648s per round** (matches the ~600s claim)
- Each 24s run breaks down approximately:
  - Selection lookup (cached): **~0s**
  - Compression box detection (10,820 calls): **~8s** (squeeze_history is O(n²))
  - 30m bar loop + indicator calls: **~10s**
  - deepcopy + object creation: **~3s**
  - Entry/exit logic, sizing, SimBroker: **~3s**

## Top 3 Optimization Opportunities

### 1. Multiprocessing — Parallelize Candidate Evaluations (~5-8× speedup)

**Problem**: Candidates are evaluated sequentially in a `for` loop (L144-155).

**Solution**: Use `concurrent.futures.ProcessPoolExecutor`. Key requirements:
- Pre-warm the selection cache by running the baseline eval first (already happens).
- Each worker needs its own `replay` object — BUT the parquet DataFrames are read-only
  and can be shared via fork (Linux) or pickled (Windows).
- On Windows, use `multiprocessing` with `spawn` — the `replay` object is large (~2GB with
  414 symbols of daily+30m data). **Better approach**: pre-warm selection cache, pickle just
  the cache dict, and have workers reconstruct minimal replay or receive pre-sliced data.
- Alternative: Use `ThreadPoolExecutor` with GIL release via numpy. Won't help much since
  the hot path is pure Python loops.
- **Best Windows approach**: `ProcessPoolExecutor(max_workers=N)` where N = CPU cores (e.g., 8).
  Pass `(mutations, cfg_kwargs)` to workers. Each worker loads data independently.
  Trade data loading time (~10s one-time) for 8× throughput.

**Estimated speedup**: 648s → ~100s (8 cores) per round.

**Risk**: Memory — 8 copies of replay data = ~16GB. Mitigate by sharing read-only data
via `multiprocessing.shared_memory` or by having workers share a single loaded replay
(thread-safe since selection cache is pre-warmed and engine.run() is stateless w.r.t. replay).

### 2. Cache Compression Box Detection Across Runs (~30% speedup, ~7s saved per eval)

**Problem**: `detect_compression_box` is called in the engine's per-date loop (L268, L306)
for every tradable symbol on every date. The box depends only on `daily_bars` (from cached
selection) and `settings` params (`min_containment`, `max_squeeze_metric`, `breakout_tolerance_pct`).
For P5 candidates, these box-affecting params are FIXED in the base mutations — none of the
27 candidates change them. So the same box is computed 27 times identically.

**Solution**: Pre-compute a `box_cache: dict[(symbol, date), Box | None]` before the greedy
loop starts (during baseline eval). Store it on the engine or pass it in. Skip
`detect_compression_box` when cache hit.

**Estimated speedup**: 8s × 27 = 216s saved per round → 648s → ~430s (33% faster).

**Implementation**: Add `box_cache` param to `ALCBIntradayEngine.__init__`. In `run()`,
check cache before calling `detect_compression_box`. Build cache during baseline eval.

### 3. Incremental AVWAP/ATR with Running Accumulators (~15% speedup)

**Problem**: `compute_campaign_avwap` does a full scan from anchor every call — O(n) where
n grows as bars accumulate. `atr_from_bars` recomputes true ranges from scratch each call.
`bars_since_anchor` creates a new filtered list each time.

**Solution**:
- **AVWAP**: Maintain running `cum_pv` and `cum_vol` per symbol. On each new bar, add
  `bar.typical_price * bar.volume` to cumulative. AVWAP = cum_pv / cum_vol. O(1) per bar.
- **ATR**: Maintain a rolling deque of true ranges. On each new bar, append new TR and
  pop oldest if > period. ATR = mean(deque). O(1) per bar.
- **bars_30m_accum trimming** (L363-364): Replace `bars_30m_accum[sym] = bars_30m_accum[sym][-200:]`
  with a `collections.deque(maxlen=200)` — O(1) trim instead of O(200) list copy.

**Estimated speedup**: 3-4s per eval → ~15% improvement.

## Additional Quick Wins

### 4. Eliminate Redundant deepcopy (~1-2s per eval)

L256: `campaigns.setdefault(item.symbol, deepcopy(item.campaign))` — called for all 20
tradable symbols every date. Since selection is cached, the same Campaign objects are
deepcopied 541 times each. Instead, deepcopy only on first encounter per symbol.
Current code already uses `setdefault` so this only copies on first set — OK.
But the `item.campaign` from cached selection gets deepcopied even if the symbol was
already seen. Actually `setdefault` prevents this — only copies if key absent. This is fine.

### 5. Skip Dates Where No Box Exists (potential 50%+ date reduction)

Most dates have 0 active breakout campaigns. The engine still iterates all 13 bars for
all tradable symbols even when no campaign exists. Adding an early `continue` when
`not any(sym in campaigns and campaigns[sym].state in (BREAKOUT, CONTINUATION) for sym in replay_symbols - set(positions))`
and no open positions would skip ~60% of dates entirely.

### 6. Profile-Guided: Skip `_compute_intraday_score` When Not Needed

The intraday score is only used for conviction sizing and min_score gate. If a candidate
doesn't change these params, the score computation could be skipped entirely when no
gates or sizing depend on it. Check `cfg.ablation.use_intraday_scoring` and
`cfg.ablation.use_conviction_sizing` early and skip the ~O(500) computation.

## Recommended Implementation Order

1. **Compression box cache** (easiest, biggest single-eval improvement, ~30%)
2. **Multiprocessing** (biggest total improvement, ~5-8×, more complex on Windows)
3. **Incremental AVWAP/ATR** (moderate effort, ~15%)

Combined: 648s → ~50-80s per round (8-13× total speedup).
