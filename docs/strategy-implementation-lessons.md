## Core implementation model

### 1. Build one shared strategy core, not separate live and backtest brains

Future strategies should put all actual decision ownership into a shared core:

```text
strategy/core/state.py        # typed state dataclasses
strategy/core/logic.py        # state machine and decision transitions
strategy/core/serializers.py  # snapshot/hydration compatibility
strategy/core/data_policy.py  # only if strategy-specific timing/session rules are needed
```

The live engine should only be an adapter around that core: async scheduling, market-data ingestion, OMS submission, persistence, instrumentation, and runtime contracts. The backtest engine should also only be an adapter: deterministic replay, simulated fills, equity/trade collection, and diagnostics.

The main lesson is that signal logic duplication is not the real enemy; **orchestration drift** is. The reports identify duplicated higher-timeframe handling, order staging, fill timing, accounting mutation, end-of-day handling, diagnostics normalization, and decision logging as the main divergence points. 

---

### 2. Make backtests thin replay drivers over the same core

A backtest should not reimplement the strategy. It should replay bars/events through the same core used live.

Use this model:

```text
ReplayStep(bar/update/fill)
    -> shared core transition
    -> neutral actions
    -> backtest execution adapter
    -> simulated fills
    -> fills returned to shared core
    -> canonical decision/trade outputs
```

Backtests should preserve existing CLI, optimization, diagnostics, and result shapes while internally becoming replay drivers plus analytics plumbing. The completion plan states that a strategy is not complete until both the live wrapper and backtest driver use the same shared decision state machine; core code alone is not enough if one side still owns real transitions separately. 

---

### 3. Centralize completed-bar and session semantics before building strategy logic

Every future strategy should receive only vetted, completed market data. Do not let individual strategies implement ad hoc `drop last bar`, daily alignment, or 4H/1H availability rules.

Required pattern:

```text
market data source
    -> shared completed-bar policy
    -> strategy-local data_policy only if needed
    -> shared core
```

The shared policy should cover:

* intraday higher-timeframe bar availability
* previous-completed-session daily exposure
* explicit incomplete-last-bar dropping
* exchange timezone and session handling
* consistent live/backtest behavior

The reports call completed-bar availability the single most important live/backtest parity issue, and specify that strategy cores should receive already-vetted completed bars rather than containing bespoke tail-filtering logic. 

---

### 4. Express orders as neutral strategy actions

Shared cores should not emit live OMS objects or backtest broker orders directly. They should emit neutral actions such as:

```text
SubmitEntry
SubmitExit
SubmitAddOnEntry
SubmitProtectiveStop
ReplaceProtectiveStop
SubmitProfitTarget
SubmitPartialExit
SubmitMarketExit
CancelAction
FlattenPosition
```

Then adapters translate those actions:

```text
shared core action
    -> live adapter -> OMSOrder / Intent
    -> backtest adapter -> SimOrder / SimBroker
```

This keeps transport-specific logic out of the core while allowing live and backtest execution semantics to be compared. The reports explicitly recommend neutral action schemas, live OMS adapters, and backtest/parity execution adapters, with equivalence tests proving that the same action produces the correct live and simulated order behavior. 

---

### 5. Preserve live infrastructure as authoritative

Do not simplify live trading by forcing it through backtest abstractions. Live execution should keep the existing production responsibilities:

* OMS persistence
* broker callback handling
* runtime preflight
* heartbeat emission
* watchdog visibility
* readiness checks
* restart recovery
* historical-data pacing and circuit breaking
* existing strategy IDs and health contracts

The target split is: **strategy logic shared, execution adapter different, market-data driver different, observability sink different**. The alignment plan is explicit that live should not be forced through `SimBroker`; live should keep OMS/Intent flows while backtests simulate the same neutral actions. 

---

### 6. Freeze baselines before structural refactors

Before changing architecture, freeze the known-correct outputs.

A future strategy migration should include:

```text
tests/fixtures/backtest_baselines/manifest.json
    - artifact ID
    - source output path
    - regeneration executor
    - entrypoint
    - arguments
    - expected hash
    - parsed key metrics
```

Every migration step must prove either:

* no output drift occurred, or
* the drift is deliberately accepted as a behavior-changing fix.

The reports emphasize manifest-driven regeneration, isolated per-artifact regeneration, deterministic output hashing, and no-drift verification as the primary protection against accidental behavior changes during refactor work. 

---

### 7. Use canonical decision and trade schemas

Future strategies should emit a common decision stream and normalized trade outcomes.

Decision events should capture:

```text
timestamp
strategy ID
symbol/instrument
decision code
reason/details
state snapshot reference
actions emitted
```

Trade outcomes should normalize:

```text
entry timestamp
decision timestamp
fill timestamp
exit timestamp
gross PnL
commission
net PnL
realized/unrealized state
exit reason
route/cohort metadata where relevant
```

The reports describe `DecisionEvent` and `TradeOutcome` as the shared internal language for parity and analytics, while warning that normalized backtest trade outcomes should not replace live OMS truth objects. 

---

### 8. Treat snapshot and hydration as first-class strategy contracts

Every shared core should support typed serialization and restart-safe hydration.

Implementation expectations:

* use explicit dataclasses or typed state objects
* serialize enums, tuples, sets, deques, nested dataclasses, and typed dict keys safely
* preserve backward compatibility for existing persisted state where needed
* test snapshot → hydrate → continue behavior
* avoid opaque payload blobs when typed state is available

The reports highlight the move from opaque lifecycle/payload scaffolding toward strategy-local typed bar/update/fill payloads and typed serializers, specifically to improve restart, parity, and optimizer auditability. 

---

### 9. Keep exceptions explicit and parity-tested

If a live path cannot or should not follow the standard architecture, do not hide it.

For any exception:

```text
- document why it exists
- define whether it is temporary or permanent
- share decision/config logic where possible
- keep execution differences explicit
- add dedicated parity tests against the backtest mirror
```

The reports use the overlay path as the model: it remains a documented live-only execution exception, but its decision/allocation behavior is shared with the backtest mirror and covered by parity tests. 

---

### 10. Optimize with replay bundles and source-fingerprinted caches

Future auto-optimization should reuse deterministic replay data rather than repeatedly rebuilding full contexts.

Recommended cache model:

```text
ReplayBundle
    - source fingerprint
    - resolved data root
    - replay-ready data
    - metadata for optimization
```

Cache keys should include:

* source fingerprint
* data directory/root
* phase
* scoring weights
* hard rejects
* relevant run context

Also keep raw final-metric caches separate from phase-scoring caches when the same mutation can be reused across diagnostics or final evaluation. The reports emphasize source-fingerprinted replay bundles, warmed contexts, worker-pool reuse, deterministic invalidation, and stable output ordering so performance improvements do not create false diagnostic drift. 

---

## Implementation checklist for a coding agent

```text
1. Start with a shared core
   - Create core/state.py, core/logic.py, core/serializers.py.
   - Put real decision transitions in the core.
   - Keep live/backtest wrappers thin.

2. Apply completed-bar policy before the core
   - No partial higher-timeframe bars.
   - No same-day daily context before session completion.
   - No strategy-local ad hoc drop-last logic unless wrapped in shared data_policy.

3. Emit neutral actions
   - Core emits SubmitEntry, SubmitExit, ReplaceProtectiveStop, FlattenPosition, etc.
   - Live adapter maps actions to OMS/Intent.
   - Backtest adapter maps actions to SimBroker orders.

4. Feed fills back into the core
   - Entry fills, partial fills, stop updates, cancels, replaces, and flattens must all return as events.
   - Do not clear working-order state until the core has consumed the fill/update.
   - Reject or quarantine unmatched fills; never let them flow through default exit paths.

5. Preserve live contracts
   - health_status() unchanged.
   - snapshot/hydrate unchanged or backward compatible.
   - _record_decision() and heartbeat fields preserved.
   - OMS, watchdog, readiness, runtime preflight, and broker-session hardening remain active.

6. Make the backtest a replay driver
   - Step deterministic bars/events through the core.
   - Use the backtest adapter for simulated execution.
   - Emit legacy result objects only at the boundary if needed.
   - Keep diagnostics and optimizer entrypoints stable.

7. Normalize observability
   - Emit DecisionEvent streams.
   - Emit canonical TradeOutcome records.
   - Keep live OMS accounting authoritative.
   - Use normalized records for parity, diagnostics, and analytics.

8. Freeze outputs before refactor
   - Add manifest entries with regeneration command, args, output path, and expected hash.
   - Run baseline regeneration before and after each migration step.
   - Treat any unexpected text or metric drift as a blocker.

9. Keep optimizer compatibility
   - Use typed ReplayBundle inputs.
   - Namespace caches by source fingerprint and run context.
   - Invalidate replay bundles, metric caches, config caches, and worker pools together.
   - Ensure warm-cache diagnostics do not force duplicate full-engine runs.

10. Prove completion
   - Unit tests for core transitions.
   - Adapter equivalence tests.
   - Replay/live decision-stream parity.
   - Baseline no-drift regeneration.
   - Runtime/OMS/watchdog/heartbeat regression tests.
   - Paper-trading evidence before production cutover.
```

## Condensed instruction for future strategy implementation

Implement the strategy once as a typed shared core that owns state transitions and emits neutral actions. Put completed-bar/session policy before the core. Let live and backtest differ only through adapters: live uses OMS, runtime, watchdog, heartbeat, and broker-session infrastructure; backtest uses replay, simulated execution, diagnostics, and optimization tooling. Freeze corrected outputs before refactors, preserve public wrappers, emit canonical decision/trade records, test replay/live equivalence, and require a paper-trading gate before deployment.


## Backtest implementation errors to prevent

### 1. Never expose incomplete higher-timeframe bars

Any lower-timeframe bar must only see higher-timeframe data that was fully complete and available at that moment. Do **not** rely on default `resample()` timestamps unless they are explicitly right-labeled/right-closed or shifted to bar-close availability.

Required safeguards:

* Higher-timeframe bars must be timestamped at their **availability time**, not the start of the window.
* A 5m/15m/hourly bar must not see the current unfinished 30m/1H/4H bar.
* Intraday logic must not see the same day’s daily OHLC before the trading day/session has ended.
* Daily bars indexed at midnight must be mapped with explicit daily-specific logic, usually exposing only the prior completed daily bar during the current session.

This was one of the largest recurring defects: higher-timeframe bars were made available too early, contaminating regime filters, state machines, entries, exits, and headline alpha claims.  

---

### 2. Do not approve and fill a trade using the same completed bar

If a signal depends on a bar’s close, high, low, or any indicator computed from the completed bar, the earliest realistic market fill is the **next bar**, unless a pre-existing resting order was already active before the bar began.

Required safeguards:

* Signal from bar `t` → market/order decision after bar `t` closes → fill on bar `t+1` or later.
* Same-bar fills are only allowed when the order was resting before the bar opened.
* If using a resting order, the decision to place it must not depend on that same bar’s final close/high/low.
* Add tests proving entries cannot use a bar’s closing information to fill earlier inside that same bar.

The reports repeatedly flagged same-bar execution optimism, including cases where the final state of a bar approved a trade that was then credited at a price touched earlier in that bar.  

---

### 3. Route every fill through one broker/execution model

Do not hand-roll entry, stop, partial, flatten, or end-of-data exits in strategy code. All trade execution should pass through a shared broker or fill engine.

Required safeguards:

* Market exits must pay adverse slippage and commission.
* Stop exits must handle gap-through-stop behavior realistically.
* Passive limit exits must not assume perfect fills from a one-tick touch or favorable gap-through unless explicitly modeled and tested.
* Partial exits must not fill at the bar high/low after the trigger is observed.
* End-of-data, stale, emergency, time-decay, regime-flip, and discretionary flattens must use the same friction model as normal exits.

Several reports found optimistic discretionary exits, partial exits priced at favorable intrabar extremes, exact-stop fills, missing exit friction, and passive-limit assumptions that were too generous.  

---

### 4. Use mark-to-market equity for all risk metrics

Do not compute drawdown, Sharpe, Sortino, Calmar, volatility, or portfolio risk from realized-only equity.

Required safeguards:

* Maintain both `realized_equity` and `mark_to_market_equity`.
* Risk metrics must use mark-to-market equity.
* Realized-only curves may be reported separately, but never as the main equity curve.
* Open positions must be marked every bar using current tradable prices.

The reports repeatedly found realized-only equity curves understating drawdowns and making Sharpe/Calmar look too good.  

---

### 5. Make PnL, commission, and equity semantics internally consistent

Every trade and summary metric must reconcile cleanly.

Required safeguards:

* `trade.pnl_dollars` should have one clearly documented meaning, preferably **net of all costs**.
* `trade.pnl_net` must equal gross PnL minus all commissions/fees not already embedded in price.
* Slippage should be embedded in fill prices, not subtracted again later.
* Commission fields must store the **per-trade commission delta**, never cumulative total commission.
* Entry, add-on, partial, stop, market exit, and flatten commissions must all be debited consistently.
* Final equity change should reconcile with summed net trade PnL plus open MTM PnL.
* Diagnostics must not label gross PnL as “net profit.”

The reports found gross-vs-net confusion, incomplete commission debits, cumulative commission stored on trade records, and headline diagnostics that summed gross fields while presenting them as clean results.  

---

### 6. Use shared-capital portfolio simulation, not independent-account recombination

Do not run each symbol as if it has its own full account and then combine the equity curves as if they shared one account.

Required safeguards:

* Portfolio backtests must use one shared cash/equity ledger.
* Position sizing must use shared available capital.
* Simultaneous positions must compete for capital and risk budget.
* Independent per-symbol runs are acceptable only for research attribution, not headline portfolio metrics.
* Diagnostics must clearly label independent-symbol aggregation if used.

The reports flagged independent-account recombination as materially inflating headline portfolio performance, especially where sizing depends on equity. 

---

### 7. Keep diagnostics cohort-pure and denominator-safe

Diagnostics should never mix incompatible populations in one funnel, acceptance rate, coverage statistic, or attribution table.

Required safeguards:

* Every funnel stage must use the same starting population unless explicitly split by cohort.
* Acceptance rates must never exceed 100% within a valid cohort.
* Coverage numerators and denominators must describe the same population.
* Missing-data share, fallback share, live-data share, accepted share, and routed share must be separately named.
* Do not describe raw PnL inside a regime/window as “alpha” unless it is actually benchmark-relative or normalized.
* Do not feed mislabeled metrics into scoring or optimization.

The reports found mixed-population funnels, mislabeled coverage stats, raw window PnL presented as alpha, and optimizer-visible metrics that rewarded contaminated or mislabeled results.  

---

### 8. Treat timestamps as part of the accounting model

All timestamps must represent real market availability and execution time.

Required safeguards:

* Store exchange-local event times and UTC conversions through one helper.
* Do not hard-code `09:30` or `16:00` as UTC unless the market actually trades in UTC.
* Enforce `exit_time >= entry_time` for all same-day trades.
* Hold-time diagnostics must reject negative durations.
* Tests should cover daylight-saving transitions and market open/close conversions.

One report found same-day exits stamped with raw UTC wall-clock times, creating impossible negative hold durations and corrupting time-based diagnostics. 

---

### 9. Separate selection attribution from realized trade attribution

Candidate/selector records and realized trade records should not be blurred.

Required safeguards:

* Keep candidate-level metadata immutable after selection.
* Store one trade-level record per executed trade.
* If a candidate can produce multiple trades or reentries, track those as child trade records.
* Score, route, and cohort metadata must be attached before the position is built.
* Do not overwrite ledger rows after trade construction and expect diagnostics to reflect the new values.

The reports found attribution issues where score diagnostics and candidate outcomes were blurred or corrupted, reducing confidence in claims about why a strategy worked. 

---

### 10. Add invariant tests before accepting any future strategy result

A coding agent implementing similar strategies should add these tests as non-negotiable gates:

```text
Data availability tests
- A lower-timeframe bar cannot access a higher-timeframe bar whose close time is after the lower bar timestamp.
- Intraday bars cannot access same-day daily OHLC before the daily bar is complete.
- Resampled bars are timestamped at availability time, not window start.

Execution causality tests
- A signal using bar t close cannot fill on bar t.
- Same-bar fills require a resting order submitted before bar t began.
- Partial exits cannot fill at bar high/low after the trigger is observed.
- Discretionary exits route through the broker and pay slippage/commission.

Accounting tests
- final_equity - initial_equity reconciles to net realized PnL plus open MTM PnL.
- trade.commission is per-trade, not cumulative.
- every entry, add-on, partial, stop, and flatten path debits costs exactly once.
- risk metrics use mark-to-market equity, not realized-only equity.

Diagnostics tests
- no displayed cohort acceptance rate can exceed 100%.
- every reported percentage uses a matching numerator and denominator.
- gross and net metrics are named accurately.
- “alpha” metrics are benchmark-relative or renamed.
- same-day trades must satisfy exit_time >= entry_time.
```


---

### 11. Do not let stale artifacts, replay shortcuts, or reporting basis become official performance

Even after core causality and shared-core architecture are improved, headline results can still be overstated by stale summaries, portfolio replay shortcuts, optimistic partial/limit execution, and realized-only or inconsistent risk reporting.

Required safeguards:

* Make live-parity entry timing explicit in every optimized config and summary. If a production path can only enter after a completed post-open bar, do not publish same-open or auction-style results as official unless a separate pre-auction strategy, order type, cutoff, and data model exist.
* Regenerate downstream strategy and portfolio artifacts whenever a source strategy timing, fill, cost, or risk-basis assumption changes. Never mix old source-trade artifacts with new portfolio summaries.
* Treat candidate/completed-trade replay as portfolio-sizing evidence only. It cannot discover source-strategy fill, order-path, or intrabar execution errors; those must be tested at the source engine level.
* Portfolio outcome layers must expose explicit cost attribution. Do not record `commission=0.0` at portfolio close unless the report clearly proves source net R already includes all costs and shows that cost lineage.
* Official portfolio drawdown, Sharpe, Calmar, heat, and risk metrics must be based on active-position MTM/MAE, not only closed-trade equity or realized engine deltas.
* Partial exits should use conservative trigger-price, next-bar, or stop-first sequencing variants for stress testing. Do not fill partials at a favorable intrabar average after observing the trigger bar.
* Passive limit and rebalance fills need queue/slippage/spread stress. Do not assume free maker fills, favorable gap-through fills, exact next-open sizing, or zero-cost overlay/rebalance execution.
* Extreme PF, high win rate, low drawdown, or high compounding results require fixed-size/no-compounding reports, widened slippage/spread reports, and untouched OOS or walk-forward validation before being considered production candidates.
* Diagnostics must reconcile phase gates, PF, drawdown units, Calmar units, and stale-vs-current run summaries. A failed gate, contradictory PF, impossible drawdown percentage, or stale summary should block promotion.
* Future reports should include an “existing safeguards” appendix showing completed-bar policy, shared-core coverage, source-fingerprinted replay, parity tests, MTM basis, baseline-regeneration status, and known execution assumptions.

Additional tests to add:

```text
Artifact hygiene tests
- optimized configs explicitly declare live-parity fill timing and auction/non-auction mode.
- stale summaries cannot be marked official after timing/fill/cost/risk-basis changes.
- portfolio artifacts identify the exact source strategy artifact hashes they consume.

Portfolio replay tests
- portfolio net PnL reconciles to source net trade economics and explicit portfolio costs.
- portfolio MTM/MAE drawdown is emitted and used as the headline risk metric.
- completed-trade replay reports are labeled as replay/sizing evidence, not full execution simulation.

Execution stress tests
- partial exits are rerun under trigger-price, next-bar-open, and stop-first variants.
- passive limit exits are rerun with queue miss, extra ticks, and spread-bps stress.
- overlay/rebalance paths include commission, slippage, and actual-fill quantity tests.

Validation tests
- fixed-size/no-compounding reports are produced for high-return strategies.
- mutation-heavy strategies require untouched OOS or walk-forward validation.
- diagnostics fail if gate status, PF, drawdown, Calmar, or current-vs-stale summaries disagree.
```

## Condensed instruction for future coding agents

Build every backtest under a strict “information available at the time” contract. Align higher-timeframe data by completion time, never by window start. Fill after the signal unless an order was already resting. Use one broker path for all fills. Charge all costs exactly once. Mark open positions to market for risk metrics. Simulate portfolios with shared capital. Keep diagnostics cohort-pure, net/gross labels honest, and timestamp invariants enforced by tests. Then rerun from scratch after any integrity fix; do not compare new optimizer rounds against contaminated baselines.
