# IARIC API Reference -- Extracted Function Signatures & Logic

---

## 1. `strategies/stock/iaric/signals.py`

### Module-Level Constants
None defined at module level. All thresholds come from `StrategySettings`.

---

### Function Signatures

```python
def detect_setup(item: WatchlistItem, market: MarketSnapshot, last_1m_bar: Bar, settings: StrategySettings) -> str | None:
```
**Description:** Checks if price is inside the AVWAP band, then classifies as PANIC_FLUSH or DRIFT_EXHAUSTION.
**Exact logic:**
1. `in_avwap_band` = bar low or close is within `[item.avwap_band_lower, item.avwap_band_upper]`
2. If not in band -> `None`
3. If `market.drop_from_hod >= settings.panic_flush_drop_pct` AND `market.minutes_since_hod <= settings.panic_flush_minutes` -> `"PANIC_FLUSH"`
4. If `market.drop_from_hod >= settings.drift_exhaustion_drop_pct` AND `market.minutes_since_hod >= settings.drift_exhaustion_minutes` -> `"DRIFT_EXHAUSTION"`
5. Otherwise -> `None`

---

```python
def lock_setup(sym: SymbolIntradayState, bar_1m: Bar, atr_5m_pct: float, reason: str = "") -> None:
```
**Description:** Locks a detected setup by computing reclaim/stop levels from the 1m bar low and ATR offset.
- `offset = max(0.003, 0.25 * atr_5m_pct)`
- `sym.setup_low = bar_1m.low`
- `sym.reclaim_level = setup_low * (1 + offset)`
- `sym.stop_level = setup_low * (1 - offset)`
- Sets `fsm_state = "SETUP_DETECTED"`, resets `acceptance_count = 0`

---

```python
def compute_location_grade(item: WatchlistItem, market: MarketSnapshot) -> str:
```
**Description:** Grades entry location as A/B/C based on AVWAP proximity and session VWAP discount.
**Exact conditions:**
- If `session_vwap` or `avwap_live` or `last_price` is None -> `"B"`
- `near_avwap` = avwap_live or last_price is within `[avwap_band_lower, avwap_band_upper]`
- `discount_pct = (session_vwap - last_price) / session_vwap`
- **A**: `near_avwap AND 0.015 <= discount_pct <= 0.045`
- **B**: `near_avwap` (but discount outside sweet spot)
- **C**: `discount_pct >= 0.015` (but NOT near_avwap)
- Default: `"B"`

---

```python
def compute_required_acceptance(
    item: WatchlistItem,
    sym: SymbolIntradayState,
    now: datetime,
    settings: StrategySettings,
    market_wide_institutional_selling: bool = False,
) -> tuple[int, list[str]]:
```
**Description:** Computes how many 5m acceptance closes are required, starting from `settings.acceptance_base_closes` and adding +1 for each adverse condition.
**Exact adders (each +1):**
1. `sym.micropressure_mode == "PROXY"` -> `"proxy_mode"`
2. `item.sponsorship_state == "STALE"` -> `"sponsorship_stale"`
3. `item.regime_tier == "B"` -> `"regime_b"`
4. `now` hour (ET) >= 14 -> `"late_day"`
5. `sym.location_grade != "A"` -> `"location_non_a"`
6. `sym.flowproxy_signal == "UNAVAILABLE"` -> `"flow_unavailable"`
7. `market_wide_institutional_selling` is True -> `"market_selling"`

Returns `(required_count, list_of_adder_names)`.

---

```python
def update_acceptance(sym: SymbolIntradayState, bar_5m: Bar) -> None:
```
**Description:** Increments `sym.acceptance_count` if the 5m bar closes at or above `sym.reclaim_level`.

---

```python
def compute_micropressure_from_ticks(ticks_window) -> str:
```
**Description:** Classifies tick-level pressure as ACCUMULATE/DISTRIBUTE/NEUTRAL.
- `uptick_value = sum(positive ticks)`
- `downtick_value = abs(sum(negative ticks))`
- If uptick > 1.5x downtick -> `"ACCUMULATE"`
- If downtick > 1.5x uptick -> `"DISTRIBUTE"`
- Else -> `"NEUTRAL"`

---

```python
def compute_micropressure_proxy(bar_5m: Bar, expected_volume: float, median20_volume: float, reclaim_level: float) -> str:
```
**Description:** Proxy micropressure from 5m bar when tick data unavailable.
- Path 1: `surge >= 1.3 AND close >= reclaim AND cpr >= 0.60 AND close > open` -> `"ACCUMULATE"`
- Path 2: `cpr >= 0.75 AND close > open AND volume >= 1.3 * median20_volume` -> `"ACCUMULATE"`
- Else -> `"NEUTRAL"`

---

```python
def compute_flowproxy_signal(flow_value: str | None) -> str:
```
**Description:** Normalizes external flow signal string to one of `{ACCUMULATE, DISTRIBUTE, NEUTRAL, STALE, UNAVAILABLE}`.

---

```python
def resolve_confidence(sym: SymbolIntradayState) -> str:
```
**Description:** Resolves final confidence to RED/GREEN/YELLOW from three signal inputs.
**Exact conditions:**
- **RED** (any one triggers):
  - `sym.sponsorship_signal == "DISTRIBUTE"`
  - `sym.micropressure_signal == "DISTRIBUTE"`
  - `sym.flowproxy_signal == "DISTRIBUTE"`
- Count positives:
  - `sponsorship_signal == "STRONG"` -> +1
  - `micropressure_signal == "ACCUMULATE"` -> +1
  - `flowproxy_signal == "ACCUMULATE"` -> +1
- If `flowproxy_signal != "UNAVAILABLE"`:
  - **GREEN** if `positives >= 2`, else **YELLOW**
- If flowproxy unavailable:
  - **GREEN** if `sponsorship == "STRONG" AND micropressure == "ACCUMULATE"`, else **YELLOW**

---

```python
def cooldown_expired(sym: SymbolIntradayState, now: datetime, settings: StrategySettings) -> bool:
```
**Description:** Returns True if enough time has passed since invalidation (or if never invalidated).
- `(now - invalidated_at).total_seconds() >= settings.invalidation_cooldown_minutes * 60`

---

```python
def reset_setup_state(sym: SymbolIntradayState) -> None:
```
**Description:** Resets all setup fields on SymbolIntradayState back to IDLE defaults.

---

```python
def alpha_step(
    item: WatchlistItem,
    sym: SymbolIntradayState,
    market: MarketSnapshot,
    bar_1m: Bar | None,
    bar_5m: Bar | None,
    now: datetime,
    atr_5m_pct: float,
    settings: StrategySettings,
    market_wide_institutional_selling: bool = False,
) -> tuple[str, list[str]]:
```
**Description:** Main FSM step function. Returns (action_code, adders). Action codes:
- `"INVALIDATE"` -- stop breached or setup stale
- `"SETUP_DETECTED"` -- new setup locked from IDLE
- `"MOVE_TO_ACCEPTING"` -- reclaim level touched, computes required acceptance
- `"READY_TO_ENTER"` -- acceptance met AND confidence != RED
- `"RESET_TO_IDLE"` -- cooldown expired after invalidation
- `"MANAGE_POSITION"` -- currently in position
- `"NO_ACTION"` -- nothing to do

---

```python
def update_symbol_tier(item: WatchlistItem, sym: SymbolIntradayState, market: MarketSnapshot, settings: StrategySettings) -> str:
```
**Description:** Classifies symbol polling tier as HOT/WARM/COLD for data subscription management.
- **HOT**: in position, in active FSM state, in AVWAP band, or drop >= `hot_drop_pct`
- **WARM**: drop >= `warm_drop_pct`, or within 2x AVWAP band distance
- **COLD**: otherwise

---

## 2. `strategies/stock/iaric/exits.py`

### Function Signatures

```python
def classify_trade(market: MarketSnapshot, position: PositionState) -> str:
```
**Description:** Classifies an open trade into a category for carry/exit decisions.
**Exact logic (checked in order):**
1. If `last_price` is None or no 5m bars -> return current `position.setup_tag`
2. `prior_high = max(high of bars[-4:-1])`, `last_bar = bars_5m[-1]`
3. If `last_bar.close > prior_high AND cpr >= 0.6` -> `"MOMENTUM_CONTINUATION"`
4. If `last_bar.close > entry_price AND last_bar.high < prior_high` -> `"MEAN_REVERSION_BOUNCE"`
5. If `avwap_live is not None AND last_price >= avwap_live` -> `"FLOW_DRIVEN_GRIND"`
6. Else -> `"FAILED"`

---

```python
def regime_adjusted_partial(regime_multiplier: float, settings: StrategySettings) -> tuple[float, float]:
```
**Description:** Interpolates partial-take trigger R-multiple and fraction based on regime strength.
- `t = clamp((regime_multiplier - 0.35) / 0.65, 0.0, 1.0)`
- `r_trigger = partial_r_min + t * (partial_r_max - partial_r_min)`
- `fraction = partial_frac_max - t * (partial_frac_max - partial_frac_min)`

---

```python
def should_take_partial(
    position: PositionState,
    market_price: float,
    settings: StrategySettings,
    regime_multiplier: float | None = None,
) -> tuple[bool, float]:
```
**Description:** Determines if a partial profit take is due.
**Exact conditions:**
- If `position.partial_taken` -> `(False, default_fraction)`
- If `regime_multiplier` provided -> use `regime_adjusted_partial()` for trigger/fraction
- Else -> use `settings.partial_r_multiple` and `settings.partial_exit_fraction`
- `triggered = market_price >= entry_price + (r_trigger * initial_risk_per_share)`
- Returns `(triggered, fraction)`

---

```python
def should_exit_for_time_stop(position: PositionState, now: datetime, market_price: float) -> bool:
```
**Description:** Time-based stop -- exit if deadline passed AND price is at or below entry.
- If `time_stop_deadline is None` -> False
- `now >= time_stop_deadline AND market_price <= entry_price`

---

```python
def should_exit_for_avwap_breakdown(bar_30m: Bar, avwap_live: float, avg_30m_volume: float, settings: StrategySettings) -> bool:
```
**Description:** AVWAP breakdown exit on heavy volume 30m bar.
- `bar_30m.close < avwap_live * (1 - settings.avwap_breakdown_pct)`
- AND `bar_30m.volume > settings.avwap_breakdown_volume_mult * avg_30m_volume`

---

```python
def carry_eligible(
    item: WatchlistItem,
    market: MarketSnapshot,
    position: PositionState,
    flow_reversal_flag: bool = False,
) -> tuple[bool, str]:
```
**Description:** Determines if a position qualifies for overnight carry.
**Exact conditions (ALL must pass, checked in order):**
1. `item.regime_tier == "A"` (else `"regime_not_a"`)
2. `flow_reversal_flag` is False (else `"flow_reversal_flag"`)
3. `position.setup_tag in {"MOMENTUM_CONTINUATION", "FLOW_DRIVEN_GRIND"}` (else `"setup_not_carry"`)
4. `last_30m_bar`, `avwap_live`, `last_price` all not None (else `"missing_close_context"`)
5. `last_price > entry_price` (else `"not_in_profit"`)
6. `last_30m_bar.close >= avwap_live` (else `"close_below_avwap"`)
7. `close_pct = (last_price - session_low) / daily_range >= 0.75` (else `"close_not_in_top_quartile"`)
8. `item.sponsorship_state == "STRONG"` (else `"sponsorship_not_strong"`)
9. `NOT item.earnings_risk_flag AND NOT item.blacklist_flag` (else `"event_risk"`)

Returns `(True, "eligible")` if all pass.

---

```python
def flow_reversal_exit_due(flow_reversal_flag: bool, now: datetime, opened_today: bool) -> bool:
```
**Description:** Forces exit at 9:31 ET next day if flow reversed on a carried position.
- If `not flow_reversal_flag or opened_today` -> False
- `now (ET) >= 09:31` -> True

---

## 3. `strategies/stock/iaric/models.py`

### Dataclasses

#### `ResearchDailyBar` (slots=True)
| Field | Type | Default |
|---|---|---|
| trade_date | date | required |
| open | float | required |
| high | float | required |
| low | float | required |
| close | float | required |
| volume | float | required |
| event_tag | str | `""` |
| **Properties:** `typical_price` = (H+L+C)/3, `cpr` = (C-L)/(H-L) |

#### `MarketResearch` (slots=True)
| Field | Type | Default |
|---|---|---|
| price_ok | bool | required |
| breadth_pct_above_20dma | float | required |
| vix_percentile_1y | float | required |
| hy_spread_5d_bps_change | float | required |
| market_wide_institutional_selling | bool | `False` |

#### `SectorResearch` (slots=True)
| Field | Type | Default |
|---|---|---|
| name | str | required |
| flow_trend_20d | float | required |
| breadth_20d | float | required |
| participation | float | required |

#### `ResearchSymbol` (slots=True)
| Field | Type | Default |
|---|---|---|
| symbol | str | required |
| exchange | str | required |
| primary_exchange | str | required |
| currency | str | required |
| tick_size | float | required |
| point_value | float | required |
| sector | str | required |
| price | float | required |
| adv20_usd | float | required |
| median_spread_pct | float | required |
| earnings_within_sessions | int \| None | required |
| blacklist_flag | bool | required |
| halted_flag | bool | required |
| severe_news_flag | bool | required |
| etf_flag | bool | `False` |
| adr_flag | bool | `False` |
| preferred_flag | bool | `False` |
| otc_flag | bool | `False` |
| hard_to_borrow_flag | bool | `False` |
| flow_proxy_history | list[float] | `[]` |
| daily_bars | list[ResearchDailyBar] | `[]` |
| sector_return_20d | float | `0.0` |
| sector_return_60d | float | `0.0` |
| intraday_atr_seed | float | `0.0` |
| average_30m_volume | float | `0.0` |
| expected_5m_volume | float | `0.0` |
| **Properties:** `trend_price`, `sma20`, `sma50`, `sma50_slope`, `stock_return_20d`, `stock_return_60d`, `daily_atr_estimate` |

#### `HeldPositionResearch` (slots=True)
| Field | Type | Default |
|---|---|---|
| symbol | str | required |
| entry_time | datetime | required |
| entry_price | float | required |
| size | int | required |
| stop | float | required |
| initial_r | float | required |
| setup_tag | str | `""` |
| carry_eligible_flag | bool | `False` |

#### `ResearchSnapshot` (slots=True)
| Field | Type | Default |
|---|---|---|
| trade_date | date | required |
| market | MarketResearch | required |
| sectors | dict[str, SectorResearch] | required |
| symbols | dict[str, ResearchSymbol] | required |
| held_positions | list[HeldPositionResearch] | `[]` |

#### `RegimeSnapshot` (slots=True, frozen=True)
| Field | Type | Default |
|---|---|---|
| score | float | required |
| tier | str | required |
| risk_multiplier | float | required |
| price_ok | bool | required |
| breadth_ok | bool | required |
| vol_ok | bool | required |
| credit_ok | bool | required |

#### `HeldPositionDirective` (slots=True)
| Field | Type | Default |
|---|---|---|
| symbol | str | required |
| entry_time | datetime | required |
| entry_price | float | required |
| size | int | required |
| stop | float | required |
| initial_r | float | required |
| setup_tag | str | required |
| time_stop_deadline | datetime \| None | required |
| carry_eligible_flag | bool | required |
| flow_reversal_flag | bool | required |

#### `WatchlistItem` (slots=True)
| Field | Type | Default |
|---|---|---|
| symbol | str | required |
| exchange | str | required |
| primary_exchange | str | required |
| currency | str | required |
| tick_size | float | required |
| point_value | float | required |
| sector | str | required |
| regime_score | float | required |
| regime_tier | str | required |
| regime_risk_multiplier | float | required |
| sector_score | float | required |
| sector_rank_weight | float | required |
| sponsorship_score | float | required |
| sponsorship_state | str | required |
| persistence | float | required |
| intensity_z | float | required |
| accel_z | float | required |
| rs_percentile | float | required |
| leader_pass | bool | required |
| trend_pass | bool | required |
| trend_strength | float | required |
| earnings_risk_flag | bool | required |
| blacklist_flag | bool | required |
| anchor_date | date | required |
| anchor_type | str | required |
| acceptance_pass | bool | required |
| avwap_ref | float | required |
| avwap_band_lower | float | required |
| avwap_band_upper | float | required |
| daily_atr_estimate | float | required |
| intraday_atr_seed | float | required |
| daily_rank | float | required |
| tradable_flag | bool | required |
| conviction_bucket | str | required |
| conviction_multiplier | float | required |
| recommended_risk_r | float | required |
| average_30m_volume | float | `0.0` |
| expected_5m_volume | float | `0.0` |
| overflow_rank | int \| None | `None` |

#### `WatchlistArtifact` (slots=True)
| Field | Type | Default |
|---|---|---|
| trade_date | date | required |
| generated_at | datetime | required |
| regime | RegimeSnapshot | required |
| items | list[WatchlistItem] | required |
| tradable | list[WatchlistItem] | required |
| overflow | list[WatchlistItem] | required |
| market_wide_institutional_selling | bool | `False` |
| held_positions | list[HeldPositionDirective] | `[]` |
| **Properties:** `by_symbol` -> dict[str, WatchlistItem], **Methods:** `tradable_symbols()` -> list[str] |

#### `Bar` (slots=True)
| Field | Type | Default |
|---|---|---|
| symbol | str | required |
| start_time | datetime | required |
| end_time | datetime | required |
| open | float | required |
| high | float | required |
| low | float | required |
| close | float | required |
| volume | float | required |
| **Properties:** `typical_price` = (H+L+C)/3, `cpr` = (C-L)/(H-L) |

#### `QuoteSnapshot` (slots=True)
| Field | Type | Default |
|---|---|---|
| ts | datetime | required |
| bid | float | required |
| ask | float | required |
| last | float | required |
| bid_size | float | `0.0` |
| ask_size | float | `0.0` |
| cumulative_volume | float | `0.0` |
| cumulative_value | float | `0.0` |
| vwap | float \| None | `None` |
| is_halted | bool | `False` |
| spread_pct | float | `0.0` |

#### `VWAPLedger` (slots=True)
| Field | Type | Default |
|---|---|---|
| cum_pv | float | `0.0` |
| cum_vol | float | `0.0` |
| value | float \| None | `None` |
| **Methods:** `update(bar: Bar) -> None` -- accumulates typical_price * volume |

#### `AVWAPLedger` (slots=True)
| Field | Type | Default |
|---|---|---|
| cum_pv | float | required |
| cum_vol | float | required |
| value | float | required |
| **Class Methods:** `bootstrap(avwap: float, volume: float = 1.0) -> AVWAPLedger` |
| **Methods:** `update(bar: Bar) -> None` |

#### `PendingOrderState` (slots=True)
| Field | Type | Default |
|---|---|---|
| oms_order_id | str | required |
| submitted_at | datetime | required |
| role | str | required |
| requested_qty | int | required |
| limit_price | float \| None | `None` |
| stop_price | float \| None | `None` |
| cancel_requested | bool | `False` |

#### `PositionState` (slots=True)
| Field | Type | Default |
|---|---|---|
| entry_price | float | required |
| qty_entry | int | required |
| qty_open | int | required |
| final_stop | float | required |
| current_stop | float | required |
| entry_time | datetime | required |
| initial_risk_per_share | float | required |
| max_favorable_price | float | required |
| max_adverse_price | float | required |
| partial_taken | bool | `False` |
| stop_order_id | str | `""` |
| trade_id | str | `""` |
| realized_pnl_usd | float | `0.0` |
| entry_commission | float | `0.0` |
| setup_tag | str | `"UNCLASSIFIED"` |
| time_stop_deadline | datetime \| None | `None` |
| **Properties:** `total_initial_risk_usd` = initial_risk_per_share * qty_entry |

#### `MarketSnapshot` (slots=True)
| Field | Type | Default |
|---|---|---|
| symbol | str | required |
| last_price | float \| None | `None` |
| bid | float | `0.0` |
| ask | float | `0.0` |
| spread_pct | float | `0.0` |
| session_high | float \| None | `None` |
| session_low | float \| None | `None` |
| session_vwap | float \| None | `None` |
| avwap_live | float \| None | `None` |
| last_quote | QuoteSnapshot \| None | `None` |
| last_1m_bar | Bar \| None | `None` |
| last_5m_bar | Bar \| None | `None` |
| last_30m_bar | Bar \| None | `None` |
| minute_bars | deque[Bar] | `deque(maxlen=390)` |
| bars_5m | deque[Bar] | `deque(maxlen=120)` |
| bars_30m | deque[Bar] | `deque(maxlen=40)` |
| tick_pressure_window | deque[tuple[datetime, float]] | `deque(maxlen=512)` |
| **Properties:** `minutes_since_hod` -> int (counts back through minute_bars to find HOD), `drop_from_hod` -> float = (session_high - last_price) / session_high |

#### `SymbolIntradayState` (slots=True)
| Field | Type | Default |
|---|---|---|
| symbol | str | required |
| tier | str | `"COLD"` |
| fsm_state | str | `"IDLE"` |
| in_position | bool | `False` |
| position_qty | int | `0` |
| avg_price | float \| None | `None` |
| setup_type | str \| None | `None` |
| setup_low | float \| None | `None` |
| reclaim_level | float \| None | `None` |
| stop_level | float \| None | `None` |
| setup_time | datetime \| None | `None` |
| invalidated_at | datetime \| None | `None` |
| acceptance_count | int | `0` |
| required_acceptance_count | int | `0` |
| location_grade | str \| None | `None` |
| session_vwap | float \| None | `None` |
| avwap_live | float \| None | `None` |
| sponsorship_signal | str | `"NEUTRAL"` |
| micropressure_signal | str | `"NEUTRAL"` |
| micropressure_mode | str | `"PROXY"` |
| flowproxy_signal | str | `"UNAVAILABLE"` |
| confidence | str \| None | `None` |
| last_1m_bar_time | datetime \| None | `None` |
| last_5m_bar_time | datetime \| None | `None` |
| active_order_id | str \| None | `None` |
| time_stop_deadline | datetime \| None | `None` |
| setup_tag | str \| None | `None` |
| expected_volume_pct | float | `0.0` |
| average_30m_volume | float | `0.0` |
| last_transition_reason | str | `""` |
| entry_order | PendingOrderState \| None | `None` |
| position | PositionState \| None | `None` |
| exit_order | PendingOrderState \| None | `None` |
| pending_hard_exit | bool | `False` |

#### `IntradayStateSnapshot` (slots=True)
| Field | Type | Default |
|---|---|---|
| trade_date | date | required |
| saved_at | datetime | required |
| symbols | list[SymbolIntradayState] | required |
| last_decision_code | str | `""` |
| meta | dict[str, Any] | `{}` |

#### `PortfolioState` (slots=True)
| Field | Type | Default |
|---|---|---|
| account_equity | float | required |
| base_risk_fraction | float | required |
| open_positions | dict[str, PositionState] | `{}` |
| pending_entry_risk | dict[str, float] | `{}` |
| regime_allows_no_new_entries | bool | `False` |
| **Methods:** `open_risk_dollars()`, `sector_position_count(symbol_to_sector, sector)`, `sector_open_risk(symbol_to_sector, sector)` |

#### `TierChange` (slots=True)
| Field | Type | Default |
|---|---|---|
| symbol | str | required |
| from_tier | str | required |
| to_tier | str | required |
| reason | str | required |
| at | datetime | required |

---

## 4. `strategies/stock/iaric/risk.py` (bonus -- contains compute_final_risk_unit)

### Function Signatures

```python
def timing_gate_allows_entry(now: datetime, settings: StrategySettings) -> bool:
```
**Description:** Returns True if current ET time is within allowed entry windows.
- Blocked before `settings.open_block_end`
- Blocked at/after `settings.entry_end`
- Blocked during `[settings.close_block_start, settings.forced_flatten]`

---

```python
def timing_multiplier(now: datetime, settings: StrategySettings) -> float:
```
**Description:** Returns sizing multiplier for current time slot from `settings.timing_sizing` list of `(start, end, multiplier)` tuples. Returns `0.0` if no slot matches.

---

```python
def compute_final_risk_unit(item: WatchlistItem, sym: SymbolIntradayState, now: datetime, settings: StrategySettings) -> float:
```
**Description:** Computes the composite sizing multiplier for position sizing.
**Exact formula:**
```
final_risk_unit = conviction * confidence * regime * timing * location * stale_penalty
```
Where:
- `conviction` = `item.conviction_multiplier`
- `confidence` = `settings.confidence_green_mult` if GREEN, else `settings.confidence_yellow_mult`
- `regime` = `item.regime_risk_multiplier`
- `timing` = `timing_multiplier(now, settings)`
- `location` = `{"A": 1.0, "B": 0.90, "C": 0.70}[sym.location_grade]` (default 0.90)
- `stale_penalty` = `settings.stale_penalty` if sponsorship STALE or flowproxy STALE, else `1.0`

---

```python
def compute_order_quantity(
    account_equity: float,
    base_risk_fraction: float,
    final_risk_unit: float,
    entry_price: float,
    stop_level: float,
) -> int:
```
**Description:** Converts risk budget to share count.
- `risk_dollars = account_equity * base_risk_fraction * final_risk_unit`
- `per_share_risk = max(entry_price - stop_level, 0.01)`
- `shares = int(risk_dollars // per_share_risk)`

---

```python
def max_positions_for_regime(tier: str, settings: StrategySettings) -> int:
```
**Description:** Returns max positions allowed: tier A or tier B from settings, else 0.

---

```python
def adjust_qty_for_portfolio_constraints(
    portfolio: PortfolioState,
    item: WatchlistItem,
    intended_qty: int,
    entry_price: float,
    stop_level: float,
    symbol_to_sector: dict[str, str],
    settings: StrategySettings,
) -> tuple[int, str]:
```
**Description:** Reduces qty to fit portfolio risk budget and sector caps. Returns `(adjusted_qty, reason)`.
Checks in order: qty_zero, regime_block, max_positions, sector_position_cap, risk_budget_cap/reduced, sector_risk_cap/reduced, or "ok".

---

```python
def pretrade_risk_check(
    portfolio: PortfolioState,
    item: WatchlistItem,
    qty: int,
    entry_price: float,
    stop_level: float,
    symbol_to_sector: dict[str, str],
    settings: StrategySettings,
) -> bool:
```
**Description:** Boolean wrapper around `adjust_qty_for_portfolio_constraints` -- returns True if any quantity would be allowed.
