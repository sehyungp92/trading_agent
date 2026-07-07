# Regime Predictor Autoresearch Protocol

## Overview

This protocol governs the 4-phase sequential optimization of the regime HMM predictor.
Each phase has its own scoring function, candidate set, and success gate.
Run phases sequentially — each builds on the cumulative mutations of prior phases.

## Prerequisites

1. **Data exists**: `python -m research.backtests.regime.cli download`
2. **Verify baseline**: `python -m research.backtests.regime.cli run --diagnostics`
3. **Check phase state if resuming**: `research/backtests/regime/auto/output/phase_state.json`
   Fresh phase runs now seed from the checked-in `recommended_full_stack` preset.

## Phase Loop

For each phase N (1 through 4):

### Step 1: Run the phase

```bash
python -m research.backtests.regime.cli phase-run --phase N --max-rounds 20 --max-workers 4
```

Uses the checked-in `recommended_full_stack` preset when starting fresh.
Pass explicit JSON only when you intentionally want a non-default baseline.

**Expected outputs:**
- `research/backtests/regime/auto/output/phase_N_result.json` — greedy result
- `research/backtests/regime/auto/output/phase_N_diagnostics.txt` — full report
- `research/backtests/regime/auto/output/phase_state.json` — updated state

### Step 2: Read diagnostics

Read `phase_N_diagnostics.txt` — focus on:
- Regime distribution (are all 4 regimes represented?)
- Transition rate (are there meaningful regime changes?)
- Mutation analysis (which candidates were accepted?)
- Crisis audit (do crisis periods get correct regimes?)

### Step 3: Check the gate

```bash
python -m research.backtests.regime.cli phase-gate --phase N
```

### Step 4: Decision

**If PASSED** → proceed to phase N+1.

**If FAILED** → examine the failure category:

#### `scoring_ineffective`
Best mutations improved financial metrics but regime health didn't improve (or got worse).
- **Action**: Increase regime health weights in `phase_scoring.py`, re-run the phase.
- **Heuristic**: If entropy improved but financials degraded >40% → scoring weights too aggressive, reduce regime health by 10%.

#### `candidates_exhausted`
Greedy stopped early but gate criteria are close (within 80% of targets).
- **Action**: Add narrower-range candidates to `phase_candidates.py` around best values, re-run.
- **Heuristic**: If only 2 regimes active after Phase 1 → sticky prior may need to go even lower (add candidates at 2-3).

#### `structural_issue`
Degenerate HMM (n_active < 2 or transition_rate ≈ 0).
- **Action**: Investigate code/data issue. Check if rolling window produces enough training data. Verify feature matrix isn't degenerate.

**Max 2 retries per phase before escalating.**

## Phase-Specific Notes

### Phase 1: Fix HMM Dynamics
- **Goal**: Break 2-state collapse → ≥3 active regimes
- **All candidates are HMM-affecting** — no cache speedup, expect ~50 minutes
- **Key signal**: transition_rate should jump from ~0.0002 to >0.008/week
- If no improvement: the problem may be in the feature space, not HMM params

### Phase 2: Fix Features
- **Goal**: 4 distinct regimes with quadrant separation
- **Depends on**: New data (DBC, DFII10) — must re-run `download` first
- **Mixed HMM/non-HMM candidates** — partial cache speedup
- If no quadrant separation: growth/inflation features may need rethinking

### Phase 3: Crisis Integration
- **Goal**: Crisis overlay responds to actual crises
- **Mostly non-HMM candidates** — fast with caching
- **Key signal**: crisis_response should be >0.4
- If crisis response stays low: the ventilator may be disconnected from regime signals

### Phase 4: Fine-Tuning
- **Goal**: Production readiness + historical alignment
- **Narrow ranges** — incremental improvements
- After Phase 4: run historical validation separately

## Post-Phase-4 Validation

### Historical validation
```bash
python -m research.backtests.regime.cli historical-validate
```
Defaults to the checked-in `recommended_full_stack` preset.
Pass `--mutations-json` only to validate an explicit saved config.
Target: alignment > 0.5, transition latency < 8 weeks for GFC/COVID/2022.

### Walk-forward validation
```bash
python -m research.backtests.regime.cli walk-forward --test-years 2
```
Target: OOS/IS ratio > 0.5.

## Decision Heuristics Summary

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| Only 2 regimes after Phase 1 | Sticky prior still too high | Add candidates at sticky_diag 2-3 |
| Entropy improved, financials tanked | Scoring too regime-heavy | Reduce regime health weight by 10% |
| Crisis response < 0.2 despite diversity | Ventilator disconnected | Check ventilator lambda and delta_rho settings |
| Historical alignment < 0.3 | Wrong features | Go back to Phase 2 |
| All candidates rejected | Hard rejects too tight | Relax phase reject thresholds |
| Gate close but not passing | Need fine-tuning | Add narrow-range candidates |

## File Reference

| File | Purpose |
|------|---------|
| `phase_scoring.py` | 4 scoring functions + `compute_regime_stats()` |
| `phase_candidates.py` | Per-phase candidate generators |
| `phase_gates.py` | Gate criteria + failure categorization |
| `phase_state.py` | JSON-persisted phase state |
| `phase_diagnostics.py` | Inter-phase delta analysis |
| `historical_validation.py` | Known timeline alignment scoring |
| `greedy_optimize.py` | Core optimizer (phase-aware via `_worker_phase`) |
| `cli.py` | CLI entry points for all phase commands |
