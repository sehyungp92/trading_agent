# Analysis Harness Program

Human-owned objective contract for offline analysis-harness improvement.

## Mission

Improve evidence-grounded analysis and proposal quality so the system proposes fewer weak
strategy, risk, and execution changes, preserves more genuinely high-value candidates, and
maximizes expected risk-adjusted returns over time.

## Scope

Harness experiments may change prompt assembly, context retrieval, learning-card selection,
generated-playbook injection, response parsing, validators, benchmark scoring, and provider
routing.

## Out Of Scope

Harness experiments must not change live trading execution, send bot commands, edit
`memory/policies`, weaken approval gates, bypass double approval, deploy strategy/config
changes directly, change objective weights, or change immutable score profiles without
human approval.

## Primary Metrics

- Fewer false-positive material strategy/config changes.
- More high-value candidates preserved through validation.
- Better confidence calibration against realized outcomes and priors.
- Better use of authoritative monthly outcomes, replay parity, coverage, and approval ledgers.
- Fewer repeated rejected or harmful ideas.
- Lower hallucinated-evidence and invalid-path rates.

## Secondary Metrics

- Cost.
- Latency.
- Report compactness.
- Provider availability.

Secondary metrics break ties only after quality and governance metrics pass.

## Keep And Discard Rules

- Keep a variant when primary benchmark score improves and no governance hard-fail regresses.
- Keep a tie only when the variant reduces complexity, cost, or latency without quality loss.
- Discard variants that improve narrow scores while increasing approval bypass, hallucinated
  evidence, deterministic-gate misses, or monthly-authority violations.
- Always record discarded experiments as learning signal.

## Anti-Overfitting Rule

Before promotion, ask: if this exact bot, date, and benchmark case disappeared, would the change
still improve decision quality across related workflows? If the answer is no, discard or keep the
experiment offline until broader evidence supports it.

## Governance Invariants

- Monthly validation remains the authority for material strategy/config changes.
- Generated learning is advisory context, never hidden policy.
- Benchmark objective weights and immutable score profiles are human-owned.
- Policy memory is read-only to autonomous learning.
- Direct live trading commands remain out of scope.
