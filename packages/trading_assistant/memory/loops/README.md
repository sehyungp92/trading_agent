# Loop Contracts

This directory is the source of truth for recurring loop intent, authority, inputs,
outputs, required checks, and stopping criteria.

Contracts are checked against `trading_assistant.orchestrator.scheduler` by:

```text
python tools/run_workspace_checks.py loop-contracts
```

The work log is generated from `memory/findings/loop_run_ledger.jsonl` and is not
an approval or lifecycle authority. `ScheduledRunStore`, `TaskRegistry`, monthly
artifacts, `ProposalLedger`, `StrategyChangeLedger`, and approval records remain
the authoritative stores.

No contract may grant live bot mutation authority or autonomous writes to
`memory/policies`.
