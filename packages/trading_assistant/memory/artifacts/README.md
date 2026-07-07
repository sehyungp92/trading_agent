# Artifact Authority Registry

`registry.yaml` documents manually named artifact authorities. The runtime
registry also imports `REQUIRED_BACKTEST_ARTIFACTS` and optional backtest artifact
constants from `trading_assistant.schemas.backtest_artifacts`; required backtest
files are not redefined here.

Authority classes:

- `approval_gate`: may satisfy a monthly approval gate when deterministic checks pass.
- `binding`: source-of-truth evidence, but not necessarily approval-gate proof.
- `advisory`: search order, priors, or context only.
- `diagnostics_only`: useful for debugging; never approval-gate proof.
- `generated`: generated context or summaries.
- `human_owned`: controlled by humans, not autonomous loops.

Monthly search briefs are advisory search-order guidance only. Monthly evidence
verification artifacts are approval-gate evidence but only after deterministic
candidate gates and model-review validation have run. `memory/policies` is
human-owned. Runner observability is diagnostics-only.
