# trading-stock-family shadow artifacts

This bundle formalizes the week-1 `trading` stock-family bridge.

- `strategy_plugin_contract.json` pins the local clean `trading` checkout, adapter hash, live-shadow fixture set, and shadow maturity.
- `deployment_metadata.json` mirrors the read-only deployment facts expected from the live side and pins the adjacent contract artifact by hash.
- The adapter runs the production live-shadow harness against a temporary copy of the pinned repo, then compares live OMS traces with replay traces across all required parity dimensions.

The contract covers `IARIC_v1` and `ALCB_v1` and remains `shadow_validated`, not `approval_ready`.
