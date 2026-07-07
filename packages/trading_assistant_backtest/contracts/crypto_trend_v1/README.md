# crypto-trend-v1 shadow artifacts

This directory is the persisted shadow contract bundle for the first wired live strategy bridge.

- `strategy_plugin_contract.json` pins the local clean `crypto_trader` checkout, adapter hash, decision API, supported symbols/timeframes, fixture set, and maturity.
- `deployment_metadata.json` mirrors the read-only facts a live bot/VPS must publish for monthly shadow runs and pins the adjacent contract artifact by hash.
- `parity_fixtures/*.json` cover normal entries, blocked trades, exits, stops, sizing/risk caps, and long/short order intent. Fixtures use live decision timeframes supported by `crypto_trader`; the contract also lists `1m` because the monthly data bundle is a 1m source slice.

The contract is intentionally `shadow_validated`, not `approval_ready`. Promotion to `approval_ready` still requires live-emitted deployment metadata from the actual VPS, broader fixtures from production incidents/trades, and a replay-backed evaluator that can score real candidates.
