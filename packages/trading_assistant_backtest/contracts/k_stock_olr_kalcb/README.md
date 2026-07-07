# k-stock-olr-kalcb shadow artifacts

This bundle formalizes the week-1 KRX bridge for `k_stock_trader`.

- `strategy_plugin_contract.json` pins the local clean `k_stock_trader` checkout, OLR/KALCB adapter hash, fixture manifests, and shadow maturity.
- `deployment_metadata.json` mirrors the read-only deployment facts expected from the live side and pins the adjacent contract artifact by hash.
- The adapter compares production KALCB/OLR artifact generation against the replay-cache decision path and emits normalized decision parity across all required dimensions.

The contract is `shadow_validated`, not `approval_ready`.
