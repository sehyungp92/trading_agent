# trading_assistant_data

`trading_assistant_data` is the canonical, versioned market-data product consumed by
`trading_assistant` and future backtest runners.

This repo owns raw-source imports, canonical parquet normalization, coverage manifests,
calendar definitions, checksum policy, fee/slippage/funding references, adjustment policy
references, compatibility exports, and monthly `DataBundleManifest` files.

It is deliberately data-only. It must not contain strategy decision logic, optimizers,
backtest runners, order routing, order placement, account mutation, approval packets, or
deployment state.

## Contract

The intended handoff is:

```text
trading_assistant writes MonthlyRunManifest
  -> manifest points to DataBundleManifest in trading_assistant_data
  -> trading_assistant_backtest reads only the manifest and bundle
  -> backtest artifacts echo the same data bundle checksum
  -> trading_assistant validates before model review or approval routing
```

Compatibility files are generated under `data/export/` for the current
`FileSystemParquetAdapter` layout:

```text
data/export/filesystem/<market>/<symbol>/<timeframe>/<YYYY-MM>.parquet
data/export/manifests/<bot_id>/<strategy_id>/<YYYY-MM>.coverage_manifest.json
```

The compatibility manifest is emitted only for a truthful single-slice bundle. Multi-slice
optimization must use the direct `DataBundleManifest` handoff.

## Bootstrap

```bash
python -m trading_assistant_data --help
python -m trading_assistant_data normalize --all
python -m trading_assistant_data finalize-slices --run-month 2026-05 --requirements-file data/requirements/strategies/crypto_portfolio/btc_1m.json
python -m trading_assistant_data build-bundle --run-month 2026-05 --bot-id crypto_portfolio --strategy-id btc_1m --requirements-file data/requirements/strategies/crypto_portfolio/btc_1m.json
python -m trading_assistant_data export-filesystem --run-month 2026-05 --bundle-manifest data/bundles/monthly/2026-05/crypto_portfolio/btc_1m/data_bundle_manifest.json
python -m trading_assistant_data reproduce-bundle --bundle-manifest data/bundles/monthly/2026-05/crypto_portfolio/phased_optimizer/data_bundle_manifest.json --json
```

All commands support `--dry-run` and `--json`, and write structured reports under
`data/validation_reports/`.

`authoritative` bundles require slice manifests finalized against a real data-content
commit SHA. A checkout with imported or canonical files but no commit can still emit
diagnostics and compatibility exports, but monthly optimizer authority remains disabled
until the data snapshot is committed, the intended strategy slice requirements are
finalized, and the bundle is rebuilt from those exact requirements.

## Storage

Parquet, CSV, DB, imported snapshots, and canonical data paths are configured for Git LFS.
JSON manifests, calendars, policies, requirements, tests, and docs remain regular git
content.
