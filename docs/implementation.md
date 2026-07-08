# trading_agent Deployment Implementation Next Steps

This is the active deployment checklist for the target topology:

- one IBKR trading bot VPS;
- one crypto trading bot VPS;
- one K-stock trading bot VPS;
- one local workstation running the trading assistant, local relay, data
  bundle reproduction, approval workflow, and monitoring.

Do not co-locate the three live bot runtimes on one VPS. Do not deploy the
assistant control plane into any live bot image. Live bot images must remain
runtime-only and must not depend on `trading-assistant` or
`trading-assistant-backtest`.

---

## 1. Deployment Principles

### 1.1 Non-negotiables

1. Deploy one bot at a time. The deployable `deployments/cutover_plan.json`
   declares `one_bot_at_a_time_artifact_only_or_paper_first`.
2. First modes are:
   - `ibkr`: `paper`
   - `crypto`: `paper`
   - `k_stock`: `artifact_only`
3. Every production deploy starts from a named commit with a clean worktree.
   Generated runtime metadata must record the deployed commit and clean
   worktree state.
4. Secrets live outside git in per-machine `.env` files, Docker secrets, or
   locked-down OS secret storage.
5. Dashboard, database, and broker ports stay bound to `127.0.0.1` unless a
   private overlay network is explicitly used.
6. The local assistant receives bot evidence through a relay-compatible
   `/events`, `/events?since=...`, and `/ack` API. Because the assistant is
   local, expose that relay over a private network or reverse tunnel, not a
   public unauthenticated port.
7. Promotion to live capital requires written evidence from the gate sequence,
   not just a green container start.

### 1.2 Deployable units

| Unit | Compose | Bot package | Runtime entrypoint | Generated config | Contracts |
|---|---|---|---|---|---|
| IBKR trading | `deployments/ibkr/docker-compose.yml` | `trading/ibkr_trader` | `ibkr-trading-runtime` | `deployments/ibkr/generated/strategies.effective.json` | `trading_swing_family`, `trading_momentum_family`, `trading_stock_family` |
| Crypto trading | `deployments/crypto/docker-compose.yml` | `trading/crypto_trader` | `crypto-trader` | `deployments/crypto/generated/live_config.effective.json` | `crypto_trend_v1`, `crypto_momentum_v1`, `crypto_breakout_v1` |
| K-stock trading | `deployments/k_stock/docker-compose.yml` | `trading/k_stock_trader` | `k-stock-olr-kalcb-runtime` | `deployments/k_stock/generated/olr_kalcb.effective.json` | `k_stock_olr_kalcb` |
| Trading assistant | local only | `packages/trading_assistant`, `packages/trading_assistant_data`, `packages/trading_assistant_backtest` | `uvicorn trading_assistant.orchestrator.app:app` | local `.env` and data bundle manifests | installed deployment metadata from all bot units |

The `deployments/*/docker-compose.yml` files define the production VPS service
surfaces: env loading, persistent volumes, restart policy, healthchecks, runtime
commands, log rotation, and required broker/database sidecars. Keep machine
secrets in the matching `.env` files copied from `.env.example`.

### 1.3 Release gates and operator inputs

Before any VPS update, Phase 0 must prove from a clean release checkout that:

- `python tools/verify_deployment_metadata.py --bot all` passes from a clean
  checkout and validates all seven runtime metadata records;
- `python tools/verify_cutover_plan.py` passes with hashes matching the
  deployable compose files and generated live configs;
- crypto and K-stock runtime metadata emitters write the artifacts expected by
  the metadata matrix;
- relay tests encode ID-ordered default delivery with priority delivery as an
  explicit opt-in;
- local relay ownership stays in `trading_assistant.relay_ingress`, with
  operator docs and package READMEs describing assistant-local relay ingress;
- crypto deploy config has a non-secret runtime config contract and
  fail-closed preflight before `crypto-trader paper`;
- monorepo bot image boundaries are enforced by dependency checks and bot image
  reports, with assistant packages absent from bot runtime images.

Current release-readiness status: the material code gates from the live/paper
audit are implemented. The remaining blockers are deployment evidence blockers,
not known code implementation blockers:

- `python tools/verify_deployment_metadata.py --bot all` must pass from a
  clean release checkout or clean deploy worktree;
- `deployments/operational_evidence.json` must be collected from the actual
  three VPSes and the local assistant deployment window;
- `python tools/verify_operational_deployment_evidence.py` must pass against
  that collected evidence before paper/live deployment is marked complete.

The operator inputs that remain outside the repo are:

- choose or create the final named release reference for the clean commit;
- emit and install live/VPS runtime deployment metadata from the clean bot
  deployments;
- collect `deployments/operational_evidence.json` from the three VPSes and the
  local assistant, including relay ingest evidence and crypto sidecar runtime
  policy evidence;
- run a production scheduled-shadow cycle for the pilot scope with installed
  live/VPS metadata, local relay ingest evidence,
  `source_kind=monthly_validation_shadow`, and `adoption_disabled=true`;
- complete production fixture breadth for the pilot scope, including a
  live/shadow telemetry source case class;
- complete approval-grade promotion evidence: each promoted scope must move
  from `shadow_validated` to `approval_ready` and provide explicit P6/P7
  optimizer evidence before `validation-matrix` can have zero
  `approval_remaining_gaps`.

---

## 2. Phase 0: Freeze The Release Candidate

Run this phase on the local workstation from the repository root.

### 2.1 Pick the deployment commit

1. Create or select the deployment branch.
2. Confirm all intended code and config changes are committed.
3. Record the commit SHA and the release name or tag:

```bash
git rev-parse HEAD
git describe --tags --exact-match HEAD || true
git status --short
```

The final production metadata must be emitted from a clean deploy checkout. If
the working tree is dirty because active development is in progress, do not
promote that checkout. Create a separate clean clone or worktree for deployment
evidence. If `python tools/verify_deployment_metadata.py --bot all` fails, stop
deployment and resolve the dirty checkout, metadata emitter, or contract
evidence before provisioning or updating a VPS.

### 2.2 Run monorepo gates

Use these commands before provisioning or updating any VPS:

```bash
python tools/workspace_import_smoke.py --all-packages --run-commands
python tools/verify_dependency_boundaries.py

python tools/generate_effective_live_configs.py
python tools/verify_effective_live_configs.py

python tools/build_bot_image.py --bot all --emit-dependency-reports --timeout-seconds 1800
python tools/verify_deployment_metadata.py --bot all
python tools/verify_cutover_plan.py

python tools/run_workspace_checks.py deployment-gate
```

Do not treat the static release checks as proof that the deployment is
operational. After the three VPSes and the local assistant are actually
running, collect the runtime evidence bundle and run:

```bash
python tools/verify_operational_deployment_evidence.py
```

For a stricter release review, add:

```bash
python tools/verify_refactor_acceptance.py --bot all --strict
```

Do not deploy a bot whose image report is not `status: pass` under:

- `deployments/ibkr/generated/dependency_report.json`
- `deployments/crypto/generated/dependency_report.json`
- `deployments/k_stock/generated/dependency_report.json`

### 2.3 Inspect generated config and package evidence

Confirm the following paths exist and match `deployments/cutover_plan.json`:

```text
deployments/ibkr/generated/strategies.effective.json
deployments/ibkr/generated/dependency_report.json

deployments/crypto/generated/live_config.effective.json
deployments/crypto/generated/dependency_report.json

deployments/k_stock/generated/olr_kalcb.effective.json
deployments/k_stock/generated/dependency_report.json
```

Do not require checked-in runtime deployment metadata here. Runtime metadata is
emitted later from the clean deploy checkout or the target VPS runtime and is
installed locally through the assistant metadata installer.

Review `deployments/cutover_plan.json` and confirm:

- each `compose_sha256` matches the compose file being deployed;
- each `live_config_hash` matches the generated effective config;
- each rollback block has a real restore command;
- no placeholder token remains.

### 2.4 Decide the private network path

The bot sidecars need to POST events to a relay, and the local assistant needs
to poll that relay. Because the assistant stays local, choose exactly one
private ingress path before turning on sidecars:

1. Preferred: private overlay network between the workstation and the three
   VPSes. Bind the local relay to the workstation overlay IP on port `8001`.
2. Acceptable: SSH reverse tunnels from each VPS to the workstation or from the
   workstation to a locked-down relay endpoint.
3. Temporary fallback: direct local orchestrator ingest for smoke tests only,
   using `POST /ingest`; do not use this as the unattended production path.

The old assistant guide used a separate relay VPS. In the new topology, that
relay service runs locally with the assistant unless you explicitly decide to
add a fourth always-on infrastructure host.

---

## 3. Phase 1: Shared Secrets And Configuration

### 3.1 Generate secrets

Generate at least these values:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Create:

- one relay read API key, `RELAY_API_KEY`;
- one local assistant API key, `ORCHESTRATOR_API_KEY`;
- one HMAC value for crypto sidecar events;
- one HMAC value for K-stock sidecar events;
- one HMAC value for IBKR sidecar events, unless the IBKR runtime is split into
  independent family services.

IBKR sidecars read the same `INSTRUMENTATION_HMAC_SECRET` env var. If all IBKR
families run inside the same runtime service, map the same HMAC value to
`swing_multi_01`, `momentum_nq_01`, and `stock_trader` in the relay.

### 3.2 Relay secret map

On the local relay, set `RELAY_SHARED_SECRETS` to a JSON object that covers the
event `bot_id` values actually emitted by each sidecar:

```json
{
  "swing_multi_01": "<IBKR_HMAC>",
  "momentum_nq_01": "<IBKR_HMAC>",
  "stock_trader": "<IBKR_HMAC>",
  "k_stock_trader": "<K_STOCK_HMAC>",
  "paper_bot_01": "<CRYPTO_HMAC>",
  "crypto": "<CRYPTO_HMAC>"
}
```

Before production, verify the crypto event bot id from the active
`live_config.json` or generated effective config. The generated crypto effective
config should contain `paper_bot_01` inside `materialized_config.live_config`.
Keep both `paper_bot_01` and `crypto` during initial integration if any
instrumentation path still uses the top-level bot id.

### 3.3 Per-machine secret files

Create one `.env` per machine:

| Machine | Secret groups |
|---|---|
| Local assistant | `RELAY_SHARED_SECRETS`, `RELAY_API_KEY`, `ORCHESTRATOR_API_KEY`, Telegram credentials, optional Discord/email credentials, read-only data source credentials |
| IBKR VPS | IBKR username/password, paper/live account id, Postgres passwords, `INSTRUMENTATION_HMAC_SECRET`, `INSTRUMENTATION_RELAY_URL` |
| Crypto VPS | Hyperliquid wallet address/private key, Postgres passwords if enabled, `relay_url`, `relay_secret`, active `live_config.json` |
| K-stock VPS | KIS app key/secret/account values, Postgres passwords, `INSTRUMENTATION_HMAC_SECRET`, `RELAY_URL` |

Minimum permissions:

```bash
chmod 600 .env
chmod 600 config/live_config.json 2>/dev/null || true
```

On Windows, store the local `.env` under
`packages/trading_assistant/.env` and keep the working tree protected with disk
encryption.

---

## 4. Phase 2: Provision The Three VPSes

Repeat this section once for each VPS. Use the same path on all three hosts so
commands are easy to compare:

```text
/opt/trading_agent
```

### 4.1 Base server setup

```bash
ssh root@<VPS_IP>
adduser trader
usermod -aG sudo trader
mkdir -p /home/trader/.ssh
cp ~/.ssh/authorized_keys /home/trader/.ssh/
chown -R trader:trader /home/trader/.ssh
chmod 700 /home/trader/.ssh
chmod 600 /home/trader/.ssh/authorized_keys

sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh || systemctl restart sshd
```

Install common packages:

```bash
ssh trader@<VPS_IP>
sudo apt update
sudo apt install -y git curl ca-certificates jq htop tmux unzip ufw unattended-upgrades
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"
exit
ssh trader@<VPS_IP>
docker --version
docker compose version
```

Firewall baseline:

```bash
sudo ufw allow OpenSSH
sudo ufw enable
sudo ufw status verbose
```

Only open additional ports on a private overlay interface. Do not expose
Postgres, dashboards, IB Gateway, KIS services, or bot health endpoints to the
public internet.

### 4.2 Clone and pin the repo

```bash
sudo mkdir -p /opt/trading_agent
sudo chown trader:trader /opt/trading_agent
cd /opt/trading_agent
git clone <YOUR_PRIVATE_REPO_URL> repo
cd repo
git checkout <DEPLOYMENT_COMMIT_SHA>
git status --short
```

On each VPS, run only lightweight validation:

```bash
python3 --version
docker compose -f deployments/ibkr/docker-compose.yml config --quiet
docker compose -f deployments/crypto/docker-compose.yml config --quiet
docker compose -f deployments/k_stock/docker-compose.yml config --quiet
```

Each VPS only starts its own bot compose file. The other compose checks are a
sanity check that the checkout is complete.

### 4.3 Create host directories

```bash
sudo mkdir -p /var/log/trading_agent /opt/trading_agent/backups
sudo chown -R trader:trader /var/log/trading_agent /opt/trading_agent/backups
mkdir -p /opt/trading_agent/repo/runtime_data
mkdir -p /opt/trading_agent/repo/runtime_state
mkdir -p /opt/trading_agent/repo/runtime_artifacts
```

Use bot-specific subdirectories if the compose file bind-mounts shared host
paths:

```text
runtime_data/ibkr
runtime_data/crypto
runtime_data/k_stock
runtime_state/ibkr
runtime_state/crypto
runtime_state/k_stock
```

---

## 5. Phase 3: Maintain Production Compose Files

The compose files have been productionized and should remain the single VPS
entrypoints for their respective bots:

- `deployments/ibkr/docker-compose.yml`
- `deployments/crypto/docker-compose.yml`
- `deployments/k_stock/docker-compose.yml`

Before starting a bot runtime, copy the relevant `.env.example` to `.env` on
that machine, fill secrets, and re-run the release gates from Phase 0. Keep the
build context as `../..` and the bot Dockerfile paths as currently declared.

### 5.1 Required compose additions for every bot

For the bot service:

1. Keep the optional `.env` loader and matching `.env.example` current.
2. Keep real paper/artifact runtime commands, not help commands.
3. Use `restart: unless-stopped` only for long-running services.
4. Maintain bind mounts or named volumes for:
   - runtime data;
   - runtime state;
   - instrumentation JSONL;
   - generated deployment metadata;
   - bot-local config that intentionally stays outside the image.
5. Keep Docker log rotation enabled:

```yaml
logging:
  driver: json-file
  options:
    max-size: "50m"
    max-file: "5"
```

6. Keep `security_opt: ["no-new-privileges:true"]` and `cap_drop: ["ALL"]`
   where the runtime does not need extra capabilities.
7. Bind dashboards and databases to `127.0.0.1`.
8. Keep assistant packages out of bot images. Re-run:

```bash
python tools/verify_dependency_boundaries.py
python tools/build_bot_image.py --bot <ibkr|crypto|k_stock> --emit-dependency-reports
```

### 5.2 Compose config checks

After each compose edit:

```bash
docker compose -f deployments/<bot>/docker-compose.yml config --quiet
python tools/verify_cutover_plan.py
```

If the compose file changed, refresh the cutover plan hash and rollback block
as part of the same reviewed change. The rollback `restore_test` must run a
bot runtime/status/preflight command; `docker compose config --quiet` is only a
syntax check and does not satisfy rollback evidence.

---

## 6. Phase 4: Local Assistant Deployment

Run this on the local workstation. The assistant is the control plane,
evidence processor, data reproducer, monthly validation scheduler, and approval
surface. It should not place trades directly.

### 6.1 Install the workspace

Preferred with `uv`:

```powershell
cd C:\Users\sehyu\Documents\Other\Projects\trading_agent
uv sync --all-packages --all-extras --dev
.venv\Scripts\Activate.ps1
```

Fallback with editable installs:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .\packages\trading_contracts
pip install -e .\packages\trading_assistant_data[dev,ibkr,hyperliquid]
pip install -e .\packages\trading_assistant_backtest[dev]
pip install -e .\packages\trading_assistant[dev,notifications]
```

The local relay server is hosted by `trading_assistant.relay_ingress`; bot
packages only need their outbound relay clients.

### 6.2 Configure local assistant env

Create `packages/trading_assistant/.env`:

```bash
BOT_IDS=ibkr,crypto,k_stock,swing_multi_01,momentum_nq_01,stock_trader,k_stock_trader,paper_bot_01
BOT_TIMEZONES=ibkr:US/Eastern,crypto:UTC,k_stock:Asia/Seoul,swing_multi_01:US/Eastern,momentum_nq_01:US/Eastern,stock_trader:US/Eastern,k_stock_trader:Asia/Seoul,paper_bot_01:UTC

BIND_HOST=127.0.0.1
ORCHESTRATOR_API_KEY=<ORCHESTRATOR_API_KEY>
ALLOW_UNAUTHENTICATED_LOCAL=false

RELAY_URL=http://127.0.0.1:8001
RELAY_API_KEY=<RELAY_API_KEY>
RELAY_POLL_INTERVAL_SECONDS=300

MONTHLY_VALIDATION_ENABLED=true
MONTHLY_VALIDATION_MODE=shadow
MONTHLY_APPROVAL_SCOPE_ALLOWLIST=trading_stock_family
MONTHLY_APPROVAL_SCOPE_MAP=ibkr:trading_stock_family
MONTHLY_DEPLOYMENT_METADATA_INSTALL_REPORTS=../trading_assistant_backtest/artifacts/validation/deployment_metadata/trading_stock_family/install_report.json
MONTHLY_OPERATIONAL_EVIDENCE_PATH=../../deployments/operational_evidence.json
MONTHLY_RELAY_INGEST_EVIDENCE_PATH=../trading_assistant_backtest/artifacts/validation/relay_ingest/trading_stock_family/relay_ingest_evidence.json
MONTHLY_VPS_HOST_ID=<IBKR_VPS_HOST_ID>
MONTHLY_ASSISTANT_HOST_ID=<LOCAL_ASSISTANT_HOST_ID>
MARKET_DATA_ROOT=../trading_assistant_data/data/export
BACKTEST_REPO_PATH=../trading_assistant_backtest
BACKTEST_ARTIFACT_ROOT=../trading_assistant_backtest/artifacts/monthly_validation

TELEGRAM_BOT_TOKEN=<TELEGRAM_BOT_TOKEN>
TELEGRAM_CHAT_ID=<TELEGRAM_CHAT_ID>
```

Keep `MONTHLY_VALIDATION_MODE=shadow` until the selected scope passes the
approval-grade audit. When testing `approval_gated`, the allowlist and map must
name only the promoted scope; other configured `BOT_IDS` remain shadow by
construction.

If you intentionally test without a relay, set `DIRECT_INGEST_ONLY=true` and
leave `RELAY_URL` blank. Do this only for local smoke tests.

### 6.3 Start the local relay

Copy `packages\trading_assistant\.env.example` to
`packages\trading_assistant\.env`, fill `RELAY_API_KEY`,
`RELAY_SHARED_SECRETS`, and `ORCHESTRATOR_API_KEY`, then start the relay API
that bot sidecars post into:

```powershell
powershell -ExecutionPolicy Bypass -File packages\trading_assistant\scripts\start-relay.ps1
```

Health check:

```powershell
curl.exe http://127.0.0.1:8001/health
```

Expose this relay to the three VPSes only through the private network decision
from Phase 0. The bot sidecars should use the private relay URL, for example:

```text
http://<WORKSTATION_PRIVATE_IP>:8001
```

The sidecars append `/events` when needed.

### 6.4 Start the assistant orchestrator

```powershell
cd packages\trading_assistant
python -m uvicorn trading_assistant.orchestrator.app:app --host 127.0.0.1 --port 8000
```

Verify:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -H "X-Api-Key: <ORCHESTRATOR_API_KEY>" http://127.0.0.1:8000/metrics
curl.exe -H "X-Api-Key: <ORCHESTRATOR_API_KEY>" http://127.0.0.1:8000/events/pending
```

### 6.5 Install Windows startup

After the relay and orchestrator start cleanly by hand:

```powershell
powershell -ExecutionPolicy Bypass -File packages\trading_assistant\scripts\install-startup.ps1
```

This registers both `TradingAssistantRelayAutoStart` and
`TradingAssistantAutoStart` for the current Windows user.

Reboot the workstation and verify:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8001/health
```

### 6.6 Local assistant validation

```bash
python tools/run_workspace_checks.py deployment-gate
python tools/run_workspace_checks.py monthly-shadow-smoke
python tools/run_workspace_checks.py validation-matrix
```

The assistant remains in `MONTHLY_VALIDATION_MODE=shadow` until runtime
deployment metadata has been installed from all bot VPSes and at least one
full shadow cycle has passed for the scope being promoted. During shadow,
`validation-matrix` may still report `approval_remaining_gaps`; record those
gaps as promotion work. Do not flip to approval-gated mode until
`approval_remaining_gaps` is zero for the promoted scope.

The final operational completion gate is
`python tools/verify_operational_deployment_evidence.py`. Its evidence file is
`deployments/operational_evidence.json`; it must reference hashed evidence
artifacts copied back from the VPSes/local assistant and must cover separate
VPS runtime status, sidecar forwarding, assistant ingest, installed metadata,
monthly shadow validation with real metadata, and rollback smoke execution.
For each bot, `assistant_ingest` must include inline `relay_ingest_evidence` or
one or more evidence paths containing event ID, bot ID, deployment ID, runtime
instance ID, effective config hash, deployment metadata hash, freshness, and
HMAC secret fingerprint. For crypto, `sidecar_forwarding` must also include
runtime policy evidence with thresholds, `incident_action=cancel_working_entry_orders`
for degraded relay state, and `open_position_action=hold_existing_positions`.

---

## 7. Phase 5: IBKR VPS Deployment

Deploy this only on the IBKR VPS.

### 7.1 IBKR prerequisites

1. IBKR paper account is active and matches `IB_ACCOUNT_ID`.
2. Market data subscriptions and API acknowledgement are complete.
3. Paper account has access to shared real-time subscriptions, or the runtime
   is approved to fall back to delayed data during paper tests.
4. The VPS has no competing TWS/Gateway session using the same account.
5. The local relay is reachable from the VPS over the private network:

```bash
curl -sf http://<WORKSTATION_PRIVATE_IP>:8001/health
```

### 7.2 IBKR `.env`

Create `/opt/trading_agent/repo/deployments/ibkr/.env` or the path referenced
by the compose file:

```bash
TRADING_MODE=paper

TWS_USERID=<IBKR_USERNAME>
TWS_PASSWORD=<IBKR_PASSWORD>
IB_HOST=127.0.0.1
IB_PORT=4002
IB_ACCOUNT_ID=<DU_ACCOUNT_ID>

POSTGRES_PASSWORD=<POSTGRES_ADMIN_PASSWORD>
POSTGRES_READER_PASSWORD=<POSTGRES_READER_PASSWORD>
POSTGRES_WRITER_PASSWORD=<POSTGRES_WRITER_PASSWORD>
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=trading

INSTRUMENTATION_RELAY_URL=http://<WORKSTATION_PRIVATE_IP>:8001/events
INSTRUMENTATION_HMAC_SECRET=<IBKR_HMAC>

TELEGRAM_BOT_TOKEN=<OPTIONAL_WATCHDOG_TOKEN>
TELEGRAM_CHAT_ID=<OPTIONAL_WATCHDOG_CHAT_ID>

PAPER_INITIAL_EQUITY=30000
STOCK_TRADER_DEPLOY_MODE=both
CONFIG_DIR=/app/trading/ibkr_trader/config
```

### 7.3 Required compose services

Keep `deployments/ibkr/docker-compose.yml` aligned with these services:

- `ib-gateway`: `ghcr.io/gnzsnz/ib-gateway:stable`, paper port bound to
  `127.0.0.1:4002`, live port bound to `127.0.0.1:4001`.
- `postgres`: if the runtime is using persistent strategy state and dashboard
  queries.
- `ibkr-trading`: image built from `trading/ibkr_trader/Dockerfile`.
- optional `ibkr-watchdog`: Telegram alerting, only after paper runtime is
  stable.

Do not put the assistant packages in this compose file.

### 7.4 Build and gateway smoke

```bash
cd /opt/trading_agent/repo
docker compose -f deployments/ibkr/docker-compose.yml build ibkr-trading
docker compose -f deployments/ibkr/docker-compose.yml up -d ib-gateway postgres
docker compose -f deployments/ibkr/docker-compose.yml ps
docker compose -f deployments/ibkr/docker-compose.yml logs ib-gateway --tail 100
```

Check IB Gateway TCP:

```bash
ss -tlnp | grep 4002
```

### 7.5 Runtime preflight

```bash
docker compose -f deployments/ibkr/docker-compose.yml run --rm ibkr-trading \
  ibkr-trading-runtime preflight \
  --config-dir config \
  --json \
  --write-registry-artifact data/strategy-registry.json
```

Expected:

- config loads;
- strategy registry loads;
- all enabled strategy manifests are visible;
- no stock artifact readiness failure remains;
- instrumentation relay URL and HMAC are present.

### 7.6 Paper runtime start

First run foreground or one-shot:

```bash
docker compose -f deployments/ibkr/docker-compose.yml run --rm ibkr-trading \
  ibkr-trading-runtime run \
  --config-dir config \
  --connect-ib \
  --once
```

If that passes, set the service command to the long-running form:

```bash
ibkr-trading-runtime run --config-dir config --connect-ib
```

Then start:

```bash
docker compose -f deployments/ibkr/docker-compose.yml up -d ibkr-trading
docker compose -f deployments/ibkr/docker-compose.yml logs -f --tail 100 ibkr-trading
```

### 7.7 IBKR evidence checks

On the VPS:

```bash
docker compose -f deployments/ibkr/docker-compose.yml ps
docker compose -f deployments/ibkr/docker-compose.yml logs ibkr-trading --tail 200
curl -sf http://<WORKSTATION_PRIVATE_IP>:8001/health
```

On the local assistant:

```powershell
curl.exe -H "X-Api-Key: <RELAY_API_KEY>" "http://127.0.0.1:8001/events?limit=10"
curl.exe -H "X-Api-Key: <ORCHESTRATOR_API_KEY>" http://127.0.0.1:8000/events/pending
```

Expected IBKR bot ids in relay events:

- `swing_multi_01`
- `momentum_nq_01`
- `stock_trader`

### 7.8 IBKR paper gate

Run for at least five full US trading sessions before any live pilot:

- all enabled strategy families heartbeat during their market windows;
- IB Gateway reconnect handling works across daily reset;
- relay receives all three family streams;
- dashboard or DB state shows current strategy state without stale heartbeat;
- watchdog alert is tested by stopping the runtime and confirming alert;
- no unexplained broker rejection or unowned position remains;
- generated runtime metadata is copied back to the local assistant and accepted
  by the metadata installer.

---

## 8. Phase 6: Crypto VPS Deployment

Deploy this only on the crypto VPS.

### 8.1 Crypto prerequisites

1. Hyperliquid testnet wallet exists and is funded for paper-mode testing.
2. Wallet address and private key are stored only in an untracked
   `config/live_config.json` or secret mount.
3. Historical warmup data exists for BTC, ETH, and SOL.
4. The local relay is reachable from the VPS over the private network:

```bash
curl -sf http://<WORKSTATION_PRIVATE_IP>:8001/health
```

### 8.2 Maintain crypto compose

`deployments/crypto/docker-compose.yml` is the active VPS compose file. Keep
production changes in this deployment file:

- build context remains `../..`;
- Dockerfile remains `trading/crypto_trader/Dockerfile`;
- bot service name should remain `crypto-trader` unless the cutover plan is
  updated;
- bind-mount the untracked secret live config, default
  `deployments/crypto/secrets/live_config.json`, into the container as
  `/app/trading/crypto_trader/config/live_config.json`;
- mount persistent `data`, `output`, and `live_state`;
- keep `postgres` enabled when `POSTGRES_DSN` instrumentation is active;
- keep `data-refresh` profile-gated under `maintenance`.

### 8.3 Crypto config

Start from `trading/crypto_trader/config/live_config.example.json`, write the
untracked live config, and verify these fields:

```json
{
  "is_testnet": true,
  "wallet_address": "0x...",
  "private_key": "0x...",
  "relay_url": "http://<WORKSTATION_PRIVATE_IP>:8001",
  "relay_secret": "<CRYPTO_HMAC>",
  "reconciliation_policy": "block",
  "strict_live_parity": false,
  "state_dir": "data/live_state",
  "portfolio_config_path": "output/portfolio/round_3/recommended_portfolio_config.json",
  "strategy_configs": {
    "trend": "output/portfolio/round_3/recommended_strategy_configs/trend.json",
    "momentum": "output/portfolio/round_3/recommended_strategy_configs/momentum.json",
    "breakout": "output/portfolio/round_3/recommended_strategy_configs/breakout.json"
  }
}
```

Mainnet later requires `asset_meta_path`, `strict_live_parity=true`, and a
separate live wallet decision.

### 8.4 Seed data

```bash
cd /opt/trading_agent/repo
docker compose -f deployments/crypto/docker-compose.yml build crypto-trader
docker compose -f deployments/crypto/docker-compose.yml run --rm crypto-trader \
  crypto-trader download \
  --coin BTC,ETH,SOL \
  --interval 15m,30m,1h,4h,1d \
  --days 90 \
  --data-dir data
```

If you copy data from local, copy only required candles and funding data, then
verify inside the container:

```bash
docker compose -f deployments/crypto/docker-compose.yml run --rm crypto-trader \
  sh -lc "find data -maxdepth 3 -type f | head"
```

### 8.5 Paper runtime start

Foreground test:

```bash
docker compose -f deployments/crypto/docker-compose.yml run --rm crypto-trader \
  crypto-trader paper --config config/live_config.json
```

Long-running service command:

```bash
crypto-trader paper --config config/live_config.json
```

Start:

```bash
docker compose -f deployments/crypto/docker-compose.yml up -d crypto-trader
docker compose -f deployments/crypto/docker-compose.yml logs -f --tail 100 crypto-trader
```

### 8.6 Crypto evidence checks

```bash
docker compose -f deployments/crypto/docker-compose.yml exec crypto-trader \
  crypto-trader status --state-dir data/live_state

docker compose -f deployments/crypto/docker-compose.yml exec crypto-trader \
  crypto-trader parity-report --state-dir data/live_state --output data/live_state/parity_report.json

docker compose -f deployments/crypto/docker-compose.yml exec crypto-trader \
  crypto-trader parity-gate --report data/live_state/parity_report.json
```

On the local assistant, verify relay events from the crypto bot id currently
configured in `live_config.json`.

### 8.7 Crypto paper gate

Run paper mode for at least 48 hours, then require:

- no missing warmup data;
- no stale candles or funding warnings;
- sidecar running and relay health included in health reports;
- parity report gate passes;
- reconciliation policy blocks unmanaged state;
- paper-status matches the wallet's actual testnet position and order state;
- generated deployment metadata is installed on the local assistant.

---

## 9. Phase 7: K-stock VPS Deployment

Deploy this only on the K-stock VPS.

### 9.1 K-stock prerequisites

1. KIS account and API app are active.
2. VPS public IP is allow-listed in the KIS developer portal.
3. `KIS_IS_PAPER=true` for paper phases.
4. KRX/KIS data directories have enough disk space.
5. Host clock is UTC; app/container logic uses `Asia/Seoul`.
6. The local relay is reachable from the VPS over the private network:

```bash
curl -sf http://<WORKSTATION_PRIVATE_IP>:8001/health
```

### 9.2 K-stock `.env`

Create `/opt/trading_agent/repo/deployments/k_stock/.env` or the path
referenced by compose:

```bash
KIS_APP_KEY=<LIVE_OR_PAPER_APP_KEY>
KIS_APP_SECRET=<LIVE_OR_PAPER_APP_SECRET>
KIS_ACCOUNT_NO=<ACCOUNT_NO>
KIS_ACCOUNT_PROD_CODE=01
KIS_IS_PAPER=true
KIS_HTS_ID=<HTS_ID>
KIS_MY_AGENT=Mozilla/5.0

POSTGRES_PASSWORD=<POSTGRES_ADMIN_PASSWORD>
POSTGRES_WRITER_PASSWORD=<POSTGRES_WRITER_PASSWORD>
POSTGRES_READER_PASSWORD=<POSTGRES_READER_PASSWORD>
OMS_ID=primary

K_STOCK_HOST_DATA_ROOT=/opt/trading_agent/runtime_data/k_stock

OLR_KALCB_PREFLIGHT_MODE=artifact_only_stage1
OLR_KALCB_RUNTIME_MODE=dry_run
OLR_KALCB_MARKET_DATA_SOURCE=auto
OLR_KALCB_DAILY_UNIVERSE_FILE=config/olr_kalcb/olr_deployment_universe_103.yaml
OLR_KALCB_BASELINE_MANIFEST=data/live_readiness/olr_kalcb/2026-05-28/baseline_manifest.json
OLR_KALCB_PORTFOLIO_POLICY=config/olr_kalcb/portfolio_policy.conservative.json
OLR_KALCB_SECTOR_MAP=config/olr/sector_map.yaml
OLR_KALCB_POLL_SECONDS=15
OLR_KALCB_KIS_WS_URL=
OLR_KALCB_WS_LEDGER_PATH=
OLR_KALCB_DEPLOYMENT_METADATA_PATH=
OLR_KALCB_DEPLOYMENT_METADATA_ENV=paper_vps
OLR_KALCB_STRATEGY_PLUGIN_CONTRACT=contracts/strategy_plugins/k_stock_olr_kalcb/strategy_plugin_contract.json
OLR_KALCB_ARTIFACT_TIMEOUT_SECONDS=3600

RELAY_URL=http://<WORKSTATION_PRIVATE_IP>:8001/events
INSTRUMENTATION_HMAC_SECRET=<K_STOCK_HMAC>
ASSISTANT_EVENT_DATA_DIR=instrumentation/data
```

### 9.3 Maintain K-stock compose

Keep `deployments/k_stock/docker-compose.yml` aligned with:

- `postgres` using the bot SQL migrations plus the env-driven
  `deployments/k_stock/postgres-init/004_roles_env.sh`;
- `oms` if paper/live orders route through the K-stock OMS service;
- `k-stock-trader` runtime image from `trading/k_stock_trader/Dockerfile`;
- profile-gated `k-stock-trader` runtime and `k-stock-preflight` one-shot
  services;
- `k-stock-sidecar` in the runtime profile, sharing the same
  `instrumentation/data` volume as `k-stock-trader` and forwarding to the local
  relay;
- named volume or bind mount for KIS rate budget state;
- bind mount for `K_STOCK_HOST_DATA_ROOT`, matching the required runtime data
  layout:
  `strategy/`, `backtests/`, `live_readiness/`, `krx_daily_parquet/`,
  `kis_intraday_parquet/`, `oms/`, and writable `paper_live/`;
- bind mounts or named volumes for `data/rate_budget`, `instrumentation/data`,
  and generated deployment metadata.

Keep the first preflight mode `artifact_only_stage1` and the first long-running
runtime mode `dry_run`. `artifact_only` and `artifact_only_stage1` are preflight
gates only; `k-stock-trader` accepts only `dry_run`, `paper`, or `live`.
Do not configure live KIS trading until artifact-only, dry-run OMS, and paper
gates pass.

### 9.4 Build and preflight

```bash
cd /opt/trading_agent/repo
docker compose -f deployments/k_stock/docker-compose.yml build k-stock-trader
docker compose -f deployments/k_stock/docker-compose.yml up -d postgres
docker compose -f deployments/k_stock/docker-compose.yml ps
```

Run an artifact-only preflight for the next KRX trade date:

```bash
export TRADE_DATE=<YYYY-MM-DD>
OLR_KALCB_TRADE_DATE="$TRADE_DATE" \
OLR_KALCB_PREFLIGHT_MODE=artifact_only_stage1 \
docker compose -f deployments/k_stock/docker-compose.yml --profile preflight run --rm k-stock-preflight
```

Before this preflight, populate `K_STOCK_HOST_DATA_ROOT` on the K-stock VPS with
the required runtime data layout: approved
`live_readiness/olr_kalcb/.../baseline_manifest.json`, generated
`strategy/kalcb` and `strategy/olr` artifact stores, KRX daily parquet, KIS
intraday parquet, OMS read-side snapshots if used, and a writable `paper_live/`
evidence directory.

### 9.5 Install KRX cron flow

After the preflight works manually, install the K-stock cron wrappers shipped in
the monorepo:

```bash
sudo mkdir -p /var/log/k_stock_trader
sudo chown trader:trader /var/log/k_stock_trader
chmod +x trading/k_stock_trader/infra/cron/olr_kalcb_premarket_restart.sh
chmod +x trading/k_stock_trader/infra/cron/olr_kalcb_afternoon_restart.sh
```

Crontab, expressed in UTC for KST:

```cron
PATH=/usr/local/bin:/usr/bin:/bin
OLR_KALCB_REPO_ROOT=/opt/trading_agent/repo/trading/k_stock_trader
30 22 * * 0-4  /opt/trading_agent/repo/trading/k_stock_trader/infra/cron/olr_kalcb_premarket_restart.sh
32 5  * * 1-5  /opt/trading_agent/repo/trading/k_stock_trader/infra/cron/olr_kalcb_afternoon_restart.sh
```

Set `OLR_KALCB_REPO_ROOT` as above; patch the wrappers only if local path
assumptions prevent them from running under `/opt/trading_agent/repo`.

### 9.6 K-stock gate sequence

Run the phases in order:

1. Artifact-only gate: at least three KRX sessions. Both daily and afternoon
   artifacts generate, preflight passes, no partial universe flags.
2. Dry-run OMS gate: at least three KRX sessions. Use captured completed bars,
   build offline replay, and require parity.
3. Paper gate: at least ten KRX sessions with `KIS_IS_PAPER=true` and
   `OLR_KALCB_RUNTIME_MODE=paper`. Each paper start must include
   `health_checks.json`, `account_state.json`, `positions.json`, and emitted
   `deployment_metadata.json` for the same trade date/session root.
4. Live pilot: only after written approval.

Dry-run command template:

```bash
docker compose -f deployments/k_stock/docker-compose.yml run --rm k-stock-trader \
  k-stock-olr-kalcb-runtime dry-run-bars \
  --trade-date <YYYY-MM-DD> \
  --bars-parquet data/paper_live/olr_kalcb/<YYYY-MM-DD>/market_bars_5m.parquet \
  --session-root data/paper_live/olr_kalcb/<YYYY-MM-DD> \
  --build-offline-replay
```

Paper watch command template:

```bash
docker compose -f deployments/k_stock/docker-compose.yml run --rm k-stock-trader \
  k-stock-olr-kalcb-runtime watch-bars \
  --trade-date <YYYY-MM-DD> \
  --mode paper \
  --market-data-source kis_websocket \
  --session-root data/paper_live/olr_kalcb/<YYYY-MM-DD> \
  --health-checks-json data/paper_live/olr_kalcb/<YYYY-MM-DD>/health_checks.json \
  --account-state-json data/paper_live/olr_kalcb/<YYYY-MM-DD>/account_state.json \
  --positions-json data/paper_live/olr_kalcb/<YYYY-MM-DD>/positions.json \
  --deployment-metadata-json data/paper_live/olr_kalcb/<YYYY-MM-DD>/deployment_metadata.json \
  --strategy-plugin-contract contracts/strategy_plugins/k_stock_olr_kalcb/strategy_plugin_contract.json \
  --deployment-metadata-environment paper_vps \
  --poll-seconds 15
```

For the compose-managed service path, set these values in
`deployments/k_stock/.env` and then start the runtime profile. The launch
module derives `data/paper_live/olr_kalcb/<YYYY-MM-DD>/` session paths by
default; override `OLR_KALCB_SESSION_ROOT` only for a non-standard capture
directory.

```bash
OLR_KALCB_RUNTIME_MODE=paper
OLR_KALCB_MARKET_DATA_SOURCE=auto
OLR_KALCB_DEPLOYMENT_METADATA_PATH=data/paper_live/olr_kalcb/<YYYY-MM-DD>/deployment_metadata.json
```

Deployed runtime should start the trader and sidecar together:

```bash
docker compose -f deployments/k_stock/docker-compose.yml --profile runtime up -d k-stock-trader k-stock-sidecar
```

### 9.7 K-stock evidence checks

After each KRX session:

- premarket log ends cleanly;
- afternoon log ends cleanly;
- KALCB daily artifact exists;
- OLR stage1 and final artifacts exist when expected;
- session manifest is hash sealed;
- offline replay report passes;
- relay has `k_stock_trader` events;
- assistant has ingested and processed the events;
- no OMS position drift or unknown allocation remains.

---

## 10. Phase 8: Deployment Metadata Back To Assistant

The local assistant can only promote or approval-gate a strategy bridge after
it has installed runtime deployment metadata from the bot runtime.

### 10.1 Emit metadata on bot VPSes

Preferred monorepo command from the deploy checkout:

```bash
python tools/run_runtime_deployment_metadata_matrix.py --bot <ibkr|crypto|k_stock>
python tools/verify_deployment_metadata.py --bot <ibkr|crypto|k_stock>
```

If emitting from inside containers, use the runtime entrypoints only after the
compose service bind-mounts the top-level `contracts/` and `deployments/`
paths into the container. Otherwise, run these from the deploy checkout on the
host with the package import paths available.

IBKR:

```bash
ibkr-trading-runtime emit-deployment-metadata \
  --contract contracts/strategy_plugins/trading_swing_family/strategy_plugin_contract.json \
  --effective-config deployments/ibkr/generated/strategies.effective.json \
  --output deployments/ibkr/generated/runtime_deployment_metadata/trading_swing_family/deployment_metadata.json \
  --repo-root . \
  --runtime-started-at-utc <UTC_ISO_TIMESTAMP> \
  --runtime-instance-id ibkr:trading_swing_family:<SHA>
```

Crypto:

```bash
crypto-trader emit-deployment-metadata \
  --effective-config deployments/crypto/generated/live_config.effective.json \
  --contract-source-root contracts/strategy_plugins \
  --contract-work-root artifacts/validation/runtime_deployment_metadata/raw/crypto_contracts \
  --state-dir artifacts/validation/runtime_deployment_metadata/raw/crypto_state \
  --repo-root . \
  --runtime-started-at-utc <UTC_ISO_TIMESTAMP>
```

K-stock:

```bash
k-stock-olr-kalcb-runtime emit-deployment-metadata \
  --contract contracts/strategy_plugins/k_stock_olr_kalcb/strategy_plugin_contract.json \
  --effective-config deployments/k_stock/generated/olr_kalcb.effective.json \
  --output deployments/k_stock/generated/runtime_deployment_metadata/k_stock_olr_kalcb/deployment_metadata.json \
  --repo-root . \
  --runtime-started-at-utc <UTC_ISO_TIMESTAMP> \
  --runtime-instance-id k_stock:k_stock_olr_kalcb:<SHA>
```

### 10.2 Install metadata locally

Copy metadata files back to the local workstation and install them through the
assistant backtest package:

```bash
trading-assistant-backtest-install-deployment-metadata \
  --agent-root . \
  --bridge-id <bridge_id> \
  --metadata <path-to-deployment_metadata.json> \
  --install
```

Repeat for:

- `trading_swing_family`
- `trading_momentum_family`
- `trading_stock_family`
- `crypto_trend_v1`
- `crypto_momentum_v1`
- `crypto_breakout_v1`
- `k_stock_olr_kalcb`

The installer is the source of truth for promotion metadata. It rejects dirty
checkouts, local/helper-emitted metadata, contract-hash mismatches, and
telemetry-schema mismatches. The runtime metadata matrix is expected to pass
from a clean release checkout for all seven contracts before metadata is copied
to the local assistant.

Do not commit stale generated `deployment_metadata.json` files under
`deployments/*/generated/runtime_deployment_metadata/` or
`artifacts/validation/runtime_deployment_metadata/raw/`. Those paths are
runtime outputs. If `python tools/verify_deployment_metadata.py --bot all`
fails because the local checkout is dirty, fix the checkout first; the verifier
will not validate pre-existing generated metadata because it may be stale. If
it fails from a clean checkout, stop the deployment and fix the emitting bot
runtime or contract evidence before continuing.

Then run:

```bash
trading-assistant-backtest-bridge-readiness --agent-root .
trading-assistant-backtest-validation-matrix --agent-root .
trading-assistant-backtest-approval-grade-audit --agent-root .
```

The installer must reject local, shadow, dry-run, dirty-checkout,
SHA-mismatched, contract-hash-mismatched, and telemetry-schema-mismatched
metadata. Treat any rejection as a deployment blocker.

---

## 11. Phase 9: Data Bundle And Monthly Validation

Run this locally after the bot VPSes are producing events and metadata.

### 11.1 Source refresh gates

Set read-only data refresh env vars locally:

```bash
TA_SOURCE_REFRESH_ALLOW_NETWORK=true
TA_SOURCE_REFRESH_ALLOW_WRITE=true
IBKR_HOST=127.0.0.1
IBKR_PORT=4002
IBKR_CLIENT_ID=107
IBKR_READ_ONLY_ACK=true
KIS_ACCOUNT_MODE=paper
KIS_APP_KEY=<READ_ONLY_OR_PAPER_KEY>
KIS_APP_SECRET=<READ_ONLY_OR_PAPER_SECRET>
KIS_READ_ONLY_ACK=true
```

Hyperliquid candles/funding are public; do not use a trading wallet private key
for local data refresh.

### 11.2 Declare and refresh source requests

```bash
trading-assistant-data --repo-root packages/trading_assistant_data \
  declare-source-requests --snapshot <YYYY-MM-DD> --json

trading-assistant-data --repo-root packages/trading_assistant_data \
  sync hyperliquid --symbols BTC,ETH,SOL --intervals 15m,30m,1h,4h,1d \
  --latest --funding --json

trading-assistant-data --repo-root packages/trading_assistant_data \
  sync ibkr --families trading_momentum,trading_swing,trading_stock --json

trading-assistant-data --repo-root packages/trading_assistant_data \
  sync kis --families k_stock_kis_intraday --intraday --json
```

### 11.3 Build and audit bundles

```bash
trading-assistant-data --repo-root packages/trading_assistant_data audit-coverage \
  --run-month <YYYY-MM> --json

trading-assistant-data --repo-root packages/trading_assistant_data compare-legacy-source \
  --families trading_momentum,trading_swing,trading_stock,k_stock_kis_intraday,crypto_portfolio \
  --latest-only --json
```

Build exact bundles for the scope being promoted. A deployable monthly bundle
must be authoritative, indexed, checksum-clean, and have deterministic
explanations for any non-exact legacy-source comparison.

### 11.4 Shadow monthly validation

```bash
trading-assistant-backtest-data-reproduction --agent-root . --scope all
trading-assistant-backtest-replay-evidence --agent-root . --scope all
trading-assistant-backtest-validation-matrix --agent-root .
trading-assistant-backtest-approval-grade-audit --agent-root .
```

Keep `MONTHLY_VALIDATION_MODE=shadow` until at least one shadow cycle has
completed with installed live/VPS deployment metadata for the promoted scope.

---

## 12. Phase 10: Promotion To Approval-Gated And Live

### 12.1 Assistant promotion

Only flip the local assistant to approval-gated after:

- relay and orchestrator survive a workstation reboot;
- all bot sidecars are forwarding;
- runtime deployment metadata is installed;
- `python tools/verify_cutover_plan.py` passes for the selected deployable
  compose files and generated live configs;
- `python tools/verify_operational_deployment_evidence.py` passes against
  `deployments/operational_evidence.json`;
- the promoted scope has a production `scheduled_shadow_cycle_report.json`
  sourced from monthly validation output with installed metadata reports,
  local relay ingest evidence, `source_kind=monthly_validation_shadow`, and
  `adoption_disabled=true`;
- the promoted scope's production fixture-set manifest covers accepted entry,
  blocked no-trade, risk/portfolio denial, exit/close, order/fill or explicit
  non-fill, and live/shadow telemetry source case classes;
- scoped live-config promotion evidence passes for the promoted strategies.
  For the first pilot this means `IARIC_v1` and `ALCB_v1`; broader unrelated
  IBKR verifier failures remain context and do not define the pilot gate;
- validation matrix reports `approval_grade_validation_complete=true`;
- `approval_remaining_gaps` is empty for every promoted scope;
- `trading-assistant-backtest-approval-evidence --agent-root . --scope
  <scope_id>` emits an eligible bundle with matching source hashes;
- each promoted strategy contract has been deliberately advanced from
  `shadow_validated` to `approval_ready`;
- explicit P6/P7 optimizer evidence exists for every promoted scope:
  purged two-fold scoring, post-ranking selection-OOS repair evidence,
  confirmatory rerank, and round_N+1 recommendation or deterministic
  no-adoption;
- a known-safe approval card has been delivered through Telegram.

Then:

```bash
# packages/trading_assistant/.env
MONTHLY_VALIDATION_MODE=approval_gated
DEPLOYMENT_MONITORING_ENABLED=true
```

Restart the orchestrator and verify `/health`, `/metrics`, relay polling, and
Telegram approval delivery.

Approval sequencing:

1. Promote `trading_stock_family` first through the guarded approval-evidence
   bundle path.
2. Repeat the same path for `k_stock_olr_kalcb`,
   `trading_momentum_family`, and `trading_swing_family` after their own
   production evidence is complete.
3. Do not promote `crypto_trader_portfolio` until its parity/head mismatch is
   resolved and it has passed the same production evidence path.

Each scope needs both guarded maturity promotion to `approval_ready` and
explicit approval-grade P6/P7 optimizer manifests before it can leave
shadow-only validation.

### 12.2 Bot live promotion order

Use this order unless a written risk decision says otherwise:

1. K-stock remains artifact-only until its KRX gate sequence is complete.
2. Crypto paper runs first live pilot with tiny size, because the venue path is
   simpler than IB Gateway and KIS but wallet risk is direct.
3. IBKR live pilot follows after paper evidence across daily reset and market
   data behavior is clean.
4. K-stock live pilot comes last because it has the strictest market-calendar,
   artifact, and KIS source-IP dependencies.

### 12.3 Live pilot checklist

For each bot:

- paper or artifact gate requirements are complete;
- deployed config hash matches generated effective config;
- deployment metadata was emitted from the VPS and installed locally;
- local assistant has fresh evidence for the bot;
- rollback command from `deployments/cutover_plan.json` was tested;
- max daily loss and per-strategy caps are documented;
- manual stop command is known and tested;
- first live session is supervised;
- post-session parity/replay report is written before increasing size.

---

## 13. Operations

### 13.1 Daily checks

Local assistant:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8001/health
curl.exe -H "X-Api-Key: <ORCHESTRATOR_API_KEY>" http://127.0.0.1:8000/metrics
curl.exe -H "X-Api-Key: <ORCHESTRATOR_API_KEY>" http://127.0.0.1:8000/events/pending
```

Each VPS:

```bash
cd /opt/trading_agent/repo
docker compose -f deployments/<bot>/docker-compose.yml ps
docker compose -f deployments/<bot>/docker-compose.yml logs --tail 100
df -h
```

### 13.2 Backups

On each VPS:

- backup `.env` manually into encrypted storage;
- backup Postgres if enabled;
- backup runtime state and session evidence;
- keep at least 30 days of JSONL instrumentation locally.

Example:

```bash
mkdir -p /opt/trading_agent/backups/$(date +%F)
tar czf /opt/trading_agent/backups/$(date +%F)/runtime_state.tgz runtime_state runtime_artifacts
```

On the local assistant:

- backup `packages/trading_assistant/data`;
- backup `packages/trading_assistant/memory`;
- backup `packages/trading_assistant_data/data/export`;
- backup installed deployment metadata and monthly validation artifacts.

### 13.3 Updates

For code updates:

```bash
git fetch origin
git checkout <NEW_DEPLOYMENT_SHA>
python tools/generate_effective_live_configs.py
python tools/verify_effective_live_configs.py
python tools/build_bot_image.py --bot <bot> --emit-dependency-reports
python tools/verify_deployment_metadata.py --bot <bot>
python tools/verify_cutover_plan.py
```

On the target VPS:

```bash
cd /opt/trading_agent/repo
git fetch origin
git checkout <NEW_DEPLOYMENT_SHA>
docker compose -f deployments/<bot>/docker-compose.yml build
docker compose -f deployments/<bot>/docker-compose.yml up -d --no-deps --force-recreate <bot-service>
docker compose -f deployments/<bot>/docker-compose.yml logs -f --tail 100 <bot-service>
```

Do not update all three bot VPSes in the same maintenance window until the
one-bot rollout process has been proven.

### 13.4 Rollback

Use the bot-specific rollback record from `deployments/cutover_plan.json`.
Baseline restore pattern:

```bash
cd /opt/trading_agent/repo
git checkout <PREVIOUS_GOOD_SHA>
docker compose -f deployments/<bot>/docker-compose.yml up -d --no-build
docker compose -f deployments/<bot>/docker-compose.yml logs --tail 100 <bot-service>
```

After rollback:

- emit or copy rollback runtime metadata if the runtime actually restarted;
- mark the deployment in the assistant ledger;
- run the bot-specific restore smoke command recorded in
  `deployments/cutover_plan.json`;
- do not re-promote until the failure has a written root cause.

---

## 14. Troubleshooting Quick Checks

### 14.1 Relay 401

Check:

- bot sidecar HMAC env var matches `RELAY_SHARED_SECRETS`;
- event `bot_id` is present in the relay secret map;
- crypto `relay_secret` matches the relay secret map;
- `X-Api-Key` is present for local assistant polling.

### 14.2 Bot is trading but assistant has no events

Check:

```bash
curl -sf http://<WORKSTATION_PRIVATE_IP>:8001/health
docker compose -f deployments/<bot>/docker-compose.yml logs --tail 200 | grep -i sidecar
```

Then on the workstation:

```powershell
curl.exe -H "X-Api-Key: <RELAY_API_KEY>" "http://127.0.0.1:8001/events?limit=10"
curl.exe -H "X-Api-Key: <ORCHESTRATOR_API_KEY>" http://127.0.0.1:8000/events/pending
```

### 14.3 Compose starts but exits immediately

Check the rendered command and mounted env/config first:

```bash
docker compose -f deployments/<bot>/docker-compose.yml config
docker compose -f deployments/<bot>/docker-compose.yml logs --tail 200 <service>
```

If the service was manually overridden back to a help command, restore the
production command from the deployment compose file. If K-stock exits with code
`64`, set `OLR_KALCB_TRADE_DATE` in the VPS `.env`.

### 14.4 Deployment metadata install fails

Common causes:

- metadata was emitted from the local dev checkout instead of the VPS;
- worktree was dirty;
- repo remote was local or missing;
- deployed SHA does not match the contract's expected SHA;
- config hash does not match the generated effective config;
- telemetry schema is missing from the strategy plugin contract.

Re-run:

```bash
python tools/verify_deployment_metadata.py --bot <bot>
trading-assistant-backtest-install-deployment-metadata --agent-root . --bridge-id <bridge_id> --metadata <file> --install
```

### 14.5 K-stock cron runs on wrong date

Check:

- host time is UTC;
- KST conversion is correct;
- `OLR_KALCB_REPO_ROOT` points at `trading/k_stock_trader` inside the monorepo;
- trade date resolver sees a valid KRX session;
- cron logs are under `/var/log/k_stock_trader`.

---

## 15. Completion Criteria

The deployment implementation is complete when:

- the final deployment commit is clean, pushed, and named by the agreed release
  reference or tag;
- all three bot VPSes are running separate paper/artifact deployments from the
  same reviewed monorepo commit;
- all bot sidecars forward to the local relay through the private network;
- the local assistant polls, ingests, deduplicates, and processes bot events;
- generated runtime deployment metadata from each VPS is installed locally;
- `python tools/verify_deployment_metadata.py --bot all` passes from the clean
  release/deploy checkout that produced the VPS runtime metadata;
- monthly validation runs in shadow mode with real bot metadata;
- production scheduled-shadow evidence for each promoted scope includes monthly
  validation output, installed metadata reports, local relay ingest evidence,
  `source_kind=monthly_validation_shadow`, and `adoption_disabled=true`;
- production fixture breadth covers the required case classes, including
  live/shadow telemetry source evidence;
- `trading-assistant-backtest-approval-evidence --agent-root . --scope
  <scope_id>` emits an eligible bundle for every scope promoted beyond shadow;
- `validation-matrix` has no `approval_remaining_gaps` for any scope being
  promoted beyond shadow;
- `deployments/cutover_plan.json` has tested rollback evidence for every bot;
- `deployments/operational_evidence.json` passes
  `python tools/verify_operational_deployment_evidence.py`;
- operational evidence includes per-bot relay ingest evidence linked to
  deployment metadata, and crypto sidecar runtime policy evidence with the
  configured degraded relay incident action;
- promotion to approval-gated mode is documented and manually approved;
- no live capital is enabled before the relevant bot-specific paper/artifact
  gate passes.

The first successful production milestone is not "containers are up." It is a
full evidence loop: bot event -> relay -> assistant ingest -> validation
artifact -> approval surface -> documented rollback path.
