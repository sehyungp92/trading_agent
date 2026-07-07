export const ACTIVE_CONFIG_FRESHNESS_HOURS = 24;

type RuntimeEnvSource = Record<string, string | undefined>;

const RUNTIME_ENV_CANDIDATES = [
  'TRADING_MODE',
  'TRADING_ENV',
  'SWING_TRADER_ENV',
  'ALGO_TRADER_ENV',
  'STOCK_TRADER_ENV',
] as const;

function normalizeEnvValue(value: string | undefined): string {
  return (value ?? '').trim().toLowerCase();
}

export function resolveDashboardRuntimeEnv(env: RuntimeEnvSource = process.env): string {
  for (const candidate of RUNTIME_ENV_CANDIDATES) {
    const runtimeEnv = normalizeEnvValue(env[candidate]);
    if (runtimeEnv) {
      return runtimeEnv;
    }
  }

  return 'dev';
}

export function getDashboardRuntimeEnv(): string {
  return resolveDashboardRuntimeEnv(process.env);
}

export function getDashboardAccountId(): string {
  return (
    process.env.DASHBOARD_ACCOUNT_ID ??
    process.env.IB_ACCOUNT_ID ??
    process.env.ACCOUNT_ID ??
    ''
  ).trim();
}
