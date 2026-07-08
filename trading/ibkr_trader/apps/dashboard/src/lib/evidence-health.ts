type EnvSource = Record<string, string | undefined>;

export type EvidenceHealthStatus = 'OK' | 'WARNING' | 'ERROR' | 'UNKNOWN';

export interface RelayHealthPayload {
  status?: string;
  pending_events?: number;
  per_bot_pending?: Record<string, number>;
  last_event_per_bot?: Record<string, string>;
  oldest_pending_age_seconds?: number;
  uptime_seconds?: number;
  db_size_bytes?: number;
}

export interface EvidencePipelineHealth {
  status: EvidenceHealthStatus;
  checked_at: string;
  warnings: string[];
  relay: {
    status: EvidenceHealthStatus;
    url: string | null;
    reachable: boolean;
    pending_events: number | null;
    oldest_pending_age_seconds: number | null;
    per_bot_pending: Record<string, number>;
    warnings: string[];
  };
  assistant: {
    status: EvidenceHealthStatus;
    required_bot_ids: string[];
    missing_bot_ids: string[];
    stale_bot_ids: string[];
    last_event_per_bot: Record<string, string>;
    warnings: string[];
  };
}

interface EvidenceHealthEvaluationInput {
  relayUrl?: string | null;
  relayReachable: boolean;
  relayPayload?: RelayHealthPayload | null;
  relayError?: string | null;
  requiredBotIds?: string[];
  now?: Date | string;
  backlogThreshold?: number;
  relayStalePendingSeconds?: number;
  assistantStaleEventSeconds?: number;
}

const DEFAULT_RELAY_HEALTH_URL = 'http://127.0.0.1:8000/health';
const DEFAULT_RELAY_BACKLOG_THRESHOLD = 500;
const DEFAULT_RELAY_STALE_PENDING_SECONDS = 600;
const DEFAULT_ASSISTANT_STALE_EVENT_SECONDS = 3600;
const DEFAULT_RELAY_TIMEOUT_MS = 2000;

function parsePositiveNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseBotList(value: string | undefined): string[] {
  return (value ?? '')
    .split(',')
    .map(v => v.trim())
    .filter(Boolean);
}

function relayHealthUrlFromBase(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, '');
  if (!trimmed) {
    return DEFAULT_RELAY_HEALTH_URL;
  }
  return trimmed.endsWith('/health') ? trimmed : `${trimmed}/health`;
}

export function getRelayHealthUrl(env: EnvSource = process.env): string {
  const explicit = (
    env.DASHBOARD_RELAY_HEALTH_URL ??
    env.INSTRUMENTATION_RELAY_HEALTH_URL ??
    env.RELAY_HEALTH_URL ??
    ''
  ).trim();
  if (explicit) {
    return relayHealthUrlFromBase(explicit);
  }

  const relayBase = (
    env.DASHBOARD_RELAY_URL ??
    env.INSTRUMENTATION_RELAY_URL ??
    env.RELAY_URL ??
    DEFAULT_RELAY_HEALTH_URL
  ).trim();
  return relayHealthUrlFromBase(relayBase);
}

export function getRequiredAssistantBotIds(env: EnvSource = process.env): string[] {
  return parseBotList(
    env.DASHBOARD_ASSISTANT_REQUIRED_BOTS ??
    env.DASHBOARD_BOT_IDS ??
    env.TRADING_ASSISTANT_BOTS,
  );
}

function statusRank(status: EvidenceHealthStatus): number {
  if (status === 'ERROR') return 3;
  if (status === 'WARNING') return 2;
  if (status === 'UNKNOWN') return 1;
  return 0;
}

function worstStatus(statuses: EvidenceHealthStatus[]): EvidenceHealthStatus {
  return statuses.reduce(
    (worst, current) => statusRank(current) > statusRank(worst) ? current : worst,
    'OK' as EvidenceHealthStatus,
  );
}

function finiteNumber(value: unknown): number | null {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseTimestampAgeSeconds(timestamp: string, nowMs: number): number | null {
  const parsed = Date.parse(timestamp);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return Math.max(0, (nowMs - parsed) / 1000);
}

export function evaluateEvidencePipelineHealth(
  input: EvidenceHealthEvaluationInput,
): EvidencePipelineHealth {
  const now = input.now instanceof Date ? input.now : new Date(input.now ?? Date.now());
  const checkedAt = Number.isFinite(now.getTime()) ? now.toISOString() : new Date().toISOString();
  const nowMs = Number.isFinite(now.getTime()) ? now.getTime() : Date.now();
  const backlogThreshold = input.backlogThreshold ?? DEFAULT_RELAY_BACKLOG_THRESHOLD;
  const relayStalePendingSeconds = input.relayStalePendingSeconds ?? DEFAULT_RELAY_STALE_PENDING_SECONDS;
  const assistantStaleEventSeconds = input.assistantStaleEventSeconds ?? DEFAULT_ASSISTANT_STALE_EVENT_SECONDS;

  const relayPayload = input.relayPayload ?? null;
  const pendingEvents = finiteNumber(relayPayload?.pending_events);
  const oldestPendingAgeSeconds = finiteNumber(relayPayload?.oldest_pending_age_seconds);
  const perBotPending = relayPayload?.per_bot_pending ?? {};
  const lastEventPerBot = relayPayload?.last_event_per_bot ?? {};

  const relayWarnings: string[] = [];
  let relayStatus: EvidenceHealthStatus = 'OK';

  if (!input.relayReachable) {
    relayStatus = 'ERROR';
    relayWarnings.push(input.relayError || 'relay health endpoint is unreachable');
  } else if (!relayPayload) {
    relayStatus = 'ERROR';
    relayWarnings.push('relay health endpoint returned no payload');
  } else {
    const relayReportedStatus = String(relayPayload.status ?? 'ok').trim().toLowerCase();
    if (relayReportedStatus && relayReportedStatus !== 'ok') {
      relayStatus = 'WARNING';
      relayWarnings.push(`relay reported status ${relayReportedStatus}`);
    }
    if (pendingEvents !== null && pendingEvents > backlogThreshold) {
      relayStatus = worstStatus([relayStatus, 'WARNING']);
      relayWarnings.push(`relay backlog ${pendingEvents} exceeds threshold ${backlogThreshold}`);
    }
    if (oldestPendingAgeSeconds !== null && oldestPendingAgeSeconds > relayStalePendingSeconds) {
      relayStatus = worstStatus([relayStatus, 'WARNING']);
      relayWarnings.push(
        `oldest pending relay event is ${Math.round(oldestPendingAgeSeconds)}s old`,
      );
    }
  }

  const requiredBotIds = input.requiredBotIds ?? [];
  const missingBotIds = requiredBotIds.filter(botId => !lastEventPerBot[botId]);
  const staleBotIds = requiredBotIds.filter(botId => {
    const timestamp = lastEventPerBot[botId];
    if (!timestamp) return false;
    const ageSeconds = parseTimestampAgeSeconds(timestamp, nowMs);
    return ageSeconds !== null && ageSeconds > assistantStaleEventSeconds;
  });
  const assistantWarnings: string[] = [];
  let assistantStatus: EvidenceHealthStatus = 'OK';

  if (!input.relayReachable) {
    assistantStatus = 'UNKNOWN';
    assistantWarnings.push('assistant ingestion freshness cannot be checked until relay is reachable');
  } else if (requiredBotIds.length > 0 && missingBotIds.length > 0) {
    assistantStatus = 'ERROR';
    assistantWarnings.push(`missing assistant ingestion evidence for ${missingBotIds.join(', ')}`);
  } else if (requiredBotIds.length === 0 && Object.keys(lastEventPerBot).length === 0) {
    assistantStatus = 'WARNING';
    assistantWarnings.push('relay has no assistant ingestion evidence for any bot');
  }

  if (staleBotIds.length > 0) {
    assistantStatus = worstStatus([assistantStatus, 'WARNING']);
    assistantWarnings.push(`stale assistant ingestion evidence for ${staleBotIds.join(', ')}`);
  }

  const warnings = [...relayWarnings, ...assistantWarnings];
  return {
    status: worstStatus([relayStatus, assistantStatus]),
    checked_at: checkedAt,
    warnings,
    relay: {
      status: relayStatus,
      url: input.relayUrl ?? null,
      reachable: input.relayReachable,
      pending_events: pendingEvents,
      oldest_pending_age_seconds: oldestPendingAgeSeconds,
      per_bot_pending: perBotPending,
      warnings: relayWarnings,
    },
    assistant: {
      status: assistantStatus,
      required_bot_ids: requiredBotIds,
      missing_bot_ids: missingBotIds,
      stale_bot_ids: staleBotIds,
      last_event_per_bot: lastEventPerBot,
      warnings: assistantWarnings,
    },
  };
}

export async function getEvidencePipelineHealth(
  env: EnvSource = process.env,
): Promise<EvidencePipelineHealth> {
  const relayUrl = getRelayHealthUrl(env);
  const timeoutMs = parsePositiveNumber(env.DASHBOARD_RELAY_HEALTH_TIMEOUT_MS, DEFAULT_RELAY_TIMEOUT_MS);
  const requiredBotIds = getRequiredAssistantBotIds(env);
  const backlogThreshold = parsePositiveNumber(
    env.DASHBOARD_RELAY_BACKLOG_THRESHOLD,
    DEFAULT_RELAY_BACKLOG_THRESHOLD,
  );
  const relayStalePendingSeconds = parsePositiveNumber(
    env.DASHBOARD_RELAY_STALE_PENDING_SECONDS,
    DEFAULT_RELAY_STALE_PENDING_SECONDS,
  );
  const assistantStaleEventSeconds = parsePositiveNumber(
    env.DASHBOARD_ASSISTANT_STALE_EVENT_SECONDS,
    DEFAULT_ASSISTANT_STALE_EVENT_SECONDS,
  );

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(relayUrl, {
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!response.ok) {
      return evaluateEvidencePipelineHealth({
        relayUrl,
        relayReachable: false,
        relayError: `relay health returned HTTP ${response.status}`,
        requiredBotIds,
        backlogThreshold,
        relayStalePendingSeconds,
        assistantStaleEventSeconds,
      });
    }
    const relayPayload = await response.json() as RelayHealthPayload;
    return evaluateEvidencePipelineHealth({
      relayUrl,
      relayReachable: true,
      relayPayload,
      requiredBotIds,
      backlogThreshold,
      relayStalePendingSeconds,
      assistantStaleEventSeconds,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'unknown relay health error';
    return evaluateEvidencePipelineHealth({
      relayUrl,
      relayReachable: false,
      relayError: message,
      requiredBotIds,
      backlogThreshold,
      relayStalePendingSeconds,
      assistantStaleEventSeconds,
    });
  } finally {
    clearTimeout(timeout);
  }
}
