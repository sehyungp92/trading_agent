-- retention.sql: Periodic cleanup (run via pg_cron or external cron).
-- Recommended schedule: daily at 03:00 UTC.

DELETE FROM equity_snapshots WHERE timestamp < now() - INTERVAL '90 days';
DELETE FROM health_snapshots WHERE timestamp < now() - INTERVAL '30 days';
DELETE FROM instrumentation_events WHERE received_at < now() - INTERVAL '30 days';
DELETE FROM trades WHERE exit_time < now() - INTERVAL '365 days';
DELETE FROM daily_snapshots WHERE trade_date < CURRENT_DATE - INTERVAL '365 days';
VACUUM ANALYZE equity_snapshots;
VACUUM ANALYZE health_snapshots;
VACUUM ANALYZE instrumentation_events;
VACUUM ANALYZE trades;
VACUUM ANALYZE daily_snapshots;
