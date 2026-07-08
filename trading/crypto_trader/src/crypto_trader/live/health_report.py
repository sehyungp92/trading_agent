"""Structured health report — periodic JSON snapshot tying data flow, funnels, and portfolio."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from crypto_trader.instrumentation.strategy_ids import assistant_strategy_id


@dataclass
class HealthAlert:
    """A single alert condition detected."""

    severity: str  # "warning", "error", "critical"
    name: str
    message: str
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HealthReport:
    """Periodic system health snapshot."""

    timestamp: str
    uptime_sec: float
    data_flow: dict[str, dict] = field(default_factory=dict)
    signal_funnels: dict[str, dict] = field(default_factory=dict)
    gate_breakdown: dict[str, dict] = field(default_factory=dict)
    positions: list[dict] = field(default_factory=list)
    portfolio: dict = field(default_factory=dict)
    system: dict = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)
    assessment: str = "healthy"

    def to_dict(self) -> dict:
        report = asdict(self)
        strategy_ids = set(self.signal_funnels) | set(self.gate_breakdown)
        aliases = {
            strategy_id: assistant_strategy_id(strategy_id)
            for strategy_id in sorted(strategy_ids)
        }
        if aliases:
            report["assistant_strategy_ids"] = aliases
        return report

    def to_text(self) -> str:
        """Format as human-readable summary."""
        lines = [f"=== System Health: {self.assessment.upper()} ==="]
        lines.append(
            f"Uptime: {self.uptime_sec:.0f}s | "
            f"Errors: {self.system.get('total_errors', 0)} | "
            f"Stale feeds: {self.system.get('stale_feed_count', 0)}"
        )

        # Data flow
        lines.append("\n--- Data Flow ---")
        for key, info in self.data_flow.items():
            status = info.get("status", "unknown")
            age = info.get("last_bar_age_sec", "?")
            lines.append(f"  {key}: {status} ({age}s)")

        # Signal funnels
        lines.append("\n--- Signal Funnels ---")
        for sid, funnel in self.signal_funnels.items():
            bars = funnel.get("bars_received", 0)
            ind = funnel.get("indicators_ready", 0)
            setups = funnel.get("setups_detected", 0)
            confirms = funnel.get("confirmations", 0)
            entries = funnel.get("entries_attempted", 0)
            fills = funnel.get("fills", 0)
            lines.append(
                f"  {sid}: bars={bars} -> indicators={ind} -> setups={setups} "
                f"-> confirms={confirms} -> entries={entries} -> fills={fills}"
            )

        # Portfolio
        if self.portfolio:
            lines.append("\n--- Portfolio ---")
            heat = self.portfolio.get("heat_R", 0)
            cap = self.portfolio.get("heat_cap_R", "?")
            daily = self.portfolio.get("daily_pnl_R", 0)
            n_open = self.portfolio.get("open_risk_count", 0)
            lines.append(
                f"  Heat: {heat:.1f}R / {cap}R cap | "
                f"Daily P&L: {daily:+.1f}R | Open: {n_open} positions"
            )

        # Alerts
        lines.append("\n--- Alerts ---")
        if self.alerts:
            for a in self.alerts:
                lines.append(f"  [{a.get('severity', '?')}] {a.get('name', '?')}: {a.get('message', '')}")
        else:
            lines.append("  (none)")

        return "\n".join(lines)


class HealthReportBuilder:
    """Builds a HealthReport from engine components."""

    # Expected bar intervals by timeframe (seconds)
    TF_INTERVALS = {
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }

    def build(
        self,
        *,
        uptime_sec: float,
        health_status: dict,
        stale_feeds: list[tuple[str, str, float]],
        funnels: dict[str, dict],
        positions: list[dict],
        portfolio_state: dict,
        tf_last_bar: dict[tuple[str, str], float],
        now_mono: float,
    ) -> HealthReport:
        """Assemble health report from component data."""
        alerts: list[HealthAlert] = []

        # Data flow section
        data_flow: dict[str, dict] = {}
        for (sym, tf), last_mono in tf_last_bar.items():
            key = f"{sym}/{tf}"
            age = now_mono - last_mono
            expected = self.TF_INTERVALS.get(tf, 900)
            status = "OK" if age <= expected * 2.0 else "STALE"
            data_flow[key] = {
                "last_bar_age_sec": round(age),
                "expected_interval_sec": expected,
                "status": status,
            }

        # Stale feed alerts
        for sym, tf, elapsed in stale_feeds:
            alerts.append(HealthAlert(
                severity="error",
                name="no_bars",
                message=f"{sym}/{tf} stale for {elapsed:.0f}s",
                context={"symbol": sym, "tf": tf, "elapsed_sec": round(elapsed)},
            ))

        # Signal funnel section + alerts
        signal_funnels: dict[str, dict] = {}
        gate_breakdown: dict[str, dict] = {}
        for sid, funnel_dict in funnels.items():
            if not funnel_dict:
                continue

            # Summarize across symbols
            total_bars = sum(funnel_dict.get("bars_received", {}).values())
            total_ind = sum(funnel_dict.get("indicators_ready", {}).values())
            total_setups = sum(funnel_dict.get("setups_detected", {}).values())
            total_confirms = sum(funnel_dict.get("confirmations", {}).values())
            total_entries = sum(funnel_dict.get("entries_attempted", {}).values())
            total_fills = sum(funnel_dict.get("fills", {}).values())

            signal_funnels[sid] = {
                "bars_received": total_bars,
                "indicators_ready": total_ind,
                "setups_detected": total_setups,
                "confirmations": total_confirms,
                "entries_attempted": total_entries,
                "fills": total_fills,
            }
            gate_breakdown[sid] = funnel_dict.get("gate_rejections", {})

            # Pipeline alerts
            if total_bars == 0:
                alerts.append(HealthAlert(
                    severity="error",
                    name="pipeline_stalled",
                    message=f"{sid}: no bars received",
                ))

        # Portfolio section
        portfolio_info = {
            "heat_R": portfolio_state.get("heat_R", 0),
            "heat_cap_R": portfolio_state.get("heat_cap_R", 0),
            "daily_pnl_R": portfolio_state.get("daily_pnl_R", 0),
            "open_risk_count": portfolio_state.get("open_risk_count", 0),
        }

        # System section
        system_info = {
            "total_errors": health_status.get("total_errors", 0),
            "consecutive_errors": health_status.get("consecutive_errors", 0),
            "stale_feed_count": len(stale_feeds),
        }

        # Error burst alert
        if health_status.get("consecutive_errors", 0) > 10:
            alerts.append(HealthAlert(
                severity="warning",
                name="error_burst",
                message=f"{health_status['consecutive_errors']} consecutive errors",
            ))

        # Positions
        position_dicts = positions

        # Assessment
        has_critical = any(a.severity == "critical" for a in alerts)
        has_error = any(a.severity == "error" for a in alerts)
        if has_critical:
            assessment = "critical"
        elif has_error:
            assessment = "degraded"
        else:
            assessment = "healthy"

        return HealthReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_sec=uptime_sec,
            data_flow=data_flow,
            signal_funnels=signal_funnels,
            gate_breakdown=gate_breakdown,
            positions=position_dicts,
            portfolio=portfolio_info,
            system=system_info,
            alerts=[a.to_dict() for a in alerts],
            assessment=assessment,
        )
