"""Source-owned data authority contracts for live-refresh adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FORBIDDEN_ORDER_METHODS = frozenset(
    {
        "buy",
        "sell",
        "cancel",
        "revise",
        "place_order",
        "submit_order",
        "modify_order",
        "close_position",
        "liquidate",
    }
)


@dataclass(frozen=True)
class SourceAuthorityContract:
    contract_id: str
    source: str
    market: str
    adapter_id: str
    credential_contract_id: str
    read_only: bool
    supports_live_trading: bool
    credential_fields: tuple[str, ...]
    required_lineage_fields: tuple[str, ...]
    pacing_policy: str
    raw_write_contract: str
    canonical_write_contract: str
    idempotency_contract: str
    checksum_contract: str
    session_policy: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        for field_name in (
            "contract_id",
            "source",
            "market",
            "adapter_id",
            "credential_contract_id",
            "pacing_policy",
            "raw_write_contract",
            "canonical_write_contract",
            "idempotency_contract",
            "checksum_contract",
            "session_policy",
        ):
            if not str(getattr(self, field_name, "") or "").strip():
                errors.append(f"{field_name} missing")
        if not self.read_only:
            errors.append("contract must be read-only")
        if self.supports_live_trading:
            errors.append("authority contract must not expose live trading")
        if not self.credential_fields:
            errors.append("credential_fields missing")
        if not self.required_lineage_fields:
            errors.append("required_lineage_fields missing")
        return errors

    def lineage_errors(self, lineage: dict[str, Any]) -> list[str]:
        return [
            f"lineage.{field_name} missing"
            for field_name in self.required_lineage_fields
            if not str(lineage.get(field_name, "") or "").strip()
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "source": self.source,
            "market": self.market,
            "adapter_id": self.adapter_id,
            "credential_contract_id": self.credential_contract_id,
            "read_only": self.read_only,
            "supports_live_trading": self.supports_live_trading,
            "credential_fields": list(self.credential_fields),
            "required_lineage_fields": list(self.required_lineage_fields),
            "pacing_policy": self.pacing_policy,
            "raw_write_contract": self.raw_write_contract,
            "canonical_write_contract": self.canonical_write_contract,
            "idempotency_contract": self.idempotency_contract,
            "checksum_contract": self.checksum_contract,
            "session_policy": self.session_policy,
            "notes": list(self.notes),
            "validation_errors": self.validation_errors(),
        }


def ibkr_cme_nq_authority_contract() -> SourceAuthorityContract:
    return SourceAuthorityContract(
        contract_id="ibkr_cme_nq_read_only_authority_v1",
        source="ibkr",
        market="cme_futures",
        adapter_id="ibkr_cme_nq_read_only_adapter_v1",
        credential_contract_id="ibkr_read_only_market_data_credentials_v1",
        read_only=True,
        supports_live_trading=False,
        credential_fields=(
            "IBKR_HOST",
            "IBKR_PORT",
            "IBKR_CLIENT_ID",
            "IBKR_READ_ONLY_ACK",
        ),
        required_lineage_fields=(
            "source_endpoint",
            "export_id",
            "pulled_at_utc",
            "config_hash",
            "session_policy",
            "market_rule_authority_checksum",
            "roll_policy",
            "contract_chain_checksum",
            "continuous_construction_checksum",
            "source_contract_coverage",
            "source_conid_coverage",
            "contract_resolution_cache",
            "credential_contract_id",
            "adapter_id",
            "pacing_policy",
            "raw_write_checksum",
            "canonical_write_checksum",
            "idempotency_key",
        ),
        pacing_policy="ibkr_historical_bars_pacing_v1",
        raw_write_contract="raw_ibkr_historical_bars_parquet_v1",
        canonical_write_contract="canonical_ohlcv_parquet_v1",
        idempotency_contract="source_symbol_timeframe_window_export_id_v1",
        checksum_contract="sha256_parquet_content_plus_schema_v1",
        session_policy="cme_equity_index_futures_v1",
        notes=(
            "Historical market-data refresh only.",
            "No order, account, position, or live-trading methods are permitted.",
        ),
    )


def ibkr_us_equity_authority_contract() -> SourceAuthorityContract:
    return SourceAuthorityContract(
        contract_id="ibkr_us_equity_read_only_authority_v1",
        source="ibkr",
        market="us_equity",
        adapter_id="ibkr_us_equity_read_only_adapter_v1",
        credential_contract_id="ibkr_read_only_market_data_credentials_v1",
        read_only=True,
        supports_live_trading=False,
        credential_fields=(
            "IBKR_HOST",
            "IBKR_PORT",
            "IBKR_CLIENT_ID",
            "IBKR_READ_ONLY_ACK",
        ),
        required_lineage_fields=(
            "source_endpoint",
            "export_id",
            "pulled_at_utc",
            "config_hash",
            "corporate_action_policy",
            "raw_adjustment_policy",
            "session_policy",
            "source_conid_coverage",
            "contract_resolution_cache",
            "source_request_params_hash",
            "returned_row_count",
            "credential_contract_id",
            "adapter_id",
            "pacing_policy",
            "raw_write_checksum",
            "canonical_write_checksum",
            "idempotency_key",
        ),
        pacing_policy="ibkr_us_equity_historical_bars_pacing_v1",
        raw_write_contract="raw_ibkr_us_equity_historical_bars_parquet_v1",
        canonical_write_contract="canonical_us_equity_ohlcv_parquet_v1",
        idempotency_contract="source_symbol_timeframe_window_export_id_v1",
        checksum_contract="sha256_parquet_content_plus_schema_v1",
        session_policy="us_equities_xnys_xnas_v1",
        notes=(
            "Historical market-data refresh only.",
            "No order, account, position, or live-trading methods are permitted.",
            "Corporate-action and raw-adjustment policy must be explicit in lineage.",
        ),
    )


def kis_krx_authority_contract() -> SourceAuthorityContract:
    return SourceAuthorityContract(
        contract_id="kis_krx_read_only_authority_v1",
        source="kis",
        market="krx_equity",
        adapter_id="kis_krx_read_only_adapter_v1",
        credential_contract_id="kis_read_only_market_data_credentials_v1",
        read_only=True,
        supports_live_trading=False,
        credential_fields=(
            "KIS_APP_KEY",
            "KIS_APP_SECRET",
            "KIS_ACCOUNT_MODE",
            "KIS_READ_ONLY_ACK",
        ),
        required_lineage_fields=(
            "source_endpoint",
            "export_id",
            "pulled_at_utc",
            "config_hash",
            "session_policy",
            "credential_contract_id",
            "adapter_id",
            "pacing_policy",
            "raw_write_checksum",
            "canonical_write_checksum",
            "idempotency_key",
        ),
        pacing_policy="kis_market_data_pacing_v1",
        raw_write_contract="raw_kis_intraday_parquet_v1",
        canonical_write_contract="canonical_krx_intraday_parquet_v1",
        idempotency_contract="source_symbol_timeframe_window_export_id_v1",
        checksum_contract="sha256_parquet_content_plus_schema_v1",
        session_policy="kis_intraday_exchange_timestamp_v1",
        notes=(
            "Market-data refresh only.",
            "Timestamp/session audit must pass before non-dry-run sync is enabled.",
        ),
    )


def order_surface_errors(adapter: object) -> list[str]:
    names = {
        name
        for name in dir(adapter)
        if callable(getattr(adapter, name, None)) and not name.startswith("_")
    }
    forbidden = sorted(names.intersection(FORBIDDEN_ORDER_METHODS))
    return [f"forbidden order method exposed: {name}" for name in forbidden]
