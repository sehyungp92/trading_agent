"""Configuration loader with validation."""
import os
import yaml
from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel
from .schemas import IBKRProfile, ContractTemplate, ExchangeRoute

T = TypeVar("T", bound=BaseModel)


class IBKRConfig:
    """Immutable config loaded once at startup."""

    def __init__(self, config_dir: Path):
        self._config_dir = Path(config_dir)
        profile_path = self._config_dir / "ibkr_profiles.yaml"
        if profile_path.exists():
            self.profile: IBKRProfile = self._load("ibkr_profiles.yaml", IBKRProfile)
        else:
            # Fall back to env-only profile when yaml file is absent (e.g. Docker)
            self.profile = IBKRProfile(
                account_id=os.environ.get("IB_ACCOUNT_ID", ""),
                host=os.environ.get("IB_HOST", "127.0.0.1"),
                port=int(os.environ.get("IB_PORT", "4002")),
            )
        self.contracts: dict[str, ContractTemplate] = self._load_map(
            "contracts.yaml", ContractTemplate
        )
        self.routes: dict[str, ExchangeRoute] = self._load_map(
            "routing.yaml", ExchangeRoute
        )
        # Allow env var override for Docker connectivity / account
        ib_host = os.environ.get("IB_HOST")
        ib_port = os.environ.get("IB_PORT")
        ib_account = os.environ.get("IB_ACCOUNT_ID")
        if ib_host or ib_port or ib_account:
            overrides = {}
            if ib_host:
                overrides["host"] = ib_host
            if ib_port:
                overrides["port"] = int(ib_port)
            if ib_account:
                overrides["account_id"] = ib_account
            self.profile = self.profile.model_copy(update=overrides)
        if not self.profile.account_id:
            raise ValueError(
                "IB_ACCOUNT_ID env var is required (not set in env or ibkr_profiles.yaml)"
            )

    def _load(self, filename: str, model_cls: type[T]) -> T:
        path = self._config_dir / filename
        with open(path) as f:
            data = yaml.safe_load(f)
        return model_cls(**data)

    def _load_map(self, filename: str, model_cls: type[T]) -> dict[str, T]:
        path = self._config_dir / filename
        with open(path) as f:
            data = yaml.safe_load(f)
        return {k: model_cls(**v) for k, v in data.items()}
