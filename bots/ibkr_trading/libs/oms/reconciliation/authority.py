"""Optional reconciliation authority leases."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass(frozen=True)
class ReconciliationAuthorityScope:
    broker: str
    account_id: str
    client_id: int
    family_id: str
    recon_kind: str


@dataclass(frozen=True)
class ReconciliationLease:
    scope: ReconciliationAuthorityScope
    owner_id: str
    acquired_at: datetime
    expires_at: datetime
    last_snapshot_id: str = ""


class ReconciliationAuthority:
    """DB-backed lease for one mutating reconciler per scope."""

    def __init__(self, pool: Any):
        self._pool = pool

    async def acquire(
        self,
        scope: ReconciliationAuthorityScope,
        owner_id: str,
        ttl_seconds: float,
        *,
        last_snapshot_id: str = "",
    ) -> ReconciliationLease | None:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)
        row = await self._pool.fetchrow(
            """
            INSERT INTO reconciliation_authority_leases (
                broker, account_id, client_id, family_id, recon_kind,
                owner_id, acquired_at, expires_at, last_snapshot_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (broker, account_id, client_id, family_id, recon_kind)
            DO UPDATE SET
                owner_id = EXCLUDED.owner_id,
                acquired_at = EXCLUDED.acquired_at,
                expires_at = EXCLUDED.expires_at,
                last_snapshot_id = EXCLUDED.last_snapshot_id
            WHERE reconciliation_authority_leases.expires_at < $7
               OR reconciliation_authority_leases.owner_id = EXCLUDED.owner_id
            RETURNING *
            """,
            scope.broker,
            scope.account_id,
            scope.client_id,
            scope.family_id,
            scope.recon_kind,
            owner_id,
            now,
            expires_at,
            last_snapshot_id,
        )
        if row is None:
            return None
        return ReconciliationLease(
            scope=scope,
            owner_id=row["owner_id"],
            acquired_at=row["acquired_at"],
            expires_at=row["expires_at"],
            last_snapshot_id=row["last_snapshot_id"] or "",
        )

    async def renew(
        self,
        lease: ReconciliationLease,
        ttl_seconds: float,
    ) -> ReconciliationLease | None:
        return await self.acquire(
            lease.scope,
            lease.owner_id,
            ttl_seconds,
            last_snapshot_id=lease.last_snapshot_id,
        )

    async def release(self, lease: ReconciliationLease) -> None:
        scope = lease.scope
        await self._pool.execute(
            """
            DELETE FROM reconciliation_authority_leases
            WHERE broker = $1
              AND account_id = $2
              AND client_id = $3
              AND family_id = $4
              AND recon_kind = $5
              AND owner_id = $6
            """,
            scope.broker,
            scope.account_id,
            scope.client_id,
            scope.family_id,
            scope.recon_kind,
            lease.owner_id,
        )

    @staticmethod
    def is_authoritative(lease: ReconciliationLease | None) -> bool:
        return bool(lease and lease.expires_at > datetime.now(timezone.utc))


class InMemoryReconciliationAuthority:
    """Test/dev fallback with lease expiry semantics."""

    def __init__(self) -> None:
        self._leases: dict[ReconciliationAuthorityScope, ReconciliationLease] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        scope: ReconciliationAuthorityScope,
        owner_id: str,
        ttl_seconds: float,
        *,
        last_snapshot_id: str = "",
    ) -> ReconciliationLease | None:
        now = datetime.now(timezone.utc)
        async with self._lock:
            current = self._leases.get(scope)
            if current and current.expires_at > now and current.owner_id != owner_id:
                return None
            lease = ReconciliationLease(
                scope=scope,
                owner_id=owner_id,
                acquired_at=now,
                expires_at=now + timedelta(seconds=ttl_seconds),
                last_snapshot_id=last_snapshot_id,
            )
            self._leases[scope] = lease
            return lease

    async def renew(
        self,
        lease: ReconciliationLease,
        ttl_seconds: float,
    ) -> ReconciliationLease | None:
        return await self.acquire(
            lease.scope,
            lease.owner_id,
            ttl_seconds,
            last_snapshot_id=lease.last_snapshot_id,
        )

    async def release(self, lease: ReconciliationLease) -> None:
        async with self._lock:
            current = self._leases.get(lease.scope)
            if current and current.owner_id == lease.owner_id:
                self._leases.pop(lease.scope, None)

    @staticmethod
    def is_authoritative(lease: ReconciliationLease | None) -> bool:
        return bool(lease and lease.expires_at > datetime.now(timezone.utc))
