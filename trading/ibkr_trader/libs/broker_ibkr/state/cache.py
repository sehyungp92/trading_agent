"""In-memory caches for contracts and order ID mappings."""
from dataclasses import dataclass, field
from ..models.types import IBContractSpec


@dataclass
class IBCache:
    """In-memory caches for contracts and order ID mappings."""

    # conId -> IBContractSpec
    contracts: dict[int, IBContractSpec] = field(default_factory=dict)

    # oms_order_id -> (broker_order_id, perm_id)
    order_map: dict[str, tuple[int, int]] = field(default_factory=dict)

    # broker_order_id -> oms_order_id (reverse lookup)
    reverse_order_map: dict[int, str] = field(default_factory=dict)

    # exec_id set for fill deduplication
    seen_exec_ids: set[str] = field(default_factory=set)

    # oms_order_id set for ack deduplication (PreSubmitted vs Submitted)
    acked_oms_ids: set[str] = field(default_factory=set)

    def register_order(self, oms_order_id: str, broker_order_id: int, perm_id: int) -> None:
        self.order_map[oms_order_id] = (broker_order_id, perm_id)
        self.reverse_order_map[broker_order_id] = oms_order_id

    def lookup_oms_id(self, broker_order_id: int) -> str | None:
        return self.reverse_order_map.get(broker_order_id)

    def lookup_broker_id(self, oms_order_id: str) -> tuple[int, int] | None:
        return self.order_map.get(oms_order_id)

    def is_fill_seen(self, exec_id: str) -> bool:
        return exec_id in self.seen_exec_ids

    def mark_fill_seen(self, exec_id: str) -> None:
        self.seen_exec_ids.add(exec_id)

    def is_acked(self, oms_order_id: str) -> bool:
        return oms_order_id in self.acked_oms_ids

    def mark_acked(self, oms_order_id: str) -> None:
        self.acked_oms_ids.add(oms_order_id)

    def clear(self) -> None:
        self.contracts.clear()
        self.order_map.clear()
        self.reverse_order_map.clear()
        self.seen_exec_ids.clear()
        self.acked_oms_ids.clear()

    async def rebuild_from_broker(
        self,
        snapshot_fetcher,
        oms_order_id_resolver=None,
        fill_exists_check=None,
        fill_importer=None,
    ) -> None:
        """H6 fix: Rebuild order mappings from broker open orders and recent executions.

        Called on startup to restore order ID mappings lost after a restart.

        Args:
            snapshot_fetcher: SnapshotFetcher instance to query broker state
            oms_order_id_resolver: Optional callable(broker_order_id: int) -> Optional[str]
                that resolves a broker order ID to an OMS order ID from the DB
            fill_exists_check: Optional async callable(exec_id: str) -> bool that
                checks whether a broker execution is already persisted in the
                fills table. Required for OMS-3 (offline execution import).
            fill_importer: Optional async callable(oms_order_id: str, exec_report)
                -> bool that imports a missing broker execution into OMS state.
                Returns True iff the execution was newly persisted. The cache
                only marks the exec_id `seen` after a successful import or a
                positive `fill_exists_check`.

        OMS-3 contract: an execution is marked seen ONLY after we confirm it is
        in OMS state (already persisted OR successfully imported). The previous
        behaviour marked every fetched execution seen unconditionally, which
        caused IBKR to re-deliver `execDetailsEvent` for fills that occurred
        while the runtime was down, and the adapter dropped them as duplicates,
        permanently losing the fill from OMS accounting.
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Rebuild from open orders
            open_orders = await snapshot_fetcher.fetch_open_orders()
            for order_event in open_orders:
                broker_id = order_event.broker_order_id
                perm_id = order_event.perm_id
                # If we already know this order, skip
                if self.lookup_oms_id(broker_id) is not None:
                    continue
                # Try to resolve via OMS DB
                if oms_order_id_resolver:
                    oms_id = await oms_order_id_resolver(broker_id)
                    if oms_id:
                        self.register_order(oms_id, broker_id, perm_id)
                        logger.info(f"Cache rebuilt: {oms_id} -> broker={broker_id}")

            # Rebuild from recent executions to deduplicate fills, importing
            # any executions that the OMS has not yet seen.
            executions = await snapshot_fetcher.fetch_executions()
            imported_count = 0
            for exec_report in executions:
                broker_id = exec_report.broker_order_id

                # First ensure the order mapping exists, so a later
                # execDetailsEvent for the same order can route correctly
                # (independent of the import path below).
                if self.lookup_oms_id(broker_id) is None and oms_order_id_resolver:
                    oms_id = await oms_order_id_resolver(broker_id)
                    if oms_id:
                        self.register_order(oms_id, broker_id, exec_report.perm_id)

                # Decide whether to mark this execution seen.
                already_in_oms: bool
                if fill_exists_check is not None:
                    already_in_oms = await fill_exists_check(exec_report.exec_id)
                else:
                    # Backward compat: callers that don't pass the check fall
                    # back to the old unconditional-mark behaviour, since the
                    # alternative (skipping the mark entirely) would re-process
                    # every recent execution on every reconnect.
                    already_in_oms = True

                if already_in_oms:
                    self.mark_fill_seen(exec_report.exec_id)
                    continue

                # Execution is missing locally; import through the OMS fill
                # pipeline before suppressing future broker replays.
                if fill_importer is None:
                    logger.warning(
                        "Broker execution %s missing in OMS but no importer "
                        "configured; NOT marking seen so the next "
                        "execDetailsEvent can carry it through",
                        exec_report.exec_id,
                    )
                    continue

                oms_id = self.lookup_oms_id(broker_id)
                if not oms_id:
                    logger.warning(
                        "Cannot import broker exec %s: no OMS order mapping "
                        "for broker_order_id=%s; leaving unmarked",
                        exec_report.exec_id, broker_id,
                    )
                    continue

                try:
                    imported = await fill_importer(oms_id, exec_report)
                except Exception as imp_exc:
                    logger.error(
                        "Failed to import broker exec %s for oms_order_id=%s: %s",
                        exec_report.exec_id, oms_id, imp_exc,
                    )
                    continue

                if imported:
                    self.mark_fill_seen(exec_report.exec_id)
                    imported_count += 1
                else:
                    confirmed = False
                    if fill_exists_check is not None:
                        try:
                            confirmed = await fill_exists_check(exec_report.exec_id)
                        except Exception as check_exc:
                            logger.error(
                                "Could not confirm broker exec %s after importer returned False: %s",
                                exec_report.exec_id, check_exc,
                            )
                    if confirmed:
                        # Importer lost a duplicate-insert race; the fill is
                        # now present in OMS, so it is safe to suppress replay.
                        self.mark_fill_seen(exec_report.exec_id)
                    else:
                        logger.error(
                            "Broker exec %s was not imported and is not present in OMS; "
                            "leaving unmarked for fail-closed replay",
                            exec_report.exec_id,
                        )

            logger.info(
                f"IBCache rebuilt: {len(self.order_map)} order mappings, "
                f"{len(self.seen_exec_ids)} seen exec IDs, "
                f"{imported_count} fills imported from broker"
            )
        except Exception as e:
            logger.error(f"Failed to rebuild IBCache from broker: {e}")
