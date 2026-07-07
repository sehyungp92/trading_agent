"""Runtime scheduler construction."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def create_scheduler(jobs: list[dict]) -> object | None:
    """Create and start an AsyncIOScheduler from job definitions."""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.warning("APScheduler not installed; scheduler disabled")
        return None

    scheduler = AsyncIOScheduler()
    for job in jobs:
        job = dict(job)
        trigger_type = job.pop("trigger")
        func = job.pop("func")
        name = job.pop("name")
        misfire_grace_time = job.pop("misfire_grace_time", None)
        coalesce = job.pop("coalesce", None)

        if trigger_type == "interval":
            trigger = IntervalTrigger(seconds=job.pop("seconds"))
        elif trigger_type == "cron":
            trigger = CronTrigger(**job)
            job = {}
        else:
            logger.warning("Unknown trigger type %s for job %s", trigger_type, name)
            continue

        add_kwargs: dict = {}
        if misfire_grace_time is not None:
            add_kwargs["misfire_grace_time"] = misfire_grace_time
        if coalesce is not None:
            add_kwargs["coalesce"] = coalesce

        scheduler.add_job(func, trigger, id=name, name=name, **add_kwargs, **job)
    scheduler.start()
    return scheduler
