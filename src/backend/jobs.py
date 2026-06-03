from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Any, Callable
from uuid import uuid4

_MAX_JOBS = 200
_JOB_TTL_SECONDS = 3600  # 1 hour


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class BridgeJob:
    job_id: str
    kind: str
    experiment_id: str
    status: str
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _created_dt: datetime = field(default_factory=_utc_now_dt, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_created_dt", None)
        return data


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, BridgeJob] = {}
        self._lock = Lock()

    def start(self, kind: str, experiment_id: str, target: Callable[[], dict[str, Any]]) -> BridgeJob:
        now = _utc_now()
        job = BridgeJob(
            job_id=f"job_{uuid4().hex}",
            kind=kind,
            experiment_id=experiment_id,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._evict_stale()

        def runner() -> None:
            self._update(job.job_id, status="running")
            try:
                result = target()
                self._update(job.job_id, status="completed", result=result)
            except Exception as exc:  # pragma: no cover
                self._update(job.job_id, status="failed", error=str(exc))

        Thread(target=runner, daemon=True).start()
        return job

    def get(self, job_id: str) -> BridgeJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"job not found: {job_id}")
            return self._jobs[job_id]

    def _update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.updated_at = _utc_now()

    def _evict_stale(self) -> None:
        """Remove completed/failed jobs older than TTL or when exceeding max count.

        Must be called while self._lock is held.
        """
        now = _utc_now_dt()
        stale_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in {"completed", "failed"}
            and (now - job._created_dt).total_seconds() > _JOB_TTL_SECONDS
        ]
        for job_id in stale_ids:
            del self._jobs[job_id]

        if len(self._jobs) > _MAX_JOBS:
            finished = sorted(
                (
                    (job_id, job)
                    for job_id, job in self._jobs.items()
                    if job.status in {"completed", "failed"}
                ),
                key=lambda item: item[1]._created_dt,
            )
            excess = len(self._jobs) - _MAX_JOBS
            for job_id, _ in finished[:excess]:
                del self._jobs[job_id]
