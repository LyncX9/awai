from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol

UTC = timezone.utc


JobAction = Callable[[], Any]


class SchedulerEventWriter(Protocol):
    def write(
        self,
        category: str,
        event_name: str,
        payload: dict[str, Any] | None = None,
        *,
        status: str = "completed",
        level: str = "INFO",
        occurred_at: datetime | None = None,
    ) -> object:
        ...


@dataclass
class JobRunResult:
    job_name: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float = 0.0
    result: Any = None
    error: str | None = None


@dataclass
class ScheduledJob:
    name: str
    interval_seconds: int
    action: JobAction
    next_run_at: datetime
    enabled: bool = True
    last_started_at: datetime | None = None
    last_finished_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_status: str = "pending"
    last_error: str | None = None
    run_count: int = 0
    failure_count: int = 0
    skipped_overlap_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def running(self) -> bool:
        acquired = self.lock.acquire(blocking=False)
        if acquired:
            self.lock.release()
            return False
        return True


class InProcessScheduler:
    """Small interval scheduler with per-job locks and optional background loop."""

    def __init__(
        self,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        event_logger: SchedulerEventWriter | None = None,
    ) -> None:
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self.sleep_fn = sleep_fn
        self.event_logger = event_logger
        self.jobs: dict[str, ScheduledJob] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._registry_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def add_interval_job(
        self,
        name: str,
        interval_seconds: int,
        action: JobAction,
        start_at: datetime | None = None,
        enabled: bool = True,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        now = self._normalize_now(start_at or self.now_fn())
        with self._registry_lock:
            self.jobs[name] = ScheduledJob(
                name=name,
                interval_seconds=interval_seconds,
                action=action,
                next_run_at=now,
                enabled=enabled,
            )

    def trigger(self, name: str) -> JobRunResult:
        job = self.jobs.get(name)
        if job is None:
            raise KeyError(f"Unknown job: {name}")
        return self._run_job(job)

    def run_due(self, now: datetime | None = None) -> list[JobRunResult]:
        current = self._normalize_now(now or self.now_fn())
        due_jobs = [
            job
            for job in list(self.jobs.values())
            if job.enabled and job.next_run_at <= current
        ]
        return [self._run_job(job, now=current) for job in due_jobs]

    def start(self, poll_seconds: float = 1.0) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(poll_seconds,),
            name="traffic-prediction-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
        self._thread = None

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "job_count": len(self.jobs),
            "jobs": {
                name: {
                    "enabled": job.enabled,
                    "running": job.running,
                    "interval_seconds": job.interval_seconds,
                    "next_run_at": job.next_run_at.isoformat(),
                    "last_started_at": job.last_started_at.isoformat() if job.last_started_at else None,
                    "last_finished_at": job.last_finished_at.isoformat() if job.last_finished_at else None,
                    "last_success_at": job.last_success_at.isoformat() if job.last_success_at else None,
                    "last_failure_at": job.last_failure_at.isoformat() if job.last_failure_at else None,
                    "last_status": job.last_status,
                    "last_error": job.last_error,
                    "run_count": job.run_count,
                    "failure_count": job.failure_count,
                    "skipped_overlap_count": job.skipped_overlap_count,
                }
                for name, job in sorted(self.jobs.items())
            },
        }

    def _run_loop(self, poll_seconds: float) -> None:
        while not self._stop_event.is_set():
            self.run_due()
            self.sleep_fn(poll_seconds)

    def _run_job(self, job: ScheduledJob, now: datetime | None = None) -> JobRunResult:
        started_at = self._normalize_now(now or self.now_fn())
        acquired = job.lock.acquire(blocking=False)
        if not acquired:
            job.skipped_overlap_count += 1
            job.last_status = "skipped_overlap"
            self._log_job_event(job, "skipped_overlap", started_at=started_at)
            return JobRunResult(job_name=job.name, status="skipped_overlap", started_at=started_at)

        monotonic_start = time.monotonic()
        job.last_started_at = started_at
        job.last_status = "running"
        try:
            result = job.action()
            finished_at = self._normalize_now(self.now_fn())
            duration = time.monotonic() - monotonic_start
            job.run_count += 1
            job.last_finished_at = finished_at
            job.last_success_at = finished_at
            job.last_status = "completed"
            job.last_error = None
            job.next_run_at = finished_at + timedelta(seconds=job.interval_seconds)
            self._log_job_event(
                job,
                "completed",
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                result=result,
            )
            return JobRunResult(
                job_name=job.name,
                status="completed",
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                result=result,
            )
        except Exception as exc:
            finished_at = self._normalize_now(self.now_fn())
            duration = time.monotonic() - monotonic_start
            job.failure_count += 1
            job.last_finished_at = finished_at
            job.last_failure_at = finished_at
            job.last_status = "failed"
            job.last_error = str(exc)
            job.next_run_at = finished_at + timedelta(seconds=job.interval_seconds)
            self._log_job_event(
                job,
                "failed",
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                error=str(exc),
                level="ERROR",
            )
            return JobRunResult(
                job_name=job.name,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
                error=str(exc),
            )
        finally:
            job.lock.release()

    @staticmethod
    def _normalize_now(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _log_job_event(
        self,
        job: ScheduledJob,
        status: str,
        *,
        started_at: datetime,
        finished_at: datetime | None = None,
        duration_seconds: float = 0.0,
        result: Any = None,
        error: str | None = None,
        level: str = "INFO",
    ) -> None:
        if self.event_logger is None:
            return
        payload = {
            "job_name": job.name,
            "enabled": job.enabled,
            "status": status,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat() if finished_at else None,
            "duration_seconds": round(duration_seconds, 6),
            "run_count": job.run_count,
            "failure_count": job.failure_count,
            "skipped_overlap_count": job.skipped_overlap_count,
            "error": error,
            "result_type": type(result).__name__ if result is not None else None,
        }
        self.event_logger.write(
            "scheduler",
            "job_run",
            payload,
            status=status,
            level=level,
            occurred_at=finished_at or started_at,
        )
