from datetime import datetime, timedelta, timezone

UTC = timezone.utc

from traffic_prediction.orchestration.scheduler import InProcessScheduler


def test_scheduler_runs_due_jobs_and_updates_status() -> None:
    now = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)
    calls: list[str] = []
    scheduler = InProcessScheduler(now_fn=lambda: now)
    scheduler.add_interval_job(
        name="tomtom_ingestion",
        interval_seconds=900,
        action=lambda: calls.append("run") or {"ok": True},
        start_at=now,
    )

    results = scheduler.run_due(now=now)
    status = scheduler.status()

    assert len(results) == 1
    assert results[0].status == "completed"
    assert results[0].result == {"ok": True}
    assert calls == ["run"]
    assert status["jobs"]["tomtom_ingestion"]["last_status"] == "completed"
    assert status["jobs"]["tomtom_ingestion"]["run_count"] == 1
    assert status["jobs"]["tomtom_ingestion"]["next_run_at"] == (now + timedelta(seconds=900)).isoformat()


def test_scheduler_prevents_overlapping_manual_trigger() -> None:
    now = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)
    scheduler = InProcessScheduler(now_fn=lambda: now)
    scheduler.add_interval_job(
        name="slow_job",
        interval_seconds=60,
        action=lambda: "done",
        start_at=now,
    )
    job = scheduler.jobs["slow_job"]
    job.lock.acquire()
    try:
        result = scheduler.trigger("slow_job")
    finally:
        job.lock.release()

    assert result.status == "skipped_overlap"
    assert scheduler.status()["jobs"]["slow_job"]["skipped_overlap_count"] == 1


def test_scheduler_records_failed_job_without_raising() -> None:
    now = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)

    def fail() -> None:
        raise RuntimeError("boom")

    scheduler = InProcessScheduler(now_fn=lambda: now)
    scheduler.add_interval_job("bad_job", 60, fail, start_at=now)

    result = scheduler.trigger("bad_job")
    status = scheduler.status()["jobs"]["bad_job"]

    assert result.status == "failed"
    assert result.error == "boom"
    assert status["failure_count"] == 1
    assert status["last_status"] == "failed"
    assert status["last_error"] == "boom"


def test_scheduler_ignores_disabled_jobs_when_running_due() -> None:
    now = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)
    calls: list[str] = []
    scheduler = InProcessScheduler(now_fn=lambda: now)
    scheduler.add_interval_job(
        name="disabled_job",
        interval_seconds=60,
        action=lambda: calls.append("run"),
        start_at=now,
        enabled=False,
    )

    results = scheduler.run_due(now=now)

    assert results == []
    assert calls == []
