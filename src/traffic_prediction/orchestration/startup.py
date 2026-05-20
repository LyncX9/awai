from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StartupResourceStatus:
    name: str
    ready: bool
    status: str
    detail: str
    critical: bool = True

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class StartupReport:
    resources: tuple[StartupResourceStatus, ...]

    @property
    def ready(self) -> bool:
        return all(resource.ready for resource in self.resources if resource.critical)

    def by_name(self) -> dict[str, StartupResourceStatus]:
        return {resource.name: resource for resource in self.resources}

    def details(self) -> dict[str, str]:
        return {resource.name: resource.detail for resource in self.resources}

    def statuses(self) -> dict[str, dict[str, str | bool]]:
        return {resource.name: resource.to_dict() for resource in self.resources}


def build_startup_report(
    *,
    roads_loaded: bool,
    model_loaded: bool,
    buffer_available: bool,
    tomtom_configured: bool,
    scheduler_registered: bool,
    scheduler_running: bool,
    scheduler_enabled: bool,
    model_version: str | None = None,
    buffer_restore_error: str | None = None,
    recovery_status: str = "completed",
    recovery_detail: str = "restart recovery checks completed",
) -> StartupReport:
    resources = [
        StartupResourceStatus(
            name="roads",
            ready=roads_loaded,
            status="loaded" if roads_loaded else "missing",
            detail="roads master loaded" if roads_loaded else "roads master unavailable",
        ),
        StartupResourceStatus(
            name="model",
            ready=model_loaded,
            status="loaded" if model_loaded else "missing",
            detail=f"active model loaded: {model_version}" if model_loaded else "model artifact not loaded",
        ),
        StartupResourceStatus(
            name="buffer",
            ready=buffer_available,
            status="available" if buffer_available else "empty",
            detail=_buffer_detail(buffer_available, buffer_restore_error),
        ),
        StartupResourceStatus(
            name="tomtom",
            ready=tomtom_configured,
            status="configured" if tomtom_configured else "missing_credentials",
            detail=(
                "TomTom credentials configured"
                if tomtom_configured
                else "TomTom credentials unavailable; live ingestion can still fail gracefully"
            ),
            critical=False,
        ),
        StartupResourceStatus(
            name="scheduler",
            ready=scheduler_registered and (scheduler_running or not scheduler_enabled),
            status=_scheduler_status(scheduler_registered, scheduler_running, scheduler_enabled),
            detail=_scheduler_detail(scheduler_registered, scheduler_running, scheduler_enabled),
            critical=False,
        ),
        StartupResourceStatus(
            name="restart_recovery",
            ready=recovery_status in {"completed", "recovered", "degraded"},
            status=recovery_status,
            detail=recovery_detail,
            critical=False,
        ),
    ]
    return StartupReport(tuple(resources))


def _buffer_detail(buffer_available: bool, restore_error: str | None) -> str:
    if restore_error:
        return f"buffer restore failed; fallback startup seeding attempted: {restore_error}"
    if buffer_available:
        return "live buffer available"
    return "live buffer unavailable"


def _scheduler_status(registered: bool, running: bool, enabled: bool) -> str:
    if not registered:
        return "not_registered"
    if running:
        return "running"
    if enabled:
        return "stopped"
    return "disabled"


def _scheduler_detail(registered: bool, running: bool, enabled: bool) -> str:
    if not registered:
        return "scheduler jobs are not registered"
    if running:
        return "scheduler loop is running"
    if enabled:
        return "scheduler enabled but not running"
    return "scheduler registered for manual triggers; automatic loop disabled by config"
