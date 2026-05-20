import os
import threading
import time
from dataclasses import dataclass
import psutil

@dataclass
class ResourceMetrics:
    process_memory_mb: float
    cpu_percent: float
    total_requests: int
    error_rate: float
    avg_prediction_latency_ms: float


class ResourceMonitor:
    """
    Lightweight, thread-safe resource and application metric monitor.
    Tracks memory, CPU, prediction latencies, and error rates.
    """

    def __init__(self) -> None:
        self.total_requests = 0
        self.error_count = 0
        self.total_latency_ms = 0.0

        self._process = psutil.Process(os.getpid())
        # Warm up the CPU percent calculation
        self._process.cpu_percent()

        self._lock = threading.Lock()

    def record_request(self, latency_ms: float, error: bool = False) -> None:
        """Record a single prediction request to update running latency and error metrics."""
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            if error:
                self.error_count += 1

    def get_metrics(self) -> ResourceMetrics:
        """Calculate and return point-in-time metrics."""
        with self._lock:
            reqs = self.total_requests
            errs = self.error_count
            latency = self.total_latency_ms

        error_rate = (errs / reqs) if reqs > 0 else 0.0
        avg_latency = (latency / reqs) if reqs > 0 else 0.0

        # Memory in MB
        mem_info = self._process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024)

        # CPU percent (since last call)
        cpu_pct = self._process.cpu_percent()

        return ResourceMetrics(
            process_memory_mb=round(mem_mb, 2),
            cpu_percent=round(cpu_pct, 2),
            total_requests=reqs,
            error_rate=round(error_rate, 4),
            avg_prediction_latency_ms=round(avg_latency, 2),
        )
