import time
from unittest.mock import MagicMock, patch

from traffic_prediction.monitoring.resources import ResourceMetrics, ResourceMonitor


def test_resource_monitor_initialization():
    monitor = ResourceMonitor()
    metrics = monitor.get_metrics()
    
    assert isinstance(metrics, ResourceMetrics)
    assert metrics.total_requests == 0
    assert metrics.error_rate == 0.0
    assert metrics.avg_prediction_latency_ms == 0.0
    assert metrics.process_memory_mb >= 0.0
    assert metrics.cpu_percent >= 0.0


def test_resource_monitor_records_requests():
    monitor = ResourceMonitor()
    
    # Record some successful requests
    monitor.record_request(latency_ms=10.0, error=False)
    monitor.record_request(latency_ms=20.0, error=False)
    
    metrics = monitor.get_metrics()
    assert metrics.total_requests == 2
    assert metrics.error_rate == 0.0
    assert metrics.avg_prediction_latency_ms == 15.0


def test_resource_monitor_records_errors():
    monitor = ResourceMonitor()
    
    # Record 4 requests, 1 is error
    monitor.record_request(latency_ms=10.0, error=False)
    monitor.record_request(latency_ms=10.0, error=True)
    monitor.record_request(latency_ms=10.0, error=False)
    monitor.record_request(latency_ms=10.0, error=False)
    
    metrics = monitor.get_metrics()
    assert metrics.total_requests == 4
    assert metrics.error_rate == 0.25
    assert metrics.avg_prediction_latency_ms == 10.0


@patch("traffic_prediction.monitoring.resources.psutil.Process")
def test_resource_monitor_reads_system_metrics(mock_process_cls):
    mock_process = MagicMock()
    # Setup mock returns
    mock_process.cpu_percent.return_value = 12.5
    mock_memory_info = MagicMock()
    mock_memory_info.rss = 104857600  # 100 MB
    mock_process.memory_info.return_value = mock_memory_info
    
    mock_process_cls.return_value = mock_process
    
    monitor = ResourceMonitor()
    metrics = monitor.get_metrics()
    
    assert metrics.process_memory_mb == 100.0
    assert metrics.cpu_percent == 12.5
