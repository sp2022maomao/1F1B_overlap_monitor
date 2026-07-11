"""Decoupled 1F1B communication-computation overlap monitoring framework."""

__version__ = "0.2.1"

from overlap_monitor_v2.analyzer.critical_path import (
    CriticalPathOverlapAnalyzer,
    CriticalPathSummary,
)
from overlap_monitor_v2.analyzer.overlap import OverlapAnalyzer
from overlap_monitor_v2.adapters.megatron import MegatronWorkAdapter
from overlap_monitor_v2.core.events import Event, EventType
from overlap_monitor_v2.core.session import MonitoringSession
from overlap_monitor_v2.core.validation import ValidationReport, validate_events
from overlap_monitor_v2.profiler.work_handle import WorkHandleRecorder

__all__ = [
    "CriticalPathOverlapAnalyzer",
    "CriticalPathSummary",
    "Event",
    "EventType",
    "MonitoringSession",
    "MegatronWorkAdapter",
    "OverlapAnalyzer",
    "WorkHandleRecorder",
    "ValidationReport",
    "validate_events",
    "__version__",
]
