"""Decoupled 1F1B communication-computation overlap monitoring framework."""

__version__ = "0.3.0"

from overlap_monitor.analyzer.critical_path import (
    CriticalPathOverlapAnalyzer,
    CriticalPathSummary,
)
from overlap_monitor.analyzer.overlap import OverlapAnalyzer
from overlap_monitor.adapters.cupti_runtime import CuptiRuntimeCollector
from overlap_monitor.adapters.megatron import MegatronWorkAdapter
from overlap_monitor.core.events import Event, EventType
from overlap_monitor.core.session import MonitoringSession
from overlap_monitor.core.validation import ValidationReport, validate_events
from overlap_monitor.profiler.work_handle import WorkHandleRecorder
from overlap_monitor.profiler.cupti import CuptiActivityParser, CuptiParseResult

__all__ = [
    "CriticalPathOverlapAnalyzer",
    "CriticalPathSummary",
    "CuptiActivityParser",
    "CuptiParseResult",
    "CuptiRuntimeCollector",
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
