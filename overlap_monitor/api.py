from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Union

from overlap_monitor.analyzer.critical_path import (
    CriticalPathOverlapAnalyzer,
    CriticalPathSummary,
)
from overlap_monitor.analyzer.overlap import OverlapAnalyzer
from overlap_monitor.core.events import Event
from overlap_monitor.core.io import JsonlFormatError, read_events_jsonl
from overlap_monitor.core.metrics import OverlapSummary
from overlap_monitor.core.validation import ValidationReport, validate_events
from overlap_monitor.profiler.cupti import CuptiActivityParser

AnalysisMode = Literal["critical-path", "timeline"]
InputFormat = Literal["auto", "events", "cupti"]
Summary = Union[CriticalPathSummary, OverlapSummary]

__all__ = [
    "AnalysisMode",
    "AnalysisResult",
    "InputFormat",
    "TraceValidationError",
    "analyze_events",
    "analyze_trace",
    "load_trace",
]


class TraceValidationError(ValueError):
    """Raised when a trace cannot be analyzed without violating invariants."""

    def __init__(self, report: ValidationReport):
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues) or "unknown"
        super().__init__(f"trace validation failed: {codes}")


@dataclass(frozen=True)
class AnalysisResult:
    """Stable result returned by the high-level analysis API."""

    events: list[Event]
    summary: Summary
    validation: ValidationReport
    analysis_mode: AnalysisMode
    input_format: str = "events"
    timestamp_unit: str = "us"
    clock_alignment_assumed: bool = False
    source_warnings: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = 1

    @property
    def overlap_ratio(self) -> float:
        """Compatibility ratio for the selected analysis mode."""
        return self.summary.overlap_ratio

    @property
    def timeline_overlap_ratio(self) -> float | None:
        if isinstance(self.summary, OverlapSummary):
            return self.summary.timeline_overlap_ratio
        return None

    @property
    def communication_hidden_ratio(self) -> float | None:
        if isinstance(self.summary, CriticalPathSummary):
            return self.summary.communication_hidden_ratio
        return None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "analysis_mode": self.analysis_mode,
            "input_format": self.input_format,
            "timestamp_unit": self.timestamp_unit,
            "clock_alignment_assumed": self.clock_alignment_assumed,
            "source_warnings": list(self.source_warnings),
            "validation": self.validation.to_dict(),
            **self.summary.to_dict(),
        }


@dataclass(frozen=True)
class _LoadedTrace:
    events: list[Event]
    input_format: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def load_trace(
    path: Path | str,
    *,
    input_format: InputFormat = "auto",
    rank: int | None = None,
    device_id: int | None = None,
    stage_id: int | None = None,
    allow_incomplete: bool = False,
) -> list[Event]:
    """Load, normalize, and filter a CUPTI or Event JSONL trace."""

    return _load_trace(
        path,
        input_format=input_format,
        rank=rank,
        device_id=device_id,
        stage_id=stage_id,
        allow_incomplete=allow_incomplete,
    ).events


def analyze_events(
    events: Iterable[Event],
    *,
    mode: AnalysisMode = "critical-path",
    assume_aligned_clocks: bool = False,
    input_format: str = "events",
    source_warnings: Iterable[str] = (),
) -> AnalysisResult:
    """Validate and analyze already normalized events."""

    normalized_events = list(events)
    report = validate_events(
        normalized_events,
        strict_clock_domain=not assume_aligned_clocks,
    )
    if not report.valid:
        raise TraceValidationError(report)
    if mode == "critical-path":
        summary: Summary = CriticalPathOverlapAnalyzer().analyze(normalized_events)
    elif mode == "timeline":
        summary = OverlapAnalyzer().analyze(normalized_events)
    else:
        raise ValueError(f"unsupported analysis mode: {mode}")
    return AnalysisResult(
        events=normalized_events,
        summary=summary,
        validation=report,
        analysis_mode=mode,
        input_format=input_format,
        clock_alignment_assumed=assume_aligned_clocks,
        source_warnings=tuple(source_warnings),
    )


def analyze_trace(
    path: Path | str,
    *,
    mode: AnalysisMode = "critical-path",
    input_format: InputFormat = "auto",
    rank: int | None = None,
    device_id: int | None = None,
    stage_id: int | None = None,
    allow_incomplete: bool = False,
    assume_aligned_clocks: bool = False,
) -> AnalysisResult:
    """Load and analyze a trace through the stable high-level API."""

    loaded = _load_trace(
        path,
        input_format=input_format,
        rank=rank,
        device_id=device_id,
        stage_id=stage_id,
        allow_incomplete=allow_incomplete,
    )
    return analyze_events(
        loaded.events,
        mode=mode,
        assume_aligned_clocks=assume_aligned_clocks,
        input_format=loaded.input_format,
        source_warnings=loaded.warnings,
    )


def _load_trace(
    path: Path | str,
    *,
    input_format: InputFormat,
    rank: int | None,
    device_id: int | None,
    stage_id: int | None,
    allow_incomplete: bool,
) -> _LoadedTrace:
    trace_path = Path(path)
    resolved_format = (
        _detect_input_format(trace_path) if input_format == "auto" else input_format
    )
    if resolved_format not in {"events", "cupti"}:
        raise ValueError(f"unsupported input format: {resolved_format}")

    warnings: tuple[str, ...] = ()
    if resolved_format == "cupti":
        parsed = CuptiActivityParser().parse_file(
            trace_path,
            default_rank=rank,
            default_stage_id=stage_id,
            strict=not allow_incomplete,
        )
        events = parsed.events
        warnings = tuple(parsed.warnings)
    else:
        events = read_events_jsonl(trace_path)

    if rank is not None:
        events = [event for event in events if event.rank == rank]
    if device_id is not None:
        events = [event for event in events if event.device_id == device_id]
    if stage_id is not None:
        events = [event for event in events if event.stage_id == stage_id]
    return _LoadedTrace(events, resolved_format, warnings)


def _detect_input_format(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlFormatError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise JsonlFormatError(
                    f"invalid JSONL record at {path}:{line_number}: expected object"
                )
            return "cupti" if "record_kind" in payload else "events"
    return "events"
