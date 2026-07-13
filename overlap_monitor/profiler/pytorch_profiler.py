from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from overlap_monitor.analyzer.classifier import KernelClassifier
from overlap_monitor.core.events import Event, EventType


class PyTorchProfilerEventParser:
    """Adapter from PyTorch profiler events to normalized Event objects.

    This module does not import torch. It only consumes objects with profiler-like
    attributes, so synthetic tests can exercise it without CUDA.
    """

    def __init__(self, classifier: KernelClassifier | None = None):
        self.classifier = classifier or KernelClassifier()

    def event_name(self, event: Any) -> str:
        return str(getattr(event, "name", "") or "")

    def event_range(self, event: Any) -> tuple[float, float] | None:
        time_range = getattr(event, "time_range", None)
        if time_range is None:
            return None
        start = getattr(time_range, "start", None)
        end = getattr(time_range, "end", None)
        if start is None or end is None:
            return None
        start = float(start)
        end = float(end)
        if end <= start:
            return None
        return start, end

    def device_id(self, event: Any) -> int | None:
        value = getattr(event, "device_index", None)
        return int(value) if value is not None else None

    def parse(
        self,
        profiler_events: Iterable[Any],
        *,
        rank: int | None = None,
        stage_id: int | None = None,
    ) -> list[Event]:
        parsed = []
        for event in profiler_events:
            interval = self.event_range(event)
            if interval is None:
                continue
            parsed.append(
                self.classifier.classify_event(
                    Event(
                        timestamp_start=interval[0],
                        timestamp_end=interval[1],
                        event_type=EventType.UNKNOWN,
                        name=self.event_name(event),
                        device_id=self.device_id(event),
                        rank=rank,
                        stage_id=stage_id,
                        metadata={
                            "collector": "pytorch_profiler",
                            "timestamp_unit": "us",
                        },
                    )
                )
            )
        return parsed
