from __future__ import annotations

from typing import Any

from overlap_monitor.analyzer import KernelClassifier
from overlap_monitor.core.events import Event, EventType


class TimelineNormalizer:
    """Normalize raw profiler-like records into classified Event objects."""

    def __init__(self, classifier: KernelClassifier | None = None):
        self.classifier = classifier or KernelClassifier()

    def normalize_records(
        self,
        records: list[dict[str, Any]],
        *,
        default_rank: int | None = None,
        default_stage_id: int | None = None,
    ) -> list[Event]:
        events = []
        for record in records:
            event = Event(
                timestamp_start=float(record["timestamp_start"]),
                timestamp_end=float(record["timestamp_end"]),
                event_type=EventType(record.get("event_type", EventType.UNKNOWN.value)),
                name=str(record.get("name", "")),
                device_id=record.get("device_id"),
                rank=record.get("rank", default_rank),
                stage_id=record.get("stage_id", default_stage_id),
                metadata=dict(record.get("metadata") or {}),
            )
            events.append(self.classifier.classify_event(event))
        return events
