from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from threading import Lock

from overlap_monitor_v2.core.events import Event
from overlap_monitor_v2.core.io import write_events_jsonl


class MonitoringSession:
    """Thread-safe, framework-neutral event buffer with explicit batch flushing."""

    def __init__(self, *, max_events: int = 1_000_000):
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.max_events = max_events
        self._events: list[Event] = []
        self._dropped_events = 0
        self._lock = Lock()

    @property
    def dropped_events(self) -> int:
        with self._lock:
            return self._dropped_events

    def emit(self, event: Event) -> bool:
        with self._lock:
            if len(self._events) >= self.max_events:
                self._dropped_events += 1
                return False
            self._events.append(event)
            return True

    def extend(self, events: Iterable[Event]) -> int:
        return sum(1 for event in events if self.emit(event))

    def snapshot(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def flush_jsonl(self, path: Path, *, clear: bool = False) -> int:
        events = self.snapshot()
        write_events_jsonl(events, path)
        if clear:
            with self._lock:
                del self._events[: len(events)]
        return len(events)
