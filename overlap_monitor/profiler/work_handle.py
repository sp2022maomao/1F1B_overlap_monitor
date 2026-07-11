from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter_ns
from typing import Any, Callable

from overlap_monitor.core.events import Event, EventType


TimeSource = Callable[[], float]


def perf_counter_us() -> float:
    return perf_counter_ns() / 1000.0


@dataclass
class WorkRecord:
    comm_id: str
    name: str
    launch_time: float
    stage_id: int | None = None
    rank: int | None = None
    device_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    wait_start: float | None = None
    wait_end: float | None = None
    completion_time: float | None = None
    completion_source: str | None = None
    lock: Lock = field(default_factory=Lock, repr=False, compare=False)


class ObservedWork:
    """Thin wrapper around an async distributed Work handle."""

    def __init__(self, work: Any, record: WorkRecord):
        self._work = work
        self.record = record

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        time_source = self.record.metadata["_time_source"]
        wait_start = time_source()
        try:
            return self._work.wait(*args, **kwargs)
        finally:
            wait_end = time_source()
            with self.record.lock:
                if self.record.wait_start is None:
                    self.record.wait_start = wait_start
                    self.record.wait_end = wait_end
                if self.record.completion_time is None:
                    self.record.completion_time = wait_end
                    self.record.completion_source = "wait_return_upper_bound"

    def mark_completed(
        self, timestamp: float | None = None, *, source: str = "adapter_observation"
    ) -> None:
        """Record a completion observed by an adapter or profiler callback."""
        time_source = self.record.metadata["_time_source"]
        completion_time = time_source() if timestamp is None else timestamp
        with self.record.lock:
            if completion_time < self.record.launch_time:
                raise ValueError("completion timestamp precedes communication launch")
            if self.record.completion_time is None or completion_time < self.record.completion_time:
                self.record.completion_time = completion_time
                self.record.completion_source = source

    def __getattr__(self, name: str) -> Any:
        return getattr(self._work, name)


class WorkHandleRecorder:
    """Records async communication lifetime and critical-path wait events.

    The recorder intentionally does not import torch, Megatron, or call a global
    CUDA synchronization.
    """

    def __init__(self, time_source: TimeSource = perf_counter_us):
        self.time_source = time_source
        self.records: list[WorkRecord] = []
        self._lock = Lock()

    def wrap(
        self,
        work: Any,
        *,
        comm_id: str,
        name: str,
        stage_id: int | None = None,
        rank: int | None = None,
        device_id: int | None = None,
        **metadata: Any,
    ) -> ObservedWork:
        record = WorkRecord(
            comm_id=comm_id,
            name=name,
            launch_time=self.time_source(),
            stage_id=stage_id,
            rank=rank,
            device_id=device_id,
            metadata={**metadata, "_time_source": self.time_source},
        )
        with self._lock:
            self.records.append(record)
        return ObservedWork(work, record)

    def events(self) -> list[Event]:
        events: list[Event] = []
        with self._lock:
            records = list(self.records)
        for record in records:
            with record.lock:
                events.extend(self._record_events(record))
        return events

    def _record_events(self, record: WorkRecord) -> list[Event]:
        events: list[Event] = []
        metadata = self._public_metadata(record)
        completion_time = record.completion_time or record.launch_time
        events.append(
            Event(
                timestamp_start=record.launch_time,
                timestamp_end=completion_time,
                event_type=EventType.COMMUNICATION,
                name=record.name,
                device_id=record.device_id,
                rank=record.rank,
                stage_id=record.stage_id,
                metadata={
                    **metadata,
                    "comm_id": record.comm_id,
                    "measurement": "async_work_lifetime",
                    "completion_observed": record.completion_time is not None,
                    "completion_source": record.completion_source or "not_observed",
                    "runtime_kind": (
                        "upper_bound"
                        if record.completion_source == "wait_return_upper_bound"
                        else "observed"
                    ),
                },
            )
        )
        if record.wait_start is not None and record.wait_end is not None:
            events.append(
                Event(
                    timestamp_start=record.wait_start,
                    timestamp_end=record.wait_end,
                    event_type=EventType.WAIT,
                    name=f"{record.name}.wait",
                    device_id=record.device_id,
                    rank=record.rank,
                    stage_id=record.stage_id,
                    metadata={
                        **metadata,
                        "comm_id": record.comm_id,
                        "event_family": EventType.WAIT.value,
                        "measurement": "critical_path_wait",
                    },
                )
            )
        return events

    def _public_metadata(self, record: WorkRecord) -> dict[str, Any]:
        return {
            key: value
            for key, value in record.metadata.items()
            if not key.startswith("_")
        }
