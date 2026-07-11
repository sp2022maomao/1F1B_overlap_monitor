from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any


class EventType(str, Enum):
    COMPUTE = "COMPUTE"
    COMMUNICATION = "COMMUNICATION"
    NCCL = "NCCL"
    GEMM = "GEMM"
    ATTENTION = "ATTENTION"
    MEMORY = "MEMORY"
    WAIT = "WAIT"
    PIPELINE = "PIPELINE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Event:
    timestamp_start: float
    timestamp_end: float
    event_type: EventType
    name: str = ""
    device_id: int | None = None
    rank: int | None = None
    stage_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_start) or not isfinite(self.timestamp_end):
            raise ValueError("event timestamps must be finite")
        if self.timestamp_end < self.timestamp_start:
            raise ValueError("event timestamp_end must be >= timestamp_start")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def duration(self) -> float:
        return max(self.timestamp_end - self.timestamp_start, 0.0)

    @property
    def interval(self) -> tuple[float, float]:
        return (self.timestamp_start, self.timestamp_end)

    def with_type(self, event_type: EventType, **metadata: Any) -> "Event":
        merged = dict(self.metadata)
        merged.update(metadata)
        return Event(
            timestamp_start=self.timestamp_start,
            timestamp_end=self.timestamp_end,
            event_type=event_type,
            name=self.name,
            device_id=self.device_id,
            rank=self.rank,
            stage_id=self.stage_id,
            metadata=merged,
        )


def event_from_mapping(payload: dict[str, Any]) -> Event:
    return Event(
        timestamp_start=float(payload["timestamp_start"]),
        timestamp_end=float(payload["timestamp_end"]),
        event_type=EventType(payload.get("event_type", EventType.UNKNOWN.value)),
        name=str(payload.get("name", "")),
        device_id=payload.get("device_id"),
        rank=payload.get("rank"),
        stage_id=payload.get("stage_id"),
        metadata=dict(payload.get("metadata") or {}),
    )
