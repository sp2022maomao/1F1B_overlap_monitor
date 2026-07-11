from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaEventSpan:
    name: str
    timestamp_start: float
    timestamp_end: float
    device_id: int | None = None
    rank: int | None = None
    stage_id: int | None = None

    @property
    def duration(self) -> float:
        return max(self.timestamp_end - self.timestamp_start, 0.0)
