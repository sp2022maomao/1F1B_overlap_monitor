from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class StageMetrics:
    stage_id: int | None
    compute_time: float
    communication_time: float
    overlap_time: float
    overlap_ratio: float
    bubble_ratio: float
    active_span: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OverlapSummary:
    compute_time: float
    communication_time: float
    overlap_time: float
    overlap_ratio: float
    bubble_ratio: float
    stage_balance: float
    stage_metrics: list[StageMetrics] = field(default_factory=list)

    @property
    def overlap_ratio_definition(self) -> str:
        return "overlap_time / min(compute_time, communication_time)"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["overlap_ratio_definition"] = self.overlap_ratio_definition
        payload["stage_metrics"] = [stage.to_dict() for stage in self.stage_metrics]
        return payload
