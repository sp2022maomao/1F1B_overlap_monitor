from __future__ import annotations

from collections import defaultdict

from overlap_monitor.core.events import Event, EventType
from overlap_monitor.core.intervals import (
    intersection_total,
    interval_total,
    merge_intervals,
    span,
)
from overlap_monitor.core.metrics import OverlapSummary, StageMetrics

COMPUTE_TYPES = {EventType.COMPUTE, EventType.GEMM, EventType.ATTENTION}
COMMUNICATION_TYPES = {EventType.COMMUNICATION, EventType.NCCL}


class OverlapAnalyzer:
    """Timeline-based communication-computation overlap analyzer."""

    def compute_events(self, events: list[Event]) -> list[Event]:
        return [
            event
            for event in events
            if event.event_type in COMPUTE_TYPES
            or event.metadata.get("event_family") == EventType.COMPUTE.value
        ]

    def communication_events(self, events: list[Event]) -> list[Event]:
        return [
            event
            for event in events
            if event.event_type in COMMUNICATION_TYPES
            or event.metadata.get("event_family") == EventType.COMMUNICATION.value
        ]

    def analyze(self, events: list[Event]) -> OverlapSummary:
        compute = self.compute_events(events)
        communication = self.communication_events(events)
        compute_intervals = merge_intervals(event.interval for event in compute)
        communication_intervals = merge_intervals(event.interval for event in communication)

        compute_time = interval_total(compute_intervals, already_merged=True)
        communication_time = interval_total(communication_intervals, already_merged=True)
        overlap_time = intersection_total(
            compute_intervals, communication_intervals, already_merged=True
        )
        denom = min(compute_time, communication_time)
        overlap_ratio = overlap_time / denom if denom > 0 else 0.0
        stage_metrics = self._stage_metrics(events)
        bubble_ratio = self._pipeline_bubble_ratio(stage_metrics)
        stage_balance = self._stage_balance(stage_metrics)

        return OverlapSummary(
            compute_time=compute_time,
            communication_time=communication_time,
            overlap_time=overlap_time,
            overlap_ratio=overlap_ratio,
            bubble_ratio=bubble_ratio,
            stage_balance=stage_balance,
            stage_metrics=stage_metrics,
        )

    def _bubble_ratio(
        self,
        compute_intervals: list[tuple[float, float]],
        communication_intervals: list[tuple[float, float]],
    ) -> float:
        active = merge_intervals([*compute_intervals, *communication_intervals])
        active_span = span(active)
        if active_span is None:
            return 0.0
        total_span = active_span[1] - active_span[0]
        if total_span <= 0:
            return 0.0
        active_time = interval_total(active, already_merged=True)
        return max(total_span - active_time, 0.0) / total_span

    def _stage_metrics(self, events: list[Event]) -> list[StageMetrics]:
        by_stage: dict[int | None, list[Event]] = defaultdict(list)
        for event in events:
            by_stage[event.stage_id].append(event)
        metrics = []
        for stage_id, stage_events in sorted(
            by_stage.items(), key=lambda item: (-1 if item[0] is None else item[0])
        ):
            compute_intervals = merge_intervals(
                event.interval for event in self.compute_events(stage_events)
            )
            communication_intervals = merge_intervals(
                event.interval for event in self.communication_events(stage_events)
            )
            compute_time = interval_total(compute_intervals, already_merged=True)
            communication_time = interval_total(communication_intervals, already_merged=True)
            overlap_time = intersection_total(
                compute_intervals, communication_intervals, already_merged=True
            )
            denom = min(compute_time, communication_time)
            active_span = span([*compute_intervals, *communication_intervals])
            span_duration = active_span[1] - active_span[0] if active_span else 0.0
            metrics.append(
                StageMetrics(
                    stage_id=stage_id,
                    compute_time=compute_time,
                    communication_time=communication_time,
                    overlap_time=overlap_time,
                    overlap_ratio=overlap_time / denom if denom > 0 else 0.0,
                    bubble_ratio=self._bubble_ratio(compute_intervals, communication_intervals),
                    active_span=span_duration,
                )
            )
        return metrics

    def _stage_balance(self, stage_metrics: list[StageMetrics]) -> float:
        spans = [metric.active_span for metric in stage_metrics if metric.stage_id is not None]
        if not spans:
            return 1.0
        max_span = max(spans)
        if max_span <= 0:
            return 1.0
        return min(spans) / max_span

    def _pipeline_bubble_ratio(self, stage_metrics: list[StageMetrics]) -> float:
        stage_metrics = [metric for metric in stage_metrics if metric.stage_id is not None]
        if not stage_metrics:
            return 0.0
        total_span = sum(metric.active_span for metric in stage_metrics)
        if total_span <= 0:
            return 0.0
        idle = sum(metric.bubble_ratio * metric.active_span for metric in stage_metrics)
        return idle / total_span
