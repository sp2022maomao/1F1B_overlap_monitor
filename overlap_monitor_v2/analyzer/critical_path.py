from __future__ import annotations

from dataclasses import asdict, dataclass, field

from overlap_monitor_v2.analyzer.overlap import COMPUTE_TYPES, COMMUNICATION_TYPES
from overlap_monitor_v2.core.events import Event, EventType
from overlap_monitor_v2.core.intervals import interval_total, intersection_total, merge_intervals, span


WAIT_FAMILY = "WAIT"


@dataclass(frozen=True)
class CriticalPathGroupMetrics:
    group_id: str
    compute_time: float
    communication_runtime: float
    exposed_communication: float
    hidden_communication: float
    overlap_ratio: float
    wait_time: float
    critical_path_span: float
    measurement_quality: str
    communication_runtime_kind: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CriticalPathSummary:
    compute_time: float
    communication_runtime: float
    exposed_communication: float
    hidden_communication: float
    overlap_ratio: float
    wait_time: float
    critical_path_span: float
    measurement_quality: str
    communication_runtime_kind: str
    warnings: list[str] = field(default_factory=list)
    group_metrics: list[CriticalPathGroupMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["group_metrics"] = [metric.to_dict() for metric in self.group_metrics]
        return payload


class CriticalPathOverlapAnalyzer:
    """Critical-path overlap analyzer for asynchronous Megatron/MoE execution.

    Communication intervals should represent async Work lifetime:
    launch -> completion. WAIT events represent the part of Work.wait() that is
    still blocking the training critical path. This avoids treating Python
    launch overhead as communication runtime.
    """

    def analyze(self, events: list[Event]) -> CriticalPathSummary:
        return self._analyze_group(events, group_id="all", include_subgroups=True)

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

    def precise_communication_events(self, events: list[Event]) -> list[Event]:
        return [
            event
            for event in events
            if event.event_type == EventType.NCCL
            or event.metadata.get("measurement") in {"kernel_timeline", "nsight_timeline"}
        ]

    def work_lifetime_events(self, events: list[Event]) -> list[Event]:
        return [
            event
            for event in events
            if event.event_type == EventType.COMMUNICATION
            and event.metadata.get("measurement") == "async_work_lifetime"
        ]

    def wait_events(self, events: list[Event]) -> list[Event]:
        return [
            event
            for event in events
            if event.event_type == EventType.WAIT
            or event.metadata.get("event_family") == WAIT_FAMILY
        ]

    def _analyze_group(
        self, events: list[Event], group_id: str, include_subgroups: bool
    ) -> CriticalPathSummary:
        compute_intervals = merge_intervals(event.interval for event in self.compute_events(events))
        precise_communication = self.precise_communication_events(events)
        work_lifetimes = self.work_lifetime_events(events)
        selected_communication = precise_communication or work_lifetimes
        if not selected_communication:
            selected_communication = self.communication_events(events)
        communication_intervals = merge_intervals(event.interval for event in selected_communication)
        work_intervals = merge_intervals(event.interval for event in work_lifetimes)
        wait_intervals = merge_intervals(event.interval for event in self.wait_events(events))

        compute_time = interval_total(compute_intervals, already_merged=True)
        communication_runtime = interval_total(communication_intervals, already_merged=True)
        wait_time = intersection_total(
            work_intervals or communication_intervals, wait_intervals, already_merged=True
        )

        warnings: list[str] = []
        if precise_communication:
            hidden_communication = intersection_total(
                compute_intervals, communication_intervals, already_merged=True
            )
            exposed_communication = max(communication_runtime - hidden_communication, 0.0)
            measurement_quality = "kernel_timeline"
            runtime_kind = "observed_kernel_runtime"
        elif wait_intervals:
            exposed_communication = wait_time
            hidden_communication = max(communication_runtime - exposed_communication, 0.0)
            upper_bound = any(
                event.metadata.get("runtime_kind", "upper_bound") == "upper_bound"
                for event in selected_communication
            )
            measurement_quality = "estimated"
            runtime_kind = "upper_bound" if upper_bound else "observed_work_window"
            if upper_bound:
                warnings.append(
                    "communication_runtime and hidden_communication are estimates because "
                    "completion was observed only when wait() returned"
                )
        else:
            hidden_communication = intersection_total(
                compute_intervals, communication_intervals, already_merged=True
            )
            exposed_communication = max(communication_runtime - hidden_communication, 0.0)
            measurement_quality = "timeline_fallback"
            runtime_kind = "generic_event_runtime"
            warnings.append("no Work.wait event was available for critical-path attribution")

        overlap_ratio = (
            hidden_communication / communication_runtime
            if communication_runtime > 0
            else 0.0
        )
        critical_span = self._critical_span(compute_intervals, communication_intervals, wait_intervals)
        group_metrics = self._group_metrics(events) if include_subgroups else []

        return CriticalPathSummary(
            compute_time=compute_time,
            communication_runtime=communication_runtime,
            exposed_communication=exposed_communication,
            hidden_communication=hidden_communication,
            overlap_ratio=overlap_ratio,
            wait_time=wait_time,
            critical_path_span=critical_span,
            measurement_quality=measurement_quality,
            communication_runtime_kind=runtime_kind,
            warnings=warnings,
            group_metrics=group_metrics,
        )

    def _group_metrics(self, events: list[Event]) -> list[CriticalPathGroupMetrics]:
        by_group: dict[str, list[Event]] = {}
        for event in events:
            group_id = self._event_group_id(event)
            by_group.setdefault(group_id, []).append(event)

        metrics = []
        for group_id, group_events in sorted(by_group.items()):
            summary = self._analyze_group(group_events, group_id, include_subgroups=False)
            metrics.append(
                CriticalPathGroupMetrics(
                    group_id=group_id,
                    compute_time=summary.compute_time,
                    communication_runtime=summary.communication_runtime,
                    exposed_communication=summary.exposed_communication,
                    hidden_communication=summary.hidden_communication,
                    overlap_ratio=summary.overlap_ratio,
                    wait_time=summary.wait_time,
                    critical_path_span=summary.critical_path_span,
                    measurement_quality=summary.measurement_quality,
                    communication_runtime_kind=summary.communication_runtime_kind,
                )
            )
        return metrics

    def _event_group_id(self, event: Event) -> str:
        microbatch_id = event.metadata.get("microbatch_id", "unknown_mb")
        stage_id = event.stage_id if event.stage_id is not None else "unknown_stage"
        phase = event.metadata.get("phase", "unknown_phase")
        return f"stage={stage_id}/microbatch={microbatch_id}/phase={phase}"

    def _critical_span(
        self,
        compute_intervals: list[tuple[float, float]],
        communication_intervals: list[tuple[float, float]],
        wait_intervals: list[tuple[float, float]],
    ) -> float:
        active_span = span([*compute_intervals, *communication_intervals, *wait_intervals])
        if active_span is None:
            return 0.0
        return active_span[1] - active_span[0]
