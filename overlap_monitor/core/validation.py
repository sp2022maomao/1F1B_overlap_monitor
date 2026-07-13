from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from overlap_monitor.core.events import Event, EventType


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    event_count: int
    clock_domains: list[str]
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "event_count": self.event_count,
            "clock_domains": self.clock_domains,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_events(events: list[Event], *, strict_clock_domain: bool = True) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if not events:
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                "empty_trace",
                "the trace contains no events after applying filters",
            )
        )
    domains = sorted({_clock_domain(event) for event in events})
    if strict_clock_domain and len(domains) > 1:
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                "mixed_clock_domains",
                "events contain multiple rank/device clock domains; filter or align them first",
            )
        )

    communication_ids = {
        str(event.metadata["comm_id"])
        for event in events
        if event.event_type in {EventType.COMMUNICATION, EventType.NCCL}
        and "comm_id" in event.metadata
    }
    for event in events:
        comm_id = event.metadata.get("comm_id")
        if event.event_type == EventType.WAIT and comm_id is not None:
            if str(comm_id) not in communication_ids:
                issues.append(
                    ValidationIssue(
                        Severity.ERROR,
                        "orphan_wait",
                        f"WAIT event references unknown comm_id={comm_id}",
                    )
                )
        if (
            event.event_type == EventType.COMMUNICATION
            and event.metadata.get("measurement") == "async_work_lifetime"
            and not event.metadata.get("completion_observed", False)
        ):
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "incomplete_work",
                    f"communication comm_id={comm_id!s} has no observed completion",
                )
            )

    return ValidationReport(len(events), domains, issues)


def _clock_domain(event: Event) -> str:
    explicit = event.metadata.get("clock_domain")
    if explicit is not None:
        return str(explicit)
    if event.rank is not None:
        return f"rank={event.rank}"
    return f"device={event.device_id!s}"
