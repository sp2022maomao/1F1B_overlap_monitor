from __future__ import annotations

import json
from pathlib import Path

from overlap_monitor_v2.core.events import Event


def to_chrome_trace(events: list[Event]) -> dict:
    trace_events = []
    for index, event in enumerate(events):
        trace_events.append(
            {
                "name": event.name or event.event_type.value,
                "cat": event.event_type.value,
                "ph": "X",
                "ts": event.timestamp_start,
                "dur": event.duration,
                "pid": event.rank if event.rank is not None else 0,
                "tid": event.stage_id if event.stage_id is not None else event.device_id or 0,
                "args": {
                    "device_id": event.device_id,
                    "rank": event.rank,
                    "stage_id": event.stage_id,
                    "event_type": event.event_type.value,
                    **event.metadata,
                },
                "id": index,
            }
        )
    return {"traceEvents": trace_events, "displayTimeUnit": "us"}


def write_chrome_trace(events: list[Event], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_chrome_trace(events), indent=2), encoding="utf-8")
