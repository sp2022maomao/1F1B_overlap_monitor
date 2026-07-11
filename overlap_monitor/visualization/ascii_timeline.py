from __future__ import annotations

from collections import defaultdict

from overlap_monitor.core.events import Event, EventType


SYMBOLS = {
    EventType.COMPUTE: "C",
    EventType.GEMM: "G",
    EventType.ATTENTION: "A",
    EventType.COMMUNICATION: "N",
    EventType.NCCL: "N",
    EventType.MEMORY: "M",
    EventType.WAIT: "W",
    EventType.PIPELINE: "P",
    EventType.UNKNOWN: "?",
}


def render_ascii_timeline(events: list[Event], width: int = 80) -> str:
    if not events:
        return ""
    start = min(event.timestamp_start for event in events)
    end = max(event.timestamp_end for event in events)
    span = max(end - start, 1.0)
    by_stage: dict[int | None, list[Event]] = defaultdict(list)
    for event in events:
        by_stage[event.stage_id].append(event)

    lines = []
    for stage_id, stage_events in sorted(
        by_stage.items(), key=lambda item: (-1 if item[0] is None else item[0])
    ):
        row = [" "] * width
        for event in stage_events:
            left = int(((event.timestamp_start - start) / span) * (width - 1))
            right = max(left + 1, int(((event.timestamp_end - start) / span) * (width - 1)))
            symbol = SYMBOLS.get(event.event_type, "?")
            for idx in range(max(left, 0), min(right + 1, width)):
                row[idx] = symbol if row[idx] == " " else "X"
        label = "Stage?" if stage_id is None else f"Stage{stage_id}"
        lines.append(f"{label:<8} |{''.join(row)}|")
    return "\n".join(lines)
