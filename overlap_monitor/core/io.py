from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from collections.abc import Iterable

from overlap_monitor.core.events import Event, EventType, event_from_mapping


def event_to_dict(event: Event) -> dict:
    return {
        "timestamp_start": event.timestamp_start,
        "timestamp_end": event.timestamp_end,
        "device_id": event.device_id,
        "rank": event.rank,
        "stage_id": event.stage_id,
        "event_type": event.event_type.value,
        "name": event.name,
        "metadata": event.metadata,
    }


class JsonlFormatError(ValueError):
    pass


def write_events_jsonl(events: Iterable[Event], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as f:
            temporary_path = Path(f.name)
            for event in events:
                f.write(json.dumps(event_to_dict(event), sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_events_jsonl(path: Path) -> list[Event]:
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    events.append(event_from_mapping(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise JsonlFormatError(
                        f"invalid event at {path}:{line_number}: {exc}"
                    ) from exc
    return events
