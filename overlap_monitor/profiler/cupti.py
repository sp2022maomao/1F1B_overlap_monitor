from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from overlap_monitor.analyzer import KernelClassifier
from overlap_monitor.core.events import Event, EventType


class CuptiFormatError(ValueError):
    pass


@dataclass(frozen=True)
class CuptiParseResult:
    events: list[Event]
    dropped_records: int = 0
    skipped_records: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.dropped_records == 0 and self.skipped_records == 0


class CuptiActivityParser:
    """Convert overlap-monitor CUPTI activity JSONL into normalized events.

    The native collector writes timestamps in nanoseconds. The parser converts
    them to microseconds and classifies kernels without importing CUDA or CUPTI.
    """

    SCHEMA_VERSION = 1
    RECORD_KINDS = {
        "collector_summary",
        "external_correlation",
        "kernel",
        "trace_metadata",
    }

    def __init__(self, classifier: KernelClassifier | None = None):
        self.classifier = classifier or KernelClassifier()

    def parse_file(
        self,
        path: Path,
        *,
        default_rank: int | None = None,
        default_stage_id: int | None = None,
        strict: bool = True,
    ) -> CuptiParseResult:
        records = self._read_records(path)
        return self.parse_records(
            records,
            default_rank=default_rank,
            default_stage_id=default_stage_id,
            strict=strict,
        )

    def parse_records(
        self,
        records: list[dict[str, Any]],
        *,
        default_rank: int | None = None,
        default_stage_id: int | None = None,
        strict: bool = True,
    ) -> CuptiParseResult:
        external_ids: dict[int, tuple[int, str]] = {}
        dropped_records = 0
        warnings: list[str] = []

        for record in records:
            self._validate_schema(record)
            kind = record.get("record_kind")
            if kind == "external_correlation":
                correlation_id = self._required_int(record, "correlation_id")
                external_ids[correlation_id] = (
                    self._required_int(record, "external_id"),
                    str(record.get("external_kind", "unknown")),
                )
            elif kind == "collector_summary":
                dropped_records += self._required_int(record, "dropped_records")

        events: list[Event] = []
        skipped_records = 0
        for record in records:
            if record.get("record_kind") != "kernel":
                continue
            start_ns = self._required_int(record, "start_ns")
            end_ns = self._required_int(record, "end_ns")
            if start_ns == 0 and end_ns == 0:
                skipped_records += 1
                warnings.append("kernel record has unavailable timestamps")
                continue
            if end_ns <= start_ns:
                raise CuptiFormatError(
                    f"kernel end_ns must be greater than start_ns: {start_ns}, {end_ns}"
                )

            correlation_id = self._required_int(record, "correlation_id")
            device_id = self._required_int(record, "device_id")
            process_id = int(record.get("process_id", 0))
            rank = record.get("rank")
            if rank is None or int(rank) < 0:
                rank = default_rank
            stage_id = record.get("stage_id", default_stage_id)
            metadata = dict(record.get("metadata") or {})
            metadata.update(
                {
                    "collector": "cupti",
                    "measurement": "kernel_timeline",
                    "runtime_kind": "observed_kernel_runtime",
                    "timestamp_unit": "us",
                    "source_timestamp_unit": "ns",
                    "clock_domain": f"cupti:pid={process_id}:device={device_id}",
                    "stream_id": self._required_int(record, "stream_id"),
                    "correlation_id": correlation_id,
                    "process_id": process_id,
                }
            )
            external = external_ids.get(correlation_id)
            if external is not None:
                metadata["external_id"] = external[0]
                metadata["external_kind"] = external[1]

            event = Event(
                timestamp_start=start_ns / 1000.0,
                timestamp_end=end_ns / 1000.0,
                event_type=EventType.UNKNOWN,
                name=str(record.get("name", "")),
                device_id=device_id,
                rank=int(rank) if rank is not None else None,
                stage_id=int(stage_id) if stage_id is not None else None,
                metadata=metadata,
            )
            events.append(self.classifier.classify_event(event))

        if dropped_records:
            warnings.append(
                f"CUPTI reported {dropped_records} dropped activity records"
            )
        if strict and (dropped_records or skipped_records):
            raise CuptiFormatError("incomplete CUPTI trace: " + "; ".join(warnings))
        return CuptiParseResult(events, dropped_records, skipped_records, warnings)

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CuptiFormatError(
                        f"invalid CUPTI JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise CuptiFormatError(
                        f"invalid CUPTI record at {path}:{line_number}: expected object"
                    )
                records.append(record)
        return records

    def _validate_schema(self, record: dict[str, Any]) -> None:
        version = record.get("schema_version")
        if version != self.SCHEMA_VERSION:
            raise CuptiFormatError(
                f"unsupported CUPTI schema_version={version!r}; expected {self.SCHEMA_VERSION}"
            )
        if "record_kind" not in record:
            raise CuptiFormatError("CUPTI record is missing record_kind")
        if record["record_kind"] not in self.RECORD_KINDS:
            raise CuptiFormatError(
                f"unsupported CUPTI record_kind={record['record_kind']!r}"
            )

    def _required_int(self, record: dict[str, Any], key: str) -> int:
        if key not in record:
            raise CuptiFormatError(
                f"CUPTI {record.get('record_kind')} record is missing {key}"
            )
        try:
            return int(record[key])
        except (TypeError, ValueError) as exc:
            raise CuptiFormatError(f"CUPTI field {key} must be an integer") from exc
