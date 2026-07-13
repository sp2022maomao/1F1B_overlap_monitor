from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from overlap_monitor.adapters.cupti_runtime import (
    CuptiRuntimeCollector,
    CuptiRuntimeError,
)
from overlap_monitor.analyzer import CriticalPathOverlapAnalyzer, OverlapAnalyzer
from overlap_monitor.core.events import EventType
from overlap_monitor.profiler.cupti import CuptiActivityParser, CuptiFormatError


class CuptiParserTests(unittest.TestCase):
    def test_parses_kernel_timestamps_and_external_correlation(self):
        records = [
            {
                "schema_version": 1,
                "record_kind": "external_correlation",
                "correlation_id": 11,
                "external_id": 99,
                "external_kind": "custom0",
            },
            {
                "schema_version": 1,
                "record_kind": "kernel",
                "start_ns": 1000,
                "end_ns": 5000,
                "name": "ncclDevKernel_AllToAll",
                "device_id": 0,
                "stream_id": 7,
                "correlation_id": 11,
                "process_id": 42,
            },
        ]
        result = CuptiActivityParser().parse_records(records, default_rank=3)

        self.assertTrue(result.complete)
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event.event_type, EventType.NCCL)
        self.assertEqual(event.timestamp_start, 1.0)
        self.assertEqual(event.timestamp_end, 5.0)
        self.assertEqual(event.rank, 3)
        self.assertEqual(event.metadata["external_id"], 99)
        self.assertEqual(event.metadata["stream_id"], 7)
        self.assertEqual(event.metadata["measurement"], "kernel_timeline")

    def test_cupti_events_feed_overlap_analyzer(self):
        records = [
            self._kernel(0, 10_000, "ncclDevKernel_AllToAll", 1),
            self._kernel(5_000, 15_000, "nvte_gemm_fp8", 2),
        ]
        events = CuptiActivityParser().parse_records(records).events
        summary = OverlapAnalyzer().analyze(events)

        self.assertEqual(summary.communication_time, 10)
        self.assertEqual(summary.compute_time, 10)
        self.assertEqual(summary.overlap_time, 5)
        self.assertEqual(summary.overlap_ratio, 0.5)

        critical = CriticalPathOverlapAnalyzer().analyze(events)
        self.assertEqual(critical.communication_runtime, 10)
        self.assertEqual(critical.compute_time, 10)
        self.assertEqual(critical.hidden_communication, 5)
        self.assertEqual(critical.exposed_communication, 5)
        self.assertEqual(critical.overlap_ratio, 0.5)
        self.assertEqual(critical.measurement_quality, "kernel_timeline")

    def test_strict_mode_rejects_dropped_records(self):
        records = [
            {
                "schema_version": 1,
                "record_kind": "collector_summary",
                "dropped_records": 2,
            }
        ]
        with self.assertRaisesRegex(CuptiFormatError, "dropped"):
            CuptiActivityParser().parse_records(records)

        result = CuptiActivityParser().parse_records(records, strict=False)
        self.assertFalse(result.complete)
        self.assertEqual(result.dropped_records, 2)

    def test_file_error_contains_line_number(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(CuptiFormatError, r"bad\.jsonl:1"):
                CuptiActivityParser().parse_file(path)

    def test_rejects_unknown_record_kind(self):
        with self.assertRaisesRegex(CuptiFormatError, "record_kind"):
            CuptiActivityParser().parse_records(
                [{"schema_version": 1, "record_kind": "future_kernel"}]
            )

    def _kernel(self, start_ns: int, end_ns: int, name: str, correlation_id: int):
        return {
            "schema_version": 1,
            "record_kind": "kernel",
            "start_ns": start_ns,
            "end_ns": end_ns,
            "name": name,
            "device_id": 0,
            "stream_id": correlation_id,
            "correlation_id": correlation_id,
            "process_id": 1,
        }


class FakeNativeLibrary:
    def __init__(self):
        self.calls = []
        self.stack = []

    def overlap_cupti_start(self, path, rank):
        self.calls.append(("start", path, rank))
        return 0

    def overlap_cupti_stop(self):
        self.calls.append(("stop",))
        return 0

    def overlap_cupti_push_external_id(self, external_id):
        value = int(external_id)
        self.stack.append(value)
        self.calls.append(("push", value))
        return 0

    def overlap_cupti_pop_external_id(self, output):
        value = self.stack.pop()
        output._obj.value = value
        self.calls.append(("pop", value))
        return 0


class CuptiRuntimeCollectorTests(unittest.TestCase):
    def test_external_range_balances_native_stack(self):
        library = FakeNativeLibrary()
        collector = CuptiRuntimeCollector(library=library)
        collector.start("trace.jsonl", rank=4)
        with collector.external_range(17):
            pass
        collector.stop()

        self.assertEqual(
            [call[0] for call in library.calls], ["start", "push", "pop", "stop"]
        )

    def test_requires_started_collector(self):
        collector = CuptiRuntimeCollector(library=FakeNativeLibrary())
        with self.assertRaises(CuptiRuntimeError):
            collector.push_external_id(1)


if __name__ == "__main__":
    unittest.main()
