from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from overlap_monitor.core.events import EventType
from overlap_monitor.core.io import JsonlFormatError, read_events_jsonl
from overlap_monitor.profiler import PyTorchProfilerEventParser, TimelineNormalizer


class FakeRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeProfilerEvent:
    def __init__(self, name, start, end, device_index=0):
        self.name = name
        self.time_range = FakeRange(start, end)
        self.device_index = device_index


class EventParserTests(unittest.TestCase):
    def test_parse_profiler_events(self):
        parser = PyTorchProfilerEventParser()
        events = parser.parse([FakeProfilerEvent("ncclKernel_AllToAll", 1, 5)], rank=2, stage_id=1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].timestamp_start, 1)
        self.assertEqual(events[0].rank, 2)
        self.assertEqual(events[0].stage_id, 1)
        self.assertEqual(events[0].event_type, EventType.NCCL)
        self.assertEqual(events[0].metadata["collector"], "pytorch_profiler")

    def test_normalizer_classifies_records(self):
        normalizer = TimelineNormalizer()
        events = normalizer.normalize_records(
            [{"timestamp_start": 0, "timestamp_end": 10, "name": "ncclKernel_AllReduce"}]
        )
        self.assertEqual(events[0].event_type, EventType.NCCL)
        self.assertEqual(events[0].metadata["event_family"], EventType.COMMUNICATION.value)

    def test_jsonl_error_contains_line_number(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "broken.jsonl"
            path.write_text("{}\n{broken}\n", encoding="utf-8")
            with self.assertRaisesRegex(JsonlFormatError, r"broken\.jsonl:1"):
                read_events_jsonl(path)


if __name__ == "__main__":
    unittest.main()
