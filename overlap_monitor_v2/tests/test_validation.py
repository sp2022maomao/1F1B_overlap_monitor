from __future__ import annotations

import unittest

from overlap_monitor_v2.core.events import Event, EventType
from overlap_monitor_v2.core.validation import validate_events


class ValidationTests(unittest.TestCase):
    def test_rejects_mixed_rank_clock_domains(self):
        events = [
            Event(0, 10, EventType.GEMM, rank=0),
            Event(0, 10, EventType.NCCL, rank=1),
        ]

        report = validate_events(events)

        self.assertFalse(report.valid)
        self.assertEqual(report.issues[0].code, "mixed_clock_domains")

    def test_detects_orphan_wait(self):
        event = Event(5, 10, EventType.WAIT, metadata={"comm_id": "missing"})

        report = validate_events([event])

        self.assertFalse(report.valid)
        self.assertEqual(report.issues[0].code, "orphan_wait")

    def test_event_rejects_reverse_interval(self):
        with self.assertRaises(ValueError):
            Event(10, 5, EventType.COMPUTE)


if __name__ == "__main__":
    unittest.main()
