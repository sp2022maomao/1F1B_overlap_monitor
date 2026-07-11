from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from overlap_monitor_v2.core.events import Event, EventType
from overlap_monitor_v2.core.io import read_events_jsonl
from overlap_monitor_v2.core.session import MonitoringSession


class MonitoringSessionTests(unittest.TestCase):
    def test_bounded_buffer_and_flush(self):
        session = MonitoringSession(max_events=2)
        self.assertTrue(session.emit(Event(0, 1, EventType.GEMM)))
        self.assertTrue(session.emit(Event(1, 2, EventType.NCCL)))
        self.assertFalse(session.emit(Event(2, 3, EventType.WAIT)))
        self.assertEqual(session.dropped_events, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            self.assertEqual(session.flush_jsonl(path, clear=True), 2)
            self.assertEqual(len(read_events_jsonl(path)), 2)
            self.assertEqual(session.snapshot(), [])

    def test_concurrent_emit_preserves_all_events(self):
        session = MonitoringSession(max_events=200)

        def emit(index: int) -> bool:
            return session.emit(Event(index, index + 1, EventType.COMPUTE))

        with ThreadPoolExecutor(max_workers=8) as executor:
            accepted = list(executor.map(emit, range(200)))

        self.assertTrue(all(accepted))
        self.assertEqual(len(session.snapshot()), 200)
        self.assertEqual(session.dropped_events, 0)


if __name__ == "__main__":
    unittest.main()
