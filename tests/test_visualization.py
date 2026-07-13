from __future__ import annotations

import unittest

from overlap_monitor.analyzer import OverlapAnalyzer
from overlap_monitor.core.events import Event, EventType
from overlap_monitor.visualization import (
    render_ascii_timeline,
    render_summary_table,
    to_chrome_trace,
)


class VisualizationTests(unittest.TestCase):
    def test_chrome_trace(self):
        trace = to_chrome_trace([Event(0, 10, EventType.GEMM, name="gemm", rank=0, stage_id=1)])
        self.assertEqual(trace["traceEvents"][0]["ph"], "X")
        self.assertEqual(trace["traceEvents"][0]["cat"], "GEMM")

    def test_ascii_timeline(self):
        text = render_ascii_timeline([Event(0, 10, EventType.NCCL, stage_id=0)], width=10)
        self.assertIn("Stage0", text)
        self.assertIn("N", text)

    def test_summary_table_names_ratio_denominator(self):
        summary = OverlapAnalyzer().analyze(
            [Event(0, 10, EventType.GEMM), Event(5, 15, EventType.NCCL)]
        )

        text = render_summary_table(summary)

        self.assertIn("overlap_ratio_definition", text)
        self.assertIn("min(compute_time, communication_time)", text)


if __name__ == "__main__":
    unittest.main()
