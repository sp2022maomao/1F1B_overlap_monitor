from __future__ import annotations

import unittest

from overlap_monitor.analyzer import OverlapAnalyzer
from overlap_monitor.core.events import Event, EventType


class OverlapAlgorithmTests(unittest.TestCase):
    def test_simple_overlap(self):
        events = [
            Event(0, 10, EventType.GEMM, stage_id=0),
            Event(5, 15, EventType.NCCL, stage_id=0),
        ]
        summary = OverlapAnalyzer().analyze(events)
        self.assertEqual(summary.compute_time, 10)
        self.assertEqual(summary.communication_time, 10)
        self.assertEqual(summary.overlap_time, 5)
        self.assertEqual(summary.overlap_ratio, 0.5)
        self.assertEqual(
            summary.to_dict()["overlap_ratio_definition"],
            "overlap_time / min(compute_time, communication_time)",
        )

    def test_stage_balance_and_bubble(self):
        events = [
            Event(0, 10, EventType.GEMM, stage_id=0),
            Event(20, 30, EventType.NCCL, stage_id=0),
            Event(0, 40, EventType.GEMM, stage_id=1),
        ]
        summary = OverlapAnalyzer().analyze(events)
        self.assertAlmostEqual(summary.bubble_ratio, 10 / 70)
        self.assertAlmostEqual(summary.stage_metrics[0].bubble_ratio, 10 / 30)
        self.assertAlmostEqual(summary.stage_balance, 0.75)


if __name__ == "__main__":
    unittest.main()
