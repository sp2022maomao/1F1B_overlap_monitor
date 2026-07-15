from __future__ import annotations

import unittest
from pathlib import Path

from overlap_monitor import (
    Event,
    EventType,
    TraceValidationError,
    analyze_events,
    analyze_trace,
    load_trace,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "traces"


class PublicApiTests(unittest.TestCase):
    def test_load_trace_auto_detects_and_filters_cupti(self):
        events = load_trace(
            EXAMPLES / "cupti_activity.jsonl",
            rank=0,
            stage_id=0,
        )

        self.assertEqual(len(events), 2)
        self.assertTrue(all(event.rank == 0 for event in events))
        self.assertTrue(all(event.stage_id == 0 for event in events))

    def test_analyze_trace_returns_stable_result(self):
        result = analyze_trace(
            EXAMPLES / "cupti_activity.jsonl",
            rank=0,
            stage_id=0,
        )
        payload = result.to_dict()

        self.assertEqual(result.input_format, "cupti")
        self.assertEqual(result.communication_hidden_ratio, 0.5)
        self.assertIsNone(result.timeline_overlap_ratio)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["communication_hidden_ratio"], 0.5)
        self.assertEqual(payload["overlap_ratio"], 0.5)
        self.assertFalse(payload["clock_alignment_assumed"])

    def test_timeline_mode_has_explicit_ratio_name(self):
        result = analyze_events(
            [
                Event(0, 10, EventType.GEMM, rank=0),
                Event(5, 15, EventType.NCCL, rank=0),
            ],
            mode="timeline",
        )

        self.assertEqual(result.timeline_overlap_ratio, 0.5)
        self.assertIsNone(result.communication_hidden_ratio)
        self.assertEqual(result.to_dict()["timeline_overlap_ratio"], 0.5)
        self.assertEqual(result.summary.overlap_ratio, 0.5)

    def test_mixed_clocks_require_explicit_assumption(self):
        events = [
            Event(0, 10, EventType.GEMM, rank=0),
            Event(5, 15, EventType.NCCL, rank=1),
        ]

        with self.assertRaises(TraceValidationError):
            analyze_events(events)

        result = analyze_events(events, assume_aligned_clocks=True)
        self.assertTrue(result.clock_alignment_assumed)
        self.assertEqual(len(result.validation.clock_domains), 2)


if __name__ == "__main__":
    unittest.main()
