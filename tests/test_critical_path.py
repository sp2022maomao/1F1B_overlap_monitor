import unittest

from overlap_monitor.analyzer.critical_path import CriticalPathOverlapAnalyzer
from overlap_monitor.core.events import Event, EventType
from overlap_monitor.profiler.work_handle import WorkHandleRecorder


class FakeWork:
    def wait(self):
        return "done"


class ManualClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0)


class CriticalPathAnalyzerTests(unittest.TestCase):
    def test_wait_marks_exposed_communication(self):
        events = [
            Event(0, 100, EventType.GEMM, stage_id=0, metadata={"microbatch_id": 0}),
            Event(
                10,
                50,
                EventType.COMMUNICATION,
                name="dispatch_a2a",
                stage_id=0,
                metadata={"microbatch_id": 0, "phase": "dispatch"},
            ),
            Event(
                45,
                50,
                EventType.WAIT,
                name="dispatch_a2a.wait",
                stage_id=0,
                metadata={"microbatch_id": 0, "phase": "dispatch"},
            ),
        ]

        summary = CriticalPathOverlapAnalyzer().analyze(events)

        self.assertEqual(summary.communication_runtime, 40)
        self.assertEqual(summary.exposed_communication, 5)
        self.assertEqual(summary.hidden_communication, 35)
        self.assertAlmostEqual(summary.overlap_ratio, 35 / 40)
        self.assertEqual(summary.measurement_quality, "estimated")

    def test_falls_back_to_compute_intersection_without_wait_events(self):
        events = [
            Event(0, 10, EventType.GEMM),
            Event(5, 15, EventType.COMMUNICATION, name="combine_a2a"),
        ]

        summary = CriticalPathOverlapAnalyzer().analyze(events)

        self.assertEqual(summary.communication_runtime, 10)
        self.assertEqual(summary.hidden_communication, 5)
        self.assertEqual(summary.exposed_communication, 5)
        self.assertEqual(summary.overlap_ratio, 0.5)

    def test_work_handle_recorder_emits_lifetime_and_wait_events(self):
        clock = ManualClock([10, 45, 50])
        recorder = WorkHandleRecorder(time_source=clock)
        work = recorder.wrap(
            FakeWork(),
            comm_id="mb0_dispatch",
            name="dispatch_a2a",
            stage_id=1,
            microbatch_id=0,
            phase="dispatch",
        )

        self.assertEqual(work.wait(), "done")
        events = recorder.events()

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, EventType.COMMUNICATION)
        self.assertEqual(events[0].interval, (10, 50))
        self.assertEqual(events[1].event_type, EventType.WAIT)
        self.assertEqual(events[1].interval, (45, 50))
        self.assertEqual(events[0].metadata["runtime_kind"], "host_wait_proxy")
        self.assertFalse(events[0].metadata["completion_observed"])
        self.assertNotIn("_time_source", events[0].metadata)

        self.assertEqual(recorder.clear(), 1)
        self.assertEqual(recorder.events(), [])

    def test_explicit_completion_avoids_late_wait_overestimate(self):
        clock = ManualClock([10, 45, 45])
        recorder = WorkHandleRecorder(time_source=clock)
        work = recorder.wrap(FakeWork(), comm_id="dispatch", name="dispatch_a2a")
        work.mark_completed(30, source="profiler_callback")
        work.wait()

        events = recorder.events()

        self.assertEqual(events[0].interval, (10, 30))
        self.assertEqual(events[0].metadata["runtime_kind"], "observed_work_window")
        self.assertEqual(events[0].metadata["completion_source"], "profiler_callback")

    def test_nccl_timeline_is_preferred_over_legacy_work_upper_bound(self):
        events = [
            Event(0, 100, EventType.GEMM),
            Event(
                10,
                90,
                EventType.COMMUNICATION,
                metadata={
                    "measurement": "async_work_lifetime",
                    "runtime_kind": "upper_bound",
                },
            ),
            Event(20, 50, EventType.NCCL, metadata={"measurement": "kernel_timeline"}),
            Event(80, 90, EventType.WAIT),
        ]

        summary = CriticalPathOverlapAnalyzer().analyze(events)

        self.assertEqual(summary.communication_runtime, 30)
        self.assertEqual(summary.hidden_communication, 30)
        self.assertEqual(summary.exposed_communication, 0)
        self.assertEqual(summary.wait_time, 10)
        self.assertEqual(summary.measurement_quality, "kernel_timeline")
        self.assertEqual(summary.communication_runtime_kind, "observed_kernel_runtime")


if __name__ == "__main__":
    unittest.main()
