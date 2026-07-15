from __future__ import annotations

import unittest

from overlap_monitor.adapters import MegatronWorkAdapter
from overlap_monitor.core.events import EventType
from overlap_monitor.profiler import WorkHandleRecorder


class FakeWork:
    def wait(self):
        return True


class MegatronAdapterTests(unittest.TestCase):
    def test_attaches_pipeline_context_without_megatron_import(self):
        recorder = WorkHandleRecorder()
        adapter = MegatronWorkAdapter(recorder, rank=3, stage_id=1, device_id=3)
        work = adapter.wrap_a2a(
            FakeWork(),
            comm_id="iter7-mb2-dispatch",
            iteration=7,
            microbatch_id=2,
            phase="dispatch",
        )
        work.wait()

        event = recorder.events()[0]
        self.assertEqual(event.rank, 3)
        self.assertEqual(event.stage_id, 1)
        self.assertEqual(event.metadata["iteration"], 7)
        self.assertEqual(event.metadata["microbatch_id"], 2)

    def test_rejects_unknown_a2a_phase(self):
        adapter = MegatronWorkAdapter(WorkHandleRecorder(), rank=0, stage_id=0)
        with self.assertRaises(ValueError):
            adapter.wrap_a2a(
                FakeWork(), comm_id="x", iteration=0, microbatch_id=0, phase="expert"
            )

    def test_wraps_pipeline_send_with_1f1b_context(self):
        recorder = WorkHandleRecorder()
        adapter = MegatronWorkAdapter(recorder, rank=1, stage_id=2, device_id=1)
        work = adapter.wrap_pipeline(
            FakeWork(),
            comm_id="iter4-mb1-forward-send",
            iteration=4,
            microbatch_id=1,
            direction="send",
            phase="forward",
            peer_rank=2,
        )
        work.wait()

        event = recorder.events()[0]
        self.assertEqual(event.name, "forward_pp_send")
        self.assertEqual(event.metadata["direction"], "send")
        self.assertEqual(event.metadata["peer_rank"], 2)
        self.assertEqual(event.metadata["operation"], "pp_send")

    def test_creates_forward_region(self):
        adapter = MegatronWorkAdapter(
            WorkHandleRecorder(), rank=3, stage_id=1, device_id=3
        )

        event = adapter.region(
            10,
            25,
            iteration=8,
            microbatch_id=2,
            phase="forward",
        )

        self.assertEqual(event.event_type, EventType.PIPELINE)
        self.assertEqual(event.interval, (10, 25))
        self.assertEqual(event.metadata["iteration"], 8)
        self.assertEqual(event.metadata["microbatch_id"], 2)

    def test_rejects_invalid_pipeline_context(self):
        adapter = MegatronWorkAdapter(WorkHandleRecorder(), rank=0, stage_id=0)

        with self.assertRaisesRegex(ValueError, "microbatch_id"):
            adapter.wrap_pipeline(
                FakeWork(),
                comm_id="bad",
                iteration=0,
                microbatch_id=-1,
                direction="send",
                phase="forward",
            )


if __name__ == "__main__":
    unittest.main()
