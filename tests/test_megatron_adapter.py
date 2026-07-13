from __future__ import annotations

import unittest

from overlap_monitor.adapters import MegatronWorkAdapter
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


if __name__ == "__main__":
    unittest.main()
