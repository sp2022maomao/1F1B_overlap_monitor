from __future__ import annotations

from typing import Any

from overlap_monitor_v2.profiler.work_handle import ObservedWork, WorkHandleRecorder


class MegatronWorkAdapter:
    """Thin metadata adapter for Megatron async communication handles.

    This module deliberately does not import Megatron or torch. The training
    integration owns Work creation; the adapter only attaches stable context.
    """

    def __init__(
        self,
        recorder: WorkHandleRecorder,
        *,
        rank: int,
        stage_id: int,
        device_id: int | None = None,
    ):
        self.recorder = recorder
        self.rank = rank
        self.stage_id = stage_id
        self.device_id = device_id

    def wrap_a2a(
        self,
        work: Any,
        *,
        comm_id: str,
        iteration: int,
        microbatch_id: int,
        phase: str,
        **metadata: Any,
    ) -> ObservedWork:
        if phase not in {"dispatch", "combine"}:
            raise ValueError("A2A phase must be 'dispatch' or 'combine'")
        return self.recorder.wrap(
            work,
            comm_id=comm_id,
            name=f"{phase}_a2a",
            rank=self.rank,
            stage_id=self.stage_id,
            device_id=self.device_id,
            iteration=iteration,
            microbatch_id=microbatch_id,
            phase=phase,
            **metadata,
        )
