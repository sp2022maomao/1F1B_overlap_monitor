from __future__ import annotations

from typing import Any

from overlap_monitor.core.events import Event, EventType
from overlap_monitor.profiler.work_handle import ObservedWork, WorkHandleRecorder


class MegatronWorkAdapter:
    """Thin metadata adapter for Megatron async communication handles.

    This module deliberately does not import Megatron or torch. The training
    integration owns Work creation; the adapter only attaches stable context.
    """

    A2A_PHASES = {"dispatch", "combine"}
    PIPELINE_DIRECTIONS = {"send", "recv"}
    PIPELINE_PHASES = {"forward", "backward"}
    REGION_PHASES = {"iteration", "forward", "backward", "expert", "drain"}
    STANDARD_PHASES = (
        A2A_PHASES | PIPELINE_PHASES | REGION_PHASES | {"send", "recv"}
    )

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
        if phase not in self.A2A_PHASES:
            raise ValueError("A2A phase must be 'dispatch' or 'combine'")
        return self.wrap_communication(
            work,
            comm_id=comm_id,
            operation="a2a",
            iteration=iteration,
            microbatch_id=microbatch_id,
            phase=phase,
            **metadata,
        )

    def wrap_pipeline(
        self,
        work: Any,
        *,
        comm_id: str,
        iteration: int,
        microbatch_id: int,
        direction: str,
        phase: str,
        peer_rank: int | None = None,
        **metadata: Any,
    ) -> ObservedWork:
        """Attach 1F1B context to an async pipeline send or receive Work."""

        if direction not in self.PIPELINE_DIRECTIONS:
            raise ValueError("pipeline direction must be 'send' or 'recv'")
        if phase not in self.PIPELINE_PHASES:
            raise ValueError("pipeline phase must be 'forward' or 'backward'")
        return self.wrap_communication(
            work,
            comm_id=comm_id,
            operation=f"pp_{direction}",
            iteration=iteration,
            microbatch_id=microbatch_id,
            phase=phase,
            direction=direction,
            peer_rank=peer_rank,
            **metadata,
        )

    def wrap_communication(
        self,
        work: Any,
        *,
        comm_id: str,
        operation: str,
        iteration: int,
        microbatch_id: int,
        phase: str,
        **metadata: Any,
    ) -> ObservedWork:
        """Attach stable Megatron context to any async communication Work."""

        self._validate_context(iteration, microbatch_id)
        if phase not in self.STANDARD_PHASES:
            allowed = ", ".join(sorted(self.STANDARD_PHASES))
            raise ValueError(f"communication phase must be one of: {allowed}")
        if not operation:
            raise ValueError("communication operation must not be empty")
        return self.recorder.wrap(
            work,
            comm_id=comm_id,
            name=f"{phase}_{operation}",
            rank=self.rank,
            stage_id=self.stage_id,
            device_id=self.device_id,
            iteration=iteration,
            microbatch_id=microbatch_id,
            phase=phase,
            operation=operation,
            **metadata,
        )

    def region(
        self,
        timestamp_start: float,
        timestamp_end: float,
        *,
        iteration: int,
        phase: str,
        microbatch_id: int | None = None,
        name: str | None = None,
        event_type: EventType = EventType.PIPELINE,
        **metadata: Any,
    ) -> Event:
        """Create a semantic 1F1B iteration or execution-region event."""

        if phase not in self.REGION_PHASES:
            allowed = ", ".join(sorted(self.REGION_PHASES))
            raise ValueError(f"region phase must be one of: {allowed}")
        if microbatch_id is None:
            if iteration < 0:
                raise ValueError("iteration must be non-negative")
        else:
            self._validate_context(iteration, microbatch_id)
        context = {
            "iteration": iteration,
            "phase": phase,
            **metadata,
        }
        if microbatch_id is not None:
            context["microbatch_id"] = microbatch_id
        return Event(
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            event_type=event_type,
            name=name or f"{phase}_region",
            rank=self.rank,
            stage_id=self.stage_id,
            device_id=self.device_id,
            metadata=context,
        )

    @staticmethod
    def _validate_context(iteration: int, microbatch_id: int) -> None:
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        if microbatch_id < 0:
            raise ValueError("microbatch_id must be non-negative")
