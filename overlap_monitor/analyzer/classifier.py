from __future__ import annotations

from overlap_monitor.core.events import Event, EventType
from overlap_monitor.te_adapter import TEAdapter


class KernelClassifier:
    def __init__(self, te_adapter: TEAdapter | None = None):
        self.te_adapter = te_adapter or TEAdapter()

    def classify_name(self, name: str) -> tuple[EventType, dict]:
        lowered = name.lower()
        metadata = {}
        if "nccl" in lowered:
            metadata["communication_library"] = "nccl"
            if "alltoall" in lowered.replace("_", "") or "all2all" in lowered.replace("_", ""):
                metadata["collective"] = "all_to_all"
            elif "allreduce" in lowered.replace("_", ""):
                metadata["collective"] = "all_reduce"
            return EventType.NCCL, metadata

        te_info = self.te_adapter.classify_kernel(name)
        if te_info.event_type is not EventType.UNKNOWN:
            if te_info.precision:
                metadata["precision"] = te_info.precision
            if te_info.region:
                metadata["region"] = te_info.region
            metadata["transformer_engine"] = te_info.is_transformer_engine
            return te_info.event_type, metadata

        if any(marker in lowered for marker in ("gemm", "cublas", "cutlass", "matmul")):
            return EventType.GEMM, metadata
        if "attention" in lowered or "flash_attn" in lowered:
            return EventType.ATTENTION, metadata
        if any(marker in lowered for marker in ("memcpy", "memset", "transpose", "permute")):
            return EventType.MEMORY, metadata
        return EventType.UNKNOWN, metadata

    def classify_event(self, event: Event) -> Event:
        event_type, metadata = self.classify_name(event.name)
        if event_type is EventType.NCCL:
            # NCCL is both a concrete NCCL kernel and communication.
            metadata.setdefault("event_family", EventType.COMMUNICATION.value)
        elif event_type in {EventType.GEMM, EventType.ATTENTION}:
            metadata.setdefault("event_family", EventType.COMPUTE.value)
        return event.with_type(event_type, **metadata)

    def classify_events(self, events: list[Event]) -> list[Event]:
        return [self.classify_event(event) for event in events]
