from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata

from overlap_monitor.core.events import EventType


@dataclass(frozen=True)
class TEKernelInfo:
    event_type: EventType
    precision: str | None = None
    region: str | None = None
    is_transformer_engine: bool = False


class TEAdapter:
    """Transformer Engine kernel classifier.

    The adapter is intentionally string-based so tests and offline parsing work
    without importing Transformer Engine. Runtime version detection is optional.
    """

    GEMM_MARKERS = (
        "transformer_engine::gemm",
        "te_gemm",
        "nvte_gemm",
        "nvte_grouped_gemm",
        "grouped_gemm",
        "fp8_gemm",
        "bf16_gemm",
        "cublaslt",
    )
    ATTENTION_MARKERS = (
        "attention",
        "flash_attn",
        "fused_attn",
        "nvte_fused_attn",
        "dot_product_attention",
    )
    MEMORY_MARKERS = (
        "cast_transpose",
        "transpose",
        "permute",
        "memcpy",
        "memset",
    )

    def detect_version(self) -> str | None:
        for package in ("transformer-engine", "transformer_engine"):
            try:
                return metadata.version(package)
            except metadata.PackageNotFoundError:
                continue
        return None

    def is_te_27_compatible(self, version: str | None = None) -> bool:
        value = version if version is not None else self.detect_version()
        if not value:
            return False
        parts = []
        for token in value.replace("-", ".").split(".")[:2]:
            try:
                parts.append(int(token))
            except ValueError:
                parts.append(0)
        while len(parts) < 2:
            parts.append(0)
        return tuple(parts) >= (2, 7)

    def classify_kernel(self, name: str) -> TEKernelInfo:
        lowered = name.lower()
        is_te = "transformer_engine" in lowered or "nvte" in lowered or "te_" in lowered
        precision = None
        if "fp8" in lowered:
            precision = "fp8"
        elif "bf16" in lowered or "bfloat16" in lowered:
            precision = "bf16"
        elif "fp16" in lowered or "half" in lowered:
            precision = "fp16"

        if any(marker in lowered for marker in self.GEMM_MARKERS):
            return TEKernelInfo(EventType.GEMM, precision, "gemm", is_te)
        if any(marker in lowered for marker in self.ATTENTION_MARKERS):
            return TEKernelInfo(EventType.ATTENTION, precision, "attention", is_te)
        if any(marker in lowered for marker in self.MEMORY_MARKERS):
            return TEKernelInfo(EventType.MEMORY, precision, "memory", is_te)
        if is_te:
            return TEKernelInfo(EventType.COMPUTE, precision, "transformer_engine", is_te)
        return TEKernelInfo(EventType.UNKNOWN)
