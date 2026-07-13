from __future__ import annotations

import ctypes
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class CuptiRuntimeError(RuntimeError):
    pass


class CuptiRuntimeCollector:
    """Thin ctypes wrapper for the optional native CUPTI collector.

    The wrapper does not import torch and never synchronizes CUDA. Callers must
    ensure GPU work is complete before stop() when a complete trace is needed.
    """

    def __init__(self, library_path: Path | str | None = None, *, library: Any = None):
        if library is None:
            if library_path is None:
                raise ValueError("library_path is required")
            library = ctypes.CDLL(str(library_path))
        self._library = library
        self._configure_signatures()
        self._started = False

    def start(self, output_path: Path | str, *, rank: int = -1) -> None:
        if self._started:
            raise CuptiRuntimeError("CUPTI collector is already started")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = self._library.overlap_cupti_start(
            str(output_path).encode("utf-8"), int(rank)
        )
        self._check(result, "start")
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        result = self._library.overlap_cupti_stop()
        self._started = False
        self._check(result, "stop")

    def push_external_id(self, external_id: int) -> None:
        self._require_started()
        self._check(
            self._library.overlap_cupti_push_external_id(int(external_id)),
            "push_external_id",
        )

    def pop_external_id(self) -> int:
        self._require_started()
        value = ctypes.c_uint64()
        self._check(
            self._library.overlap_cupti_pop_external_id(ctypes.byref(value)),
            "pop_external_id",
        )
        return int(value.value)

    @contextmanager
    def external_range(self, external_id: int) -> Iterator[None]:
        self.push_external_id(external_id)
        try:
            yield
        finally:
            popped = self.pop_external_id()
            if popped != external_id:
                raise CuptiRuntimeError(
                    f"external correlation stack mismatch: expected {external_id}, got {popped}"
                )

    def __enter__(self) -> "CuptiRuntimeCollector":
        if not self._started:
            raise CuptiRuntimeError(
                "call start() before entering the collector context"
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def _configure_signatures(self) -> None:
        functions = (
            "overlap_cupti_start",
            "overlap_cupti_stop",
            "overlap_cupti_push_external_id",
            "overlap_cupti_pop_external_id",
        )
        missing = [name for name in functions if not hasattr(self._library, name)]
        if missing:
            raise CuptiRuntimeError(
                "native CUPTI library is missing symbols: " + ", ".join(missing)
            )
        try:
            self._library.overlap_cupti_start.argtypes = [ctypes.c_char_p, ctypes.c_int]
            self._library.overlap_cupti_start.restype = ctypes.c_int
            self._library.overlap_cupti_stop.argtypes = []
            self._library.overlap_cupti_stop.restype = ctypes.c_int
            self._library.overlap_cupti_push_external_id.argtypes = [ctypes.c_uint64]
            self._library.overlap_cupti_push_external_id.restype = ctypes.c_int
            self._library.overlap_cupti_pop_external_id.argtypes = [
                ctypes.POINTER(ctypes.c_uint64)
            ]
            self._library.overlap_cupti_pop_external_id.restype = ctypes.c_int
        except AttributeError:
            # Test doubles expose callables without ctypes signature attributes.
            pass

    def _require_started(self) -> None:
        if not self._started:
            raise CuptiRuntimeError("CUPTI collector is not started")

    def _check(self, result: int, operation: str) -> None:
        if result != 0:
            raise CuptiRuntimeError(
                f"CUPTI collector {operation} failed with code {result}"
            )
