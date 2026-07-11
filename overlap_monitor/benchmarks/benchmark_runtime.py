from __future__ import annotations

import argparse
import json
from statistics import median
from time import perf_counter_ns

from overlap_monitor.analyzer import OverlapAnalyzer
from overlap_monitor.core.events import Event, EventType
from overlap_monitor.profiler import WorkHandleRecorder


class ImmediateWork:
    def wait(self) -> bool:
        return True


def ns_per_call(operation, iterations: int, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        for _ in range(iterations):
            operation()
        samples.append((perf_counter_ns() - start) / iterations)
    return median(samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU microbenchmark for overlap_monitor")
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    work = ImmediateWork()
    bare_wait = ns_per_call(work.wait, args.iterations, args.repeats)

    recorder = WorkHandleRecorder()
    counter = 0

    def recorded_wait() -> None:
        nonlocal counter
        observed = recorder.wrap(work, comm_id=str(counter), name="a2a")
        observed.wait()
        counter += 1

    recorded = ns_per_call(recorded_wait, args.iterations, args.repeats)

    events = [
        Event(float(i), float(i + 2), EventType.GEMM if i % 2 == 0 else EventType.NCCL)
        for i in range(10_000)
    ]
    analyzer = OverlapAnalyzer()
    analyzer_ns = ns_per_call(lambda: analyzer.analyze(events), 20, args.repeats)

    result = {
        "iterations": args.iterations,
        "repeats": args.repeats,
        "bare_wait_ns_per_call": bare_wait,
        "recorded_wrap_wait_ns_per_call": recorded,
        "recorder_overhead_ns_per_call": max(recorded - bare_wait, 0.0),
        "analyzer_10000_events_ms": analyzer_ns / 1_000_000.0,
        "scope": "local CPU-only microbenchmark; excludes CUDA profiler and GPU overhead",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
