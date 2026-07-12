# CUPTI Measurement Path

The optional CUPTI path records GPU kernel execution intervals directly and
feeds them into the existing timeline analyzers. It is decoupled from Megatron,
PyTorch, and the Work-handle recorder.

## Architecture

```text
Megatron / PyTorch workload
  -> optional external correlation ranges
  -> native CUPTI Activity collector
       -> concurrent kernel records (start/end/stream/correlation)
       -> dropped-record accounting
  -> CUPTI activity JSONL
  -> CuptiActivityParser
  -> overlap-monitor Event JSONL
  -> OverlapAnalyzer / CriticalPathOverlapAnalyzer
```

The native callback only copies completed activity records into a bounded
in-memory buffer. Kernel classification, JSON parsing, and overlap calculation
run after collection.

## What It Measures

CUPTI kernel records provide nanosecond timestamps for observed GPU execution:

```text
communication_runtime = duration(union(classified NCCL kernels))
compute_runtime       = duration(union(classified compute kernels))
overlap_time          = duration(intersection(NCCL, compute))
```

The parser converts timestamps to microseconds and adds:

```json
{
  "collector": "cupti",
  "measurement": "kernel_timeline",
  "runtime_kind": "observed_kernel_runtime",
  "stream_id": 11,
  "correlation_id": 101,
  "source_timestamp_unit": "ns"
}
```

This path replaces the Work/wait completion estimate for kernel-runtime and
timeline-overlap metrics. It does not prove that every uncovered communication
interval is on the end-to-end training critical path. That attribution still
requires iteration, stage, microbatch, phase, and dependency context.

## Build

Requirements:

```text
Linux
CMake >= 3.20
CUDA Toolkit with CUPTI headers and libcupti (target: CUDA 12.9)
```

Build the optional shared library:

```bash
cmake -S native/cupti_collector -B build/cupti -DCMAKE_BUILD_TYPE=Release
cmake --build build/cupti --parallel
```

The output is normally:

```text
build/cupti/liboverlap_cupti.so
```

The Python package does not link against CUPTI. Systems without CUDA can still
install and test all offline analysis modules.

## Runtime Collection

Load the library explicitly and bound the profiling window:

```python
from pathlib import Path

from overlap_monitor import CuptiRuntimeCollector

collector = CuptiRuntimeCollector("build/cupti/liboverlap_cupti.so")
collector.start(Path("cupti_rank0.jsonl"), rank=0)

with collector.external_range(1001):
    run_profiled_steps()

# Complete outstanding GPU work before stop(). The collector does not add an
# implicit CUDA synchronization because that would change the measured path.
torch.cuda.synchronize()
collector.stop()
```

External IDs are user-defined. A Megatron integration can map them to
iteration/stage/microbatch/phase metadata in a sidecar file. CUPTI correlates
the external ID with CUDA API and kernel correlation IDs.

The collector defaults to 2,097,152 in-memory records. Override the bound with:

```bash
export OVERLAP_CUPTI_MAX_RECORDS=4194304
```

Use a short, representative profiling window rather than tracing an entire
training run.

## Import and Analyze

Convert native records into the stable event schema:

```bash
overlap-monitor import-cupti \
  --input cupti_rank0.jsonl \
  --output events_rank0.jsonl \
  --rank 0 \
  --stage-id 0
```

Analyze observed kernel overlap:

```bash
overlap-monitor analyze \
  --input events_rank0.jsonl \
  --mode timeline \
  --output-json summary_rank0.json
```

The importer rejects traces containing dropped records or unavailable kernel
timestamps. `--allow-incomplete` is available for debugging, but results from
an incomplete trace must not be used for accuracy claims.

## Raw JSONL Schema

Kernel record:

```json
{
  "schema_version": 1,
  "record_kind": "kernel",
  "start_ns": 1000,
  "end_ns": 5000,
  "name": "ncclDevKernel_AllToAll",
  "device_id": 0,
  "stream_id": 11,
  "correlation_id": 101,
  "process_id": 1234,
  "rank": 0
}
```

External correlation record:

```json
{
  "schema_version": 1,
  "record_kind": "external_correlation",
  "correlation_id": 101,
  "external_id": 1001,
  "external_kind": "custom0"
}
```

Collector summary:

```json
{
  "schema_version": 1,
  "record_kind": "collector_summary",
  "dropped_records": 0,
  "cupti_dropped_records": 0,
  "client_dropped_records": 0
}
```

## Validation Protocol

Local tests prove schema validation, timestamp conversion, external
correlation, dropped-record rejection, and integration with the interval
analyzers. They do not prove native collector compatibility or GPU accuracy.

GPU validation should use the same bounded workload in four modes:

1. monitor off;
2. Work-only recorder;
3. native CUPTI collector;
4. Nsight Systems reference trace.

Record at least iteration p50/p95, tokens/s, kernel count, dropped records,
NCCL duration, compute duration, and overlap duration. Suggested acceptance
gates are less than 5% interval/overlap error against Nsight and less than 2%
throughput loss for the CUPTI-on run. These are project targets, not currently
validated claims.

## References

- [CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)
- [CUPTI kernel activity record](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel10.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
