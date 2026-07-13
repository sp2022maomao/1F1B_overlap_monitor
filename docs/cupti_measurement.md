# CUPTI Measurement

The optional native collector records GPU kernel intervals and feeds them into
the same analyzers used by normalized profiler events. It is isolated from
Megatron, PyTorch, and the Work/wait adapter.

## Profiler Granularity

| Tool | Data granularity | Main role |
| --- | --- | --- |
| CUPTI | Low-level CUDA activity and kernel records | Capture raw NCCL and compute kernel intervals |
| Nsight Systems | System-level CPU/GPU/stream timeline | Inspect and validate cross-stream concurrency |
| PyTorch Profiler | Framework operator/module/autograd level | Connect framework operations to CUDA activity |

CUPTI is the lowest-level collection interface. Nsight Systems provides an
integrated system view, while PyTorch Profiler preserves framework semantics.
The monitor normalizes all supported sources but keeps their evidence quality
explicit.

## Data Flow

```text
Megatron or PyTorch workload
  -> native CUPTI Activity collector
  -> raw CUPTI JSONL
  -> parser and kernel classifier
  -> normalized Event JSONL (microseconds)
  -> validation
  -> overlap analyzer
```

The callback copies completed activity records into a bounded in-memory buffer.
Kernel classification and overlap calculation happen after collection.

## Requirements

```text
Linux
CMake >= 3.20
CUDA Toolkit with CUPTI headers and libcupti
```

The target deployment is CUDA 12.9. The Python package itself does not link
against CUPTI and remains usable on systems without CUDA.

## Build

```bash
cmake -S native/cupti_collector -B build/cupti -DCMAKE_BUILD_TYPE=Release
cmake --build build/cupti --parallel
```

Expected output:

```text
build/cupti/liboverlap_cupti.so
```

## Capture

Keep the profiling window short and representative:

```python
from pathlib import Path

import torch

from overlap_monitor import CuptiRuntimeCollector

collector = CuptiRuntimeCollector("build/cupti/liboverlap_cupti.so")
collector.start(Path("cupti_rank0.jsonl"), rank=0)

with collector.external_range(1001):
    run_profiled_steps()

# The caller owns synchronization. The collector does not alter training order.
torch.cuda.synchronize()
collector.stop()
```

External IDs can represent an iteration, stage, microbatch, or phase. Their
meaning should be recorded in experiment metadata or a sidecar file.

The default in-memory limit is 2,097,152 records:

```bash
export OVERLAP_CUPTI_MAX_RECORDS=4194304
```

Increase it only when the bounded trace reports dropped client records.

## Analyze

The normal workflow is one command:

```bash
overlap-monitor analyze \
  --input cupti_rank0.jsonl \
  --rank 0 \
  --stage-id 0 \
  --events-output events_rank0.jsonl \
  --output-json summary_rank0.json \
  --trace-json timeline_rank0.json \
  --table
```

The CLI auto-detects CUPTI records, converts nanoseconds to microseconds,
validates the trace, and runs the critical-path analyzer. `--events-output` is
optional and is useful for reproducibility.

For the legacy two-step workflow:

```bash
overlap-monitor import-cupti \
  --input cupti_rank0.jsonl \
  --output events_rank0.jsonl \
  --rank 0 \
  --stage-id 0

overlap-monitor analyze --input events_rank0.jsonl --table
```

Incomplete traces are rejected by default. `--allow-incomplete` is for parser
debugging only and invalidates accuracy claims.

## Metrics

Normalized CUPTI timestamps use microseconds (`us`):

```text
communication_runtime = duration(union(NCCL kernels))
compute_time           = duration(union(compute kernels))
hidden_communication   = duration(intersection(NCCL, compute))
exposed_communication  = communication_runtime - hidden_communication
overlap_ratio          = hidden_communication / communication_runtime
```

These metrics measure observed kernel concurrency. They do not prove that every
uncovered NCCL interval delays the end-to-end iteration; that attribution also
requires stage, microbatch, phase, and dependency context.

## Raw Record Schema

Kernel record:

```json
{"schema_version":1,"record_kind":"kernel","start_ns":1000,"end_ns":5000,"name":"ncclDevKernel_AllToAll","device_id":0,"stream_id":11,"correlation_id":101,"process_id":1234,"rank":0}
```

External correlation record:

```json
{"schema_version":1,"record_kind":"external_correlation","correlation_id":101,"external_id":1001,"external_kind":"custom0"}
```

The collector ends the file with a summary containing CUPTI and client dropped
record counts. The parser requires zero dropped records for a complete trace.

## Validation Checklist

Compare the same bounded workload in four modes:

1. Monitor off.
2. Work/wait adapter only.
3. Native CUPTI collector.
4. Nsight Systems reference trace.

Record iteration p50/p95, throughput, kernel count, dropped records, NCCL time,
compute time, and overlap time. Current project targets are less than 5% metric
error against Nsight and less than 2% throughput loss. These are acceptance
targets, not validated production claims.

Current GPU evidence is documented in
[validation.md](validation.md). CUDA 12.9,
Transformer Engine 2.7, real Megatron 1F1B, and Nsight comparison remain open.

## References

- [CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)
- [CUPTI kernel activity record](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel10.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
