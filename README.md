# overlap-monitor-v2

Decoupled 1F1B communication-computation overlap monitoring framework for Megatron/MoE experiments.

This package is independent from the original Megatron codebase. It does not modify original benchmark scripts, original tests, or original output formats.

## Installation

Install the CPU-only analysis package:

```bash
python3 -m pip install .
```

For future runtime experiments:

```text
Python >= 3.10
PyTorch with CUDA profiler support
transformer-engine==2.7.0
```

Run the synthetic test suite from a source checkout:

```bash
python3 -m unittest discover -s overlap_monitor_v2/tests -p 'test_*.py'
```

## Architecture

```text
1F1B Execution
  -> Event Collector
  -> Timeline Normalizer
  -> Kernel Classifier
  -> Overlap Analyzer
  -> Metric Reporter
```

Package layout:

```text
overlap_monitor_v2/
├── adapters/
├── core/
├── profiler/
├── analyzer/
├── te_adapter/
├── visualization/
├── tests/
├── configs/
├── benchmarks/
└── tests/
```

Runtime and offline analysis are deliberately separated:

```text
Megatron async Work -> MegatronWorkAdapter -> WorkHandleRecorder
                                         -> MonitoringSession -> events.jsonl
events.jsonl / profiler trace -> validator -> analyzer -> report/trace
```

## Event Model

Events use a common schema:

```text
timestamp_start
timestamp_end
device_id
rank
stage_id
event_type
name
metadata
```

Event types:

```text
COMPUTE
COMMUNICATION
NCCL
GEMM
ATTENTION
MEMORY
WAIT
PIPELINE
UNKNOWN
```

## Overlap Algorithm

v2 keeps two analyzers because Megatron/MoE execution is asynchronous.

### Kernel Timeline Overlap

Given compute intervals `C` and communication intervals `N`, v2 computes:

```text
compute_time = union(C)
communication_time = union(N)
overlap_time = intersection(C, N)
overlap_ratio = overlap_time / min(compute_time, communication_time)
bubble_ratio = idle stage-time / total stage timeline span
stage_balance = min(stage active span) / max(stage active span)
```

This is useful for profiler/Nsight-like timelines, but it does not always equal
the training critical path.

### Critical Path Overlap

For async Megatron communication, prefer:

```text
COMMUNICATION = async Work launch -> completion
WAIT          = Work.wait start -> Work.wait end
```

When an NCCL kernel timeline is available:

```text
communication_runtime = union(NCCL kernels)
hidden_communication = intersection(NCCL, compute)
exposed_communication = communication_runtime - hidden_communication
```

With Work/wait events only:

```text
communication_runtime = union(Work launch -> observed completion)
exposed_communication = intersection(COMMUNICATION, WAIT)
hidden_communication = communication_runtime - exposed_communication
overlap_ratio = hidden_communication / communication_runtime
```

The second path is labeled `estimated/upper_bound` when completion is observed
only after `wait()` returns. Reports never present that window as exact NCCL
kernel runtime.

`profiler.WorkHandleRecorder` records `work.wait()` without forcing a global CUDA
synchronization. `analyzer.CriticalPathOverlapAnalyzer` turns those events into
critical-path metrics.

`adapters.MegatronWorkAdapter` adds rank/stage/iteration/microbatch context while
remaining free of Megatron and torch imports.

## Validation

Validate clock domains and Work correlations before analysis:

```bash
overlap-monitor-v2 validate --input events.jsonl
```

The validator rejects mixed unaligned rank clock domains and orphan WAIT events,
and warns when a Work completion is only an upper-bound observation.

Current local verification is documented in `docs/validation_report.md`. It
covers synthetic correctness and CPU overhead only; a real TE 2.7.0 Megatron GPU
comparison is still required before claiming production accuracy or training
speedup.

Run the bundled CPU microbenchmark with:

```bash
python3 overlap_monitor_v2/benchmarks/benchmark_runtime.py
```

## Transformer Engine 2.7.0 Support

`te_adapter/te_adapter.py` maps TE kernels such as:

```text
transformer_engine::gemm
nvte_grouped_gemm
nvte_fused_attn
```

to:

```text
GEMM
ATTENTION
MEMORY
COMPUTE
```

It also tags precision hints such as `fp8`, `bf16`, and `fp16` when visible in kernel names.

## Visualization

The framework can generate:

```text
Chrome tracing JSON
ASCII timeline
Markdown summary table
```

Chrome trace output is compatible with:

```text
chrome://tracing
```

## Migration From Original Monitor

Original monitor behavior:

```text
PyTorch profiler -> in-process parser -> CSV/JSONL metrics
```

v2 migration path:

```text
PyTorch profiler events
  -> profiler.PyTorchProfilerEventParser
  -> profiler.TimelineNormalizer
  -> analyzer.OverlapAnalyzer
  -> visualization/reporting
```

For async A2A critical-path measurement:

```text
torch.distributed async Work
  -> profiler.WorkHandleRecorder
  -> analyzer.CriticalPathOverlapAnalyzer
  -> exposed/hidden communication metrics
```

Original output fields can be preserved in an adapter, while v2 adds stage-aware metrics and visualization.

## Future GPU Experiment Plan

Target hardware:

```text
8 x A100-40G
```

Target models/runtime:

```text
Mixtral MoE
Qwen MoE
Megatron-Core
Transformer Engine 2.7.0
```

Experiment directions:

```text
EP AllToAll overlap
1F1B pipeline schedule analysis
NCCL kernel timeline analysis
GEMM/attention overlap analysis
pipeline bubble and stage imbalance measurement
```
