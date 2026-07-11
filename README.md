# 1F1B Overlap Monitor

[![tests](https://github.com/sp2022maomao/1F1B_overlap_monitor/actions/workflows/tests.yml/badge.svg)](https://github.com/sp2022maomao/1F1B_overlap_monitor/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`overlap-monitor` is a decoupled 1F1B communication-computation overlap
monitoring framework for Megatron/MoE experiments.

The project is designed for asynchronous training systems where direct Python
timers or single-stream CUDA events can misrepresent real exposed communication.
Its main focus is critical-path overlap analysis for Megatron-style async
`Work.wait()` behavior, while still supporting kernel timeline overlap from
PyTorch profiler or Nsight-like traces.

## Status

Open-source alpha, version `0.3.0`.

The Python event model, analyzers, CLI, adapters, and synthetic tests are
validated locally. Real GPU validation against TE 2.7.0, Megatron, and Nsight
Systems is still required before making production accuracy or speedup claims.

## Why This Exists

In MoE and 1F1B pipeline training, communication may launch asynchronously and
finish later on another stream. Measuring only:

```python
start.record()
dispatch()
end.record()
```

can capture launch overhead or a local stream window instead of exposed
communication on the training critical path.

This project separates three concepts:

```text
communication runtime
hidden communication
exposed communication
```

and labels measurement quality explicitly:

```text
kernel_timeline          exact NCCL/GPU kernel interval is available
estimated/upper_bound    only Work launch/wait observations are available
```

## Installation

Install from a source checkout:

```bash
python3 -m pip install .
```

For editable development:

```bash
python3 -m pip install -e .
```

Optional GPU/runtime integrations may use:

```text
Python >= 3.10
PyTorch with CUDA profiler support
transformer-engine==2.7.0
Nsight Systems traces
```

## Quick Start

Run the synthetic test suite:

```bash
python3 -m unittest discover -s overlap_monitor/tests -p 'test_*.py'
```

Analyze a bundled async critical-path example:

```bash
overlap-monitor analyze \
  --input overlap_monitor/examples/critical_path_events.jsonl \
  --mode critical-path \
  --table \
  --ascii
```

Validate event quality before analysis:

```bash
overlap-monitor validate --input events.jsonl
```

Run the CPU microbenchmark:

```bash
python3 -m overlap_monitor.benchmarks.benchmark_runtime
```

## Migrating from 0.2.x

Version 0.3.0 adopts a version-independent public name. Replace imports and
the CLI command as follows:

```text
overlap_monitor_v2  -> overlap_monitor
overlap-monitor-v2  -> overlap-monitor
```

The event schema and output formats are unchanged.

## Architecture

```text
1F1B Execution
  -> Event Collector
  -> Timeline Normalizer
  -> Kernel Classifier
  -> Overlap Analyzer
  -> Metric Reporter
```

Runtime and offline analysis are deliberately separated:

```text
Megatron async Work -> MegatronWorkAdapter -> WorkHandleRecorder
                                         -> MonitoringSession -> events.jsonl
events.jsonl / profiler trace -> validator -> analyzer -> report/trace
```

Package layout:

```text
overlap_monitor/
├── adapters/
├── analyzer/
├── benchmarks/
├── configs/
├── core/
├── examples/
├── profiler/
├── te_adapter/
├── tests/
└── visualization/
```

Core modules do not import Megatron, PyTorch, Transformer Engine, or Nsight.
Framework integration should happen through thin adapters.

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

## Overlap Algorithms

### Kernel Timeline Overlap

Given compute intervals `C` and communication intervals `N`, the analyzer
computes:

```text
compute_time = union(C)
communication_time = union(N)
overlap_time = intersection(C, N)
overlap_ratio = overlap_time / min(compute_time, communication_time)
bubble_ratio = idle stage-time / total stage timeline span
stage_balance = min(stage active span) / max(stage active span)
```

This is useful for profiler/Nsight timelines, but it does not always equal the
training critical path.

### Critical-Path Overlap

For async Megatron communication:

```text
COMMUNICATION = async Work launch -> completion
WAIT          = Work.wait start -> Work.wait end
```

When NCCL kernel intervals are available:

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

The Work-only path is reported as an estimated upper bound when true completion
is only observed after `wait()` returns.

## Public API

```python
from overlap_monitor import (
    CriticalPathOverlapAnalyzer,
    Event,
    EventType,
    MegatronWorkAdapter,
    MonitoringSession,
    OverlapAnalyzer,
    WorkHandleRecorder,
)
```

Example Work-handle wrapping:

```python
recorder = WorkHandleRecorder()
work = torch.distributed.all_to_all_single(..., async_op=True)
work = recorder.wrap(
    work,
    comm_id="mb0_dispatch",
    name="dispatch_a2a",
    stage_id=0,
    microbatch_id=0,
    phase="dispatch",
)
work.wait()
events = recorder.events()
```

`WorkHandleRecorder` does not call global CUDA synchronize.

## Transformer Engine 2.7.0 Support

`te_adapter/te_adapter.py` maps TE kernel names such as:

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

It also tags visible precision hints such as `fp8`, `bf16`, and `fp16`.

## Visualization

The framework can generate:

```text
Chrome tracing JSON
ASCII timeline
Markdown summary table
```

Chrome trace output can be opened with `chrome://tracing`.

## Documentation

- [Async measurement design](docs/async_measurement_design.md)
- [Industrial reusable module design](docs/industrial_reusable_module_design.md)
- [Validation report](docs/validation_report.md)
- [Similar projects report](docs/similar_projects_report.md)
- [Roadmap](ROADMAP.md)
- [Contributing guide](CONTRIBUTING.md)

## Similar Tools

This project is not a replacement for PyTorch profiler, Kineto, Nsight Systems,
Megatron-Core, or DeepEP. Those tools are upstream runtimes or full profiler
systems. `overlap-monitor` is a small analysis library focused on
stage-aware 1F1B critical-path overlap metrics and async Work/wait measurement
semantics.

See `docs/similar_projects_report.md` for the comparison.

## License

MIT License. See [LICENSE](LICENSE).
