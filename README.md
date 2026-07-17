# 1F1B Overlap Monitor

[![tests](https://github.com/sp2022maomao/1F1B_overlap_monitor/actions/workflows/tests.yml/badge.svg)](https://github.com/sp2022maomao/1F1B_overlap_monitor/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

`overlap-monitor` analyzes communication-computation overlap in asynchronous
Megatron/MoE and 1F1B traces. It accepts normalized Event JSONL or native CUPTI
activity records and reports NCCL runtime, hidden and exposed communication,
overlap ratios, and pipeline-stage metrics.

> **Status:** open-source alpha (`0.3.0`). The offline pipeline and a bounded
> two-GPU RTX 4090 CUPTI test are validated. CUDA 12.9, Transformer Engine 2.7,
> real Megatron 1F1B, and Nsight cross-validation remain planned work.

## Install

Python 3.10 or newer is required.

```bash
git clone https://github.com/sp2022maomao/1F1B_overlap_monitor.git
cd 1F1B_overlap_monitor
python3 -m pip install .
```

The offline analyzer has no mandatory third-party dependencies. Native CUPTI
collection additionally requires Linux, CMake, and a CUDA toolkit with CUPTI.

## Quick Start

Analyze the bundled trace. Input format is detected automatically, timestamps
are normalized to microseconds (`us`), and validation runs before analysis.

```bash
overlap-monitor analyze \
  --input examples/traces/cupti_activity.jsonl \
  --rank 0 \
  --stage-id 0 \
  --table \
  --output-json summary.json
```

Use the same pipeline from Python:

```python
from overlap_monitor import analyze_trace

result = analyze_trace(
    "examples/traces/cupti_activity.jsonl",
    rank=0,
    stage_id=0,
)
print(result.communication_hidden_ratio)
```

For a normalized Event trace, the command is identical:

```bash
overlap-monitor validate --input events_rank0.jsonl
overlap-monitor analyze --input events_rank0.jsonl --table --ascii
```

See the [API and CLI reference](docs/api.md) for filters, output files, modes,
schemas, and clock-alignment rules.

## Choose A Data Source

| Source | Use it for | Guide |
| --- | --- | --- |
| CUPTI | Precise NCCL and compute kernel intervals | [CUPTI measurement](docs/cupti_measurement.md) |
| PyTorch Profiler | Framework/operator attribution | [Integration](docs/integration.md#pytorch-profiler) |
| Work/wait adapter | Low-overhead async dependency proxy | [Integration](docs/integration.md#async-work-handles) |
| Nsight Systems | Cross-stream visual and accuracy checks | [Validation](docs/validation.md) |

These sources answer different questions. CUPTI provides GPU activity timing;
`Work.wait()` observes a host-side wait and is not an exact NCCL kernel timer.

## Metrics

| Metric | Definition |
| --- | --- |
| `communication_runtime` | Union of selected communication intervals |
| `hidden_communication` | Communication concurrent with classified compute |
| `exposed_communication` | Communication not hidden by classified compute |
| `communication_hidden_ratio` | `hidden_communication / communication_runtime` |
| `timeline_overlap_ratio` | `overlap_time / min(compute_time, communication_time)` |
| `wait_time` | Host-side Work wait proxy |

Use the explicit ratio fields above. `overlap_ratio` is retained only as a
mode-specific compatibility alias. Every summary records the active definition,
measurement quality, runtime kind, and validation result.

## Measurement Contracts

- `critical-path` is the default mode and reports communication hiding.
- `timeline` reports symmetric interval overlap and pipeline-stage metrics.
- Multiple rank/device clock domains are rejected unless the caller explicitly
  asserts external alignment with `--assume-aligned-clocks`.
- Incomplete CUPTI traces are rejected. `--allow-incomplete` is for debugging,
  not reported measurements.
- `kernel_timeline` is observed GPU evidence. `estimated` and
  `host_wait_proxy` results must not be presented as precise NCCL runtime.

Read [measurement semantics](docs/measurement.md) before comparing results
across modes or profilers.

## Documentation

| Document | Purpose |
| --- | --- |
| [API and CLI](docs/api.md) | Analyze traces from Python or the command line |
| [Integration](docs/integration.md) | Instrument Work handles, Megatron, or PyTorch Profiler |
| [CUPTI measurement](docs/cupti_measurement.md) | Build, collect, validate, and analyze GPU activity |
| [Measurement semantics](docs/measurement.md) | Interpret overlap and critical-path metrics |
| [Architecture](docs/architecture.md) | Understand module boundaries and extension points |
| [Validation](docs/validation.md) | Review verified behavior and current limitations |

The complete index is in [docs/README.md](docs/README.md). References and the
implementation map are in [references/README.md](references/README.md).

## Development

```bash
python3 -m pip install -e ".[dev]"
ruff check .
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [ROADMAP.md](ROADMAP.md), and
[CITATION.cff](CITATION.cff). Licensed under the [MIT License](LICENSE).
