# 1F1B Overlap Monitor

[![tests](https://github.com/sp2022maomao/1F1B_overlap_monitor/actions/workflows/tests.yml/badge.svg)](https://github.com/sp2022maomao/1F1B_overlap_monitor/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

`overlap-monitor` measures communication-computation overlap in asynchronous
Megatron/MoE and 1F1B training traces. It accepts normalized event JSONL or raw
records from the included CUPTI collector and reports NCCL runtime, hidden and
exposed communication, overlap ratio, and stage-level metrics.

> **Status:** open-source alpha (`0.3.0`). The analysis pipeline and bounded
> two-GPU RTX 4090 CUPTI test are validated. CUDA 12.9, Transformer Engine 2.7,
> real Megatron 1F1B, and Nsight cross-validation are still pending.

## Why

Asynchronous collectives may return on the host before their NCCL kernels finish
on the GPU. Python timers, a single CUDA stream, and `Work.wait()` therefore
answer different questions.

This project keeps three quantities separate:

```text
communication runtime
hidden communication
exposed communication
```

Observed GPU kernel timelines are preferred. Host-side `Work.wait()` events are
reported only as proxies when a kernel timeline is unavailable.

## Install

```bash
git clone https://github.com/sp2022maomao/1F1B_overlap_monitor.git
cd 1F1B_overlap_monitor
python3 -m pip install .
```

The offline analyzer has no mandatory third-party dependencies. Native CUPTI
collection requires Linux, CMake, and a CUDA toolkit with CUPTI.

## Quick Start

Analyze the bundled CUPTI trace directly:

```bash
overlap-monitor analyze \
  --input examples/traces/cupti_activity.jsonl \
  --rank 0 \
  --stage-id 0 \
  --table \
  --output-json summary.json
```

`analyze` automatically detects native CUPTI records and normalized Event JSONL.
The command validates the trace before calculating metrics. All normalized
timestamps and duration fields use microseconds (`us`).

Analyze the bundled asynchronous Work/wait example:

```bash
overlap-monitor analyze \
  --input examples/traces/critical_path_events.jsonl \
  --table \
  --ascii
```

## Analyze Your Trace

### Native CUPTI JSONL

Use one command for the normal path:

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

`--events-output` is optional. It preserves the normalized events used for the
analysis. The older two-step `import-cupti` command remains supported.

Incomplete CUPTI traces are rejected by default. `--allow-incomplete` is for
debugging only and should not be used for reported measurements.

### Normalized Event JSONL

```bash
overlap-monitor validate --input events_rank0.jsonl
overlap-monitor analyze --input events_rank0.jsonl --table
```

Use `--rank`, `--device-id`, or `--stage-id` to select one aligned clock domain.
Use `--mode timeline` for symmetric timeline and pipeline metrics; the default
critical-path mode reports communication hiding and exposed communication.

## Data Sources

| Tool | Granularity | Primary role |
| --- | --- | --- |
| CUPTI | CUDA activity and kernel records | Raw NCCL and compute kernel intervals |
| Nsight Systems | System-level CPU/GPU/stream timeline | Cross-stream validation and visual inspection |
| PyTorch Profiler | Framework operator/module/autograd level | Framework attribution and CUDA event export |
| Work/wait adapter | Host-side distributed Work lifecycle | Low-overhead dependency proxy |

These sources are complementary, not equivalent timers. CUPTI is the lowest
level, Nsight Systems integrates the system timeline, and PyTorch Profiler keeps
the strongest framework semantics.

## Metrics

| Metric | Meaning |
| --- | --- |
| `communication_runtime` | Union of selected NCCL/kernel communication intervals |
| `hidden_communication` | Communication concurrent with classified compute |
| `exposed_communication` | Communication not hidden by classified compute |
| `overlap_ratio` in critical-path mode | `hidden_communication / communication_runtime` |
| `overlap_ratio` in timeline mode | `overlap_time / min(compute_time, communication_time)` |
| `wait_time` | Host-side Work wait proxy |
| `critical_path_span` | Span covered by the analyzed compute, communication, and wait events |

Every result includes `overlap_ratio_definition` and its validation report.
Critical-path results also include `measurement_quality` and
`communication_runtime_kind`.
`kernel_timeline` is observed GPU evidence; `estimated` and `host_wait_proxy`
must not be presented as precise NCCL runtime.

## CUPTI Collection

Build the optional native collector:

```bash
cmake -S native/cupti_collector -B build/cupti -DCMAKE_BUILD_TYPE=Release
cmake --build build/cupti --parallel
```

See [CUPTI measurement](docs/cupti_measurement.md) for runtime collection,
schema, dropped-record handling, and validation guidance.

## Architecture

```text
collector or profiler trace
  -> parser and classifier
  -> normalized Event JSONL
  -> validation
  -> timeline or critical-path analyzer
  -> JSON, Markdown table, ASCII, or Chrome trace
```

Core analysis modules do not import Megatron, PyTorch, Transformer Engine,
Nsight, CUDA, or CUPTI. Runtime integrations remain thin adapters.

## Documentation

Start with the [documentation index](docs/README.md).

- [Architecture](docs/architecture.md)
- [Integration guide](docs/integration.md)
- [Measurement semantics](docs/measurement.md)
- [CUPTI collection and analysis](docs/cupti_measurement.md)
- [Validation](docs/validation.md)
- [Reference implementation map](references/README.md)
- [Roadmap](ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## Development

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
ruff check .
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m build
```

## License

MIT. See [LICENSE](LICENSE). Citation metadata is available in [CITATION.cff](CITATION.cff).
