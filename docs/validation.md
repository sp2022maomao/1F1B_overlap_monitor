# Validation

This page records what has been demonstrated and what remains an open claim.
Synthetic correctness and GPU smoke tests are intentionally separated from
production accuracy and overhead validation.

## Automated Coverage

The hardware-independent suite covers:

- event parsing, validation, and atomic JSONL I/O;
- interval union and intersection math;
- timeline and critical-path metrics;
- Work/wait lifecycle and explicit completion observations;
- mixed clock-domain rejection;
- CUPTI schema, external correlation, and dropped-record rejection;
- PyTorch profiler and TE kernel classification;
- Megatron metadata attachment;
- CLI auto-detection and output generation;
- ASCII, Markdown, and Chrome trace output;
- bounded and concurrent event buffering.

Run it with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
ruff check .
python3 -m build
```

## RTX 4090 Smoke Test

Date: 2026-07-13

Environment:

| Item | Value |
| --- | --- |
| OS | Ubuntu 24.04.2 LTS, Linux 6.17.0-14-generic |
| GPU | 2 x NVIDIA GeForce RTX 4090 |
| Topology | `NODE`, no NVLink reported |
| Driver | 570.211.01 |
| PyTorch | 2.4.1+cu121 |
| NCCL | 2.20.5 |
| CUPTI | `libcupti.so.12` |

The native collector compiled and loaded against the available CUDA 12.1
runtime/CUPTI packages. Start/stop, external correlation, JSONL output, and
dropped-record accounting succeeded.

### Single-GPU GEMM

Ten FP16 `512 x 512` matrix multiplications produced:

| Metric | Result |
| --- | ---: |
| kernel records | 10 |
| dropped records | 0 |
| per-kernel duration | 5.376-6.112 us |
| compute interval union | 55.25 us |
| classified as GEMM | 10/10 |

### Two-GPU All-to-All and GEMM

Five iterations submitted `all_to_all_single(async_op=True)` and GEMM on a
separate CUDA stream:

| Metric | rank 0 | rank 1 |
| --- | ---: | ---: |
| NCCL kernels | 5 | 5 |
| GEMM kernels | 5 | 5 |
| dropped records | 0 | 0 |
| externally correlated kernels | 10/10 | 10/10 |
| communication union | 219.75 us | 413.50 us |
| compute union | 39.00 us | 44.75 us |
| interval intersection | 31.50 us | 38.75 us |

Both GPUs were occupied by unrelated high-utilization jobs. These values prove
the end-to-end collection path works; they are not stable performance results.
Docker required `--shm-size=1g` for NCCL initialization.

The reproducible workload is in
[`benchmarks/cupti_gpu_smoke.py`](../benchmarks/cupti_gpu_smoke.py).

## CPU Microbenchmark

The benchmark measures the framework-neutral Work wrapper and offline analyzer:

```bash
python3 -m benchmarks.benchmark_runtime --iterations 20000 --repeats 5
```

Results depend on the host and Python build. They must not be used to infer GPU
iteration speedup.

On the 2026-07-13 local CPU run, three independent runs with seven repeats each
produced:

| Metric | Observed range |
| --- | ---: |
| Work recorder incremental cost | 3.47-3.68 us/call |
| 10,000-event offline analysis | 21.13-22.63 ms |
| 10,000 events through individual `emit()` | 3.50-3.85 ms |
| 10,000 events through batched `extend()` | 0.092-0.097 ms |
| batched ingestion speedup | 37.2-39.8x |

The ingestion comparison runs both implementations in the same process. It
measures Python lock/list overhead only, not end-to-end training throughput.

## Not Yet Validated

The following remain open:

- Transformer Engine 2.7 kernel classification on a real workload;
- real Megatron 1F1B/MoE schedule attribution;
- CUPTI interval and overlap error against the same Nsight Systems trace;
- monitor-on throughput and memory overhead on an idle target system;
- CUDA 12.9 and 8 x A100/H100 EP=8 execution;
- aligned multi-node timelines.

Until those checks pass, describe the project as a validated alpha with real GPU
smoke coverage, not as a production-accurate profiler.
