# Similar Projects Report

Date: 2026-07-12

This report summarizes existing related projects and the positioning of
`overlap-monitor`.

## Search Summary

Searches were run for:

```text
Megatron overlap profiler
NCCL profiler PyTorch
nsight systems python trace analyzer
DeepEP alltoall overlap
PyTorch chrome trace analyzer profiler
```

I did not find an open-source project that directly matches the exact scope:

```text
standalone 1F1B + Megatron/MoE + async Work/wait critical-path overlap monitor
```

The closest projects are either full training frameworks, low-level profiler
systems, trace analyzers, or NCCL/kernel-focused tools.

## Comparison

| Project | What it provides | Relation to this project |
| --- | --- | --- |
| [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) / Megatron-Core | Large-scale transformer training framework with pipeline, tensor, expert parallelism, and overlap optimizations. | Upstream runtime target. This project should stay decoupled and provide thin adapters rather than modifying Megatron core directly. |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | Expert-parallel communication library for MoE workloads, including high-throughput and low-latency dispatch/combine paths. | Runtime whose communication behavior may be measured. It is not a standalone 1F1B critical-path monitor. |
| [PyTorch Kineto](https://github.com/pytorch/kineto) | PyTorch profiling backend for CPU/GPU activity collection and trace generation. | Potential trace source. This project consumes or normalizes trace-like events and adds 1F1B/MoE-specific overlap semantics. |
| [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) | User-facing profiler API for collecting PyTorch execution traces. | Useful for kernel timeline events. It does not by itself define exposed communication or async Work/wait critical-path metrics. |
| [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) | Analyzes PyTorch profiler traces for distributed training performance, including straggler and trace-level analysis. | Similar in being an offline trace analyzer, but broader and trace-centric. This project is narrower and focuses on 1F1B/MoE critical-path overlap. |
| [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) | System-wide CPU/GPU profiler and timeline viewer. | Best ground truth for GPU validation. This project should compare against Nsight, not replace it. |
| [NVIDIA Nsight Systems recipes](https://github.com/NVIDIA/nsys-recipes) and community recipes such as [hyxcl/nsys_recipes](https://github.com/hyxcl/nsys_recipes) | Post-processing examples for Nsight Systems profiling data. | Useful reference for trace post-processing; not Megatron 1F1B-specific. |
| [NVIDIA NCCL tests](https://github.com/NVIDIA/nccl-tests) | Benchmarks NCCL collective performance. | Useful for communication microbenchmarks; not a training critical-path overlap analyzer. |
| [CollectiveTrace](https://github.com/jayden1711/CollectiveTrace) | Profiles CUDA kernels blocking NCCL collectives during distributed training/inference. | Related NCCL-focused diagnostic tool. This project complements it by reporting hidden/exposed communication in 1F1B/MoE schedules. |
| [fzyzcjy/torch_utils](https://github.com/fzyzcjy/torch_utils) | PyTorch utilities including trace and memory profiling helpers with NCCL-aware details. | Related utility collection; not a dedicated 1F1B overlap framework. |

## Differentiation

`overlap-monitor` should position itself as:

```text
a lightweight, framework-decoupled overlap analysis library for Megatron/MoE
1F1B schedules, with explicit async Work/wait measurement semantics.
```

The differentiators are:

- stage-aware 1F1B event model
- critical-path exposed communication metrics
- Work-handle upper-bound labeling
- kernel timeline and Work/wait analysis paths in one library
- no hard dependency on Megatron, torch, Transformer Engine, or Nsight in core
- small synthetic tests that verify overlap math independent of GPU hardware

## Recommended Open-Source Positioning

Use careful wording:

```text
This project provides a validated local/synthetic alpha implementation. GPU
accuracy claims require comparison with Nsight Systems or PyTorch profiler
kernel traces on real Megatron/MoE workloads.
```

Avoid claiming:

```text
production-accurate overlap measurement
training speedup
exact NCCL runtime from Work-only events
```

until the GPU validation plan in `docs/validation_report.md` is completed.

## Next Competitive Step

The next release should include:

1. A minimal Megatron-Core integration patch example.
2. One public synthetic 1F1B trace fixture.
3. One real GPU validation report comparing Work-only, PyTorch profiler, and
   Nsight Systems timelines.
4. A small benchmark table showing monitor-on overhead.
