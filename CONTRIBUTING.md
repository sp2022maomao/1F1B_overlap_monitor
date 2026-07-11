# Contributing

Thanks for helping improve `overlap-monitor`.

## Project Scope

This project focuses on lightweight, decoupled 1F1B communication-computation
overlap measurement for Megatron/MoE-style training. The core package must stay
independent from Megatron, PyTorch, Transformer Engine, and Nsight imports.

## Development Setup

```bash
python3 -m pip install -e .
python3 -m unittest discover -s overlap_monitor/tests -p 'test_*.py'
```

Optional GPU integrations may depend on:

```text
torch
transformer-engine==2.7.0
Nsight Systems / PyTorch profiler traces
```

## Contribution Rules

- Keep the original benchmark behavior reproducible.
- Do not add global CUDA synchronization in runtime recorders.
- Do not import Megatron, torch, or transformer_engine from core modules.
- Mark Work-handle completion windows as estimated upper bounds unless a true
  NCCL/GPU kernel timeline is available.
- Add or update tests for every behavior change.
- Update docs when changing public API, metrics, or measurement semantics.

## Validation Expectations

For pure Python changes:

```bash
python3 -m unittest discover -s overlap_monitor/tests -p 'test_*.py'
python3 -m overlap_monitor.benchmarks.benchmark_runtime
```

For GPU changes, include:

- monitor-off vs monitor-on iteration time
- tokens/s or samples/s delta
- Nsight/PyTorch profiler alignment evidence
- known hardware, driver, CUDA, PyTorch, TE, and Megatron versions

## Pull Request Checklist

- Tests pass locally.
- New code is decoupled from original Megatron training code.
- Measurement quality is explicit in output.
- Documentation reflects any new metric or warning.
- Generated files such as `build/`, `dist/`, `*.egg-info`, and `__pycache__/`
  are not committed.
