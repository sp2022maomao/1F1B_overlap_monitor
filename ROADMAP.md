# Roadmap

## Completed in 0.3.x

- Stable package and CLI names: `overlap_monitor` and `overlap-monitor`.
- Decoupled event model, timeline analyzer, and critical-path analyzer.
- Explicit `kernel_timeline`, `estimated`, and `host_wait_proxy` semantics.
- Native CUPTI collector, strict JSONL parser, and direct CLI analysis path.
- Bounded two-GPU RTX 4090 NCCL/GEMM smoke validation.
- Python 3.10-3.12 CI, Ruff checks, and wheel/sdist build checks.
- Stable high-level Python API shared with the CLI, versioned schemas, and
  explicit mode-specific overlap ratios.

## 0.4.0: Target-Stack Validation

- Build and run the native CUPTI collector on CUDA 12.9.
- Validate Transformer Engine 2.7 kernel classification.
- Add a minimal real Megatron-Core/MoE 1F1B integration example.
- Compare CUPTI metrics and overhead against Nsight Systems on the same trace.
- Publish A100-class monitor-off, Work-only, CUPTI, and Nsight results.

## 0.5.0: Trace Interoperability

- Improve PyTorch Profiler trace import.
- Add optional Nsight SQLite ingestion if dependencies remain isolated.
- Add richer Chrome trace categories and rank-group summaries.
- Add schema migrations when stage, microbatch, iteration, or phase metadata
  evolves.

## 1.0.0: Stable API

- Freeze the event schema, timestamp units, and summary fields.
- Publish validated accuracy and overhead bounds.
- Add versioned migrations for downstream Megatron integrations.
