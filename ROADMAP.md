# Roadmap

## v0.2.x: Open-Source Alpha

- Keep the standalone Python package installable.
- Keep unit tests passing on Python 3.10 and 3.12.
- Maintain strict separation between core analysis and Megatron/PyTorch runtime
  integrations.
- Document measurement quality and upper-bound semantics clearly.

## v0.3.0: GPU Validation

- Validate Work-handle critical-path metrics against Nsight Systems traces.
- Add a minimal Megatron-Core/MoE integration example.
- Add an A100-class benchmark report with monitor-off, Work-only, and
  profiler/Nsight modes.
- Define acceptable error and overhead thresholds.

## v0.4.0: Trace Interoperability

- Improve PyTorch profiler trace import.
- Validate and stabilize the initial native CUPTI collector on CUDA 12.9.
- Add richer Chrome trace export categories.
- Add optional Nsight SQLite ingestion if the dependency footprint stays clean.
- Add stage-aware 1F1B summaries across rank groups.

## v1.0.0: Stable API

- Freeze the public event schema and summary fields.
- Publish GPU-validated accuracy/overhead claims.
- Add versioned migration notes for downstream Megatron integrations.
