# Architecture

`overlap-monitor` separates runtime observation from offline analysis. The core
package does not import PyTorch, Megatron, Transformer Engine, CUDA, or CUPTI.

## Data Flow

```text
runtime or profiler
  -> source-specific parser / adapter
  -> normalized Event JSONL
  -> validation and clock-domain checks
  -> timeline or critical-path analyzer
  -> JSON, Markdown, ASCII, or Chrome trace
```

The normalized event is the boundary between framework-specific collection and
framework-neutral analysis.

## Repository Layout

```text
overlap_monitor/         installable Python package
  adapters/              thin runtime metadata adapters
  analyzer/              timeline and critical-path algorithms
  core/                  event schema, intervals, I/O, validation
  profiler/              CUPTI, PyTorch profiler, and Work parsers
  te_adapter/            Transformer Engine kernel classification
  visualization/         text and Chrome trace output
native/cupti_collector/  optional C++ CUPTI Activity collector
examples/                trace fixtures and usage notes
tests/                   hardware-independent regression tests
benchmarks/              CPU overhead and bounded GPU smoke tests
docs/                    user and design documentation
references/              papers and upstream implementation map
```

Tests, examples, and benchmarks are deliberately outside the installable
package, following the layout used by projects such as Megatron-LM,
TransformerEngine, and DeepSpeed.

## Runtime Boundary

Runtime code is intentionally small:

- `WorkHandleRecorder` wraps an asynchronous Work handle and records host-side
  launch and wait observations.
- `MegatronWorkAdapter` attaches rank, stage, iteration, microbatch, and phase
  metadata without importing Megatron.
- `CuptiRuntimeCollector` controls the optional native collector through
  `ctypes` without importing PyTorch.
- `MonitoringSession` provides a bounded, thread-safe event buffer.

Runtime components never add a global CUDA synchronization. A caller may
synchronize after a bounded profiling window, immediately before flushing a
CUPTI trace, when a complete trace is required.

## Analysis Boundary

The analysis package has two algorithms:

| Analyzer | Evidence | Question answered |
| --- | --- | --- |
| `OverlapAnalyzer` | aligned compute and communication intervals | How much do the two activity classes overlap on the timeline? |
| `CriticalPathOverlapAnalyzer` | NCCL kernel intervals or Work/wait observations | How much communication is hidden or exposed under the available evidence? |

Both merge intervals before summing them, preventing concurrent kernels of the
same class from being double counted.

## Stable Contracts

The current public contracts are:

- `Event` and `EventType`;
- normalized Event JSONL in microseconds;
- analyzer summary dataclasses and `to_dict()` output;
- `overlap-monitor analyze`, `validate`, and `import-cupti`;
- explicit `measurement_quality`, `communication_runtime_kind`, and
  `overlap_ratio_definition` fields.

The project is still alpha. Schema changes must remain backward compatible or
be accompanied by a versioned migration before 1.0.

## Extension Rules

New trace sources should:

1. live behind a parser or adapter;
2. emit normalized `Event` objects;
3. state timestamp units and clock domains;
4. preserve source metadata needed for attribution;
5. avoid importing optional frameworks from `core` or `analyzer`;
6. include synthetic regression tests that run without a GPU.

See [Integration](integration.md) for concrete examples and
[Measurement semantics](measurement.md) before interpreting results.
