# Reference Materials

This directory documents the research and implementation context behind
`overlap-monitor`. It is intended for users who need to understand the metric
semantics, inspect the relevant Megatron paths, or cite the underlying systems
work.

## Contents

| Document | Purpose |
| --- | --- |
| [Metric semantics](metric_design.md) | Defines timeline and critical-path metrics exactly as implemented. |
| [Implementation map](implementation_map.md) | Maps monitor concepts to stable upstream symbols and integration points. |
| [Annotated bibliography](references.md) | Groups primary papers, official documentation, and related systems by topic. |
| [BibTeX](references.bib) | Provides starter citation entries for reports and papers. |
| [Directory manifest](file_list.md) | Records the scope and maintenance policy of this directory. |

## Measurement Modes

The project deliberately separates two evidence levels:

| Mode | Evidence | Valid conclusion |
| --- | --- | --- |
| Kernel timeline | NCCL and compute kernel intervals from a GPU trace | Observed kernel overlap and uncovered communication time |
| Async Work/wait | Launch, observed completion, and blocking `wait()` intervals | Critical-path wait and an estimated communication window |
| Generic timeline fallback | Classified communication and compute events | Approximate overlap with an explicit quality warning |

A `Work` interval ending only when `wait()` returns is an upper bound, not an
exact NCCL kernel runtime. Reports must retain the emitted
`measurement_quality` and `communication_runtime_kind` fields.

## Scope

- These files explain measurement methodology; they do not vendor papers or
  third-party source code.
- Upstream paths are referenced by symbol rather than line number because
  Megatron evolves quickly.
- The source code and tests remain authoritative if a document and a released
  implementation disagree.
- External links were last reviewed on 2026-07-12.

For project usage, start with the repository [README](../README.md). For
measurement caveats, read [Metric semantics](metric_design.md) before reporting
results.
