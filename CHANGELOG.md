# Changelog

## Unreleased

- Streamline the English and Chinese quick starts, API/CLI reference,
  integration guide, and documentation index around task-oriented workflows.
- Add stable `load_trace()`, `analyze_trace()`, and `analyze_events()` entry
  points with a shared `AnalysisResult` used by the CLI.
- Add versioned Event and summary JSON schemas, explicit
  `timeline_overlap_ratio` and `communication_hidden_ratio` fields, and an
  explicit clock-alignment assumption in serialized output.
- Replace `--allow-mixed-clock-domains` with the clearer
  `--assume-aligned-clocks` spelling while retaining the old flag as a hidden
  compatibility alias.
- Extend the Megatron adapter with pipeline send/receive Work metadata and
  semantic iteration/forward/backward regions.
- Move tests, examples, and benchmarks out of the installable package and
  consolidate project documentation around architecture, integration,
  measurement semantics, and validation.
- Classify PyTorch Profiler events on import, tighten Transformer Engine kernel
  detection, optimize batched event recording, add bounded-window recorder
  cleanup, and make overlap-ratio denominators explicit in every summary.
- Let `overlap-monitor analyze` auto-detect native CUPTI JSONL and analyze it
  without a separate import step.
- Add normalized-event output, explicit microsecond units, `--version`, empty
  trace rejection, and concise CLI errors.
- Restructure the English and Chinese READMEs around one primary workflow and
  add a documentation index.
- Add an optional native CUPTI Activity collector for concurrent GPU kernel
  timestamps and external correlation IDs.
- Add strict CUPTI JSONL import, dropped-record detection, runtime wrapper,
  synthetic activity examples, and overlap integration tests.
- Document CUPTI build/runtime use and the remaining Nsight/GPU validation
  boundary.
- Restructure the reference materials for public use with implementation-aligned
  metric semantics, stable upstream source links, an annotated bibliography,
  and maintained BibTeX entries.

## 0.3.0 - 2026-07-12

- Rename the public Python package from `overlap_monitor_v2` to
  `overlap_monitor`.
- Rename the distribution from `overlap-monitor-v2` to `overlap-monitor` and
  the CLI command from `overlap-monitor-v2` to `overlap-monitor`.
- Update tests, examples, package metadata, and documentation to use the stable
  version-independent name.

## 0.2.1 - 2026-07-12

- Prepare the repository as an open-source alpha release.
- Add MIT license, citation metadata, roadmap, contribution guide, security
  policy, issue templates, and pull request template.
- Add a public-facing README with project scope, measurement semantics, and
  validation commands.
- Add a similar-projects report that positions the project against Megatron,
  DeepEP, PyTorch/Kineto, HTA, Nsight, and NCCL trace tools.

## 0.2.0 - 2026-07-11

- Separate precise NCCL kernel timelines from Work-lifetime upper bounds.
- Add measurement quality, runtime semantics, and warnings to critical-path reports.
- Add thread-safe Work recording and bounded monitoring sessions.
- Add event-stream validation for clock domains and Work/wait correlation.
- Add atomic JSONL output and contextual parser errors.
- Add a Megatron metadata adapter without importing Megatron or torch.
- Add CLI validation, CPU microbenchmark, and expanded synthetic tests.

## 0.1.0

- Initial decoupled event model, analyzers, TE classifier, CLI, and visualization.
