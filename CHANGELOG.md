# Changelog

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
