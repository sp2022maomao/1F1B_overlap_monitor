# Documentation

Start with the root [README](../README.md) or [中文 README](../README_zh.md),
then use the shortest path below for the task at hand.

## Use The Project

| Document | Purpose |
| --- | --- |
| [Python API and CLI](api.md) | Use stable high-level entry points and versioned outputs |
| [Integration](integration.md) | Add Event, Work/wait, Megatron, or PyTorch Profiler inputs |
| [CUPTI measurement](cupti_measurement.md) | Build the native collector and analyze a bounded GPU trace |
| [Measurement semantics](measurement.md) | Interpret timeline, hidden, exposed, and proxy metrics correctly |

## Understand The Project

| Document | Purpose |
| --- | --- |
| [Architecture](architecture.md) | Module boundaries, data flow, and extension rules |
| [Validation](validation.md) | Current evidence, GPU smoke results, and open limitations |
| [Ecosystem](ecosystem.md) | Relationship to Megatron, DeepEP, Kineto, HTA, and Nsight |
| [References](../references/README.md) | Papers, upstream implementations, metric map, and BibTeX |

## Recommended Order

1. Run the root README example.
2. Read measurement semantics before comparing overlap ratios.
3. Follow integration for the least invasive data source.
4. Use CUPTI only when precise kernel intervals are required.
5. Check validation before making accuracy or overhead claims.

Development and release material is in [CONTRIBUTING](../CONTRIBUTING.md),
[ROADMAP](../ROADMAP.md), and [CHANGELOG](../CHANGELOG.md).
