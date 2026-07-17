# Documentation

Start with the root [README](../README.md) or [中文 README](../README_zh.md).
Then follow the shortest path for your task.

## I Want To...

| Task | Read |
| --- | --- |
| Analyze an existing trace | [Python API and CLI](api.md) |
| Instrument Work handles or Megatron | [Integration](integration.md) |
| Capture precise GPU kernel intervals | [CUPTI measurement](cupti_measurement.md) |
| Interpret overlap ratios correctly | [Measurement semantics](measurement.md) |
| Verify supported and unverified behavior | [Validation](validation.md) |

## Design And Research

| Document | Purpose |
| --- | --- |
| [Architecture](architecture.md) | Module boundaries, data flow, and extension rules |
| [Ecosystem](ecosystem.md) | Relationship to Megatron, DeepEP, Kineto, HTA, and Nsight |
| [References](../references/README.md) | Papers, upstream implementations, metric map, and BibTeX |

## Recommended Workflow

1. Run the root README example.
2. Choose the least invasive data source that answers the question.
3. Read measurement semantics before comparing ratios.
4. Check validation before making accuracy or overhead claims.

Development and release material is in [CONTRIBUTING](../CONTRIBUTING.md),
[ROADMAP](../ROADMAP.md), and [CHANGELOG](../CHANGELOG.md).
