# Reference Directory Manifest

This directory contains documentation and citation metadata only. It does not
contain copied papers, profiler traces, generated benchmark results, or
third-party source code.

| File | Maintained content |
| --- | --- |
| `README.md` | Entry point, scope, and evidence levels |
| `metric_design.md` | Equations, output semantics, and reporting rules |
| `implementation_map.md` | Upstream code and tool integration map |
| `references.md` | Annotated primary-source bibliography |
| `references.bib` | BibTeX starter entries |

## Maintenance Policy

When the analyzer behavior changes:

1. Update `metric_design.md` in the same pull request.
2. Preserve the distinction between observed kernel runtime and estimated Work
   lifetime.
3. Prefer stable symbols and repository paths over line numbers.
4. Link to official documentation, original papers, or canonical repositories.
5. Record the link-review date in `README.md`.

BibTeX entries are provided as a convenience. Authors should verify venue,
author list, DOI, and formatting against the target publication style before
submission.
