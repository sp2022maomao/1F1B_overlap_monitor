# Python API and CLI

The high-level Python API and CLI share the same loading, filtering, validation,
and analysis implementation. Use the high-level API for applications and the
lower-level classes only when constructing custom pipelines.

## Stable Python Entry Points

The package root exports:

```python
from overlap_monitor import (
    AnalysisResult,
    Event,
    EventType,
    TraceValidationError,
    analyze_events,
    analyze_trace,
    load_trace,
)
```

### Load a trace

```python
events = load_trace(
    "trace.jsonl",
    input_format="auto",
    rank=0,
    device_id=0,
    stage_id=0,
)
```

`load_trace()` accepts normalized Event JSONL and native CUPTI JSONL. It
auto-detects the source by default, normalizes CUPTI timestamps to microseconds,
and applies rank/device/stage filters. It does not analyze the events.

### Load and analyze

```python
result = analyze_trace(
    "trace.jsonl",
    mode="critical-path",
    rank=0,
    stage_id=0,
)

print(result.communication_hidden_ratio)
print(result.validation.valid)
print(result.to_dict())
```

`analyze_trace()` returns `AnalysisResult`, which contains:

| Field | Meaning |
| --- | --- |
| `events` | normalized and filtered events used by the analyzer |
| `summary` | `CriticalPathSummary` or `OverlapSummary` |
| `validation` | event count, clock domains, warnings, and errors |
| `analysis_mode` | `critical-path` or `timeline` |
| `input_format` | detected `events` or `cupti` source |
| `clock_alignment_assumed` | whether cross-domain alignment was asserted by the caller |
| `source_warnings` | parser warnings such as incomplete CUPTI records |

`result.to_dict()` produces the versioned summary JSON used by the CLI.

### Analyze in-memory events

```python
result = analyze_events(
    events,
    mode="timeline",
)
print(result.timeline_overlap_ratio)
```

Invalid event sets raise `TraceValidationError`. The exception exposes the full
report as `exc.report`.

## Ratio Fields

The explicit fields should be used by new integrations:

```text
timeline mode:
  timeline_overlap_ratio
  = overlap_time / min(compute_time, communication_time)

critical-path mode:
  communication_hidden_ratio
  = hidden_communication / communication_runtime
```

`overlap_ratio` remains as a compatibility alias and has the same value as the
mode-specific field. Every serialized result also contains
`overlap_ratio_definition`.

## Clock Domains

Multiple rank/device clock domains are rejected by default. Only bypass this
check after timestamps have been aligned externally:

```python
result = analyze_trace(
    "aligned-events.jsonl",
    assume_aligned_clocks=True,
)
assert result.clock_alignment_assumed
```

The CLI equivalent is `--assume-aligned-clocks`. The old
`--allow-mixed-clock-domains` spelling remains a hidden compatibility alias but
should not be used in new scripts.

## CLI Mapping

The standard command maps directly to `analyze_trace()`:

```bash
overlap-monitor analyze \
  --input trace.jsonl \
  --input-format auto \
  --mode critical-path \
  --rank 0 \
  --stage-id 0 \
  --output-json summary.json
```

Validation uses the same loader and accepts either trace format:

```bash
overlap-monitor validate \
  --input trace.jsonl \
  --input-format auto \
  --rank 0
```

`import-cupti` is retained for workflows that need a persistent normalized
Event JSONL before analysis. The normal CUPTI path needs only `analyze`.

## Versioned Schemas

New normalized Event records include `schema_version: 1`. Readers still accept
legacy records with no version and reject unknown explicit versions.

The package ships:

```text
overlap_monitor/schemas/event-v1.schema.json
overlap_monitor/schemas/summary-v1.schema.json
```

Standard Event metadata fields are `iteration`, `microbatch_id`, `phase`,
`comm_id`, `measurement`, `runtime_kind`, `clock_domain`, and
`timestamp_unit`. Additional source-specific metadata remains allowed.
