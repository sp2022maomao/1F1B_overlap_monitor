# Python API and CLI

The Python API and CLI share one loading, filtering, validation, and analysis
pipeline. Applications should use the package-level functions below; lower-level
parsers and analyzers are extension points, not the primary interface.

## API At A Glance

| Entry point | Use case | Returns |
| --- | --- | --- |
| `load_trace()` | Load, normalize, and filter a trace | `list[Event]` |
| `analyze_trace()` | Analyze Event or CUPTI JSONL end to end | `AnalysisResult` |
| `analyze_events()` | Analyze normalized in-memory events | `AnalysisResult` |

```python
from overlap_monitor import analyze_trace

result = analyze_trace(
    "trace.jsonl",
    mode="critical-path",
    rank=0,
    stage_id=0,
)

print(result.communication_hidden_ratio)
print(result.validation.valid)
summary = result.to_dict()
```

### Common arguments

| Argument | Values | Default | Meaning |
| --- | --- | --- | --- |
| `input_format` | `auto`, `events`, `cupti` | `auto` | Input parser; automatic detection is recommended |
| `mode` | `critical-path`, `timeline` | `critical-path` | Metric family to compute |
| `rank` | integer or `None` | `None` | Keep one distributed rank |
| `device_id` | integer or `None` | `None` | Keep one CUDA device |
| `stage_id` | integer or `None` | `None` | Keep one pipeline stage |
| `allow_incomplete` | boolean | `False` | Permit incomplete CUPTI data for debugging |
| `assume_aligned_clocks` | boolean | `False` | Assert that multiple clock domains were externally aligned |

`load_trace()` accepts the input and filter arguments, but does not take `mode`
or `assume_aligned_clocks` because it does not run validation or analysis.

## In-Memory Events

```python
from overlap_monitor import Event, EventType, analyze_events

events = [
    Event(0, 10, EventType.GEMM, rank=0, stage_id=0),
    Event(5, 15, EventType.NCCL, rank=0, stage_id=0),
]
result = analyze_events(events, mode="timeline")
print(result.timeline_overlap_ratio)  # 0.5
```

Invalid inputs raise `TraceValidationError`; the complete validation report is
available as `exc.report`.

## Result Contract

`AnalysisResult` contains both the normalized evidence and the serialized
summary contract:

| Field | Meaning |
| --- | --- |
| `events` | Filtered events used in the calculation |
| `summary` | Mode-specific metric object |
| `validation` | Event count, clock domains, warnings, and errors |
| `analysis_mode` | `critical-path` or `timeline` |
| `input_format` | Detected `events` or `cupti` format |
| `clock_alignment_assumed` | Whether external clock alignment was asserted |
| `source_warnings` | Parser warnings, including incomplete CUPTI evidence |
| `timestamp_unit` | Always `us` for normalized events |

`result.to_dict()` returns the versioned JSON object written by the CLI.

Use the explicit ratio for the selected mode:

```text
critical-path: communication_hidden_ratio
               = hidden_communication / communication_runtime

timeline:      timeline_overlap_ratio
               = overlap_time / min(compute_time, communication_time)
```

`overlap_ratio` remains a compatibility alias and equals the active explicit
field. Serialized output always includes `overlap_ratio_definition`.

## CLI

The normal path needs one command:

```bash
overlap-monitor analyze \
  --input trace.jsonl \
  --input-format auto \
  --mode critical-path \
  --rank 0 \
  --stage-id 0 \
  --events-output events_rank0.jsonl \
  --output-json summary_rank0.json \
  --trace-json timeline_rank0.json \
  --table
```

| Option | Effect |
| --- | --- |
| `--events-output` | Save the normalized Event JSONL used in analysis |
| `--output-json` | Save the versioned metric summary |
| `--trace-json` | Save Chrome Trace format output |
| `--table` | Print a Markdown summary table |
| `--ascii` | Print a compact stage timeline |

Validate without calculating metrics:

```bash
overlap-monitor validate --input trace.jsonl --rank 0
```

Run `overlap-monitor analyze --help` for the complete option list.
`import-cupti` remains available when a workflow needs persistent normalized
events before analysis; direct `analyze` is preferred otherwise.

## Safety Rules

- Analyze one rank/device clock domain at a time unless timestamps were aligned
  externally. In that case, use `--assume-aligned-clocks`; the assumption is
  recorded in the output.
- Keep incomplete CUPTI records rejected for reported measurements.
  `--allow-incomplete` is a debugging escape hatch.
- Treat `host_wait_proxy` and `estimated` results as proxies, not precise NCCL
  kernel duration. See [measurement semantics](measurement.md).

## Schemas

Normalized Event and summary records use `schema_version: 1`:

```text
overlap_monitor/schemas/event-v1.schema.json
overlap_monitor/schemas/summary-v1.schema.json
```

Standard Event metadata keys are `iteration`, `microbatch_id`, `phase`,
`comm_id`, `measurement`, `runtime_kind`, `clock_domain`, and
`timestamp_unit`. Readers accept legacy records without a version, reject
unknown explicit versions, and preserve additional source metadata.
