# Integration Guide

Choose the least invasive data source that answers the experiment question.
The normal workflow is to collect a bounded trace, analyze it offline, and use
Nsight Systems only for cross-validation.

## 1. Use The High-Level API

Applications should normally use `analyze_trace()`:

```python
from overlap_monitor import analyze_trace

result = analyze_trace(
    "trace.jsonl",
    mode="critical-path",
    rank=0,
    stage_id=0,
)
print(result.communication_hidden_ratio)
```

For custom in-memory pipelines, use `analyze_events()`:

```python
from overlap_monitor import Event, EventType, analyze_events

events = [
    Event(0, 10, EventType.GEMM, rank=0, stage_id=0),
    Event(5, 15, EventType.NCCL, rank=0, stage_id=0),
]
result = analyze_events(events, mode="timeline")
print(result.timeline_overlap_ratio)
```

The CLI calls the same public API:

```bash
overlap-monitor analyze --input events_rank0.jsonl --table
```

## 2. Observe Async Work Handles

Wrap the `Work` returned by an asynchronous collective:

```python
from pathlib import Path

from overlap_monitor import WorkHandleRecorder
from overlap_monitor.core.io import write_events_jsonl

recorder = WorkHandleRecorder()
raw_work = torch.distributed.all_to_all_single(
    output,
    input,
    async_op=True,
)
work = recorder.wrap(
    raw_work,
    comm_id="iter12-mb3-dispatch",
    name="dispatch_a2a",
    rank=rank,
    stage_id=pp_rank,
    iteration=12,
    microbatch_id=3,
    phase="dispatch",
)
work.wait()
events = recorder.events()
write_events_jsonl(events, Path("events_rank0.jsonl"))
recorder.clear()
```

The wrapper delegates every unknown attribute to the original Work object and
does not synchronize CUDA. Persist a bounded window and call `clear()` before
the next window to release recorded history. Its timing is reported as a
host-side proxy unless an independent completion timestamp or kernel timeline
is available.

## 3. Attach Megatron Context

`MegatronWorkAdapter` keeps Megatron-specific metadata out of the recorder:

```python
from overlap_monitor import EventType, MegatronWorkAdapter, WorkHandleRecorder

recorder = WorkHandleRecorder()
adapter = MegatronWorkAdapter(
    recorder,
    rank=rank,
    stage_id=pp_rank,
    device_id=device,
)

work = adapter.wrap_a2a(
    raw_work,
    comm_id=f"iter{iteration}-mb{microbatch_id}-dispatch",
    iteration=iteration,
    microbatch_id=microbatch_id,
    phase="dispatch",
)

send_work = adapter.wrap_pipeline(
    raw_send_work,
    comm_id=f"iter{iteration}-mb{microbatch_id}-forward-send",
    iteration=iteration,
    microbatch_id=microbatch_id,
    direction="send",
    phase="forward",
    peer_rank=next_pipeline_rank,
)

forward_region = adapter.region(
    forward_start_us,
    forward_end_us,
    iteration=iteration,
    microbatch_id=microbatch_id,
    phase="forward",
    event_type=EventType.PIPELINE,
)
```

Add this only where the original code already returns an async Work handle.
Do not alter routing, collective selection, or schedule order in a baseline run.
`wrap_pipeline()` supports forward/backward send/recv Work handles, while
`region()` creates semantic iteration, forward, backward, expert, or drain
regions without changing execution. Region timestamps must use the same clock
domain as events analyzed with them. Pass `event_type=EventType.COMPUTE` only
when the interval represents measured compute rather than a semantic boundary.

## 4. Parse PyTorch Profiler Events

The parser consumes profiler-like objects without importing PyTorch itself:

```python
from overlap_monitor.profiler import PyTorchProfilerEventParser

events = PyTorchProfilerEventParser().parse(
    profiler.events(),
    rank=rank,
    stage_id=pp_rank,
)
```

Kernel names are classified into NCCL, GEMM, attention, memory, and unknown
events. Inspect unknown events before relying on coverage-sensitive metrics such
as bubble ratio.

## 5. Collect CUPTI Activity

Use CUPTI when precise NCCL and compute kernel intervals are required. The
collector is optional and isolated from the Python analysis package.

See [CUPTI measurement](cupti_measurement.md) for build, runtime, schema,
correlation, and dropped-record handling.

## 6. Export Results

One CLI command can write all common outputs:

```bash
overlap-monitor analyze \
  --input trace.jsonl \
  --rank 0 \
  --stage-id 0 \
  --events-output events_rank0.jsonl \
  --output-json summary_rank0.json \
  --trace-json timeline_rank0.json \
  --table
```

Use `--mode timeline` for the symmetric interval metric. The default
`critical-path` mode reports communication hiding. Read
[Measurement semantics](measurement.md) before comparing the two ratios.
Use `--assume-aligned-clocks` only after externally aligning multiple rank or
device clocks; the assumption is persisted in summary JSON.

## Production Checklist

- Bound the profiling window after warmup.
- Record rank, stage, iteration, microbatch, and phase.
- Keep one clock domain per analysis unless timestamps are explicitly aligned.
- Reject incomplete CUPTI traces for published results.
- Measure monitor-off and monitor-on throughput.
- Compare at least one representative trace against Nsight Systems.
