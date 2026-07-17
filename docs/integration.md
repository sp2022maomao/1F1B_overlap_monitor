# Integration Guide

Instrument the smallest surface that answers the experiment question. Keep the
profiling window bounded, analyze offline, and use Nsight Systems for at least
one representative cross-check.

## Choose An Integration

| Need | Integration | Evidence quality |
| --- | --- | --- |
| Exact NCCL/compute kernel intervals | [CUPTI](#cupti-activity) | Observed GPU timeline |
| Framework operator attribution | [PyTorch Profiler](#pytorch-profiler) | Framework and CUDA events |
| Async collective dependency timing | [Work handle](#async-work-handles) | Host-side proxy |
| 1F1B/MoE semantic metadata | [Megatron adapter](#megatron-context) | Metadata over another source |

All integrations produce normalized `Event` objects. Analyze them through
`analyze_trace()` or `analyze_events()` as documented in [API and CLI](api.md).

## Async Work Handles

Wrap the `Work` returned by an existing asynchronous collective:

```python
from pathlib import Path

from overlap_monitor import WorkHandleRecorder
from overlap_monitor.core.io import write_events_jsonl

recorder = WorkHandleRecorder()
raw_work = torch.distributed.all_to_all_single(output, input, async_op=True)
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

write_events_jsonl(recorder.events(), Path("events_rank0.jsonl"))
recorder.clear()
```

The wrapper delegates unknown attributes to the original Work object and does
not synchronize CUDA. Record a bounded window and call `clear()` between
windows. Work lifetime and `wait()` duration are host observations; without an
independent GPU completion timestamp, the result is a proxy rather than exact
NCCL kernel time.

## Megatron Context

`MegatronWorkAdapter` adds stable 1F1B and MoE metadata without importing or
modifying Megatron:

```python
from overlap_monitor import EventType, MegatronWorkAdapter, WorkHandleRecorder

adapter = MegatronWorkAdapter(
    WorkHandleRecorder(),
    rank=rank,
    stage_id=pp_rank,
    device_id=device,
)

dispatch_work = adapter.wrap_a2a(
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

Attach wrappers only where the original code already returns async Work. Do not
change routing, collective type, stream dependencies, or schedule order in a
baseline run. `wrap_pipeline()` supports forward/backward send and receive;
`region()` marks iteration, forward, backward, expert, or drain intervals.

Region timestamps must share the analyzed clock domain. Use
`EventType.COMPUTE` only for measured compute, not a semantic boundary.

## PyTorch Profiler

The parser accepts profiler-like events without importing PyTorch itself:

```python
from overlap_monitor.profiler import PyTorchProfilerEventParser

events = PyTorchProfilerEventParser().parse(
    profiler.events(),
    rank=rank,
    stage_id=pp_rank,
)
```

It classifies NCCL, GEMM, attention, memory, and unknown events. Review unknown
kernels before using coverage-sensitive metrics such as bubble ratio.

## CUPTI Activity

Use CUPTI when GPU kernel intervals, stream concurrency, and correlation IDs are
required. The native collector is optional and isolated from the Python package.

Follow [CUPTI measurement](cupti_measurement.md) for build, runtime collection,
dropped-record handling, and the direct analysis command.

## Analyze The Output

```bash
overlap-monitor analyze \
  --input events_rank0.jsonl \
  --rank 0 \
  --stage-id 0 \
  --output-json summary_rank0.json \
  --trace-json timeline_rank0.json \
  --table
```

The default `critical-path` mode reports communication hiding. Use
`--mode timeline` for symmetric interval overlap and pipeline-stage metrics.
Read [measurement semantics](measurement.md) before comparing these ratios.

## Production Checklist

- Profile a bounded window after warmup.
- Record rank, stage, iteration, microbatch, phase, and `comm_id`.
- Keep one clock domain per analysis unless timestamps were explicitly aligned.
- Reject incomplete CUPTI traces for reported results.
- Benchmark monitor-off and monitor-on throughput with the same workload.
- Cross-check one representative trace with Nsight Systems.
