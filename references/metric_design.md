# Metric Semantics

This document defines the metrics emitted by `overlap-monitor` 0.3.x. The
definitions follow the implementation in `OverlapAnalyzer` and
`CriticalPathOverlapAnalyzer`.

## Event Requirements

All events compared in one analysis must use the same clock domain and time
unit. The bundled recorder uses microseconds, but the analyzers operate on
unitless intervals and preserve the input unit.

Recommended metadata includes:

```text
rank
device_id
stage_id
microbatch_id
phase              # dispatch, expert, combine, forward, backward
comm_id             # correlates launch, completion, and wait
measurement         # kernel_timeline, async_work_lifetime, critical_path_wait
runtime_kind        # observed_kernel_runtime, observed_work_window, upper_bound
```

Before analysis, intervals of the same class are merged. This prevents
concurrent events in the same class from being counted more than once.

## Timeline Mode

Use timeline mode when communication and compute intervals come from a GPU
trace. Let `C` be the union of compute intervals and `N` the union of
communication intervals:

```text
compute_time       = duration(C)
communication_time = duration(N)
overlap_time       = duration(intersection(C, N))
overlap_ratio      = overlap_time / min(compute_time, communication_time)
```

The ratio answers: "How much of the smaller activity class overlaps the other
class?" It is symmetric and bounded by `[0, 1]`. It is not the same as the
fraction of communication hidden from the critical path.

Per-stage metrics use the same equations. Pipeline metrics are:

```text
stage_balance = min(stage_active_span) / max(stage_active_span)
bubble_ratio  = weighted idle fraction across stages
```

Here, idle time means a gap between the first and last classified compute or
communication event in a stage. Uninstrumented work can therefore appear as a
bubble; interpret this metric only when event coverage is known.

## Critical-Path Mode

Critical-path mode is designed for asynchronous distributed `Work` handles.
It separates communication lifetime from time spent blocking at `wait()`.

### Exact kernel timeline available

When NCCL kernel intervals are present, they take precedence over Work-lifetime
events:

```text
communication_runtime = duration(union(NCCL kernel intervals))
hidden_communication   = duration(intersection(compute, NCCL))
exposed_communication  = communication_runtime - hidden_communication
overlap_ratio          = hidden_communication / communication_runtime
measurement_quality    = kernel_timeline
```

This reports observed kernel overlap. "Exposed" here means communication not
covered by classified compute; proving that every uncovered interval lies on
the end-to-end iteration critical path still requires schedule context.

### Work/wait observations only

When the collector observes launch and `wait()` but no kernel timeline:

```text
wait_time              = duration(intersection(work_lifetime, wait_interval))
exposed_communication  = wait_time
hidden_communication   = communication_runtime - exposed_communication
overlap_ratio          = hidden_communication / communication_runtime
measurement_quality    = estimated
```

If completion is first observed when `wait()` returns, the launch-to-return
window is marked `upper_bound`. It can include queueing, unrelated host work,
and time after the underlying collective completed. In that case,
`communication_runtime` and `hidden_communication` are estimates; `wait_time`
is the directly observed blocking interval.

### Generic fallback

Without precise NCCL or `wait()` events, the analyzer intersects generic
communication and compute intervals and emits `timeline_fallback` plus a
warning. Do not compare fallback results directly with kernel-timeline results
without stating the evidence difference.

## Grouping

Critical-path metrics are also computed for:

```text
stage_id / microbatch_id / phase
```

Producers should populate all three fields. Missing values are grouped under
explicit `unknown_*` labels rather than silently discarded.

## Reporting Checklist

Every published result should include:

- analyzer mode and `overlap-monitor` version;
- Megatron, PyTorch, CUDA, NCCL, and Transformer Engine versions;
- rank, stage, microbatch, and profiler window;
- timestamp unit and clock domain;
- `measurement_quality` and `communication_runtime_kind`;
- profiler configuration and measured instrumentation overhead;
- both absolute durations and ratios.

Do not report Python launch duration as NCCL runtime, combine all NCCL
collectives under an A2A label, or introduce global CUDA synchronization in the
measured path.
