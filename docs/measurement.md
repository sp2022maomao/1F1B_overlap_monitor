# Measurement Semantics

Asynchronous training exposes several different notions of time. Python wall
time, CUDA events, `Work.wait()`, CUPTI, and Nsight Systems are complementary;
they are not interchangeable timers.

## Event Requirements

Events analyzed together must share a timestamp unit and aligned clock domain.
The CLI normalizes supported inputs to microseconds (`us`). Recommended context
is:

```text
rank, device_id, stage_id
iteration, microbatch_id, phase
comm_id, measurement, runtime_kind
```

Intervals of the same class are merged before metrics are calculated.

## Timeline Mode

Use timeline mode for aligned GPU compute and communication intervals. Let `C`
be the union of compute intervals and `N` the union of communication intervals:

```text
compute_time       = duration(C)
communication_time = duration(N)
overlap_time       = duration(intersection(C, N))
timeline_overlap_ratio
                   = overlap_time / min(compute_time, communication_time)
overlap_ratio      = timeline_overlap_ratio  # compatibility alias
```

This symmetric ratio answers how much of the smaller activity class overlaps
the other. It does not directly mean that communication is hidden from the
training critical path.

Pipeline metrics are derived from classified stage intervals:

```text
stage_balance = min(stage_active_span) / max(stage_active_span)
bubble_ratio  = weighted idle fraction inside stage spans
```

Uninstrumented work appears idle, so bubble ratio is valid only when trace
coverage is known.

## Critical-Path Mode

Critical-path mode reports the fraction of communication hidden under the
available evidence:

```text
communication_hidden_ratio
              = hidden_communication / communication_runtime
overlap_ratio = communication_hidden_ratio  # compatibility alias
```

The denominator intentionally differs from timeline mode. New integrations
should consume the explicit mode-specific field. Every JSON summary includes
`overlap_ratio_definition`, while `overlap_ratio` remains compatible with
earlier releases.

### NCCL kernel timeline available

Observed NCCL kernel intervals take precedence over host Work events:

```text
communication_runtime = duration(union(NCCL))
hidden_communication   = duration(intersection(compute, NCCL))
exposed_communication  = communication_runtime - hidden_communication
measurement_quality    = kernel_timeline
```

Here, exposed means not covered by classified compute. Proving that every
uncovered interval lies on the end-to-end iteration critical path still
requires schedule and dependency context.

### Work/wait observations only

When only asynchronous Work observations are available:

```text
wait_time              = duration(intersection(work_lifetime, wait_interval))
exposed_communication  = wait_time
hidden_communication   = communication_runtime - exposed_communication
measurement_quality    = estimated
runtime_kind           = host_wait_proxy
```

`Work.wait()` is the point where the program observes or establishes a
dependency on the operation. Under ProcessGroupNCCL it does not necessarily
mean the NCCL kernel has completed when the host call returns. The resulting
communication lifetime is therefore a host proxy, not precise kernel runtime.

### Generic fallback

Without NCCL kernels or wait events, the analyzer intersects generic
communication and compute events and emits `timeline_fallback` with a warning.
Do not compare this directly with kernel-timeline results without stating the
evidence difference.

## Source Selection

| Source | Best use | Limitation |
| --- | --- | --- |
| CUPTI Activity | precise GPU kernel intervals | needs semantic correlation and validation |
| Nsight Systems | system-wide reference timeline | heavier offline workflow |
| PyTorch Profiler | framework/operator attribution | trace overhead and framework coverage |
| Work/wait adapter | low-overhead dependency observation | host proxy, not exact NCCL duration |

## Reporting Checklist

Report the following with every experiment:

- analyzer mode and package version;
- absolute durations and ratio;
- `overlap_ratio_definition`;
- `measurement_quality` and `communication_runtime_kind`;
- rank, stage, microbatch, phase, and profiling window;
- CUDA, NCCL, PyTorch, TE, Megatron, and profiler versions;
- monitor-on overhead and dropped-record count;
- whether timestamps were aligned across ranks.

Do not introduce a global CUDA synchronization inside the measured path or
label Python launch duration as NCCL runtime.
