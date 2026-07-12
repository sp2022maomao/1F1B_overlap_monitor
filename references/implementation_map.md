# Implementation Map

This map connects the monitor's event model to upstream Megatron execution
paths. It identifies observation points; it does not require changes to
Megatron's communication or scheduling semantics.

## Megatron-Core

Canonical repository: [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

| Concern | Upstream path or symbol | Monitor use |
| --- | --- | --- |
| Combined forward/backward scheduling | [`combined_1f1b.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/combined_1f1b.py) | Associate events with 1F1B phases and microbatches. |
| Pipeline schedule selection | [`schedules.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py) | Confirm which schedule is active at runtime. |
| MoE token dispatch and combine | [`token_dispatcher.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py) | Attach dispatch/combine phase metadata to communication events. |
| Overlap configuration | [`model_parallel_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/model_parallel_config.py) | Record overlap-related configuration with each run. |

Prefer searching for these symbols instead of relying on line numbers:

```text
combined_1f1b_schedule_for_no_pipelining
combined_1f1b_schedule_for_interleaved_pipelining
overlap_moe_expert_parallel_comm
delay_wgrad_compute
token_dispatch
token_combine
```

The exact symbols available depend on the Megatron-Core revision. Every
experiment should record its upstream commit SHA.

## Observation Boundaries

The least invasive integration order is:

1. Import an existing PyTorch profiler or Nsight Systems timeline for exact GPU
   kernel intervals.
2. Add NVTX ranges for iteration, stage, microbatch, dispatch, expert compute,
   and combine when the trace lacks semantic boundaries.
3. Wrap asynchronous distributed `Work` handles to observe launch and blocking
   `wait()` behavior when a kernel timeline is unavailable.
4. Keep event normalization and overlap analysis offline.

The runtime collector must not add a global `torch.cuda.synchronize()` to a
performance run. Synchronization changes the schedule being measured and can
destroy the overlap under study.

## Related Systems

| Project | Relevant capability | Relationship to this project |
| --- | --- | --- |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | MoE dispatch/combine communication | Optimization backend and comparison target, not the baseline monitor. |
| [FLUX](https://github.com/bytedance/flux) | Collective/GEMM fusion and overlap | Implementation reference for communication-compute scheduling. |
| [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) | Operator and CUDA activity traces | Input source for normalized timeline events. |
| [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | System-wide CUDA/NCCL timeline | Preferred external validation source for GPU timing. |
| [Kineto](https://github.com/pytorch/kineto) | PyTorch profiling backend | Trace infrastructure, complementary to this project's analysis layer. |
| [CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html) | Direct CUDA kernel timestamps and correlation IDs | Native input path for observed NCCL/compute kernel overlap. |

## Baseline Integrity

Do not describe a run as an unmodified Megatron baseline after changing the
token dispatcher, routing policy, collective backend, parallel configuration,
or optimizer behavior. DeepEP and FLUX experiments should be reported as
separate configurations. Instrumentation overhead and profiler settings should
also be recorded with each result.
