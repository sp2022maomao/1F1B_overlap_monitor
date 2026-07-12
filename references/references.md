# Annotated Bibliography

This bibliography prioritizes original papers, official documentation, and
canonical repositories. Entries are grouped by the role they play in the
design of `overlap-monitor`.

## Megatron and Communication Overlap

- [Megatron-Core MoE User Guide](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/moe.html), NVIDIA. Official reference for expert parallelism, token dispatchers, and MoE overlap features.
- [Megatron Bridge: Communication Overlap](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/communication-overlap.html), NVIDIA. Configuration-level overview of communication overlap in the Megatron software stack.
- [Megatron-Core Technical Report](https://arxiv.org/abs/2603.07685), NVIDIA Megatron-Core contributors. System-level context for current Megatron-Core training techniques.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), Narayanan et al. Background on combined tensor, pipeline, and data parallel training.

## Pipeline Scheduling

- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965), Huang et al. Introduces microbatch pipeline execution and bubble analysis.
- [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://doi.org/10.1145/3341301.3359646), Narayanan et al. Foundational reference for interleaved forward/backward pipeline scheduling.

These papers provide scheduling context. They do not by themselves define the
project's overlap formulas; those definitions are documented in
[Metric semantics](metric_design.md).

## MoE Communication Systems

- [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596), Rajbhandari et al. Discusses scalable MoE training and inference, where token dispatch and combine communication are central costs.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), DeepSeek-AI. Describes a modern MoE training system and communication-hiding schedule.
- [DeepEP](https://github.com/deepseek-ai/DeepEP), DeepSeek-AI. Canonical repository for an expert-parallel communication library.
- [FLUX](https://github.com/bytedance/flux), ByteDance. Canonical repository for communication-computation fusion and overlap kernels.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), Rajbhandari et al. Related background on distributed training communication and state partitioning; not a direct A2A-overlap reference.

DeepEP and FLUX change execution or communication behavior. They are useful
comparison systems but should not be silently introduced into an unmodified
Megatron baseline.

## Profiling and Trace Infrastructure

- [NVIDIA CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html). Primary API reference for asynchronous CUDA activity records, correlation IDs, and dropped-record accounting.
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html). Primary reference for CUDA, NCCL, NVTX, and system timeline collection.
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html). Official API documentation for collecting operator and CUDA activity traces.
- [Kineto](https://github.com/pytorch/kineto). Canonical repository for the profiling backend used by PyTorch.
- [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis). Offline analysis tooling for distributed training traces.
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests). Useful for validating collective performance independently of the training framework.

## Suggested Citation Set

For a short systems report, cite:

1. the Megatron-Core MoE documentation or technical report;
2. the Megatron-LM paper;
3. GPipe or PipeDream for pipeline scheduling;
4. Nsight Systems or PyTorch Profiler for the trace methodology;
5. the specific MoE communication backend used in the experiment.

The accompanying [BibTeX file](references.bib) is a starting point. Verify
metadata against the target venue's required citation format before publishing.
