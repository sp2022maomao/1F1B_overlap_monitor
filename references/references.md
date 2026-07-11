# 参考文献与资料

## 1. NVIDIA / Megatron 官方资料

### Megatron-Core MoE 文档

链接：https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/features/moe.html

作用：

- 说明 Megatron-Core MoE 支持 Expert Parallelism、token dispatcher、GroupedGEMM、A2A overlap 等特性。
- 与本项目最直接相关的是 `--overlap-moe-expert-parallel-comm` 和 `--delay-wgrad-compute`。
- 这个文档是判断 “1F1B A2A overlap 是否按 Megatron 官方方式打开” 的第一参考。

### Megatron Bridge Communication Overlap 文档

链接：https://docs.nvidia.com/nemo/megatron-bridge/latest/training/communication-overlap.html

作用：

- 说明训练系统中通信和计算 overlap 的配置方式。
- 虽然重点不只 MoE A2A，但它给出了 Megatron 系列软件栈里 overlap 配置的官方语义。

### Megatron-Core Technical Report

链接：https://arxiv.org/abs/2603.07685

作用：

- 这是 Megatron-Core 系统设计的总报告。
- 和本 monitor 相关的重点是 MoE training optimization、expert parallel communication、1F1B FWD/BWD overlap。
- 它说明 1F1B overlap 的目标不是改变模型语义，而是重排/合并 forward-backward 执行，让 EP A2A 通信被其他 microbatch 的计算隐藏。

## 2. Pipeline / 1F1B 相关论文

### GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

链接：https://arxiv.org/abs/1811.06965

作用：

- 早期 pipeline parallelism 代表工作。
- 关注 microbatch、pipeline bubble、stage utilization。
- 对本项目的意义：1F1B overlap monitor 里的 “pipeline bubble / exposed time” 思路可追溯到这类 pipeline 利用率分析。

### PipeDream: Generalized Pipeline Parallelism for DNN Training

链接：https://dl.acm.org/doi/10.1145/3341301.3359646

作用：

- 代表性 1F1B pipeline schedule 工作。
- 对本项目的意义：1F1B 的核心是让 forward/backward 交错执行，提高流水线利用率；Megatron 的 MoE A2A overlap 可以看作把这种思想用于 MoE EP 通信隐藏。

### Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

链接：https://arxiv.org/abs/2104.04473

作用：

- Megatron-LM 系统论文。
- 讨论 tensor / pipeline / data parallel 组合，以及大模型训练中的通信优化。
- 对本项目的意义：提供 Megatron 系列训练系统中 pipeline schedule 和通信隐藏的背景。

## 3. MoE / A2A / 通信隐藏相关论文与实现

### DeepSpeed ZeRO

链接：https://arxiv.org/abs/1910.02054

作用：

- 重点是 optimizer states、gradient、parameter partition。
- 对本项目的意义：虽然不是 MoE A2A，但 exposed communication、communication overlap、distributed optimizer memory/communication tradeoff 都是相同系统问题。

### DeepSpeed-MoE

链接：https://arxiv.org/abs/2201.05596

作用：

- MoE 训练和推理系统代表工作。
- 对本项目的意义：MoE 的核心通信瓶颈通常来自 token dispatch / combine 的 AllToAll。

### DeepSeek-V3 Technical Report

链接：https://arxiv.org/abs/2412.19437

作用：

- 报告中讨论 DualPipe、MoE 训练和通信隐藏。
- 对本项目的意义：提供 “通过 pipeline / 双向调度隐藏通信” 的现代 MoE 系统背景。

### DeepEP

链接：https://github.com/deepseek-ai/DeepEP

作用：

- DeepSeek 的专家并行通信库。
- 对本项目的意义：它是 MoE AllToAll / dispatch-combine 优化实现的重要参考，但本项目 baseline monitor 不应直接改变 Megatron 原生 A2A 语义。

### FLUX

链接：https://github.com/bytedance/flux

作用：

- GPU 通信和计算 overlap 的开源实现参考。
- 对本项目的意义：可参考它如何把 collective communication 与 GEMM 调度到可重叠路径，但本 monitor 只记录指标，不改训练 kernel。

## 4. Timeline / profiler 工具资料

### NVIDIA Nsight Systems

链接：https://docs.nvidia.com/nsight-systems/UserGuide/index.html

作用：

- 标准 GPU timeline 工具。
- 对本项目的意义：NCCL kernel、cuBLAS/cuBLASLt GEMM kernel、CUDA stream、NVTX range 的时间轴并发图都可以从 Nsight Systems trace 中得到。

### PyTorch Profiler

链接：https://docs.pytorch.org/docs/stable/profiler.html

作用：

- PyTorch 官方 profiler，可导出 Chrome trace。
- 对本项目的意义：快速实现第一版 monitor，记录 `aten::all_to_all_single`、NCCL kernel、GEMM kernel 的时间区间，再计算 overlap ratio。

## 5. 本项目推荐引用顺序

如果只需要短报告引用，建议优先引用：

1. Megatron-Core MoE 文档
2. Megatron-Core Technical Report
3. Megatron-LM large-scale training paper
4. PipeDream 或 GPipe
5. Nsight Systems / PyTorch profiler 文档

如果要写更研究化的说明，再补：

1. DeepSpeed-MoE
2. DeepSeek-V3 / DeepEP
3. FLUX
