# 已有实现与源码入口对照

这个文件把 “论文/文档里的概念” 映射到 “能检查的实现入口”。这里列的是参考入口，不表示都要合入当前 baseline。

## 1. Megatron-Core / Megatron-LM 本地实现入口

### 1F1B combined schedule

文件：

`Megatron-LM/megatron/core/pipeline_parallel/combined_1f1b.py`

相关位置：

- line 18：`combined_1f1b_schedule_for_no_pipelining`
- line 34-41：说明 forward/backward co-schedule，用 backward compute 隐藏 forward EP A2A，反向同理。
- line 134-137：说明 combined forward/backward step 只在 `overlap_moe_expert_parallel_comm` 打开时调用。

为什么重要：

- 这是 Megatron 原生 1F1B MoE A2A overlap 的核心调度文件。
- monitor 如果要按 step / microbatch / layer 对齐，应该优先对照这个 schedule。

### schedule 入口选择

文件：

`Megatron-LM/megatron/core/pipeline_parallel/schedules.py`

相关位置：

- line 575：`config.overlap_moe_expert_parallel_comm` 打开时进入 `combined_1f1b_schedule_for_no_pipelining`。
- line 1302：interleaved pipeline 场景下进入 `combined_1f1b_schedule_for_interleaved_pipelining`。

为什么重要：

- 可以验证启动参数是否真的走到 1F1B combined schedule。
- 监测代码若只看到命令行参数，不如在这个路径上加轻量日志更可靠。

### MoE token dispatch / AllToAll

文件：

`Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py`

相关位置：

- line 675：`token_dispatch`
- line 690-703：Expert Parallel AllToAll dispatch，调用 `all_to_all` 交换 token 和 probability。
- line 997-1009：HybridEP dispatcher 的 dispatch/combine 流程说明。

为什么重要：

- AllToAll 阶段的通信时间窗要从这里或 profiler 里的对应 kernel/event 识别。
- 最好记录 dispatch 和 combine 两类 A2A，而不是只记录一个总通信时间。

### 配置项定义

文件：

`Megatron-LM/megatron/core/model_parallel_config.py`

相关位置：

- line 241：`overlap_moe_expert_parallel_comm`
- line 246：`delay_wgrad_compute`

为什么重要：

- 这两个开关共同决定 MoE EP A2A overlap 的调度和 weight-gradient 延迟策略。

## 2. 外部开源实现参考

### DeepEP

链接：https://github.com/deepseek-ai/DeepEP

参考点：

- MoE expert parallel dispatch/combine 通信库。
- 可参考其对 A2A 通信的抽象、event 记录、dispatch/combine 分离方式。

不建议直接用于 baseline 的原因：

- 本项目目标是 Megatron 原生 Mixtral baseline 的 overlap profiling。
- DeepEP 会改变通信 backend/实现路径，适合之后作为方法对比，不适合作为 original baseline。

### FLUX

链接：https://github.com/bytedance/flux

参考点：

- 面向 GPU collective communication 与 GEMM overlap 的实现。
- 可参考它如何区分 communication stream、compute stream，以及如何看 GEMM 与 collective 的并发。

不建议直接用于 baseline 的原因：

- FLUX 是优化实现，不是 monitor。
- 当前阶段只需要记录 Megatron 原生行为。

### NVIDIA Nsight Systems

链接：https://docs.nvidia.com/nsight-systems/UserGuide/index.html

参考点：

- CUDA kernel timeline。
- NCCL kernel 与 cuBLAS/cuBLASLt GEMM kernel 的可视化并发区间。
- NVTX range 可用于标记 train step、layer、dispatch、combine。

建议用途：

- 作为最终 profiling 的可信 trace。
- 用来验证 PyTorch profiler 或自定义 monitor 算出的 overlap ratio 是否可信。

### PyTorch Profiler

链接：https://docs.pytorch.org/docs/stable/profiler.html

参考点：

- 可导出 Chrome trace。
- 快速解析 `aten::all_to_all_single`、NCCL kernel、GEMM kernel。

建议用途：

- 第一版轻量 monitor。
- 输出 `jsonl/csv` 指标，再可选导出 trace 文件。

## 3. 本项目 monitor 推荐插入层级

推荐优先级：

1. 不改训练语义：使用 PyTorch profiler / Nsight Systems 收集 timeline。
2. 少量 NVTX 标记：只标记 step / layer / MoE dispatch / combine，不改通信逻辑。
3. 避免在 AllToAll 前后强制 `torch.cuda.synchronize()` 做正式性能测量，因为会破坏 overlap；只有 debug 校验时可以用。

## 4. 不建议作为 baseline 的改动

- 替换 token dispatcher backend。
- 使用 DeepEP/FLUX 作为 original baseline。
- 修改 MoE routing、expert assignment、AllToAll 实现。
- 为了测量方便长期保留 `torch.cuda.synchronize()`。
- 把 optimizer、TP/EP、tokenizer 改掉后还称为 original baseline。
