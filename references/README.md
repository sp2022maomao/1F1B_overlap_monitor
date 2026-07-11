# 1F1B / A2A Overlap Monitor 参考文献

这个文件夹用于说明 `1F1B_overlap_monitor` 的设计依据：为什么要测 NCCL AllToAll 和 GEMM 的时间重叠、应该看哪些论文/文档、已有实现在哪里、以及最终 monitor 输出哪些指标。

核心结论：

- Megatron-Core MoE 的 1F1B A2A overlap 对应启动参数主要是 `--overlap-moe-expert-parallel-comm` 和 `--delay-wgrad-compute`。
- 监测对象不是普通 iteration time，而是 AllToAll 通信 kernel 与 Expert GEMM / MLP compute kernel 在 GPU timeline 上的并发区间。
- 推荐的数值指标是论文和系统 profiling 中常见的 communication-computation overlap ratio：

```text
alltoall_overlap_ratio = overlap_time(AllToAll NCCL, Expert GEMM) / alltoall_comm_time
exposed_comm_time      = alltoall_comm_time - overlap_time
```

## 文件列表

- `file_list.md`：本文件夹内容说明。
- `references.md`：论文、官方文档和工具文档列表。
- `implementation_map.md`：已有实现/源码入口和本地 Megatron 对照文件。
- `metric_design.md`：monitor 应输出的指标定义和计算方法。
- `references.bib`：可复制到论文/报告里的简化 BibTeX 条目。

## 最相关的本地代码入口

本地仓库中和 1F1B / A2A overlap 最直接相关的文件：

- `Megatron-LM/megatron/core/pipeline_parallel/combined_1f1b.py`
- `Megatron-LM/megatron/core/pipeline_parallel/schedules.py`
- `Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py`
- `Megatron-LM/megatron/core/model_parallel_config.py`
