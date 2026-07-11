# 文件介绍列表

## README.md

用途：总览文件夹目的，说明 1F1B / A2A overlap monitor 的核心指标和本地代码入口。

适合阅读对象：第一次看这个 monitor 设计的人。

## references.md

用途：列出设计参考的官方文档、论文和 profiling 工具文档。

重点内容：

- NVIDIA Megatron-Core MoE 文档
- Megatron-Core MoE technical report
- Megatron-LM / GPipe / PipeDream / DeepSpeed / DeepSeek-V3 等系统论文
- Nsight Systems / PyTorch profiler 这类 timeline 工具文档

## implementation_map.md

用途：把论文/文档和可检查的实现入口对应起来。

重点内容：

- Megatron-Core 1F1B combined schedule 对照文件
- MoE token dispatcher / AllToAll 对照文件
- DeepEP / FLUX / Nsight / PyTorch profiler 等外部实现或工具入口

## metric_design.md

用途：定义 overlap monitor 应该记录哪些字段，以及如何计算。

重点内容：

- `alltoall_comm_time`
- `gemm_time`
- `overlap_time`
- `alltoall_overlap_ratio`
- `exposed_comm_time`
- timeline / chrome trace 输出建议

## references.bib

用途：简化 BibTeX，方便后续写报告或论文引用。

注意：这里只放短条目和 URL，正式论文版本建议之后再按目标格式补 DOI、会议名和页码。
