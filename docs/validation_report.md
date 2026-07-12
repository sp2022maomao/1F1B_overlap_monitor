# overlap_monitor 验证报告

## 验证范围

本报告记录 v0.3.0 的纯 Python 基础验证。后续 RTX 4090 CUPTI smoke test
见 [`validation_4090_20260713.md`](validation_4090_20260713.md)。TE 2.7.0、
Megatron 真实训练吞吐和 Nsight 精度对照仍未完成。

## 结果

| 项目 | 结果 |
| --- | --- |
| 单元/集成测试 | 31/31 通过 |
| Python compileall | 通过 |
| 公共包导入 | 通过 |
| wheel 构建与隔离安装 | `overlap_monitor-0.3.0-py3-none-any.whl` 通过 |
| torch/Megatron/TE 直接依赖扫描 | 0 处 |
| synthetic critical-path 结果 | 符合预期 |
| mixed-rank clock-domain 防护 | 通过 |
| wait-return host proxy 识别 | 通过 |
| NCCL timeline 优先策略 | 通过 |
| CUPTI JSONL schema/parser | synthetic trace 通过 |
| CUPTI dropped-record 防护 | 通过 |
| native CUPTI collector | NVIDIA CUDA 12.8 headers 语法检查通过；未链接/运行 |

Synthetic 示例结果：

```text
communication_runtime = 80 us
exposed_communication = 35 us
hidden_communication = 45 us
overlap_ratio = 0.5625
measurement_quality = estimated
communication_runtime_kind = host_wait_proxy
```

这里特意标为 `estimated/host_wait_proxy`，因为示例没有真实 NCCL completion timestamp。

## CPU 微基准

环境：本地 `/usr/bin/python3 -S`，10,000 次操作，5 次取中位数。

| 指标 | 结果 |
| --- | ---: |
| bare fake Work.wait | 58.7 ns/call |
| recorder wrap + wait | 4,594.9 ns/call |
| recorder 增量成本 | 4,536.2 ns/call，约 4.54 us |
| 10,000 events 离线分析 | 21.67 ms |

该微基准只测 CPU 代码路径，不能推导 GPU iteration speedup。实际运行开销通常还受 PyTorch profiler 采样策略影响。

## 有效性判断

已证明：事件模型、区间算法、Work/wait 关联、质量分级、CLI、JSONL 和 adapter 在 synthetic 场景有效；模块不会导入或修改原始 Megatron。

尚未证明：native collector 在 CUDA 12.9 上的编译/运行，以及在 8×A100、TE 2.7.0、真实 1F1B/MoE A2A 下，CUPTI 指标与 Nsight Systems 的误差范围和训练吞吐开销。

## GPU 验收门槛

1. 同一短 workload 分别运行 monitor off、Work-only、PyTorch profiler/Nsight。
2. 至少预热 10 step、采样 20 step，比较 iteration p50/p95 和 tokens/s。
3. 将 NCCL/GEMM overlap、exposed wait 与 Nsight timeline 对齐。
4. 建议门槛：关键 interval 误差小于 5%，monitor-on 吞吐下降小于 2%，无额外全局 CUDA synchronize。
5. 达标后再将状态从“本地验证”升级为“GPU 验证完成”。
