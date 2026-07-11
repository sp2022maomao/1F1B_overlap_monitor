# 1F1B overlap 测量方法与异步误差设计

## 结论

Mixtral/MoE 训练里的 A2A、GEMM、pipeline schedule 通常是异步执行的。直接在原始训练代码里用 Python timer 或简单 `cudaEventRecord(start/end)` 包住某个函数，容易测到 launch 开销或单 stream 局部时间，而不是训练 iteration 关键路径上的真实通信开销。

因此当前主线采用两层设计：

1. `OverlapAnalyzer`：分析 profiler/kernel timeline 的区间 overlap。
2. `CriticalPathOverlapAnalyzer`：分析异步 Work 生命周期和 `work.wait()` 暴露等待，更适合 Megatron/MoE 的 critical path overlap。

## 常见方法

| 方法 | 能回答的问题 | 主要风险 |
| --- | --- | --- |
| Python timer | 代码段墙钟时间 | 容易只测 launch 或 CPU 调度 |
| 简单 CUDA event | 某个 stream 上的局部时间 | 跨 stream/async dependency 容易错 |
| Nsight Systems | 完整 GPU timeline | 开销较高，适合离线验证 |
| Kernel interval overlap | NCCL/GEMM 是否在时间线上相交 | 不一定等价于 iteration critical path |
| Work handle / critical path | 通信是否真的阻塞训练推进 | 需要在 async Work 创建和 wait 位置插桩 |

## 推荐指标

不要只报告 A2A kernel duration，应该同时报告：

- `communication_runtime`：有 kernel timeline 时为 NCCL 区间；只有 Work 时为带质量标记的生命周期观测窗口。
- `hidden_communication`：没有暴露在关键路径上的通信。
- `exposed_communication`：`work.wait()` 等待中仍未完成的通信。
- `overlap_ratio`：`hidden_communication / communication_runtime`。
- `critical_path_span`：当前 stage/microbatch/phase 的关键路径时间范围。

## 设计如何避免异步测量不准确

### 1. 不强制全局同步

运行期 recorder 不调用 `cudaDeviceSynchronize()`，避免为了测量而破坏 Megatron 原始 overlap 行为。

### 2. 包装 Work，而不是包住 Python 函数

在 A2A 返回 async Work 的位置记录：

```text
launch_time
```

在 `work.wait()` 前后记录：

```text
wait_start
wait_end
```

由此生成两个事件；若 completion 仅在 wait 返回时得到，它是上界：

```text
COMMUNICATION: launch_time -> observed completion
WAIT:          wait_start  -> wait_end
```

`WAIT` 是关键路径阻塞证据；精确 NCCL runtime 和 NCCL/GEMM overlap 仍以 profiler/Nsight kernel timeline 为准。

### 3. 将 kernel overlap 和 critical path overlap 分开

kernel timeline 可以回答：

```text
NCCL 和 GEMM 是否重叠
```

critical path 可以回答：

```text
通信是否真的拖慢 iteration
```

这两个问题相关，但不等价。

### 4. 以 stage/microbatch/phase 分组

事件 metadata 建议至少包含：

```text
stage_id
microbatch_id
phase = dispatch | combine | send_recv | drain
rank
device_id
```

这样可以区分 1F1B pipeline 中不同 stage 的暴露通信、bubble 和 imbalance。

## 独立代码位置

当前独立代码库：

```text
overlap_monitor/
```

新增关键模块：

```text
analyzer/critical_path.py
profiler/work_handle.py
tests/test_critical_path.py
```

这部分不依赖 Megatron，不修改原始 benchmark。后续接入 Megatron 时，只需要在原始 A2A async Work 返回位置增加一个薄 adapter。
