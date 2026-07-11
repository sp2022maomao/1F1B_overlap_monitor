# 1F1B / A2A Overlap Monitor 指标设计

## 1. 监测目标

目标是回答：

```text
Megatron 原生 Mixtral MoE 训练中，EP AllToAll 通信有多少被 GEMM/MLP 计算隐藏？
```

因此 monitor 不只看单步耗时，还要看 GPU timeline 上两类事件的并发：

- Communication：NCCL AllToAll / all_to_all_single / dispatch-combine 通信 kernel。
- Compute：Expert MLP GEMM / GroupedGEMM / cuBLAS / cuBLASLt / TransformerEngine GEMM kernel。

## 2. 事件分类

### AllToAll 通信事件

来源：

- PyTorch profiler 中的 `aten::all_to_all_single`
- NCCL kernel 名称中的 `nccl`
- Megatron token dispatcher 中 dispatch / combine 附近的 NVTX range

建议字段：

```text
step
rank
layer_id
microbatch_id
phase            # dispatch / combine / unknown
start_us
end_us
duration_us
event_name
stream_id
```

### GEMM 计算事件

来源：

- cuBLAS / cuBLASLt kernel
- TransformerEngine GEMM kernel
- GroupedGEMM kernel
- profiler 中名称包含 `gemm`, `matmul`, `cublas`, `cutlass`, `grouped_gemm` 的 CUDA kernel

建议字段：

```text
step
rank
layer_id
microbatch_id
start_us
end_us
duration_us
event_name
stream_id
```

## 3. 核心指标

设 AllToAll 通信区间集合为 `C`，GEMM 计算区间集合为 `G`。

先对同类区间做 union，避免重叠 kernel 被重复计时：

```text
C_union = union(C)
G_union = union(G)
```

### AllToAll 通信时间

```text
alltoall_comm_time_us = total_duration(C_union)
```

### GEMM 时间

```text
gemm_time_us = total_duration(G_union)
```

### 重叠时间

```text
overlap_time_us = total_duration(intersection(C_union, G_union))
```

### AllToAll overlap ratio

这是最重要指标：

```text
alltoall_overlap_ratio = overlap_time_us / alltoall_comm_time_us
```

含义：

- `0.0`：AllToAll 完全暴露，没有被 GEMM 隐藏。
- `1.0`：AllToAll 完全被 GEMM 覆盖。

### Exposed communication time

```text
exposed_comm_time_us = alltoall_comm_time_us - overlap_time_us
```

含义：

- 这是仍然暴露在 critical path 上的通信时间。
- 论文/系统报告里经常称为 exposed communication 或 communication stall。

### 可选：总 NCCL overlap ratio

```text
nccl_overlap_ratio = overlap_time(NCCL, GEMM) / nccl_time
```

注意：

- 它比 `alltoall_overlap_ratio` 更宽泛，因为 NCCL 还可能包含 all-reduce、reduce-scatter、all-gather。
- MoE A2A 分析应优先看 `alltoall_overlap_ratio`。

## 4. 推荐输出文件

### 每步 JSONL

```text
overlap_metrics/overlap_rank0.jsonl
```

每行一个 step 或一个 profiler window：

```json
{
  "step": 120,
  "rank": 0,
  "alltoall_comm_time_us": 1593147.529,
  "gemm_time_us": 78624.553,
  "overlap_time_us": 18331.320,
  "alltoall_overlap_ratio": 0.01151,
  "exposed_comm_time_us": 1574816.209,
  "num_alltoall_events": 720,
  "num_gemm_kernels": 1014
}
```

### 汇总 CSV

```text
overlap_metrics/summary.csv
```

建议字段：

```text
step,rank,alltoall_comm_time_us,gemm_time_us,overlap_time_us,alltoall_overlap_ratio,exposed_comm_time_us,num_alltoall_events,num_gemm_kernels
```

### Timeline

建议同时保存：

```text
overlap_metrics/trace_rank0.json
overlap_metrics/nsys_rank0.qdrep
```

用途：

- JSON/CSV 给数值指标。
- trace 给时间轴图，方便复查 NCCL kernel 和 GEMM kernel 是否真的并发。

## 5. 解释口径

报告时建议使用这几句话：

```text
We quantify A2A communication-computation overlap using timeline overlap ratio:
overlap_time(AllToAll NCCL, Expert GEMM) / AllToAll communication time.
The remaining non-overlapped communication is reported as exposed communication time.
```

中文：

```text
我们用时间轴重叠比例衡量 A2A 通信-计算重叠：
AllToAll NCCL 与 Expert GEMM 的并发时间 / AllToAll 通信总时间。
未被 GEMM 覆盖的部分记为 exposed communication time。
```

## 6. 注意事项

- 单位建议统一用 `us`，最终展示可换算为 `ms`。
- 不能只看 Python 函数耗时；需要 CUDA kernel timeline。
- 不能把所有 NCCL 都当 AllToAll；最好结合 `aten::all_to_all_single` 或 NVTX range 区分。
- 正式性能测量不要在 AllToAll 前后强制 `torch.cuda.synchronize()`，否则会破坏 overlap。
- 如果 profiler 开销太大，可以只 profile 少量 step，例如 50-55。
