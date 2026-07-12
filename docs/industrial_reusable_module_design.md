# overlap_monitor 工业级可复用模块说明

## 当前结论

`overlap_monitor` 已形成可安装、可测试、与 Megatron 解耦的复用模块。当前成熟度是“本地逻辑与接口验证完成，等待真实 GPU 对照验证”，不能仅凭 synthetic 测试称为生产验证完成。

它重点解决的问题是：

```text
Megatron/MoE 的通信和计算高度异步，简单 Python timer 或单 stream cudaEvent 容易测错。
```

因此模块同时支持两类分析：

1. `OverlapAnalyzer`：kernel timeline overlap，适合 profiler/Nsight 事件。
2. `CriticalPathOverlapAnalyzer`：critical-path overlap，适合 async Work / `work.wait()` 事件。

## 目录结构

```text
overlap_monitor/
├── adapters/
│   └── megatron.py
├── core/
│   ├── events.py
│   ├── intervals.py
│   ├── io.py
│   └── metrics.py
├── profiler/
│   ├── work_handle.py
│   ├── pytorch_profiler.py
│   ├── normalizer.py
│   └── cuda_event.py
├── analyzer/
│   ├── overlap.py
│   ├── critical_path.py
│   └── classifier.py
├── te_adapter/
│   └── te_adapter.py
├── visualization/
│   ├── ascii_timeline.py
│   ├── chrome_trace.py
│   └── summary_table.py
├── examples/
├── tests/
├── configs/
├── benchmarks/
│   └── benchmark_runtime.py
├── cli.py
└── README.md
```

## 模块边界

| 模块 | 职责 | 是否依赖 Megatron |
| --- | --- | --- |
| `core` | 通用事件、interval、JSONL IO、metric dataclass | 否 |
| `adapters` | 仅附加 Megatron rank/stage/microbatch 上下文 | 否 |
| `profiler` | 运行期事件采集和外部 profiler 解析 | 否 |
| `analyzer` | overlap、critical path、kernel 分类 | 否 |
| `te_adapter` | Transformer Engine kernel 名称识别 | 否 |
| `visualization` | trace、ASCII timeline、表格输出 | 否 |
| `cli.py` | 命令行分析入口 | 否 |

Megatron 接入应通过独立 adapter 完成，不应该把 Megatron import 放进核心模块。

## 核心事件模型

统一事件格式：

```text
timestamp_start
timestamp_end
device_id
rank
stage_id
event_type
name
metadata
```

关键 `event_type`：

```text
COMPUTE
COMMUNICATION
NCCL
GEMM
ATTENTION
MEMORY
WAIT
PIPELINE
UNKNOWN
```

`metadata` 推荐字段：

```text
microbatch_id
phase = dispatch | expert | combine | send_recv | drain
comm_id
measurement
```

## 异步准确性设计

### 不推荐的测法

```python
start.record()
dispatch()
end.record()
```

在异步执行里，这常常只反映 kernel launch 或某个 stream 的局部时间。

### 推荐测法

对 async communication Work 记录：

```text
launch_time
wait_start
wait_end
```

生成 Work 观测事件：

```text
COMMUNICATION: launch_time -> completion observation / wait-return proxy
WAIT:          wait_start  -> wait_end
```

若只有 `wait()` 返回这一 host-side 观测，报告标记：

```text
measurement_quality = estimated
communication_runtime_kind = host_wait_proxy
```

若同时取得 PyTorch profiler/Nsight NCCL kernel timeline，则优先使用 kernel 区间：

```text
communication_runtime = union(NCCL)
hidden_communication = NCCL ∩ COMPUTE
exposed_communication = communication_runtime - hidden_communication
measurement_quality = kernel_timeline
```

这样不会把 Work host proxy 与真实 NCCL kernel runtime 混为一谈。

## 工业化保护

- `Event` 拒绝 NaN、Inf 和反向时间区间。
- JSONL 原子写入，解析错误包含文件和行号。
- `MonitoringSession` 提供线程安全、有容量上限的事件缓冲。
- `WorkHandleRecorder` 支持并发记录、重复 `wait()` 和显式 `mark_completed()`。
- validator 拒绝未对齐的多 rank 时钟域和 orphan WAIT。
- 每份 critical-path summary 都携带质量等级、runtime 语义和 warning。

## 公共 API

```python
from overlap_monitor import (
    CriticalPathOverlapAnalyzer,
    Event,
    EventType,
    OverlapAnalyzer,
    WorkHandleRecorder,
    MegatronWorkAdapter,
    MonitoringSession,
)
```

### Critical path 分析

```python
summary = CriticalPathOverlapAnalyzer().analyze(events)
print(summary.exposed_communication)
print(summary.hidden_communication)
print(summary.overlap_ratio)
```

### Work handle 采集

```python
recorder = WorkHandleRecorder()
work = torch.distributed.all_to_all_single(..., async_op=True)
work = recorder.wrap(
    work,
    comm_id="mb0_dispatch",
    name="dispatch_a2a",
    stage_id=0,
    microbatch_id=0,
    phase="dispatch",
)
work.wait()
events = recorder.events()
```

`WorkHandleRecorder` 不调用全局 CUDA synchronize，不改变原始异步调度。

### Megatron 薄适配

```python
adapter = MegatronWorkAdapter(recorder, rank=rank, stage_id=pp_rank, device_id=device)
work = adapter.wrap_a2a(
    original_work,
    comm_id=f"iter{iteration}-mb{microbatch_id}-dispatch",
    iteration=iteration,
    microbatch_id=microbatch_id,
    phase="dispatch",
)
```

训练代码只需在 async Work 返回处加这一层；分析、校验、存储均在库内或离线执行。

## CLI 用法

```bash
python3 -m overlap_monitor.cli analyze \
  --input overlap_monitor/examples/critical_path_events.jsonl \
  --mode critical-path \
  --table \
  --ascii
```

输出 JSON：

```bash
python3 -m overlap_monitor.cli analyze \
  --input overlap_monitor/examples/critical_path_events.jsonl \
  --mode critical-path \
  --output-json /tmp/critical_path_summary.json
```

## 验证方法

本地验证：

```bash
python3 -m unittest discover \
  -s overlap_monitor/tests \
  -p 'test_*.py'
```

当前 synthetic 验证覆盖：

- event parser
- timeline overlap
- critical path exposed/hidden communication
- Work handle wrapper
- TE kernel classification
- Chrome trace / ASCII timeline
- CLI summary output
- 多 rank clock-domain 拒绝策略
- 显式 completion 与 late wait 场景
- NCCL kernel timeline 优先级
- bounded session 与原子 JSONL IO
- Megatron metadata adapter

完整结果见 `docs/validation_report.md`。

## 已知边界

`WorkHandleRecorder` 在没有底层 CUDA/NCCL completion timestamp 时，用 `wait_end` 结束 host-side 观测窗口，并标记为 `host_wait_proxy`。默认 ProcessGroupNCCL 下该时间既不保证是 completion，也不保证是上界，不能将其作为精确 kernel runtime 或 GPU stall 引用。

如果需要更精确的通信 kernel runtime，应同时导入 PyTorch profiler 或 Nsight Systems 事件，再用 `OverlapAnalyzer` 做 kernel timeline overlap，对照 critical-path 结果。

## Megatron 接入建议

现有 `adapters/megatron.py` 只做三件事：

1. 在 A2A async Work 返回后调用 `WorkHandleRecorder.wrap(...)`。
2. 填入 `stage_id / microbatch_id / phase / rank / device_id`。
3. iteration 结束后导出 JSONL，并离线调用 CLI 或 analyzer。

这样可以保持 benchmark 调度真实性，也方便与原始训练代码解耦。
