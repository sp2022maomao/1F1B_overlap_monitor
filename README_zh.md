# 1F1B Overlap Monitor

[![tests](https://github.com/sp2022maomao/1F1B_overlap_monitor/actions/workflows/tests.yml/badge.svg)](https://github.com/sp2022maomao/1F1B_overlap_monitor/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

`overlap-monitor` 用于分析 Megatron/MoE 和 1F1B 异步训练中的通信-计算重叠。它可以直接读取统一 Event JSONL 或项目内置 CUPTI collector 生成的原始记录，输出 NCCL 运行时间、隐藏/暴露通信、overlap ratio 和 stage 级指标。

> **当前状态：**开源 Alpha（`0.3.0`）。离线分析链路和双 RTX 4090 CUPTI 小规模测试已验证；CUDA 12.9、Transformer Engine 2.7、真实 Megatron 1F1B 以及 Nsight 精度对照仍待完成。

## 要解决的问题

MoE AllToAll 等通信通常异步发起。Python 函数或 `Work.wait()` 返回时，NCCL kernel 不一定已经在 GPU 上完成。因此，Python timer、单 stream CUDA event、Work/wait 和 GPU kernel timeline 测到的不是同一件事。

本项目将三个量分开：

```text
通信实际运行时间
被计算隐藏的通信
未被隐藏的暴露通信
```

分析器优先使用 GPU kernel timeline。没有 timeline 时，`Work.wait()` 只作为 host-side proxy，不会被标记成精确 NCCL 时间。

## 安装

```bash
git clone https://github.com/sp2022maomao/1F1B_overlap_monitor.git
cd 1F1B_overlap_monitor
python3 -m pip install .
```

离线分析器没有强制的第三方依赖。Native CUPTI 采集需要 Linux、CMake 和带 CUPTI 的 CUDA Toolkit。

## 30 秒上手

直接分析仓库自带的 CUPTI 样例：

```bash
overlap-monitor analyze \
  --input examples/traces/cupti_activity.jsonl \
  --rank 0 \
  --stage-id 0 \
  --table \
  --output-json summary.json
```

`analyze` 会自动识别原始 CUPTI JSONL 和统一 Event JSONL，并在分析前做完整性检查。规范化时间戳和所有 duration 字段的单位都是微秒（`us`）。

分析 Work/wait 样例：

```bash
overlap-monitor analyze \
  --input examples/traces/critical_path_events.jsonl \
  --table \
  --ascii
```

Python API 与 CLI 共用同一套加载和校验逻辑：

```python
from overlap_monitor import analyze_trace

result = analyze_trace(
    "examples/traces/cupti_activity.jsonl",
    rank=0,
    stage_id=0,
)
print(result.communication_hidden_ratio)
```

## 分析真实 Trace

常用的 CUPTI 主流程只需一条命令：

```bash
overlap-monitor analyze \
  --input cupti_rank0.jsonl \
  --rank 0 \
  --stage-id 0 \
  --events-output events_rank0.jsonl \
  --output-json summary_rank0.json \
  --trace-json timeline_rank0.json \
  --table
```

`--events-output` 可选，用于保存实际参与分析的规范化事件。原有 `import-cupti` 两步流程仍然保留兼容。

CUPTI 丢记录或时间戳缺失时默认拒绝分析。`--allow-incomplete` 只适合调试，不能用于正式实验数据。

已有 Event JSONL 时：

```bash
overlap-monitor validate --input events_rank0.jsonl
overlap-monitor analyze --input events_rank0.jsonl --table
```

`--rank`、`--device-id` 和 `--stage-id` 用于选择同一时钟域中的事件。默认 critical-path 模式输出隐藏/暴露通信；`--mode timeline` 用于对称 timeline overlap 和 pipeline stage 指标。

## 三类 Profiling 工具的粒度

| 工具 | 数据粒度 | 主要用途 |
| --- | --- | --- |
| CUPTI | 底层 CUDA activity / kernel | 提供 NCCL 和计算 kernel 原始区间 |
| Nsight Systems | CPU/GPU/stream 系统级时间轴 | 跨 stream 观察和精度对照 |
| PyTorch Profiler | operator/module/autograd 框架级 | 对应 PyTorch 算子语义和 CUDA 耗时 |

CUPTI 数据最底层，Nsight Systems 是系统级综合时间轴，PyTorch Profiler 最容易对应框架算子。三者是互补数据源，不是等价计时器。

## 核心指标

| 指标 | 含义 |
| --- | --- |
| `communication_runtime` | NCCL/通信 kernel 区间并集 |
| `hidden_communication` | 与已识别计算并发的通信 |
| `exposed_communication` | 没有被计算覆盖的通信 |
| `communication_hidden_ratio` | `hidden_communication / communication_runtime` |
| `timeline_overlap_ratio` | `overlap_time / min(compute_time, communication_time)` |
| `overlap_ratio` | 当前模式明确比例字段的兼容别名 |
| `wait_time` | Work 在主机侧的等待 proxy |
| `critical_path_span` | 本次分析覆盖的活动时间范围 |

所有输出都包含 `overlap_ratio_definition` 和校验报告。critical-path 输出还包含
`measurement_quality` 和 `communication_runtime_kind`。`kernel_timeline` 表示真实 GPU
时间轴；`estimated/host_wait_proxy` 不能当作精确 NCCL runtime。

## CUPTI 采集

```bash
cmake -S native/cupti_collector -B build/cupti -DCMAKE_BUILD_TYPE=Release
cmake --build build/cupti --parallel
```

完整采集方式、JSONL schema、丢记录处理和验证方法见 [CUPTI 测量文档](docs/cupti_measurement.md)。

## 架构

```text
采集器或 profiler trace
  -> 解析与 kernel 分类
  -> 统一 Event JSONL
  -> 数据验证
  -> timeline / critical-path 分析
  -> JSON / Markdown / ASCII / Chrome trace
```

核心分析模块不导入 Megatron、PyTorch、Transformer Engine、Nsight、CUDA 或 CUPTI，运行时集成通过薄 adapter 完成。

## 文档与开发

文档入口：[docs/README.md](docs/README.md)。

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
ruff check .
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m build
```

开发规范见 [CONTRIBUTING.md](CONTRIBUTING.md)，路线图见 [ROADMAP.md](ROADMAP.md)。项目使用 [MIT License](LICENSE)。
