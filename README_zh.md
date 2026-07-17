# 1F1B Overlap Monitor

[![tests](https://github.com/sp2022maomao/1F1B_overlap_monitor/actions/workflows/tests.yml/badge.svg)](https://github.com/sp2022maomao/1F1B_overlap_monitor/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

`overlap-monitor` 用于分析 Megatron/MoE 和 1F1B 异步训练中的通信-计算重叠。
它可读取统一 Event JSONL 或原始 CUPTI activity 记录，并输出 NCCL 运行时间、
隐藏/暴露通信、overlap ratio 和 pipeline stage 指标。

> **状态：**开源 Alpha（`0.3.0`）。离线分析链路和双 RTX 4090 CUPTI
> 小规模测试已验证；CUDA 12.9、Transformer Engine 2.7、真实 Megatron 1F1B
> 和 Nsight 精度对照仍待完成。

## 安装

需要 Python 3.10 或更高版本。

```bash
git clone https://github.com/sp2022maomao/1F1B_overlap_monitor.git
cd 1F1B_overlap_monitor
python3 -m pip install .
```

离线分析器没有强制第三方依赖。Native CUPTI 采集还需要 Linux、CMake 和包含
CUPTI 的 CUDA Toolkit。

## 30 秒上手

直接分析仓库样例。程序会自动识别输入格式，将时间统一为微秒（`us`），并在
计算前校验 trace。

```bash
overlap-monitor analyze \
  --input examples/traces/cupti_activity.jsonl \
  --rank 0 \
  --stage-id 0 \
  --table \
  --output-json summary.json
```

Python API 使用同一套加载、校验和分析流程：

```python
from overlap_monitor import analyze_trace

result = analyze_trace(
    "examples/traces/cupti_activity.jsonl",
    rank=0,
    stage_id=0,
)
print(result.communication_hidden_ratio)
```

已有统一 Event JSONL 时，调用方式不变：

```bash
overlap-monitor validate --input events_rank0.jsonl
overlap-monitor analyze --input events_rank0.jsonl --table --ascii
```

过滤条件、输出文件、分析模式、schema 和时钟对齐规则见
[API 与 CLI 文档](docs/api.md)。

## 选择数据源

| 数据源 | 适用场景 | 文档 |
| --- | --- | --- |
| CUPTI | 精确 NCCL 和计算 kernel 区间 | [CUPTI 测量](docs/cupti_measurement.md) |
| PyTorch Profiler | 框架/operator 归因 | [集成指南](docs/integration.md#pytorch-profiler) |
| Work/wait adapter | 低开销异步依赖 proxy | [集成指南](docs/integration.md#async-work-handles) |
| Nsight Systems | 跨 stream 观察与精度对照 | [验证说明](docs/validation.md) |

这些数据源测量的对象不同。CUPTI 提供 GPU activity 时间；`Work.wait()` 观察的是
主机侧等待，不能当作精确 NCCL kernel 时间。

## 核心指标

| 指标 | 定义 |
| --- | --- |
| `communication_runtime` | 通信区间的并集 |
| `hidden_communication` | 与已识别计算并发的通信 |
| `exposed_communication` | 未被计算覆盖的通信 |
| `communication_hidden_ratio` | `hidden_communication / communication_runtime` |
| `timeline_overlap_ratio` | `overlap_time / min(compute_time, communication_time)` |
| `wait_time` | Work 的主机侧等待 proxy |

新代码应使用两个含义明确的 ratio 字段。`overlap_ratio` 仅作为当前模式的兼容别名。
每份 summary 都记录公式、测量质量、runtime 类型和校验结果。

## 测量约束

- 默认 `critical-path` 模式报告通信隐藏情况。
- `timeline` 模式报告对称区间 overlap 和 pipeline stage 指标。
- 多 rank/device 时钟域默认拒绝分析；只有完成外部对齐后才能显式使用
  `--assume-aligned-clocks`。
- 不完整 CUPTI trace 默认拒绝；`--allow-incomplete` 只能用于调试。
- `kernel_timeline` 是 GPU 观测证据；`estimated` 和 `host_wait_proxy` 不能表述为
  精确 NCCL runtime。

跨模式或跨 profiler 比较前，请先阅读[测量语义](docs/measurement.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [API 与 CLI](docs/api.md) | 使用 Python 或命令行分析 trace |
| [集成指南](docs/integration.md) | 接入 Work、Megatron 或 PyTorch Profiler |
| [CUPTI 测量](docs/cupti_measurement.md) | 构建、采集、校验和分析 GPU activity |
| [测量语义](docs/measurement.md) | 正确解释 overlap 和 critical-path 指标 |
| [架构](docs/architecture.md) | 模块边界与扩展方式 |
| [验证](docs/validation.md) | 已验证行为和当前限制 |

完整索引见 [docs/README.md](docs/README.md)，参考资料见
[references/README.md](references/README.md)。

## 开发

```bash
python3 -m pip install -e ".[dev]"
ruff check .
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m build
```

开发和发布信息见 [CONTRIBUTING.md](CONTRIBUTING.md)、[ROADMAP.md](ROADMAP.md)
和 [CITATION.cff](CITATION.cff)。项目采用 [MIT License](LICENSE)。
