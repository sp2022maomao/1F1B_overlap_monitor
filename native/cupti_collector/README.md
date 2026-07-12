# Native CUPTI Collector

This optional Linux shared library captures concurrent CUDA kernel activity and
writes the raw JSONL schema consumed by `CuptiActivityParser`.

```bash
cmake -S native/cupti_collector -B build/cupti -DCMAKE_BUILD_TYPE=Release
cmake --build build/cupti --parallel
```

The collector targets CUDA 12.9 and contains a CUDA 13 kernel-record alias. It
has no Megatron or PyTorch dependency. See
[`docs/cupti_measurement.md`](../../docs/cupti_measurement.md) for runtime use,
data integrity rules, and validation limits.
