# Benchmarks

Benchmarks are kept outside the installable package and are not production
accuracy claims.

## CPU Runtime Overhead

```bash
python3 -m benchmarks.benchmark_runtime \
  --iterations 20000 \
  --batch-size 10000 \
  --repeats 5
```

This measures the framework-neutral Work wrapper, offline interval analyzer,
and batched session ingestion against one-lock-per-event ingestion.

## CUPTI GPU Smoke Test

Run from an environment with PyTorch, NCCL, at least two visible NVIDIA GPUs,
and a built `liboverlap_cupti.so`:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m benchmarks.cupti_gpu_smoke \
  --library build/cupti/liboverlap_cupti.so \
  --trace-dir traces/a2a \
  --iterations 5
```

Docker runs may need a larger shared-memory allocation such as
`--shm-size=1g`. Use idle, exclusively allocated GPUs for overhead comparisons.
