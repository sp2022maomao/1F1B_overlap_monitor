from __future__ import annotations

import argparse
import os
from pathlib import Path

from overlap_monitor import CuptiRuntimeCollector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded NCCL/GEMM smoke test for the native CUPTI collector"
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--elements", type=int, default=262_144)
    args = parser.parse_args(argv)

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError("CUPTI NCCL smoke test requires at least two ranks")
    if args.elements % world_size:
        raise ValueError("--elements must be divisible by WORLD_SIZE")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        comm_input = torch.randn(args.elements, device="cuda", dtype=torch.float16)
        comm_output = torch.empty_like(comm_input)
        matrix_a = torch.randn(
            (args.matrix_size, args.matrix_size), device="cuda", dtype=torch.float16
        )
        matrix_b = torch.randn_like(matrix_a)
        compute_stream = torch.cuda.Stream()

        warmup = dist.all_to_all_single(comm_output, comm_input, async_op=True)
        with torch.cuda.stream(compute_stream):
            torch.mm(matrix_a, matrix_b)
        warmup.wait()
        torch.cuda.synchronize()

        args.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = args.trace_dir / f"cupti_rank{rank}.jsonl"
        collector = CuptiRuntimeCollector(args.library)
        collector.start(trace_path, rank=rank)
        try:
            for iteration in range(args.iterations):
                external_id = rank * 1_000_000 + iteration
                with collector.external_range(external_id):
                    work = dist.all_to_all_single(
                        comm_output, comm_input, async_op=True
                    )
                    with torch.cuda.stream(compute_stream):
                        torch.mm(matrix_a, matrix_b)
                    work.wait()
            torch.cuda.synchronize()
        finally:
            collector.stop()
        print(f"rank={rank} trace={trace_path}", flush=True)
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
