from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from overlap_monitor import __version__
from overlap_monitor.api import TraceValidationError, analyze_trace, load_trace
from overlap_monitor.core.io import (
    JsonlFormatError,
    write_events_jsonl,
)
from overlap_monitor.core.validation import validate_events
from overlap_monitor.profiler.cupti import CuptiActivityParser, CuptiFormatError
from overlap_monitor.visualization.ascii_timeline import render_ascii_timeline
from overlap_monitor.visualization.chrome_trace import write_chrome_trace
from overlap_monitor.visualization.summary_table import render_summary_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overlap-monitor",
        description="Analyze 1F1B communication-computation overlap events.",
        epilog=(
            "example: overlap-monitor analyze --input trace.jsonl --table "
            "--output-json summary.json"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a trace JSONL file.")
    _add_trace_input_arguments(analyze)
    analyze.add_argument(
        "--mode",
        choices=("critical-path", "timeline"),
        default="critical-path",
        help="Analysis mode. Use critical-path for async Work/wait events.",
    )
    analyze.add_argument(
        "--events-output",
        type=Path,
        help="Write normalized events JSONL before analysis.",
    )
    _add_clock_alignment_arguments(analyze)
    analyze.add_argument("--output-json", type=Path, help="Write summary JSON.")
    analyze.add_argument("--trace-json", type=Path, help="Write Chrome trace JSON.")
    analyze.add_argument(
        "--ascii",
        action="store_true",
        help="Print an ASCII timeline before the summary.",
    )
    analyze.add_argument(
        "--table",
        action="store_true",
        help="Print a Markdown summary table.",
    )

    validate = subparsers.add_parser("validate", help="Validate a trace JSONL file.")
    _add_trace_input_arguments(validate)
    _add_clock_alignment_arguments(validate)

    import_cupti = subparsers.add_parser(
        "import-cupti", help="Convert native CUPTI activity JSONL into event JSONL."
    )
    import_cupti.add_argument("--input", required=True, type=Path)
    import_cupti.add_argument("--output", required=True, type=Path)
    import_cupti.add_argument("--rank", type=int)
    import_cupti.add_argument("--stage-id", type=int)
    import_cupti.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Import traces with dropped records or unavailable timestamps.",
    )
    return parser


def _add_trace_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input", required=True, type=Path, help="Input trace JSONL path."
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "events", "cupti"),
        default="auto",
        help="Input format. Auto detects native CUPTI records (default: auto).",
    )
    parser.add_argument("--rank", type=int, help="Select one rank.")
    parser.add_argument("--device-id", type=int, help="Select one device.")
    parser.add_argument(
        "--stage-id",
        type=int,
        help="Select one stage; also assigns missing CUPTI stage IDs.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow CUPTI traces with dropped records or unavailable timestamps.",
    )


def _add_clock_alignment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--assume-aligned-clocks",
        action="store_true",
        help="Analyze multiple clock domains only when the caller has aligned them.",
    )
    parser.add_argument(
        "--allow-mixed-clock-domains",
        dest="assume_aligned_clocks",
        action="store_true",
        help=argparse.SUPPRESS,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "analyze":
            return _analyze(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "import-cupti":
            return _import_cupti(args)
    except TraceValidationError as exc:
        print(
            json.dumps(exc.report.to_dict(), indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    except (JsonlFormatError, CuptiFormatError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def _analyze(args: argparse.Namespace) -> int:
    result = analyze_trace(
        args.input,
        mode=args.mode,
        input_format=args.input_format,
        rank=args.rank,
        device_id=args.device_id,
        stage_id=args.stage_id,
        allow_incomplete=args.allow_incomplete,
        assume_aligned_clocks=args.assume_aligned_clocks,
    )
    if args.events_output:
        write_events_jsonl(result.events, args.events_output)
    payload = result.to_dict()
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.trace_json:
        write_chrome_trace(result.events, args.trace_json)

    if args.ascii:
        print(render_ascii_timeline(result.events))
    if args.table:
        print(
            render_summary_table(
                result.summary,
                timestamp_unit=result.timestamp_unit,
            )
        )
    if not args.output_json and not args.table:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate(args: argparse.Namespace) -> int:
    events = load_trace(
        args.input,
        input_format=args.input_format,
        rank=args.rank,
        device_id=args.device_id,
        stage_id=args.stage_id,
        allow_incomplete=args.allow_incomplete,
    )
    report = validate_events(
        events, strict_clock_domain=not args.assume_aligned_clocks
    )
    payload = {
        "clock_alignment_assumed": args.assume_aligned_clocks,
        **report.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.valid else 2


def _import_cupti(args: argparse.Namespace) -> int:
    result = CuptiActivityParser().parse_file(
        args.input,
        default_rank=args.rank,
        default_stage_id=args.stage_id,
        strict=not args.allow_incomplete,
    )
    write_events_jsonl(result.events, args.output)
    print(
        json.dumps(
            {
                "complete": result.complete,
                "dropped_records": result.dropped_records,
                "event_count": len(result.events),
                "output": str(args.output),
                "skipped_records": result.skipped_records,
                "warnings": result.warnings,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
