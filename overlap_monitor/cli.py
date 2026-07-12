from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from overlap_monitor.analyzer.critical_path import CriticalPathOverlapAnalyzer
from overlap_monitor.analyzer.overlap import OverlapAnalyzer
from overlap_monitor.core.io import read_events_jsonl
from overlap_monitor.core.io import write_events_jsonl
from overlap_monitor.core.validation import validate_events
from overlap_monitor.profiler.cupti import CuptiActivityParser, CuptiFormatError
from overlap_monitor.visualization.ascii_timeline import render_ascii_timeline
from overlap_monitor.visualization.chrome_trace import write_chrome_trace
from overlap_monitor.visualization.summary_table import render_summary_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overlap-monitor",
        description="Analyze 1F1B communication-computation overlap events.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze an events JSONL file.")
    analyze.add_argument(
        "--input", required=True, type=Path, help="Input events JSONL path."
    )
    analyze.add_argument(
        "--mode",
        choices=("critical-path", "timeline"),
        default="critical-path",
        help="Analysis mode. Use critical-path for async Work/wait events.",
    )
    analyze.add_argument("--rank", type=int, help="Analyze one rank only.")
    analyze.add_argument("--device-id", type=int, help="Analyze one device only.")
    analyze.add_argument(
        "--allow-mixed-clock-domains",
        action="store_true",
        help="Allow timestamps from multiple ranks. Use only for aligned clocks.",
    )
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

    validate = subparsers.add_parser("validate", help="Validate an events JSONL file.")
    validate.add_argument(
        "--input", required=True, type=Path, help="Input events JSONL path."
    )
    validate.add_argument(
        "--allow-mixed-clock-domains",
        action="store_true",
        help="Treat multiple rank clock domains as valid.",
    )

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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "validate":
        return _validate(args)
    if args.command == "import-cupti":
        return _import_cupti(args)
    return 2


def _analyze(args: argparse.Namespace) -> int:
    events = read_events_jsonl(args.input)
    if args.rank is not None:
        events = [event for event in events if event.rank == args.rank]
    if args.device_id is not None:
        events = [event for event in events if event.device_id == args.device_id]
    report = validate_events(
        events, strict_clock_domain=not args.allow_mixed_clock_domains
    )
    if not report.valid:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True), file=sys.stderr)
        return 2
    if args.mode == "critical-path":
        summary = CriticalPathOverlapAnalyzer().analyze(events)
    else:
        summary = OverlapAnalyzer().analyze(events)

    payload = summary.to_dict()
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    if args.trace_json:
        write_chrome_trace(events, args.trace_json)

    if args.ascii:
        print(render_ascii_timeline(events))
    if args.table:
        print(render_summary_table(summary))
    if not args.output_json and not args.table:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate(args: argparse.Namespace) -> int:
    events = read_events_jsonl(args.input)
    report = validate_events(
        events, strict_clock_domain=not args.allow_mixed_clock_domains
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.valid else 2


def _import_cupti(args: argparse.Namespace) -> int:
    try:
        result = CuptiActivityParser().parse_file(
            args.input,
            default_rank=args.rank,
            default_stage_id=args.stage_id,
            strict=not args.allow_incomplete,
        )
    except CuptiFormatError as exc:
        print(str(exc), file=sys.stderr)
        return 2
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
