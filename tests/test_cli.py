from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from overlap_monitor import analyze_trace
from overlap_monitor.cli import main
from overlap_monitor.core.events import Event, EventType
from overlap_monitor.core.io import write_events_jsonl

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "traces"


class CliTests(unittest.TestCase):
    def test_critical_path_cli_writes_expected_summary(self):
        input_path = EXAMPLES / "critical_path_events.jsonl"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.json"
            exit_code = main(
                [
                    "analyze",
                    "--input",
                    str(input_path),
                    "--mode",
                    "critical-path",
                    "--output-json",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["communication_runtime"], 80)
            self.assertEqual(summary["exposed_communication"], 35)
            self.assertEqual(summary["hidden_communication"], 45)
            self.assertAlmostEqual(summary["overlap_ratio"], 45 / 80)
            self.assertAlmostEqual(summary["communication_hidden_ratio"], 45 / 80)
            self.assertEqual(summary["analysis_mode"], "critical-path")
            self.assertEqual(summary["input_format"], "events")
            self.assertEqual(summary["schema_version"], 1)
            self.assertFalse(summary["clock_alignment_assumed"])
            self.assertTrue(summary["validation"]["valid"])
            self.assertGreater(summary["validation"]["event_count"], 0)
            self.assertEqual(
                summary["overlap_ratio_definition"],
                "hidden_communication / communication_runtime",
            )

    def test_validate_cli_accepts_example(self):
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "validate",
                    "--input",
                    str(EXAMPLES / "critical_path_events.jsonl"),
                ]
            )
        self.assertEqual(exit_code, 0)

    def test_validate_cli_auto_detects_cupti(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "validate",
                    "--input",
                    str(EXAMPLES / "cupti_activity.jsonl"),
                    "--rank",
                    "0",
                    "--stage-id",
                    "0",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["event_count"], 2)

    def test_import_cupti_converts_activity_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir, redirect_stdout(io.StringIO()):
            output_path = Path(tmpdir) / "events.jsonl"
            exit_code = main(
                [
                    "import-cupti",
                    "--input",
                    str(EXAMPLES / "cupti_activity.jsonl"),
                    "--output",
                    str(output_path),
                    "--rank",
                    "0",
                    "--stage-id",
                    "0",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

    def test_analyze_auto_detects_cupti_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir, redirect_stdout(io.StringIO()):
            summary_path = Path(tmpdir) / "summary.json"
            events_path = Path(tmpdir) / "events.jsonl"
            exit_code = main(
                [
                    "analyze",
                    "--input",
                    str(EXAMPLES / "cupti_activity.jsonl"),
                    "--rank",
                    "0",
                    "--stage-id",
                    "0",
                    "--events-output",
                    str(events_path),
                    "--output-json",
                    str(summary_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(events_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["measurement_quality"], "kernel_timeline")
            self.assertEqual(summary["overlap_ratio"], 0.5)
            first_event = json.loads(
                events_path.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(first_event["schema_version"], 1)

    def test_cli_summary_matches_public_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            exit_code = main(
                [
                    "analyze",
                    "--input",
                    str(EXAMPLES / "cupti_activity.jsonl"),
                    "--rank",
                    "0",
                    "--stage-id",
                    "0",
                    "--output-json",
                    str(summary_path),
                ]
            )
            expected = analyze_trace(
                EXAMPLES / "cupti_activity.jsonl",
                rank=0,
                stage_id=0,
            ).to_dict()

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(summary_path.read_text()), expected)

    def test_clock_alignment_requires_explicit_assumption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "mixed.jsonl"
            output_path = Path(tmpdir) / "summary.json"
            write_events_jsonl(
                [
                    Event(0, 10, EventType.GEMM, rank=0),
                    Event(5, 15, EventType.NCCL, rank=1),
                ],
                input_path,
            )
            with redirect_stderr(io.StringIO()):
                rejected = main(["analyze", "--input", str(input_path)])
            accepted = main(
                [
                    "analyze",
                    "--input",
                    str(input_path),
                    "--assume-aligned-clocks",
                    "--output-json",
                    str(output_path),
                ]
            )

            self.assertEqual(rejected, 2)
            self.assertEqual(accepted, 0)
            self.assertTrue(
                json.loads(output_path.read_text())["clock_alignment_assumed"]
            )

    def test_legacy_clock_flag_remains_compatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "mixed.jsonl"
            write_events_jsonl(
                [
                    Event(0, 10, EventType.GEMM, rank=0),
                    Event(5, 15, EventType.NCCL, rank=1),
                ],
                input_path,
            )
            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "analyze",
                        "--input",
                        str(input_path),
                        "--allow-mixed-clock-domains",
                    ]
                )

            self.assertEqual(exit_code, 0)

    def test_analyze_reports_missing_input_without_traceback(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = main(["analyze", "--input", "missing.jsonl"])

        self.assertEqual(exit_code, 2)
        self.assertIn("error:", stderr.getvalue())
        self.assertIn("missing.jsonl", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
