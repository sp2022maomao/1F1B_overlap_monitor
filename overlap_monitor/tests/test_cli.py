from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from overlap_monitor.cli import main


class CliTests(unittest.TestCase):
    def test_critical_path_cli_writes_expected_summary(self):
        root = Path(__file__).resolve().parents[1]
        input_path = root / "examples" / "critical_path_events.jsonl"
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

    def test_validate_cli_accepts_example(self):
        root = Path(__file__).resolve().parents[1]
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                ["validate", "--input", str(root / "examples" / "critical_path_events.jsonl")]
            )
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
