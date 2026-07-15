from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, validate

from overlap_monitor import Event, EventType, analyze_events, analyze_trace
from overlap_monitor.core.io import (
    JsonlFormatError,
    read_events_jsonl,
    write_events_jsonl,
)

SCHEMAS = Path(__file__).resolve().parents[1] / "overlap_monitor" / "schemas"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "traces"


class SchemaTests(unittest.TestCase):
    def test_schema_documents_are_valid_json(self):
        event_schema = json.loads((SCHEMAS / "event-v1.schema.json").read_text())
        summary_schema = json.loads((SCHEMAS / "summary-v1.schema.json").read_text())

        Draft202012Validator.check_schema(event_schema)
        Draft202012Validator.check_schema(summary_schema)
        self.assertEqual(event_schema["properties"]["schema_version"]["const"], 1)
        self.assertIn("analysis_mode", summary_schema["required"])

    def test_event_writer_emits_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            write_events_jsonl([Event(0, 1, EventType.GEMM)], path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            schema = json.loads((SCHEMAS / "event-v1.schema.json").read_text())

            validate(payload, schema)
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(read_events_jsonl(path)[0].event_type, EventType.GEMM)

    def test_analysis_result_matches_summary_schema(self):
        result = analyze_trace(
            EXAMPLES / "cupti_activity.jsonl",
            rank=0,
            stage_id=0,
        )
        schema = json.loads((SCHEMAS / "summary-v1.schema.json").read_text())

        validate(result.to_dict(), schema)

        timeline = analyze_events(
            [
                Event(0, 10, EventType.GEMM, rank=0),
                Event(5, 15, EventType.NCCL, rank=0),
            ],
            mode="timeline",
        )
        validate(timeline.to_dict(), schema)

    def test_event_reader_rejects_unknown_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "timestamp_start": 0,
                        "timestamp_end": 1,
                        "event_type": "GEMM",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(JsonlFormatError, "schema_version=2"):
                read_events_jsonl(path)


if __name__ == "__main__":
    unittest.main()
