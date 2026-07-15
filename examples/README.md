# Examples

The `traces/` directory contains small deterministic fixtures for the CLI and
tests. All normalized timestamps are microseconds.

Analyze the CUPTI fixture:

```bash
overlap-monitor analyze \
  --input examples/traces/cupti_activity.jsonl \
  --rank 0 \
  --stage-id 0 \
  --table
```

Analyze the host Work/wait fixture:

```bash
overlap-monitor analyze \
  --input examples/traces/critical_path_events.jsonl \
  --table \
  --ascii
```

The fixtures demonstrate file formats and known metric values. They are not
performance benchmarks.

Run the equivalent high-level Python API example after installing the package:

```bash
python3 examples/python_api.py
```
