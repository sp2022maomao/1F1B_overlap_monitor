from __future__ import annotations

import json
from pathlib import Path

from overlap_monitor import analyze_trace


def main() -> None:
    trace = Path(__file__).parent / "traces" / "cupti_activity.jsonl"
    result = analyze_trace(trace, rank=0, stage_id=0)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
