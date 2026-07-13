from __future__ import annotations

from typing import Any


def render_summary_table(summary: Any, *, timestamp_unit: str | None = None) -> str:
    rows = _summary_rows(summary)
    width = max(len(name) for name, _ in rows)
    lines = ["metric | value", "--- | ---:"]
    if timestamp_unit:
        lines.append(f"timestamp_unit | {timestamp_unit}")
    for name, value in rows:
        lines.append(f"{name:<{width}} | {value:.6g}")
    if hasattr(summary, "overlap_ratio_definition"):
        lines.append(f"overlap_ratio_definition | {summary.overlap_ratio_definition}")
    if hasattr(summary, "measurement_quality"):
        lines.append(f"measurement_quality | {summary.measurement_quality}")
        lines.append(
            f"communication_runtime_kind | {summary.communication_runtime_kind}"
        )
        for warning in summary.warnings:
            lines.append(f"warning | {warning}")

    if getattr(summary, "stage_metrics", None):
        lines.extend(["", "stage | compute | communication | overlap | bubble | span"])
        lines.append("---: | ---: | ---: | ---: | ---: | ---:")
        for stage in summary.stage_metrics:
            label = "none" if stage.stage_id is None else str(stage.stage_id)
            lines.append(
                f"{label} | {stage.compute_time:.6g} | {stage.communication_time:.6g} | "
                f"{stage.overlap_time:.6g} | {stage.bubble_ratio:.6g} | {stage.active_span:.6g}"
            )
    if getattr(summary, "group_metrics", None):
        lines.extend(
            [
                "",
                "group | compute | comm runtime | exposed | hidden | overlap | wait | span | quality",
            ]
        )
        lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---")
        for group in summary.group_metrics:
            lines.append(
                f"{group.group_id} | {group.compute_time:.6g} | "
                f"{group.communication_runtime:.6g} | {group.exposed_communication:.6g} | "
                f"{group.hidden_communication:.6g} | {group.overlap_ratio:.6g} | "
                f"{group.wait_time:.6g} | {group.critical_path_span:.6g} | "
                f"{group.measurement_quality}"
            )
    return "\n".join(lines)


def _summary_rows(summary: Any) -> list[tuple[str, float]]:
    if hasattr(summary, "communication_runtime"):
        return [
            ("compute_time", summary.compute_time),
            ("communication_runtime", summary.communication_runtime),
            ("exposed_communication", summary.exposed_communication),
            ("hidden_communication", summary.hidden_communication),
            ("overlap_ratio", summary.overlap_ratio),
            ("wait_time", summary.wait_time),
            ("critical_path_span", summary.critical_path_span),
        ]
    return [
        ("compute_time", summary.compute_time),
        ("communication_time", summary.communication_time),
        ("overlap_time", summary.overlap_time),
        ("overlap_ratio", summary.overlap_ratio),
        ("bubble_ratio", summary.bubble_ratio),
        ("stage_balance", summary.stage_balance),
    ]
