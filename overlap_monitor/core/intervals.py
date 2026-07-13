from __future__ import annotations

from collections.abc import Iterable

Interval = tuple[float, float]


def valid_interval(interval: Interval) -> bool:
    start, end = interval
    return end > start


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(interval for interval in intervals if valid_interval(interval)):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_total(intervals: Iterable[Interval], *, already_merged: bool = False) -> float:
    source = list(intervals) if already_merged else merge_intervals(intervals)
    return sum(end - start for start, end in source)


def intersection_total(
    left_intervals: Iterable[Interval],
    right_intervals: Iterable[Interval],
    *,
    already_merged: bool = False,
) -> float:
    left = list(left_intervals) if already_merged else merge_intervals(left_intervals)
    right = list(right_intervals) if already_merged else merge_intervals(right_intervals)
    i = 0
    j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def span(intervals: Iterable[Interval]) -> Interval | None:
    merged = merge_intervals(intervals)
    if not merged:
        return None
    return (merged[0][0], merged[-1][1])
