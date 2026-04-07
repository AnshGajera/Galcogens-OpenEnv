"""Grader for priority detection task."""

from __future__ import annotations


PRIORITY_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def grade(output: dict, expected: dict) -> float:
    """Score priority predictions with exact and partial credit."""
    if not isinstance(output, dict) or not isinstance(expected, dict):
        return 0.0

    predicted = str(output.get("priority", "")).strip().lower()
    target = str(expected.get("priority", "")).strip().lower()

    if predicted not in PRIORITY_RANK or target not in PRIORITY_RANK:
        return 0.0
    if predicted == target:
        return 1.0

    distance = abs(PRIORITY_RANK[predicted] - PRIORITY_RANK[target])
    if distance == 1:
        return 0.5
    return _clamp(0.0)
