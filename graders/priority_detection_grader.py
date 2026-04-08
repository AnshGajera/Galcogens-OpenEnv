"""Grader for priority detection task."""

from __future__ import annotations


PRIORITY_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def _clamp(score: float) -> float:
    return max(0.01, min(0.99, float(score)))


def _strict_score(score: float) -> float:
    """Map any score to strict open interval (0, 1)."""
    bounded = _clamp(score)
    if bounded <= 0.0:
        return 0.01
    if bounded >= 1.0:
        return 0.99
    return bounded


def grade(output: dict, expected: dict) -> float:
    """Score priority predictions with exact and partial credit."""
    if not isinstance(output, dict) or not isinstance(expected, dict):
        return _strict_score(0.01)

    predicted = str(output.get("priority", "")).strip().lower()
    target = str(expected.get("priority", "")).strip().lower()

    if predicted not in PRIORITY_RANK or target not in PRIORITY_RANK:
        return _strict_score(0.01)
    if predicted == target:
        return _strict_score(0.99)

    distance = abs(PRIORITY_RANK[predicted] - PRIORITY_RANK[target])
    if distance == 1:
        return _strict_score(0.5)
    return _strict_score(0.01)
