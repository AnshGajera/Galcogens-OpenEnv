"""Grader for email classification task."""

from __future__ import annotations


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def grade(output: dict, expected: dict) -> float:
    """Score output label against expected label.

    Scoring:
    - exact label match: 1.0
    - both labels in valid label set but not exact: 0.5
    - invalid or missing prediction: 0.0
    """
    if not isinstance(output, dict) or not isinstance(expected, dict):
        return 0.0

    predicted = str(output.get("label", "")).strip().lower()
    target = str(expected.get("label", "")).strip().lower()
    valid = {"archive", "keep"}

    if predicted == target and predicted in valid:
        return 1.0
    if predicted in valid and target in valid:
        return 0.5
    return _clamp(0.0)
