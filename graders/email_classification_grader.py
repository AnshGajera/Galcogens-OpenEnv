"""Grader for email classification task."""

from __future__ import annotations


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
    """Score output label against expected label.

    Scoring:
    - exact label match: ~0.99
    - both labels in valid label set but not exact: ~0.5
    - invalid or missing prediction: ~0.01
    """
    if not isinstance(output, dict) or not isinstance(expected, dict):
        return _strict_score(0.01)

    predicted = str(output.get("label", "")).strip().lower()
    target = str(expected.get("label", "")).strip().lower()
    valid = {"archive", "keep"}

    if predicted == target and predicted in valid:
        return _strict_score(0.99)
    if predicted in valid and target in valid:
        return _strict_score(0.5)
    return _strict_score(0.01)
