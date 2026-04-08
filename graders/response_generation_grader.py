"""Grader for deterministic response generation task."""

from __future__ import annotations


def _clamp(score: float) -> float:
    return max(0.01, min(0.99, float(score)))


def grade(output: dict, expected: dict) -> float:
    """Score generated response with exact and partial credit."""
    if not isinstance(output, dict) or not isinstance(expected, dict):
        return 0.001

    predicted = str(output.get("response", "")).strip().lower()
    target = str(expected.get("response", "")).strip().lower()

    if not predicted:
        return 0.001
    if predicted == target:
        return 0.99

    partial_checks = [
        "thank" in predicted,
        "confirm" in predicted,
        len(predicted) >= 30,
    ]
    score = 0.5 if sum(partial_checks) >= 2 else 0.001
    return _clamp(score)
