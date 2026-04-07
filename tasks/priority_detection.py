"""Deterministic task: detect email priority from content and sender."""

from __future__ import annotations


TASK_ID = "priority_detection"


EXAMPLE_INPUT = {
    "subject": "URGENT: production outage",
    "body": "Service is down for customers. Please respond ASAP.",
    "sender": "ops@company.com",
}


EXPECTED_OUTPUT = {
    "priority": "high",
}


def solve(payload: dict) -> dict:
    """Return deterministic priority label in {low, medium, high}."""
    subject = str(payload.get("subject", "")).lower()
    body = str(payload.get("body", "")).lower()
    sender = str(payload.get("sender", "")).lower()
    text = " ".join([subject, body, sender])

    high_markers = (
        "urgent",
        "asap",
        "outage",
        "critical",
        "ceo",
        "incident",
    )
    medium_markers = (
        "meeting",
        "timeline",
        "review",
        "schedule",
        "follow up",
    )

    if any(marker in text for marker in high_markers):
        priority = "high"
    elif any(marker in text for marker in medium_markers):
        priority = "medium"
    else:
        priority = "low"

    return {"priority": priority}


def get_task() -> dict:
    """Expose validator-friendly task metadata and expected output."""
    return {
        "id": TASK_ID,
        "input": EXAMPLE_INPUT,
        "expected": EXPECTED_OUTPUT,
        "solver": "tasks.priority_detection:solve",
    }
