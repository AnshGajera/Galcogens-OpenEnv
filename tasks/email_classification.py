"""Deterministic task: classify whether an email should be archived or kept."""

from __future__ import annotations


TASK_ID = "email_classification"


EXAMPLE_INPUT = {
    "subject": "Flash sale this weekend",
    "body": "Get 70% off if you buy today.",
    "sender": "promo@shop.example",
}


EXPECTED_OUTPUT = {
    "label": "archive",
}


def solve(payload: dict) -> dict:
    """Return deterministic classification for a single email payload."""
    subject = str(payload.get("subject", "")).lower()
    body = str(payload.get("body", "")).lower()
    sender = str(payload.get("sender", "")).lower()
    text = " ".join([subject, body, sender])

    spam_markers = (
        "sale",
        "discount",
        "promo",
        "newsletter",
        "unsubscribe",
        "lottery",
        "free",
    )
    label = "archive" if any(marker in text for marker in spam_markers) else "keep"
    return {"label": label}


def get_task() -> dict:
    """Expose validator-friendly task metadata and expected output."""
    return {
        "id": TASK_ID,
        "input": EXAMPLE_INPUT,
        "expected": EXPECTED_OUTPUT,
        "solver": "tasks.email_classification:solve",
    }
