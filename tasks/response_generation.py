"""Deterministic task: generate a short professional acknowledgement email."""

from __future__ import annotations


TASK_ID = "response_generation"


EXAMPLE_INPUT = {
    "sender_name": "Alex",
    "topic": "kickoff meeting",
    "slot": "2026-04-03 10:00",
}


EXPECTED_OUTPUT = {
    "response": "Thank you Alex. I confirm the kickoff meeting for 2026-04-03 10:00.",
}


def solve(payload: dict) -> dict:
    """Return deterministic response text from normalized payload fields."""
    sender_name = str(payload.get("sender_name", "Team")).strip() or "Team"
    topic = str(payload.get("topic", "request")).strip() or "request"
    slot = str(payload.get("slot", "TBD")).strip() or "TBD"

    response = f"Thank you {sender_name}. I confirm the {topic} for {slot}."
    return {"response": response}


def get_task() -> dict:
    """Expose validator-friendly task metadata and expected output."""
    return {
        "id": TASK_ID,
        "input": EXAMPLE_INPUT,
        "expected": EXPECTED_OUTPUT,
        "solver": "tasks.response_generation:solve",
    }
