"""Deterministic benchmark task definitions for pre-submission validation."""

from .email_classification import get_task as get_email_classification_task
from .priority_detection import get_task as get_priority_detection_task
from .response_generation import get_task as get_response_generation_task

TASK_REGISTRY = {
    "email_classification": get_email_classification_task,
    "priority_detection": get_priority_detection_task,
    "response_generation": get_response_generation_task,
}


def list_tasks() -> list[dict]:
    """Return all task definitions in a deterministic order."""
    ordered_ids = [
        "email_classification",
        "priority_detection",
        "response_generation",
    ]
    return [TASK_REGISTRY[task_id]() for task_id in ordered_ids]
