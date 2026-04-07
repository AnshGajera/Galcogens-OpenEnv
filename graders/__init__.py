"""Task graders that return normalized scores in [0.0, 1.0]."""

from .email_classification_grader import grade as grade_email_classification
from .priority_detection_grader import grade as grade_priority_detection
from .response_generation_grader import grade as grade_response_generation

GRADER_REGISTRY = {
    "email_classification": grade_email_classification,
    "priority_detection": grade_priority_detection,
    "response_generation": grade_response_generation,
}
