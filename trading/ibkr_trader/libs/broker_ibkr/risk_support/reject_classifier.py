"""Converts IB rejection events into actionable categories."""
from ..error_map import classify_error
from ..models.types import RejectCategory


class RejectClassifier:
    """Converts IB rejection events into actionable categories for OMS."""

    @staticmethod
    def classify(error_code: int, error_msg: str) -> dict:
        category, retryable = classify_error(error_code, error_msg)
        return {
            "category": category,
            "retryable": retryable,
            "code": error_code,
            "message": error_msg,
            "is_misconfiguration": category
            in {
                RejectCategory.CONTRACT,
                RejectCategory.PERMISSIONS,
                RejectCategory.INVALID_PRICE,
            },
        }
