"""Validation helpers for gsfit_2pt submissions."""

from __future__ import annotations

from typing import Any

from tasks.gsfit_2pt.interface import Pion2PtGroundStateFit, validate_config


def validate_submission(submission: Any) -> bool:
    """Return ``True`` when a gsfit submission satisfies the fixed-fit contract."""

    if not isinstance(submission, Pion2PtGroundStateFit):
        return False

    try:
        _ = submission.meta
        validate_config(submission.config)
    except (AttributeError, TypeError, ValueError):
        return False
    return True
