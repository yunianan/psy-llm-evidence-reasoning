"""Canonical baseline ranking; independent of knowledge and reward verification.

Use original confidence descending, then case-sensitive disorder ID ascending.
This key neither rounds confidence nor adds clinical ranking signals. Traversal
and node-cache record order are provenance, not ranked candidate lists.
"""
from collections.abc import Mapping
from typing import Any


def candidate_sort_key(candidate: Any) -> tuple[float, str]:
    """Accept validated candidate mappings or model objects without mutating them."""
    if isinstance(candidate, Mapping):
        return -candidate["confidence"], candidate["disorder"]
    return -candidate.confidence, candidate.disorder
