"""Executable diagnostic standards.

The engine stays standard-agnostic: trees, labels, catalogs and entry profiles
all carry the same standard string, and only this module knows which strings are
executable.  Adding a standard here adds no clinical content; the content lives
in a versioned data pack under ``knowledge/paper_dtree``.
"""
from __future__ import annotations

SUPPORTED_STANDARDS = {
    "DSM-5": "DSM-5",
    "DSM-5-TR": "DSM-5-TR",
    "ICD-10": "ICD-10",
}


def check_standard(value, *, name: str = "standard") -> str:
    if not isinstance(value, str) or value not in SUPPORTED_STANDARDS:
        raise ValueError(f"{name} must be one of {sorted(SUPPORTED_STANDARDS)}; got {value!r}")
    return value


def non_standard_labels(standard: str) -> tuple[str, str]:
    """Category and kind for a retained output that is not a diagnosis of this standard."""
    if standard.startswith("DSM"):
        return "Non-DSM", "non_dsm"
    if standard.startswith("ICD"):
        return f"Non-{standard}", "non_icd"
    return "Non-standard", "non_standard"