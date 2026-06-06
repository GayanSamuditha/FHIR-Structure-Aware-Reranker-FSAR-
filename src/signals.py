"""Pure scoring signal functions for FSAR reranker. Each returns float in [0, 1]."""

import math
from datetime import datetime
from typing import Optional


def _parse_naive(date_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 date string to a timezone-naive datetime."""
    if not date_str:
        return None
    try:
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(date_str)
        return dt.replace(tzinfo=None)
    except Exception:
        return None


def semantic(q_vec, c_vec) -> float:
    """Cosine similarity between two 1D numpy arrays."""
    import numpy as np
    q_norm = np.linalg.norm(q_vec)
    c_norm = np.linalg.norm(c_vec)
    if q_norm == 0.0 or c_norm == 0.0:
        return 0.0
    return float(np.dot(q_vec, c_vec) / (q_norm * c_norm))


def temporal_proximity(
    anchor_date_str: Optional[str],
    direction: Optional[str],
    c_date_str: Optional[str],
) -> float:
    """
    Exponential decay score based on day distance from anchor.

    Hard cutoff: returns 0.0 if the candidate is on the wrong side
    of a directional query ("before" / "after").
    Returns 0.5 (neutral) if either date is missing.
    """
    anchor = _parse_naive(anchor_date_str)
    c_date = _parse_naive(c_date_str)

    if anchor is None or c_date is None:
        return 0.5

    if direction == "before" and c_date > anchor:
        return 0.0
    if direction == "after" and c_date < anchor:
        return 0.0

    delta_days = abs((anchor - c_date).days)
    score = math.exp(-delta_days / 180.0)
    return max(0.0, min(1.0, score))


_TYPE_PRIOR_TABLE = {
    "lab":        {"Observation"},
    "diagnostic": {"Condition", "DiagnosticReport", "Observation"},
    "treatment":  {"MedicationRequest", "Procedure"},
}


def type_prior(resource_type: str, intent: Optional[str]) -> float:
    """Lookup-table prior: 1.0 if resource_type matches intent, 0.3 if not, 0.5 if intent unknown."""
    if not intent or intent not in _TYPE_PRIOR_TABLE:
        return 0.5
    return 1.0 if resource_type in _TYPE_PRIOR_TABLE[intent] else 0.3


def code_overlap(query_codes: list, c_codes: list) -> float:
    """Jaccard overlap on exact (system, code) pairs."""
    q_set = {(c.get("system", ""), c.get("code", "")) for c in (query_codes or [])}
    c_set = {(c.get("system", ""), c.get("code", "")) for c in (c_codes or [])}
    union = q_set | c_set
    if not union:
        return 0.0
    return len(q_set & c_set) / len(union)


def reference_coherence(c_id: str, anchor_id: Optional[str], refgraph: dict) -> float:
    """
    1.0  — same encounter as anchor
    0.6  — in anchor's adjacency neighbors
    0.0  — otherwise (or no anchor)
    """
    if not anchor_id:
        return 0.0

    # Same encounter check
    by_encounter = refgraph.get("by_encounter", {})
    for enc_resources in by_encounter.values():
        if anchor_id in enc_resources and c_id in enc_resources:
            return 1.0

    # Adjacency check
    adjacency = refgraph.get("adjacency", {})
    if c_id in adjacency.get(anchor_id, []):
        return 0.6
    if anchor_id in adjacency.get(c_id, []):
        return 0.6

    return 0.0


def specificity(c_codes: list, resource_type: str) -> float:  # noqa: ARG001
    """Small boost for resources with more coded detail."""
    n = len(c_codes or [])
    if n >= 2:
        return 1.0
    if n == 1:
        return 0.7
    return 0.3
