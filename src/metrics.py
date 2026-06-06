"""Retrieval evaluation metrics over a ranked list and graded relevance."""

import math
from datetime import datetime
from typing import Dict, List, Optional


def _parse_naive(date_str: Optional[str]) -> Optional[datetime]:
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


def recall_at_k(ranked_ids: List[str], relevance: Dict[str, int], k: int) -> float:
    """Fraction of grade>=2 gold items appearing in top-k."""
    gold = {rid for rid, grade in relevance.items() if grade >= 2}
    if not gold:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in gold)
    return hits / len(gold)


def precision_at_k(ranked_ids: List[str], relevance: Dict[str, int], k: int) -> float:
    """Fraction of top-k items that have grade>=2."""
    if k == 0:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if relevance.get(rid, 0) >= 2)
    return hits / k


def mrr(ranked_ids: List[str], relevance: Dict[str, int]) -> float:
    """Reciprocal rank of the first grade>=2 item; 0 if none found."""
    for rank, rid in enumerate(ranked_ids, 1):
        if relevance.get(rid, 0) >= 2:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: List[str], relevance: Dict[str, int], k: int) -> float:
    """Graded nDCG@k using grades 0/1/2 and log2(rank+1) denominator."""
    def dcg(ids: List[str]) -> float:
        return sum(
            relevance.get(rid, 0) / math.log2(rank + 1)
            for rank, rid in enumerate(ids[:k], 2)   # rank starts at 1 → denominator log2(rank+1)
        )

    actual = dcg(ranked_ids)
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(g / math.log2(r + 1) for r, g in enumerate(ideal_grades, 2))
    return actual / ideal if ideal > 0 else 0.0


def temporal_precision_at_k(
    ranked_ids: List[str],
    candidates_meta: Dict[str, dict],
    anchor_date: Optional[str],
    direction: Optional[str],
    k: int,
) -> Optional[float]:
    """
    Fraction of top-k candidates whose date is on the correct side of anchor_date.
    Returns None if direction is None (query has no temporal constraint).
    """
    if direction is None:
        return None
    anchor = _parse_naive(anchor_date)
    if anchor is None:
        return None
    correct = 0
    for rid in ranked_ids[:k]:
        c_date = _parse_naive((candidates_meta.get(rid) or {}).get("date"))
        if c_date is None:
            continue
        if direction == "before" and c_date <= anchor:
            correct += 1
        elif direction == "after" and c_date >= anchor:
            correct += 1
    return correct / k if k > 0 else 0.0


def coherence_at_k(
    ranked_ids: List[str],
    candidates_meta: Dict[str, dict],
    gold_encounter_id: Optional[str],
    k: int,
) -> Optional[float]:
    """
    Fraction of top-k candidates that share gold_encounter_id.
    Returns None if gold_encounter_id is None.
    """
    if gold_encounter_id is None:
        return None
    matching = sum(
        1 for rid in ranked_ids[:k]
        if (candidates_meta.get(rid) or {}).get("encounter_id") == gold_encounter_id
    )
    return matching / k if k > 0 else 0.0
