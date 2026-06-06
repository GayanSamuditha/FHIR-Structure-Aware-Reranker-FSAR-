"""Weighted-fusion reranker over a candidate pool."""

from typing import Optional

import numpy as np

from src import signals as sig

_DEFAULT_WEIGHTS = {
    "semantic": 0.40,
    "temporal": 0.20,
    "reference": 0.15,
    "code_overlap": 0.15,
    "type_prior": 0.07,
    "specificity": 0.03,
}


def rerank(
    query: dict,
    candidates: list,
    q_vec: np.ndarray,
    c_vecs: np.ndarray,
    refgraph: dict,
    weights: Optional[dict] = None,
) -> list:
    """
    Score and sort candidates using weighted signal fusion.

    Args:
        query: dict with query_text, meta (anchor_date, direction, intent,
               anchor_id), and optionally codes.
        candidates: list of record dicts (id, resource_type, date, codes, …).
        q_vec: 1D query embedding.
        c_vecs: 2D array, one row per candidate (same order as candidates).
        refgraph: loaded refgraph dict.
        weights: optional weight override; defaults to _DEFAULT_WEIGHTS.

    Returns:
        List of dicts sorted descending by final_score.
        Each dict: {id, final_score, breakdown: {signal: value, …}}.
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}
    meta = query.get("meta", {})
    anchor_date = meta.get("anchor_date")
    direction = meta.get("direction")
    intent = meta.get("intent")
    anchor_id = meta.get("anchor_id")
    query_codes = query.get("codes", []) or []

    # Raw cosine scores for min-max normalisation across the pool
    raw_cosines = np.array([
        sig.semantic(q_vec, c_vecs[i]) for i in range(len(candidates))
    ])
    cos_min, cos_max = raw_cosines.min(), raw_cosines.max()
    if cos_max > cos_min:
        norm_cosines = (raw_cosines - cos_min) / (cos_max - cos_min)
    else:
        norm_cosines = np.ones(len(candidates)) * 0.5

    results = []
    for i, cand in enumerate(candidates):
        sem = float(norm_cosines[i])
        temp = sig.temporal_proximity(anchor_date, direction, cand.get("date"))
        tp = sig.type_prior(cand.get("resource_type", ""), intent)
        co = sig.code_overlap(query_codes, cand.get("codes") or [])
        rc = sig.reference_coherence(cand.get("id", ""), anchor_id, refgraph)
        sp = sig.specificity(cand.get("codes") or [], cand.get("resource_type", ""))

        final = (
            w["semantic"] * sem
            + w["temporal"] * temp
            + w["type_prior"] * tp
            + w["code_overlap"] * co
            + w["reference"] * rc
            + w["specificity"] * sp
        )

        results.append({
            "id": cand.get("id", ""),
            "final_score": final,
            "breakdown": {
                "semantic": sem,
                "temporal": temp,
                "type_prior": tp,
                "code_overlap": co,
                "reference": rc,
                "specificity": sp,
            },
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results
