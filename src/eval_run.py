"""Run evaluation: naive vs FSAR full vs ablations, emit results table."""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.embed import embed_query
from src.index import NumpyIndex
from src.rerank import rerank
from src.metrics import (
    ndcg_at_k,
    recall_at_k,
    mrr,
    temporal_precision_at_k,
    coherence_at_k,
)

# ---------------------------------------------------------------------------
# Weight configs: naive uses semantic-only ordering; ablations zero one signal
# and redistribute its weight to semantic.
# ---------------------------------------------------------------------------
_BASE = dict(
    semantic=0.40,
    temporal=0.20,
    reference=0.15,
    code_overlap=0.15,
    type_prior=0.07,
    specificity=0.03,
)

CONDITIONS = {
    "naive":            None,   # sentinel: keep index order (semantic-sorted)
    "FSAR_full":        _BASE,
    "FSAR_no_temporal": {**_BASE, "temporal": 0.0,     "semantic": _BASE["semantic"] + 0.20},
    "FSAR_no_reference":{**_BASE, "reference": 0.0,    "semantic": _BASE["semantic"] + 0.15},
    "FSAR_no_code":     {**_BASE, "code_overlap": 0.0, "semantic": _BASE["semantic"] + 0.15},
    "FSAR_no_type":     {**_BASE, "type_prior": 0.0,   "semantic": _BASE["semantic"] + 0.07},
}

K = config.TOP_K_FINAL          # 10
RETRIEVE_K = config.TOP_K_RETRIEVE  # 50

EVAL_RESULTS_PATH = config.ARTIFACTS_DIR / "eval_results.json"
TFIDF_PATH = config.ARTIFACTS_DIR / "tfidf_vectorizer.pkl"


def _load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(vals: list) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def run_eval() -> dict:
    """Run all conditions on the test split and return aggregated metrics."""
    print("Loading artifacts...")
    df = pd.read_parquet(config.STATEMENTS_PATH)
    all_vectors = np.load(config.VECTORS_PATH)

    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(config.REFGRAPH_PATH) as f:
        refgraph = json.load(f)

    queries = _load_jsonl(config.ARTIFACTS_DIR / "eval_queries_test.jsonl")
    print(f"Loaded {len(queries)} test queries")
    print(f"  Template 1: {sum(1 for q in queries if q['template'] == 1)}")
    print(f"  Template 2: {sum(1 for q in queries if q['template'] == 2)}")
    print(f"  Template 4: {sum(1 for q in queries if q['template'] == 4)}")

    index = NumpyIndex(all_vectors, df)

    # metric buckets  {condition: {metric: [per-query values]}}
    buckets: dict = {
        cond: {"ndcg": [], "recall": [], "mrr": [], "temp_prec": [], "coh": []}
        for cond in CONDITIONS
    }

    for qi, query in enumerate(queries):
        if qi % 100 == 0:
            print(f"  processing query {qi}/{len(queries)}...")

        q_vec = embed_query(query["query_text"], model=vectorizer)
        indices, _ = index.search(q_vec, k=RETRIEVE_K)

        candidates = df.iloc[indices].to_dict("records")
        c_vecs = all_vectors[indices]

        relevance: dict = query["relevance"]
        meta: dict = query.get("meta", {})
        anchor_date = meta.get("anchor_date")
        direction = meta.get("direction")
        anchor_id = meta.get("anchor_id")
        template = query.get("template")

        # build id→record lookup for metric helpers
        cands_meta = {c["id"]: c for c in candidates}

        for cond_name, weights in CONDITIONS.items():
            if weights is None:
                # naive: semantic order from index (already descending cosine)
                ranked_ids = [c["id"] for c in candidates]
            else:
                ranked = rerank(query, candidates, q_vec, c_vecs, refgraph, weights=weights)
                ranked_ids = [r["id"] for r in ranked]

            buckets[cond_name]["ndcg"].append(ndcg_at_k(ranked_ids, relevance, K))
            buckets[cond_name]["recall"].append(recall_at_k(ranked_ids, relevance, K))
            buckets[cond_name]["mrr"].append(mrr(ranked_ids, relevance))

            # temporal precision — template 1 only (direction is always "before")
            if template == 1 and direction is not None:
                tp = temporal_precision_at_k(ranked_ids, cands_meta, anchor_date, direction, K)
                if tp is not None:
                    buckets[cond_name]["temp_prec"].append(tp)

            # coherence — template 2 only (anchor_id is the encounter)
            if template == 2:
                gold_enc = anchor_id if (anchor_id and anchor_id.startswith("Encounter/")) else None
                coh = coherence_at_k(ranked_ids, cands_meta, gold_enc, K)
                if coh is not None:
                    buckets[cond_name]["coh"].append(coh)

    # aggregate
    results = {}
    for cond_name, b in buckets.items():
        results[cond_name] = {
            "ndcg_at_10":          _mean(b["ndcg"]),
            "recall_at_10":        _mean(b["recall"]),
            "mrr":                 _mean(b["mrr"]),
            "temporal_prec_at_10": _mean(b["temp_prec"]),
            "coherence_at_10":     _mean(b["coh"]),
        }

    return results


def print_table(results: dict) -> None:
    conds = list(results.keys())
    header = f"{'Condition':<22} {'nDCG@10':>9} {'Recall@10':>10} {'MRR':>8} {'TempPrec@10':>13} {'Coh@10':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for cond in conds:
        r = results[cond]

        def fmt(v):
            return f"{v:.4f}" if v is not None else "   N/A"

        print(
            f"{cond:<22} {fmt(r['ndcg_at_10']):>9} {fmt(r['recall_at_10']):>10}"
            f" {fmt(r['mrr']):>8} {fmt(r['temporal_prec_at_10']):>13} {fmt(r['coherence_at_10']):>9}"
        )
    print("=" * len(header))


def main():
    results = run_eval()
    print_table(results)

    print(f"\nSaving results to {EVAL_RESULTS_PATH}")
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("✓ Done")


if __name__ == "__main__":
    main()
