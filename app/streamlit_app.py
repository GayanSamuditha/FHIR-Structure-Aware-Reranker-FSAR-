"""FSAR Streamlit dashboard — Query Explorer + Ablation Results."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make src importable when launched from project root or app/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.embed import embed_query
from src.index import NumpyIndex
from src.rerank import rerank, _DEFAULT_WEIGHTS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS = ROOT / "artifacts"
STATEMENTS_PATH = ARTIFACTS / "statements.parquet"
VECTORS_PATH = ARTIFACTS / "vectors.npy"
TFIDF_PATH = ARTIFACTS / "tfidf_vectorizer.pkl"
REFGRAPH_PATH = ARTIFACTS / "refgraph.json"
EVAL_RESULTS_PATH = ARTIFACTS / "eval_results.json"

SIGNAL_COLORS = {
    "semantic": "#4C72B0",
    "temporal": "#DD8452",
    "reference": "#55A868",
    "code_overlap": "#C44E52",
    "type_prior": "#8172B2",
    "specificity": "#937860",
}

# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    df = pd.read_parquet(STATEMENTS_PATH)
    vectors = np.load(VECTORS_PATH)
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(REFGRAPH_PATH) as f:
        refgraph = json.load(f)
    with open(EVAL_RESULTS_PATH) as f:
        eval_results = json.load(f)
    index = NumpyIndex(vectors, df)
    return df, vectors, vectorizer, refgraph, eval_results, index


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FSAR — FHIR Structure-Aware Reranker",
    page_icon="🏥",
    layout="wide",
)

df, all_vectors, vectorizer, refgraph, eval_results, index = load_artifacts()

page = st.sidebar.radio("Page", ["Query Explorer", "Ablation Results"])

# ===========================================================================
# PAGE 1: Query Explorer
# ===========================================================================

if page == "Query Explorer":
    st.title("Query Explorer")
    st.caption(
        "Stage-1: TF-IDF top-50 retrieval · Stage-2: FSAR weighted reranking "
        "(temporal + reference + type + code)"
    )

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("Query controls")
        query_text = st.text_input("Clinical query", value="What lab results did the patient have in the 6 months before starting Metformin?")
        intent = st.selectbox("Intent", ["lab", "diagnostic", "treatment", "none"])
        direction = st.selectbox("Temporal direction", ["none", "before", "after"])
        anchor_date = st.text_input("Anchor date (YYYY-MM-DD, optional)", value="")
        top_k = st.slider("Top-k results", min_value=5, max_value=20, value=10)
        search_btn = st.button("Search", type="primary", use_container_width=True)

    if not search_btn:
        st.info("Enter a query in the sidebar and click **Search**.")
        st.stop()

    if not query_text.strip():
        st.error("Query text is required.")
        st.stop()

    # --- Build meta ---
    meta = {
        "anchor_date": anchor_date.strip() or None,
        "direction": None if direction == "none" else direction,
        "intent": None if intent == "none" else intent,
        "anchor_id": None,
    }
    query_dict = {"query_text": query_text, "meta": meta, "codes": []}

    # --- Retrieve ---
    q_vec = embed_query(query_text, model=vectorizer)
    indices, _ = index.search(q_vec, k=50)
    candidates = df.iloc[indices].to_dict("records")
    c_vecs = all_vectors[indices]

    # --- FSAR rerank ---
    fsar_results = rerank(query_dict, candidates, q_vec, c_vecs, refgraph)

    # --- Naive order (semantic only) ---
    naive_weights = {k: 0.0 for k in _DEFAULT_WEIGHTS}
    naive_weights["semantic"] = 1.0
    naive_results = rerank(query_dict, candidates, q_vec, c_vecs, refgraph, weights=naive_weights)

    top_result = fsar_results[0]
    naive_top = naive_results[0]

    # --- Callout: naive vs FSAR for top-1 ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            label="Naive top-1 score (semantic only)",
            value=f"{naive_top['final_score']:.3f}",
            help=f"Resource: {naive_top['id']}",
        )
    with col_b:
        delta = top_result["final_score"] - naive_top["final_score"]
        st.metric(
            label="FSAR top-1 score (all signals)",
            value=f"{top_result['final_score']:.3f}",
            delta=f"{delta:+.3f} vs naive",
            help=f"Resource: {top_result['id']}",
        )

    if naive_top["id"] != top_result["id"]:
        st.info(
            f"**Ranking changed.** Naive top-1: `{naive_top['id'].split('/')[0]}` "
            f"→ FSAR top-1: `{top_result['id'].split('/')[0]}`"
        )
    else:
        st.success("Top-1 result is the same for both naive and FSAR.")

    st.divider()

    # --- Part A: Results table ---
    st.subheader(f"Top-{top_k} results")

    id_to_meta = {c["id"]: c for c in candidates}

    table_rows = []
    for rank, r in enumerate(fsar_results[:top_k], 1):
        cand = id_to_meta.get(r["id"], {})
        text = cand.get("text", "")
        table_rows.append({
            "Rank": rank,
            "Type": cand.get("resource_type", ""),
            "Date": cand.get("date", "") or "",
            "Statement": text[:100] + ("…" if len(text) > 100 else ""),
            "Score": round(r["final_score"], 4),
        })

    st.dataframe(
        pd.DataFrame(table_rows).set_index("Rank"),
        use_container_width=True,
        height=min(60 + len(table_rows) * 38, 500),
    )

    st.divider()

    # --- Part B: Signal breakdown per result ---
    st.subheader("Signal breakdown")

    for rank, r in enumerate(fsar_results[:top_k], 1):
        cand = id_to_meta.get(r["id"], {})
        rt = cand.get("resource_type", "?")
        label = f"#{rank}  {rt}  —  score {r['final_score']:.3f}  |  {r['id']}"
        with st.expander(label):
            bd = r["breakdown"]
            signals = list(bd.keys())
            values = [bd[s] for s in signals]
            colors = [SIGNAL_COLORS.get(s, "#888") for s in signals]

            fig = go.Figure(go.Bar(
                x=values,
                y=signals,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                cliponaxis=False,
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 1.15], title="Score"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=40, t=10, b=10),
                height=220,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"signal_chart_{rank}")

            cols = st.columns(len(signals))
            for col, sig, val in zip(cols, signals, values):
                col.metric(sig, f"{val:.3f}")

# ===========================================================================
# PAGE 2: Ablation Results
# ===========================================================================

else:
    st.title("Ablation Results")
    st.caption("Evaluated on 446 held-out test queries · k = 10")

    CONDITIONS_DISPLAY = {
        "naive":             "Naive (TF-IDF)",
        "FSAR_full":         "FSAR full",
        "FSAR_no_temporal":  "− temporal",
        "FSAR_no_reference": "− reference",
        "FSAR_no_code":      "− code_overlap",
        "FSAR_no_type":      "− type_prior",
    }

    fsar = eval_results["FSAR_full"]
    naive = eval_results["naive"]

    # --- Metric cards ---
    st.subheader("FSAR full — primary metrics vs naive")
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "nDCG@10",
        f"{fsar['ndcg_at_10']:.4f}",
        delta=f"{fsar['ndcg_at_10'] - naive['ndcg_at_10']:+.4f} vs naive",
    )
    m2.metric(
        "Recall@10",
        f"{fsar['recall_at_10']:.4f}",
        delta=f"{fsar['recall_at_10'] - naive['recall_at_10']:+.4f} vs naive",
    )
    m3.metric(
        "MRR",
        f"{fsar['mrr']:.4f}",
        delta=f"{fsar['mrr'] - naive['mrr']:+.4f} vs naive",
    )

    st.divider()

    # --- Plotly grouped bar chart ---
    st.subheader("nDCG@10 / Recall@10 / MRR by condition")

    cond_keys = list(CONDITIONS_DISPLAY.keys())
    cond_labels = [CONDITIONS_DISPLAY[k] for k in cond_keys]

    ndcg_vals   = [eval_results[k]["ndcg_at_10"]   for k in cond_keys]
    recall_vals = [eval_results[k]["recall_at_10"] for k in cond_keys]
    mrr_vals    = [eval_results[k]["mrr"]          for k in cond_keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="nDCG@10",   x=cond_labels, y=ndcg_vals))
    fig.add_trace(go.Bar(name="Recall@10", x=cond_labels, y=recall_vals))
    fig.add_trace(go.Bar(name="MRR",       x=cond_labels, y=mrr_vals))
    fig.update_layout(barmode="group", title="Metrics by condition")
    st.plotly_chart(fig, use_container_width=True, key="ablation_chart")

    st.divider()

    # --- Full results table ---
    st.subheader("Full results table")

    rows = []
    col_metrics = ["ndcg_at_10", "recall_at_10", "mrr", "temporal_prec_at_10", "coherence_at_10"]
    col_display = ["nDCG@10", "Recall@10", "MRR", "TempPrec@10", "Coh@10"]

    for k, display in CONDITIONS_DISPLAY.items():
        r = eval_results[k]
        row = {"Condition": display}
        for cm, cd in zip(col_metrics, col_display):
            v = r.get(cm)
            row[cd] = round(v, 4) if v is not None else None
        rows.append(row)

    results_df = pd.DataFrame(rows).set_index("Condition")

    # Highlight max per column
    def highlight_max(s):
        is_max = s == s.max()
        return ["font-weight: bold; color: #1a7f4b" if v else "" for v in is_max]

    styled = results_df.style.apply(highlight_max, axis=0)
    st.dataframe(styled, use_container_width=True)

    st.caption(
        "Eval measured on 446 held-out queries auto-generated from Synthea structure. "
        "TF-IDF lexical baseline; bge-small embeddings pending PyTorch fix."
    )
