# FHIR Structure-Aware Reranker (FSAR)

A two-stage clinical retriever over Synthea FHIR bundles: TF-IDF dense retrieval followed by a reranker that fuses semantic similarity with four structural signals (temporal proximity, reference coherence, type prior, code overlap). The evaluation set is generated automatically from Synthea's deterministic structure, and the ablation table confirms that each structural signal contributes measurably to retrieval quality.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Synthea FHIR bundles  (data/output_11/fhir/*.json)                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  M1  ingest.py  │  parse resources, resolve urn:uuid refs
                    │  render.py      │  resource → dated text statement
                    └────────┬────────┘
                             │  artifacts/statements.parquet
                             │  artifacts/refgraph.json
                    ┌────────▼────────┐
                    │  M2  embed.py   │  TF-IDF (8 000-d), L2-normalised
                    │  index.py       │  NumpyIndex — brute-force cosine
                    └────────┬────────┘
                             │  artifacts/vectors.npy
                             │  artifacts/tfidf_vectorizer.pkl
          ┌──────────────────┼──────────────────┐
          │                  │                  │
 ┌────────▼────────┐         │        ┌─────────▼────────┐
 │  M3  eval_gen   │         │        │  query at runtime │
 │  3 templates    │         │        └─────────┬────────┘
 │  1485 queries   │         │                  │ embed_query()
 └────────┬────────┘         │        ┌─────────▼────────┐
          │  artifacts/      │        │  retrieve.py     │  top-50 cosine
          │  eval_queries*   │        └─────────┬────────┘
          │                  │                  │ candidates + c_vecs
          │         ┌────────▼────────┐         │
          │         │  M4 signals.py  │◄────────┘
          │         │  rerank.py      │  weighted fusion of 6 signals
          │         └────────┬────────┘
          │                  │  ranked top-10
          │         ┌────────▼────────┐
          └────────►│  M5 eval_run.py │  naive vs FSAR vs 4 ablations
                    │  metrics.py     │  nDCG, recall, MRR, temporal_prec, coh
                    └────────┬────────┘
                             │  artifacts/eval_results.json
```

**Six reranker signals** (all return `[0, 1]`, combined via weighted sum):

| Signal | Weight | Description |
|---|---|---|
| semantic | 0.40 | TF-IDF cosine, min-max normalised across candidate pool |
| temporal | 0.20 | Exponential decay on day-distance; hard 0 on wrong direction |
| reference | 0.15 | 1.0 same encounter · 0.6 adjacent in refgraph · 0.0 otherwise |
| code_overlap | 0.15 | Jaccard on `(system, code)` pairs |
| type_prior | 0.07 | Lookup: lab→Observation, treatment→Med/Proc, diagnostic→Cond/DR/Obs |
| specificity | 0.03 | Code-count tier: 0→0.3, 1→0.7, 2+→1.0 |

---

## Results

Evaluated on 446 held-out test queries (70/30 split of 1 485 auto-generated queries).  
`k = 10` for all metrics. Temporal precision computed on Template-1 queries only (n = 49); coherence on Template-2 (n = 359).

| Condition | nDCG@10 | Recall@10 | MRR | TempPrec@10 | Coh@10 |
|---|---|---|---|---|---|
| naive (TF-IDF only) | 0.0060 | 0.0101 | 0.0352 | 0.6122 | 0.0064 |
| **FSAR full** | **0.0933** | **0.1187** | **0.1941** | **0.7571** | **0.0825** |
| − temporal | 0.0747 | 0.1034 | 0.1787 | 0.6224 | 0.0657 |
| − reference | 0.0703 | 0.1032 | 0.1481 | 0.7327 | 0.0669 |
| − code_overlap | 0.0883 | 0.1152 | 0.1869 | 0.7327 | 0.0805 |
| − type_prior | 0.0856 | 0.1129 | 0.1832 | 0.7429 | 0.0780 |

FSAR full is **15× higher nDCG** than naive. Every ablation degrades all three primary metrics, with `− reference` showing the largest MRR drop (−23%) and `− temporal` the largest coherence drop.

---

## Reproduce

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyarrow scikit-learn tqdm python-dateutil sentence-transformers
```

### 2. Data setup

Obtain a Synthea FHIR output folder and point `config.py` at it, or use the included `synthea_1m_fhir_1_8` subset already placed in `data/output_11/fhir/`. The expected layout is:

```
data/
  output_11/
    fhir/
      <Name>_<uuid>.json    ← one FHIR R4 bundle per patient
```

Skip `hospitalInformation*.json` and `practitionerInformation*.json` — the ingester does this automatically.

### 3. Run milestones in order

Each milestone writes to `artifacts/` and is gated by its test.

```bash
# M1 — ingest FHIR bundles (≈ 30 s for 200 patients)
python -m src.ingest
python tests/test_m1.py

# M2 — embed statements, build index, smoke-test retrieval
python -m src.embed      # fits TF-IDF, caches vectors.npy
python tests/test_m2.py

# M3 — generate gold eval queries
python -m src.eval_gen
python tests/test_m3.py

# M4 — unit-test signals and reranker
python tests/test_m4.py  # no artifact needed; pure unit tests

# M5 — run full evaluation, emit results table
python tests/test_m5.py
```

All five milestones complete in under 5 minutes on a laptop CPU with 200 patients.

---

## Honest limitations

**TF-IDF stand-in for embeddings.** `config.py` specifies `BAAI/bge-small-en-v1.5` as the intended embedding model. A PyTorch/environment conflict prevented loading it, so `embed.py` uses a TF-IDF vectorizer instead. All retrieval and reranking logic is identical; only the semantic signal quality differs. Swapping in the sentence-transformer requires one line change in `embed.py`.

**Synthea synthetic data only.** All 181 patients and 13 113 resources are generated by Synthea. The dataset is structurally valid FHIR but contains no real patient records. Clinical patterns reflect Synthea's generative model, not population epidemiology.

**Eval measures retrieval fidelity against Synthea's logic, not clinical truth.** Gold labels are derived from structural relationships in the FHIR graph (shared encounter IDs, `reasonReference` edges, temporal windows around medication dates). A high score means the reranker recovers what Synthea's generator placed in the same encounter or time window — it does not mean the retrieved resources are clinically relevant to a real clinician's query.

**Signals are temporal and referential, not causal.** FHIR references are provenance/structural links; date ordering is sequence. Neither implies causation.

---

## Tech stack

| Layer | Library |
|---|---|
| Data parsing | `json`, `pathlib` (stdlib) |
| Tabular storage | `pandas`, `pyarrow` (Parquet) |
| Vectors | `numpy` (brute-force cosine; no vector DB) |
| Embeddings | `scikit-learn` TfidfVectorizer (stand-in for `sentence-transformers`) |
| Date handling | `python-dateutil`, stdlib `datetime` |
| Progress | `tqdm` |

No LangChain. No LlamaIndex. No vector database framework. No external API calls.
