# FHIR-Structure-Aware Reranker (FSAR) — Build Spec

A from-scratch two-stage clinical retriever over Synthea FHIR, with a reranker that
fuses semantic similarity with FHIR-structure signals (temporal, resource-type,
code-overlap, reference-coherence), plus a **self-generating gold eval set** built
from Synthea's deterministic structure and an **ablation** proving the reranker beats
naive dense retrieval.

No LangChain, no LlamaIndex, no vector-DB framework. Raw embeddings, a hand-built
index, and explicit signal code.

> Naming note: signals are **temporal + referential**, NOT causal. FHIR references are
> structural/provenance links and date order is sequence — neither establishes
> causation. Do not label anything "causal." Metrics measure retrieval fidelity
> against Synthea's generative logic, not clinical truth.

---

## 0. Decisions / knobs (set in `config.py`)

| Knob | Default | Notes |
|---|---|---|
| `SUBSET_PATIENTS` | 200 (M1–M5), 2000 (M6) | See scale math below. Start tiny. |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` (384-d, CPU-fast) | Clinical alt: `pritamdeka/S-PubMedBert-MS-MARCO`. |
| `USE_CROSS_ENCODER` | `False` | If True: `cross-encoder/ms-marco-MiniLM-L-6-v2` as the semantic component. Default keeps base = bi-encoder cosine (cheaper, cleaner ablation). |
| `TOP_K_RETRIEVE` | 50 | Stage-1 candidate pool. |
| `TOP_K_FINAL` | 10 | After rerank. |
| `INDEX_BACKEND` | `numpy` (M2–M5), `faiss` (M6) | numpy brute-force cosine is fine at the scales below. |
| `WEIGHTS` | see §5 | Tune on held-out split; do not treat defaults as discovered. |

**Scale math (so you don't OOM in Claude Code):** float32, 384-d → ~1.5 KB/vector.
~100 resources/patient. 200 patients ≈ 20k vectors ≈ 30 MB. 2000 patients ≈ 200k ≈
300 MB. Both fit in RAM with numpy. Stay ≤ ~2k patients before switching to FAISS.

---

## 1. Repo layout

```
fhir-rerank/
  data/                 # raw Synthea bundles (gitignored)
  artifacts/            # statements.parquet, vectors.npy, refgraph.json, eval_queries.jsonl (gitignored)
  src/
    config.py           # all paths, model names, k-values, WEIGHTS
    ingest.py           # bundles -> records + reference graph  -> artifacts/
    render.py           # one resource -> one natural-language statement
    embed.py            # batch-embed statements -> vectors.npy (cached)
    index.py            # build + query (numpy cosine; faiss optional)
    retrieve.py         # stage-1 dense top-k
    signals.py          # individual scoring signals (pure functions)
    rerank.py           # weighted fusion + per-signal breakdown
    eval_gen.py         # auto-generate gold query/answer pairs
    eval_run.py         # run naive vs reranked vs ablations -> results table
    metrics.py          # recall@k, precision@k, MRR, nDCG@k (graded)
    cli.py              # thin entrypoints per milestone
  tests/                # one acceptance test per milestone
  requirements.txt
  README.md
```

`requirements.txt`: `numpy pandas pyarrow sentence-transformers scikit-learn tqdm python-dateutil` (add `faiss-cpu` only at M6).

---

## 2. Synthea facts (hand these to Claude Code so it does NOT explore)

- Output is FHIR **R4**. One JSON per patient: `output/fhir/{Name}_{uuid}.json`.
  **Skip** `hospitalInformation*.json` and `practitionerInformation*.json` (orgs/practitioners, not patient journeys).
- Each file is a `Bundle`, `type: "transaction"`, with `entry: [{fullUrl, resource, request}]`.
- **Intra-bundle references are `urn:uuid:<uuid>`** matching some `entry.fullUrl`. To resolve
  a reference, first build a per-bundle map `fullUrl -> (resourceType, localId)`, then resolve
  every `reference` string through it. Some refs are `"ResourceType/id"`; handle both.
- Primary clinical dates by type:
  - `Condition.onsetDateTime`
  - `Observation.effectiveDateTime`
  - `MedicationRequest.authoredOn`
  - `Procedure.performedDateTime` OR `Procedure.performedPeriod.start`
  - `Encounter.period.start`
  - `DiagnosticReport.effectiveDateTime`
- Codes live at `resource.code.coding[]` = `{system, code, display}`; medications at
  `resource.medicationCodeableConcept.coding[]`. Systems:
  SNOMED `http://snomed.info/sct`, LOINC `http://loinc.org`,
  RxNorm `http://www.nlm.nih.gov/research/umls/rxnorm`.
- Reference edges that matter for coherence:
  `*.encounter -> Encounter`, `MedicationRequest.reasonReference -> Condition`,
  `Procedure.reasonReference -> Condition`, `Encounter.reasonReference -> Condition`,
  `DiagnosticReport.result[] -> Observation`.

**Resource types to ingest:** Condition, Observation, MedicationRequest, Procedure,
Encounter, DiagnosticReport. (Optionally AllergyIntolerance, Immunization, CarePlan.)
Ignore Claim/ExplanationOfBenefit/Coverage — billing noise.

---

## 3. Canonical record schema (output of `ingest.py`)

One row per resource, written to `artifacts/statements.parquet`:

```python
{
  "id": str,            # f"{resource_type}/{local_id}"  (stable, unique)
  "patient_id": str,    # the Patient resource id
  "resource_type": str,
  "text": str,          # rendered statement from render.py
  "date": str | None,   # ISO 8601, the primary clinical date for this type
  "codes": list[dict],  # [{system, code, display}]
  "encounter_id": str | None,
  "references": list[str],  # resolved ids this resource points to
}
```

`artifacts/refgraph.json`:
```python
{
  "by_encounter": {encounter_id: [resource_id, ...]},   # co-encounter grouping
  "adjacency":   {resource_id: [neighbor_id, ...]},      # union of out+in references
}
```

---

## 4. render.py — resource -> statement

Compact, structured, date-stamped. Carry codes inline. Example targets:

- Observation: `"2019-03-14 | lab | Serum creatinine 1.8 mg/dL (LOINC 2160-0)"`
- Condition: `"2017-06-02 | diagnosis | Type 2 diabetes mellitus (SNOMED 44054006)"`
- MedicationRequest: `"2019-04-01 | medication started | Metformin 500mg (RxNorm 860975)"`
- Procedure: `"2018-11-20 | procedure | Echocardiography (SNOMED 40701008)"`

**Anti-leakage rule:** statements describe the resource; **eval query text must never
copy a statement** (see §6), or semantic retrieval wins trivially.

---

## 5. The reranker (signals.py + rerank.py)

All signals return `[0,1]`. Final score = weighted sum; keep per-signal values for the
explainability breakdown (used in demo + writeup).

```python
final = (W.semantic   * semantic        # cosine (or cross-encoder if enabled)
       + W.temporal    * temporal_proximity
       + W.type_prior  * type_prior
       + W.code_overlap * code_overlap
       + W.reference    * reference_coherence
       + W.specificity * specificity)
```

Default `WEIGHTS` (tune later): semantic .40, temporal .20, reference .15,
code_overlap .15, type_prior .07, specificity .03.

Signal definitions (pure functions, each unit-tested):

- `semantic(q_vec, c_vec)` — cosine, min-max normalized over the candidate pool.
- `temporal_proximity(anchor_date, direction, c_date)` — if the query carries an anchor
  date + direction (`"before"`/`"after"`/`None`): exponential decay on day-delta;
  **return 0 if c_date is on the wrong side** of a directional query. No anchor → 0.5 (neutral).
- `type_prior(resource_type, intent)` — lookup: `diagnostic -> {Condition, DiagnosticReport}`,
  `treatment -> {MedicationRequest, Procedure}`, `lab -> {Observation}`. Match → 1, else baseline.
- `code_overlap(query_codes, c_codes)` — overlap on normalized `(system, code)`; partial
  credit for shared SNOMED prefix (keep simple; full ontology optional).
- `reference_coherence(c_id, anchor_id, refgraph)` — 1 if same encounter as anchor, 0.6 if
  in anchor's adjacency, else 0. (Anchor = the query's pinned resource, supplied by eval gen.)
- `specificity(c_codes, resource_type)` — small boost for richer/coded resources.

Intent + anchor for the **eval path are supplied as query metadata** (no NLP needed).
A free-text intent classifier + date parser is an optional later phase — do not build it now.

---

## 6. eval_gen.py — the self-generating gold set (the differentiator)

Fully automatic from parsed data. Each query template emits: `query_text`,
`meta` (`anchor_date`, `direction`, `intent`, `anchor_id`, `relevant_codes`),
and **graded gold** `relevance: {resource_id: grade}` (2 = exact, 1 = related, 0 = distractor).

Templates:
1. **Temporal window.** Patient with MedicationRequest M (date D). Gold(2) = that patient's
   Observations in `[D - window, D]`; gold(1) = same patient's labs outside window;
   distractors = other patients' matching labs. Query: *"What lab results did the patient have
   in the {window} before starting {med.display}?"* `direction="before"`.
2. **Encounter coherence.** Condition C at Encounter E. Gold(2) = resources with
   `encounter_id == E` or `reasonReference == C`; gold(1) = same patient, other encounters.
   Query: *"What happened during the encounter where the patient was diagnosed with {C.display}?"*
3. **Code/specificity.** Query: *"Find {condition.display} diagnoses for the patient."*
   Gold(2) = Conditions with matching SNOMED code; gold(1) = text-mentioning-but-uncoded resources.
4. **Treatment-after (intent + direction).** Query: *"What treatments followed the {C.display}
   diagnosis?"* `direction="after"`, `intent="treatment"`. Gold(2) = Med/Procedure after onset.

**Validity gates (assert in the generator):**
- Each query has ≥3 gold(2) and ≥10 distractors (else skip — guarantees a non-trivial pool).
- **Leakage check:** reject if `query_text` shares a long n-gram with any gold statement.
- Only emit queries for patients meeting the data preconditions (e.g., ≥N labs straddling D).

Write to `artifacts/eval_queries.jsonl`. Split 70/30 into `tune`/`test`; tune `WEIGHTS`
on `tune` only.

---

## 7. metrics.py + eval_run.py

Metrics over `TOP_K_FINAL`: `recall@k`, `precision@k`, `MRR`, `nDCG@k` (graded, uses the
2/1/0 grades). Plus two project-specific ones: **temporal_precision@k** (fraction of top-k
in the correct window for temporal queries) and **coherence@k** (fraction sharing the gold
encounter for coherence queries).

`eval_run.py` produces one table:

| condition | nDCG@10 | recall@10 | MRR | temporal_prec@10 | coherence@10 |
|---|---|---|---|---|---|
| naive (dense only) | … | … | … | … | … |
| FSAR (full) | … | … | … | … | … |
| FSAR − temporal | … | … | … | … | … |
| FSAR − reference | … | … | … | … | … |
| FSAR − code_overlap | … | … | … | … | … |

Ablations = drop one signal (weight→0) and re-score. This table IS the deliverable / the
screenshot. **Acceptance: FSAR(full) ≥ naive on nDCG@10**, and each ablation row shows a
measurable drop attributable to its signal.

---

## 8. Milestones (build one per Claude Code session; gate on the test)

- **M1 ingest** → `statements.parquet` + `refgraph.json` for 200 patients.
  *Accept:* row count > 0; 5 spot-checked statements read correctly; refgraph has edges; refs resolve (no `urn:uuid:` left unresolved).
- **M2 embed + index + retrieve** → top-k for a hardcoded query.
  *Accept:* `vectors.npy` shape matches row count; neighbors are topically sensible.
- **M3 eval_gen** → `eval_queries.jsonl`.
  *Accept:* ≥50 queries; every query passes the §6 validity gates; leakage check passes.
- **M4 signals + rerank** → rerank one candidate list with per-signal breakdown printed.
  *Accept:* each signal unit-tested; directional temporal returns 0 on wrong side.
- **M5 eval_run** → the §7 table.
  *Accept:* FSAR ≥ naive on nDCG@10; ablations behave.
- **M6 (optional)** scale to ~2k patients + FAISS; Streamlit demo (3 preset queries with
  before/after); README with the table and one before/after example.

---

## 9. Token-economy rules for the Claude Code sessions

- Treat **this file as the spec**; do not let Claude Code re-explore the dataset or re-derive schemas — §2/§3 already give it everything.
- **One module/milestone per session.** Do not ask for the whole repo at once.
- **Cache aggressively:** embeddings → `artifacts/vectors.npy`; never re-embed on rerun (check mtime/hash).
- **Gate with tests, not prints.** Each milestone writes to disk and asserts; do not paste big DataFrames/JSON into the chat.
- Keep **one source of truth = `config.py`**. No magic numbers scattered in modules.
- Tell it explicitly: *"don't print large outputs; write to artifacts/ and assert shapes/counts."*
- Fixed small subset (200) until M5 passes; only scale at M6.
- Pin function signatures from this spec so it doesn't invent its own interfaces.
