"""Configuration for FHIR-Structure-Aware Reranker (FSAR)."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "output_11" / "fhir"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure artifacts directory exists
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Artifact paths
STATEMENTS_PATH = ARTIFACTS_DIR / "statements.parquet"
REFGRAPH_PATH = ARTIFACTS_DIR / "refgraph.json"
VECTORS_PATH = ARTIFACTS_DIR / "vectors.npy"
EVAL_QUERIES_PATH = ARTIFACTS_DIR / "eval_queries.jsonl"

# Dataset configuration
SUBSET_PATIENTS = 200  # M1-M5: 200, M6: 2000

# Embedding model
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-d, CPU-fast
USE_CROSS_ENCODER = False
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval parameters
TOP_K_RETRIEVE = 50  # Stage-1 candidate pool
TOP_K_FINAL = 10     # After rerank

# Index backend
INDEX_BACKEND = "numpy"  # M2-M5: numpy, M6: faiss

# Reranker weights (tune on held-out split)
class WEIGHTS:
    semantic = 0.40
    temporal = 0.20
    reference = 0.15
    code_overlap = 0.15
    type_prior = 0.07
    specificity = 0.03

# Resource types to ingest
RESOURCE_TYPES = [
    "Condition",
    "Observation",
    "MedicationRequest",
    "Procedure",
    "Encounter",
    "DiagnosticReport",
]

# Optional resource types (can be enabled later)
OPTIONAL_RESOURCE_TYPES = [
    "AllergyIntolerance",
    "Immunization",
    "CarePlan",
]

# Skip these file patterns
SKIP_FILE_PATTERNS = [
    "hospitalInformation",
    "practitionerInformation",
]

# Date field mapping by resource type
DATE_FIELDS = {
    "Condition": "onsetDateTime",
    "Observation": "effectiveDateTime",
    "MedicationRequest": "authoredOn",
    "Procedure": ["performedDateTime", "performedPeriod.start"],
    "Encounter": "period.start",
    "DiagnosticReport": "effectiveDateTime",
}

# Code field mapping by resource type
CODE_FIELDS = {
    "MedicationRequest": "medicationCodeableConcept",
    # All others use "code"
}

# Reference fields that matter for coherence
REFERENCE_FIELDS = [
    "encounter",
    "reasonReference",
    "result",
]

# Code systems
CODE_SYSTEMS = {
    "SNOMED": "http://snomed.info/sct",
    "LOINC": "http://loinc.org",
    "RxNorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
}
