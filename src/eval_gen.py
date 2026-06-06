"""Auto-generate gold eval query/answer pairs from parsed FHIR data."""

import json
import random
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from src import config


def load_data() -> Tuple[pd.DataFrame, dict]:
    """
    Load statements and reference graph.

    Returns:
        Tuple of (statements_df, refgraph)
    """
    print(f"Loading statements from {config.STATEMENTS_PATH}...")
    statements_df = pd.read_parquet(config.STATEMENTS_PATH)

    # Drop resources with empty IDs (e.g., "ResourceType/") — they are un-addressable
    statements_df = statements_df[~statements_df["id"].str.endswith("/")].copy()
    # Drop any remaining duplicates; keep first occurrence
    statements_df = statements_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    print(f"Loading reference graph from {config.REFGRAPH_PATH}...")
    with open(config.REFGRAPH_PATH, "r") as f:
        refgraph = json.load(f)

    return statements_df, refgraph


def get_ngrams(text: str, n: int = 4) -> set:
    """
    Extract n-grams from text.

    Args:
        text: Input text
        n: N-gram size

    Returns:
        Set of n-grams
    """
    words = text.lower().split()
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i + n])
        ngrams.add(ngram)
    return ngrams


def has_ngram_overlap(query_text: str, resource_texts: List[str], n: int = 4) -> bool:
    """
    Check if query shares any n-grams with resource texts.

    Args:
        query_text: Query text
        resource_texts: List of resource text strings
        n: N-gram size

    Returns:
        True if there is overlap, False otherwise
    """
    query_ngrams = get_ngrams(query_text, n)
    if not query_ngrams:
        return False

    for text in resource_texts:
        text_ngrams = get_ngrams(text, n)
        if query_ngrams & text_ngrams:
            return True

    return False


def parse_date(date_str: str | None) -> datetime | None:
    """
    Parse ISO 8601 date string to datetime.

    Args:
        date_str: ISO 8601 date string

    Returns:
        datetime object or None
    """
    if not date_str:
        return None

    try:
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(date_str)
        # Strip timezone info so all comparisons use naive datetimes
        return dt.replace(tzinfo=None)
    except Exception:
        return None


def generate_template1_temporal_window(
    statements_df: pd.DataFrame,
    refgraph: dict,
    window_days: int = 180,
) -> List[dict]:
    """
    TEMPLATE 1: Temporal window - lab results before medication.

    Args:
        statements_df: Statements DataFrame
        refgraph: Reference graph
        window_days: Window size in days

    Returns:
        List of query dictionaries
    """
    queries = []

    # Get medications with dates
    meds = statements_df[
        (statements_df["resource_type"] == "MedicationRequest") &
        (statements_df["date"].notna())
    ].copy()

    print(f"  Found {len(meds)} medications with dates")

    for idx, med in meds.iterrows():
        med_date = parse_date(med["date"])
        if not med_date:
            continue

        patient_id = med["patient_id"]
        med_id = med["id"]

        # Get med display name
        med_display = "medication"
        if med["codes"] and len(med["codes"]) > 0:
            med_display = med["codes"][0].get("display", "medication")

        # Find observations for this patient in the window
        patient_obs = statements_df[
            (statements_df["patient_id"] == patient_id) &
            (statements_df["resource_type"] == "Observation") &
            (statements_df["date"].notna())
        ].copy()

        if len(patient_obs) == 0:
            continue

        # Categorize by window
        gold_2 = []  # In window
        gold_1 = []  # Outside window

        for obs_idx, obs in patient_obs.iterrows():
            obs_date = parse_date(obs["date"])
            if not obs_date:
                continue

            days_diff = (med_date - obs_date).days

            if 0 <= days_diff <= window_days:
                gold_2.append(obs["id"])
            else:
                gold_1.append(obs["id"])

        # Check validity: need ≥3 gold(2)
        if len(gold_2) < 3:
            continue

        # Get distractors from other patients (exclude any IDs that appear in gold sets)
        gold_ids = set(gold_2) | set(gold_1)
        other_patients_obs = statements_df[
            (statements_df["patient_id"] != patient_id) &
            (statements_df["resource_type"] == "Observation") &
            (~statements_df["id"].isin(gold_ids))
        ]

        distractors = other_patients_obs["id"].tolist()[:20]

        # Check validity: need ≥10 distractors
        if len(distractors) < 10:
            continue

        # Generate query text
        query_text = f"What lab results did the patient have in the 6 months before starting {med_display}?"

        # Check for n-gram overlap with gold resources
        gold_texts = patient_obs[patient_obs["id"].isin(gold_2)]["text"].tolist()
        if has_ngram_overlap(query_text, gold_texts):
            continue

        # Build relevance dict — distractors first so gold grades always win
        relevance = {}
        for rid in distractors:
            relevance[rid] = 0
        for rid in gold_1:
            relevance[rid] = 1
        for rid in gold_2:
            relevance[rid] = 2

        # Create query
        query = {
            "query_id": f"T1_{len(queries)}",
            "template": 1,
            "query_text": query_text,
            "meta": {
                "anchor_date": med["date"],
                "direction": "before",
                "intent": "lab",
                "anchor_id": med_id,
                "window_days": window_days,
            },
            "relevance": relevance,
        }

        queries.append(query)

    return queries


def generate_template2_encounter_coherence(
    statements_df: pd.DataFrame,
    refgraph: dict,
) -> List[dict]:
    """
    TEMPLATE 2: Encounter coherence - what happened during encounter.

    Args:
        statements_df: Statements DataFrame
        refgraph: Reference graph

    Returns:
        List of query dictionaries
    """
    queries = []

    # Get encounters with ≥2 other resources
    by_encounter = refgraph.get("by_encounter", {})

    print(f"  Found {len(by_encounter)} encounters in refgraph")

    for encounter_id, resource_ids in by_encounter.items():
        # Need at least 3 resources for the encounter (≥2 other resources plus encounter itself)
        if len(resource_ids) < 3:
            continue

        # Get encounter resource
        encounter = statements_df[statements_df["id"] == encounter_id]
        if len(encounter) == 0:
            continue

        encounter = encounter.iloc[0]
        patient_id = encounter["patient_id"]
        encounter_date = encounter.get("date", "unknown")

        # Gold(2) = resources with same encounter_id (excluding the encounter itself),
        # deduped so duplicate IDs don't inflate count
        gold_2_set = set(rid for rid in resource_ids if rid != encounter_id)
        gold_2 = list(gold_2_set)

        # Check validity: need ≥3 gold(2)
        if len(gold_2) < 3:
            continue

        # Gold(1) = same patient, different encounters
        same_patient = statements_df[
            (statements_df["patient_id"] == patient_id) &
            (statements_df["encounter_id"].notna()) &
            (statements_df["encounter_id"] != encounter_id) &
            (~statements_df["id"].isin(gold_2_set))
        ]
        gold_1 = same_patient["id"].tolist()[:10]

        # Distractors = other patients' resources (exclude all gold IDs)
        gold_ids = gold_2_set | set(gold_1)
        other_patients = statements_df[
            (statements_df["patient_id"] != patient_id) &
            (~statements_df["id"].isin(gold_ids))
        ]
        distractors = other_patients["id"].tolist()[:20]

        # Check validity: need ≥10 distractors
        if len(distractors) < 10:
            continue

        # Generate query text
        encounter_date_str = encounter_date.split("T")[0] if encounter_date else "unknown date"
        query_text = f"What happened during the encounter where the patient was seen on {encounter_date_str}?"

        # Check for n-gram overlap
        gold_texts = statements_df[statements_df["id"].isin(gold_2)]["text"].tolist()
        if has_ngram_overlap(query_text, gold_texts):
            continue

        # Build relevance dict — distractors first so gold grades always win
        relevance = {}
        for rid in distractors:
            relevance[rid] = 0
        for rid in gold_1:
            relevance[rid] = 1
        for rid in gold_2:
            relevance[rid] = 2

        # Create query
        query = {
            "query_id": f"T2_{len(queries)}",
            "template": 2,
            "query_text": query_text,
            "meta": {
                "anchor_date": encounter_date,
                "direction": None,
                "intent": "diagnostic",
                "anchor_id": encounter_id,
            },
            "relevance": relevance,
        }

        queries.append(query)

    return queries


def generate_template4_treatment_after(
    statements_df: pd.DataFrame,
    refgraph: dict,
    window_days: int = 365,
) -> List[dict]:
    """
    TEMPLATE 4: Treatment after condition diagnosis.

    Args:
        statements_df: Statements DataFrame
        refgraph: Reference graph
        window_days: Window size in days

    Returns:
        List of query dictionaries
    """
    queries = []

    # Get conditions with dates
    conditions = statements_df[
        (statements_df["resource_type"] == "Condition") &
        (statements_df["date"].notna())
    ].copy()

    print(f"  Found {len(conditions)} conditions with dates")

    for idx, condition in conditions.iterrows():
        condition_date = parse_date(condition["date"])
        if not condition_date:
            continue

        patient_id = condition["patient_id"]
        condition_id = condition["id"]

        # Get condition display name
        condition_display = "condition"
        if condition["codes"] and len(condition["codes"]) > 0:
            condition_display = condition["codes"][0].get("display", "condition")

        # Find treatments (MedicationRequest or Procedure) for this patient after condition
        patient_treatments = statements_df[
            (statements_df["patient_id"] == patient_id) &
            (statements_df["resource_type"].isin(["MedicationRequest", "Procedure"])) &
            (statements_df["date"].notna())
        ].copy()

        if len(patient_treatments) == 0:
            continue

        # Categorize by window
        gold_2 = []  # Within window after
        gold_1 = []  # Outside window

        for tx_idx, tx in patient_treatments.iterrows():
            tx_date = parse_date(tx["date"])
            if not tx_date:
                continue

            days_diff = (tx_date - condition_date).days

            if 0 <= days_diff <= window_days:
                gold_2.append(tx["id"])
            else:
                gold_1.append(tx["id"])

        # Check validity: need ≥3 gold(2)
        if len(gold_2) < 3:
            continue

        # Get distractors from other patients (exclude gold IDs)
        gold_ids = set(gold_2) | set(gold_1)
        other_patients_tx = statements_df[
            (statements_df["patient_id"] != patient_id) &
            (statements_df["resource_type"].isin(["MedicationRequest", "Procedure"])) &
            (~statements_df["id"].isin(gold_ids))
        ]

        distractors = other_patients_tx["id"].tolist()[:20]

        # Check validity: need ≥10 distractors
        if len(distractors) < 10:
            continue

        # Generate query text
        query_text = f"What treatments followed the {condition_display} diagnosis?"

        # Check for n-gram overlap
        gold_texts = patient_treatments[patient_treatments["id"].isin(gold_2)]["text"].tolist()
        if has_ngram_overlap(query_text, gold_texts):
            continue

        # Build relevance dict — distractors first so gold grades always win
        relevance = {}
        for rid in distractors:
            relevance[rid] = 0
        for rid in gold_1:
            relevance[rid] = 1
        for rid in gold_2:
            relevance[rid] = 2

        # Create query
        query = {
            "query_id": f"T4_{len(queries)}",
            "template": 4,
            "query_text": query_text,
            "meta": {
                "anchor_date": condition["date"],
                "direction": "after",
                "intent": "treatment",
                "anchor_id": condition_id,
                "window_days": window_days,
            },
            "relevance": relevance,
        }

        queries.append(query)

    return queries


def generate_all_queries(
    statements_df: pd.DataFrame,
    refgraph: dict,
) -> List[dict]:
    """
    Generate all queries from all templates.

    Args:
        statements_df: Statements DataFrame
        refgraph: Reference graph

    Returns:
        List of all query dictionaries
    """
    all_queries = []

    print("\nGenerating Template 1 (Temporal window) queries...")
    t1_queries = generate_template1_temporal_window(statements_df, refgraph)
    print(f"  Generated {len(t1_queries)} queries")
    all_queries.extend(t1_queries)

    print("\nGenerating Template 2 (Encounter coherence) queries...")
    t2_queries = generate_template2_encounter_coherence(statements_df, refgraph)
    print(f"  Generated {len(t2_queries)} queries")
    all_queries.extend(t2_queries)

    print("\nGenerating Template 4 (Treatment after) queries...")
    t4_queries = generate_template4_treatment_after(statements_df, refgraph)
    print(f"  Generated {len(t4_queries)} queries")
    all_queries.extend(t4_queries)

    return all_queries


def split_queries(queries: List[dict], train_ratio: float = 0.7) -> Tuple[List[dict], List[dict]]:
    """
    Split queries into tune and test sets.

    Args:
        queries: List of query dictionaries
        train_ratio: Ratio for training set

    Returns:
        Tuple of (tune_queries, test_queries)
    """
    # Shuffle with fixed seed for reproducibility
    shuffled = queries.copy()
    random.seed(42)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    tune_queries = shuffled[:split_idx]
    test_queries = shuffled[split_idx:]

    return tune_queries, test_queries


def save_queries(queries: List[dict], output_path: str) -> None:
    """
    Save queries to JSONL file.

    Args:
        queries: List of query dictionaries
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        for query in queries:
            f.write(json.dumps(query) + "\n")


def main():
    """Main entry point for M3 eval generation."""
    # Load data
    statements_df, refgraph = load_data()
    print(f"Loaded {len(statements_df)} statements")

    # Generate queries
    all_queries = generate_all_queries(statements_df, refgraph)

    print("\n" + "=" * 60)
    print(f"Total queries generated: {len(all_queries)}")
    print("=" * 60)

    # Split into tune/test
    tune_queries, test_queries = split_queries(all_queries)
    print(f"Tune queries: {len(tune_queries)}")
    print(f"Test queries: {len(test_queries)}")

    # Save all queries
    all_path = config.ARTIFACTS_DIR / "eval_queries.jsonl"
    tune_path = config.ARTIFACTS_DIR / "eval_queries_tune.jsonl"
    test_path = config.ARTIFACTS_DIR / "eval_queries_test.jsonl"

    print(f"\nSaving queries to {config.ARTIFACTS_DIR}...")
    save_queries(all_queries, all_path)
    save_queries(tune_queries, tune_path)
    save_queries(test_queries, test_path)

    print("✓ Queries saved")

    # Print summary by template
    print("\n" + "=" * 60)
    print("QUERY GENERATION SUMMARY")
    print("=" * 60)
    template_counts = {}
    for query in all_queries:
        template = query["template"]
        template_counts[template] = template_counts.get(template, 0) + 1

    for template in sorted(template_counts.keys()):
        print(f"Template {template}: {template_counts[template]} queries")

    print(f"Total: {len(all_queries)} queries")
    print("=" * 60)


if __name__ == "__main__":
    main()
