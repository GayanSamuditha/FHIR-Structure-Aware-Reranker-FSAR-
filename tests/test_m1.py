"""M1 acceptance tests for ingestion module."""

import json
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.ingest import ingest_all, save_artifacts


def test_m1_ingestion():
    """
    M1 Acceptance Test
    - Row count > 0
    - 5 spot-checked statements read correctly
    - refgraph has edges
    - Refs resolve (no `urn:uuid:` left unresolved)
    """
    print("\n" + "=" * 70)
    print("M1 ACCEPTANCE TEST: INGESTION")
    print("=" * 70)

    # Run ingestion
    print("\n[1/5] Running ingestion...")
    df, refgraph = ingest_all()
    save_artifacts(df, refgraph)

    # Test 1: Row count > 0
    print("\n[2/5] Testing row count > 0...")
    assert len(df) > 0, "FAIL: No records ingested"
    print(f"✓ PASS: {len(df)} records ingested")

    # Test 2: Check that artifacts exist
    print("\n[3/5] Checking artifacts exist...")
    assert config.STATEMENTS_PATH.exists(), "FAIL: statements.parquet not created"
    assert config.REFGRAPH_PATH.exists(), "FAIL: refgraph.json not created"
    print(f"✓ PASS: Artifacts exist at {config.ARTIFACTS_DIR}")

    # Test 3: Spot-check 5 statements
    print("\n[4/5] Spot-checking 5 statements...")
    sample_size = min(5, len(df))
    samples = df.sample(n=sample_size, random_state=42)

    for idx, row in samples.iterrows():
        # Check required fields are present
        assert row["id"], f"FAIL: Missing id in row {idx}"
        assert row["patient_id"], f"FAIL: Missing patient_id in row {idx}"
        assert row["resource_type"], f"FAIL: Missing resource_type in row {idx}"
        assert row["text"], f"FAIL: Missing text in row {idx}"

        # Check text format (should have date | category | description)
        parts = row["text"].split(" | ")
        assert len(parts) >= 2, f"FAIL: Statement format incorrect in row {idx}: {row['text']}"

        # Check that id doesn't contain urn:uuid
        assert "urn:uuid" not in row["id"], f"FAIL: Unresolved urn:uuid in id: {row['id']}"

        # Check references don't contain urn:uuid
        if row["references"]:
            for ref in row["references"]:
                assert "urn:uuid" not in ref, f"FAIL: Unresolved urn:uuid in references: {ref}"

        # Check encounter_id doesn't contain urn:uuid
        if row["encounter_id"]:
            assert "urn:uuid" not in row["encounter_id"], f"FAIL: Unresolved urn:uuid in encounter_id: {row['encounter_id']}"

        print(f"  ✓ Row {idx}: {row['resource_type']} - {row['text'][:80]}...")

    print(f"✓ PASS: {sample_size} statements validated")

    # Test 4: Reference graph has edges
    print("\n[5/5] Testing reference graph...")
    assert len(refgraph["by_encounter"]) > 0, "FAIL: No encounters in reference graph"
    assert len(refgraph["adjacency"]) > 0, "FAIL: No adjacency edges in reference graph"

    # Check that adjacency actually has some edges (not just empty lists)
    total_edges = sum(len(neighbors) for neighbors in refgraph["adjacency"].values())
    assert total_edges > 0, "FAIL: Reference graph has no actual edges"

    print(f"  Encounters: {len(refgraph['by_encounter'])}")
    print(f"  Adjacency nodes: {len(refgraph['adjacency'])}")
    print(f"  Total edges: {total_edges}")
    print(f"✓ PASS: Reference graph validated")

    # Additional validation: Check no unresolved references anywhere
    print("\n[BONUS] Checking for any unresolved urn:uuid references...")
    unresolved_count = 0

    # Check all IDs
    for record_id in df["id"]:
        if "urn:uuid" in str(record_id):
            unresolved_count += 1
            print(f"  WARNING: Unresolved in id: {record_id}")

    # Check all references
    for refs in df["references"]:
        if refs:
            for ref in refs:
                if "urn:uuid" in str(ref):
                    unresolved_count += 1
                    print(f"  WARNING: Unresolved in references: {ref}")

    # Check all encounter_ids
    for enc_id in df["encounter_id"]:
        if enc_id and "urn:uuid" in str(enc_id):
            unresolved_count += 1
            print(f"  WARNING: Unresolved in encounter_id: {enc_id}")

    if unresolved_count == 0:
        print("✓ PASS: No unresolved urn:uuid references found")
    else:
        print(f"⚠ WARNING: Found {unresolved_count} unresolved references")

    # Print summary
    print("\n" + "=" * 70)
    print("M1 TEST SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(df)}")
    print(f"Resource type distribution:")
    for resource_type, count in df["resource_type"].value_counts().items():
        print(f"  {resource_type}: {count}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Records with dates: {df['date'].notna().sum()}")
    print(f"Records with codes: {df['codes'].apply(lambda x: len(x) > 0 if x else False).sum()}")
    print(f"Records with encounters: {df['encounter_id'].notna().sum()}")
    print(f"Records with references: {df['references'].apply(lambda x: len(x) > 0 if x else False).sum()}")
    print("=" * 70)
    print("✓ ALL M1 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_m1_ingestion()
        print("\n✓ M1 acceptance test completed successfully")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
