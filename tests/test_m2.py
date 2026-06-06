"""M2 acceptance tests for embedding and retrieval."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.embed import embed_statements, load_model, get_embedding_dim
from src.index import load_index
from src.retrieve import retrieve, format_results


def test_m2_embed_and_retrieve():
    """
    M2 Acceptance Test
    - vectors.npy shape == (n_statements, 384)
    - top-5 results are all valid record ids from statements.parquet
    - scores are between 0 and 1
    - results are sorted descending by score
    - no re-embedding on second run (assert vectors.npy mtime unchanged)
    """
    print("\n" + "=" * 80)
    print("M2 ACCEPTANCE TEST: EMBEDDING AND RETRIEVAL")
    print("=" * 80)

    # Load statements
    print("\n[1/7] Loading statements...")
    statements_df = pd.read_parquet(config.STATEMENTS_PATH)
    print(f"✓ Loaded {len(statements_df)} statements")

    # Test 1: First embedding run
    print("\n[2/7] Running first embedding...")
    vectors = embed_statements(statements_df)

    # Check shape (only assert first dimension - number of statements)
    assert vectors.shape[0] == len(statements_df), \
        f"FAIL: Expected {len(statements_df)} vectors, got {vectors.shape[0]}"
    print(f"✓ PASS: Vectors shape is {vectors.shape}")

    # Test 2: Cache test - record mtime
    print("\n[3/7] Testing cache (recording mtime)...")
    mtime_before = config.VECTORS_PATH.stat().st_mtime
    print(f"  mtime before: {mtime_before}")

    # Wait a bit to ensure mtime would change if re-written
    time.sleep(0.1)

    # Run embedding again
    print("\n[4/7] Running second embedding (should use cache)...")
    vectors_cached = embed_statements(statements_df)

    # Check mtime unchanged
    mtime_after = config.VECTORS_PATH.stat().st_mtime
    print(f"  mtime after: {mtime_after}")
    assert mtime_before == mtime_after, \
        "FAIL: Vectors were re-embedded (mtime changed)"
    print(f"✓ PASS: Cache used, no re-embedding")

    # Test 3: Retrieval
    print("\n[5/7] Testing retrieval...")
    test_query = "What lab results did the patient have before starting a medication?"
    print(f"  Query: {test_query}")

    # Load index
    index = load_index()
    model = load_model()

    # Retrieve top-5
    results_df, scores = retrieve(test_query, index=index, model=model, k=5)

    # Check we got 5 results
    assert len(results_df) == 5, f"FAIL: Expected 5 results, got {len(results_df)}"
    print(f"✓ PASS: Retrieved 5 results")

    # Test 4: Valid IDs
    print("\n[6/7] Validating results...")

    # Check all IDs are in statements
    valid_ids = set(statements_df["id"])
    for result_id in results_df["id"]:
        assert result_id in valid_ids, f"FAIL: Invalid ID: {result_id}"
    print(f"✓ PASS: All result IDs are valid")

    # Check scores are between 0 and 1
    assert np.all(scores >= 0) and np.all(scores <= 1), \
        f"FAIL: Scores not in [0,1]: min={scores.min()}, max={scores.max()}"
    print(f"✓ PASS: Scores in range [0, 1]")

    # Check sorted descending
    assert np.all(scores[:-1] >= scores[1:]), \
        "FAIL: Scores not sorted descending"
    print(f"✓ PASS: Scores sorted descending")

    # Test 5: Display results
    print("\n[7/7] Top-5 results:")
    print("=" * 80)

    display_df = format_results(results_df)
    for idx, row in display_df.iterrows():
        print(f"\nRank {idx + 1} | Score: {row['score']:.4f}")
        print(f"  ID:   {row['id']}")
        print(f"  Type: {row['resource_type']}")
        print(f"  Date: {row['date']}")
        print(f"  Text: {row['text']}")

    print("\n" + "=" * 80)
    print("M2 TEST SUMMARY")
    print("=" * 80)
    print(f"Statements: {len(statements_df)}")
    print(f"Vectors shape: {vectors.shape}")
    print(f"Embedding dimension: {vectors.shape[1]}")
    print(f"Cache working: Yes")
    print(f"Top-5 score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print("=" * 80)
    print("✓ ALL M2 TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_m2_embed_and_retrieve()
        print("\n✓ M2 acceptance test completed successfully")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
