"""M3 acceptance tests for eval query generation."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.eval_gen import (
    load_data,
    generate_all_queries,
    split_queries,
    save_queries,
    has_ngram_overlap,
)


def load_queries(path: Path) -> list[dict]:
    """Load queries from JSONL file."""
    queries = []
    with open(path, "r") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def test_m3_eval_generation():
    """
    M3 Acceptance Test
    - Total queries >= 30
    - Every query has >= 3 grade-2 entries
    - Every query has >= 10 grade-0 entries
    - No 4-gram overlap between query_text and gold resource texts
    - tune + test files exist and counts sum to total
    """
    print("\n" + "=" * 80)
    print("M3 ACCEPTANCE TEST: EVAL QUERY GENERATION")
    print("=" * 80)

    # Generate queries
    print("\n[1/6] Loading data and generating queries...")
    statements_df, refgraph = load_data()

    all_queries = generate_all_queries(statements_df, refgraph)
    print(f"✓ Generated {len(all_queries)} queries")

    # Test 1: Total queries >= 30
    print("\n[2/6] Testing total query count >= 30...")
    assert len(all_queries) >= 30, f"FAIL: Only {len(all_queries)} queries generated, need >= 30"
    print(f"✓ PASS: {len(all_queries)} queries generated")

    # Test 2 & 3: Check each query has >= 3 gold(2) and >= 10 distractors
    print("\n[3/6] Validating query relevance grades...")
    for query in all_queries:
        query_id = query["query_id"]
        relevance = query["relevance"]

        # Count by grade
        grade_counts = defaultdict(int)
        for resource_id, grade in relevance.items():
            grade_counts[grade] += 1

        # Check grade-2 (gold)
        assert grade_counts[2] >= 3, \
            f"FAIL: Query {query_id} has only {grade_counts[2]} grade-2 entries, need >= 3"

        # Check grade-0 (distractors)
        assert grade_counts[0] >= 10, \
            f"FAIL: Query {query_id} has only {grade_counts[0]} grade-0 entries, need >= 10"

    print(f"✓ PASS: All queries have >= 3 gold(2) and >= 10 distractors")

    # Test 4: No 4-gram overlap
    print("\n[4/6] Checking for 4-gram overlap...")
    overlap_count = 0

    for query in all_queries:
        query_text = query["query_text"]
        relevance = query["relevance"]

        # Get gold resources (grade 2)
        gold_ids = [rid for rid, grade in relevance.items() if grade == 2]

        # Get their texts
        gold_texts = statements_df[statements_df["id"].isin(gold_ids)]["text"].tolist()

        # Check overlap
        if has_ngram_overlap(query_text, gold_texts, n=4):
            overlap_count += 1
            print(f"  WARNING: Query {query['query_id']} has 4-gram overlap")

    assert overlap_count == 0, f"FAIL: {overlap_count} queries have 4-gram overlap"
    print(f"✓ PASS: No 4-gram overlap found")

    # Test 5: Split and save
    print("\n[5/6] Splitting and saving queries...")
    tune_queries, test_queries = split_queries(all_queries)

    all_path = config.ARTIFACTS_DIR / "eval_queries.jsonl"
    tune_path = config.ARTIFACTS_DIR / "eval_queries_tune.jsonl"
    test_path = config.ARTIFACTS_DIR / "eval_queries_test.jsonl"

    save_queries(all_queries, all_path)
    save_queries(tune_queries, tune_path)
    save_queries(test_queries, test_path)

    print(f"  Saved {len(all_queries)} queries to {all_path.name}")
    print(f"  Saved {len(tune_queries)} queries to {tune_path.name}")
    print(f"  Saved {len(test_queries)} queries to {test_path.name}")

    # Test 6: Verify files exist and counts sum
    print("\n[6/6] Verifying saved files...")
    assert all_path.exists(), f"FAIL: {all_path} does not exist"
    assert tune_path.exists(), f"FAIL: {tune_path} does not exist"
    assert test_path.exists(), f"FAIL: {test_path} does not exist"

    # Load and verify counts
    loaded_all = load_queries(all_path)
    loaded_tune = load_queries(tune_path)
    loaded_test = load_queries(test_path)

    assert len(loaded_all) == len(all_queries), \
        f"FAIL: Loaded {len(loaded_all)} queries from all file, expected {len(all_queries)}"
    assert len(loaded_tune) == len(tune_queries), \
        f"FAIL: Loaded {len(loaded_tune)} queries from tune file, expected {len(tune_queries)}"
    assert len(loaded_test) == len(test_queries), \
        f"FAIL: Loaded {len(loaded_test)} queries from test file, expected {len(test_queries)}"

    assert len(loaded_tune) + len(loaded_test) == len(loaded_all), \
        f"FAIL: Tune + test counts don't sum to total"

    print(f"✓ PASS: All files saved and counts verified")

    # Summary by template
    print("\n" + "=" * 80)
    print("M3 TEST SUMMARY")
    print("=" * 80)
    print(f"Total queries: {len(all_queries)}")
    print(f"Tune queries: {len(tune_queries)} (70%)")
    print(f"Test queries: {len(test_queries)} (30%)")
    print()
    print("Query counts by template:")

    template_counts = defaultdict(int)
    for query in all_queries:
        template = query["template"]
        template_counts[template] += 1

    for template in sorted(template_counts.keys()):
        template_name = {
            1: "Temporal window",
            2: "Encounter coherence",
            4: "Treatment after",
        }.get(template, f"Template {template}")
        print(f"  Template {template} ({template_name}): {template_counts[template]}")

    print("=" * 80)
    print("✓ ALL M3 TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_m3_eval_generation()
        print("\n✓ M3 acceptance test completed successfully")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
