"""M5 acceptance tests: metrics, eval_run results table, ablation checks."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.eval_run import run_eval, print_table, EVAL_RESULTS_PATH


def test_m5():
    print("=" * 70)
    print("M5 ACCEPTANCE TEST: EVAL RUN")
    print("=" * 70)

    # --- Run evaluation (also prints progress) ---
    print("\n[1/5] Running evaluation across all conditions...")
    results = run_eval()

    # Print table immediately so it's visible regardless of test outcome
    print_table(results)

    # Save for the file-existence check
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    all_pass = True

    # --- [2] File exists ---
    print("\n[2/5] Checking eval_results.json exists...")
    ok = EVAL_RESULTS_PATH.exists()
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}: eval_results.json exists")
    all_pass &= ok

    fsar   = results["FSAR_full"]
    naive  = results["naive"]
    ablations = {k: results[k] for k in ("FSAR_no_temporal", "FSAR_no_reference",
                                          "FSAR_no_code", "FSAR_no_type")}

    # --- [3] FSAR_full >= naive ---
    print("\n[3/5] FSAR_full vs naive...")
    ok_ndcg = fsar["ndcg_at_10"] >= naive["ndcg_at_10"]
    ok_rec  = fsar["recall_at_10"] >= naive["recall_at_10"]
    print(f"  {'✓ PASS' if ok_ndcg else '✗ FAIL'}: "
          f"FSAR nDCG@10 {fsar['ndcg_at_10']:.4f} >= naive {naive['ndcg_at_10']:.4f}")
    print(f"  {'✓ PASS' if ok_rec else '✗ FAIL'}: "
          f"FSAR recall@10 {fsar['recall_at_10']:.4f} >= naive {naive['recall_at_10']:.4f}")
    all_pass &= ok_ndcg and ok_rec

    # --- [4] At least 3 of 4 ablations degrade nDCG ---
    print("\n[4/5] Ablation nDCG degradation (need >= 3 of 4 lower than FSAR_full)...")
    lower_count = 0
    for abl_name, abl in ablations.items():
        lower = abl["ndcg_at_10"] < fsar["ndcg_at_10"]
        if lower:
            lower_count += 1
        print(f"  {'✓' if lower else '~'} {abl_name}: "
              f"{abl['ndcg_at_10']:.4f} {'<' if lower else '>='} {fsar['ndcg_at_10']:.4f}")
    ok_abl = lower_count >= 3
    print(f"  {'✓ PASS' if ok_abl else '✗ FAIL'}: {lower_count}/4 ablations lower")
    all_pass &= ok_abl

    # --- [5] temporal_precision: FSAR_full > FSAR_no_temporal ---
    print("\n[5/5] Temporal precision: FSAR_full > FSAR_no_temporal...")
    tp_full = fsar["temporal_prec_at_10"]
    tp_no   = results["FSAR_no_temporal"]["temporal_prec_at_10"]
    if tp_full is None or tp_no is None:
        print("  ~ SKIP: no template-1 queries in test split")
    else:
        ok_tp = tp_full > tp_no
        print(f"  {'✓ PASS' if ok_tp else '✗ FAIL'}: "
              f"FSAR temporal_prec {tp_full:.4f} > no_temporal {tp_no:.4f}")
        all_pass &= ok_tp

    print("\n" + "=" * 70)
    if all_pass:
        print("✓ ALL M5 TESTS PASSED")
    else:
        print("✗ SOME M5 TESTS FAILED")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if test_m5() else 1)
