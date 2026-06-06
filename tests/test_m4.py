"""M4 acceptance tests: unit tests for signals.py and integration test for rerank.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals import (
    semantic,
    temporal_proximity,
    type_prior,
    code_overlap,
    reference_coherence,
    specificity,
)
from src.rerank import rerank

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def _check(label: str, condition: bool) -> bool:
    print(f"  {PASS if condition else FAIL}: {label}")
    return condition


def test_temporal_proximity():
    print("\n[temporal_proximity]")
    ok = True

    # Hard cutoff: candidate after anchor with direction="before" must return 0.0
    ok &= _check(
        "before-cutoff: c > anchor => 0.0",
        temporal_proximity("2020-01-01", "before", "2020-06-01") == 0.0,
    )
    # Hard cutoff: candidate before anchor with direction="after" must return 0.0
    ok &= _check(
        "after-cutoff: c < anchor => 0.0",
        temporal_proximity("2020-06-01", "after", "2020-01-01") == 0.0,
    )
    # Same day => exp(0) = 1.0
    ok &= _check(
        "same-day => 1.0",
        abs(temporal_proximity("2020-03-15", "before", "2020-03-15") - 1.0) < 1e-9,
    )
    # Missing date => neutral 0.5
    ok &= _check(
        "missing c_date => 0.5",
        temporal_proximity("2020-01-01", "before", None) == 0.5,
    )
    ok &= _check(
        "missing anchor => 0.5",
        temporal_proximity(None, "after", "2020-01-01") == 0.5,
    )
    # direction=None: no cutoff, still decays
    score_none = temporal_proximity("2020-06-01", None, "2020-01-01")
    ok &= _check(
        "direction=None: no cutoff, score > 0",
        0.0 < score_none < 1.0,
    )
    # direction="before" in-window should be > 0
    score_in = temporal_proximity("2020-06-01", "before", "2020-03-01")
    ok &= _check(
        "before in-window (91 days) => 0 < score < 1",
        0.0 < score_in < 1.0,
    )
    return ok


def test_type_prior():
    print("\n[type_prior]")
    ok = True
    ok &= _check("lab + Observation => 1.0", type_prior("Observation", "lab") == 1.0)
    ok &= _check("lab + Condition => 0.3", type_prior("Condition", "lab") == 0.3)
    ok &= _check("diagnostic + DiagnosticReport => 1.0", type_prior("DiagnosticReport", "diagnostic") == 1.0)
    ok &= _check("treatment + MedicationRequest => 1.0", type_prior("MedicationRequest", "treatment") == 1.0)
    ok &= _check("treatment + Observation => 0.3", type_prior("Observation", "treatment") == 0.3)
    ok &= _check("None intent => 0.5", type_prior("Observation", None) == 0.5)
    ok &= _check("unknown intent => 0.5", type_prior("Observation", "unknown") == 0.5)
    return ok


def test_reference_coherence():
    print("\n[reference_coherence]")
    ok = True
    refgraph = {
        "by_encounter": {
            "Encounter/enc1": ["Observation/obs1", "Condition/cond1", "Procedure/proc1"],
            "Encounter/enc2": ["Observation/obs2"],
        },
        "adjacency": {
            "Condition/cond1": ["MedicationRequest/med1"],
            "MedicationRequest/med1": ["Condition/cond1"],
        },
    }

    ok &= _check(
        "same encounter => 1.0",
        reference_coherence("Observation/obs1", "Condition/cond1", refgraph) == 1.0,
    )
    ok &= _check(
        "adjacent => 0.6",
        reference_coherence("MedicationRequest/med1", "Condition/cond1", refgraph) == 0.6,
    )
    ok &= _check(
        "adjacent (reverse edge) => 0.6",
        reference_coherence("Condition/cond1", "MedicationRequest/med1", refgraph) == 0.6,
    )
    ok &= _check(
        "unrelated => 0.0",
        reference_coherence("Observation/obs2", "Condition/cond1", refgraph) == 0.0,
    )
    ok &= _check(
        "no anchor => 0.0",
        reference_coherence("Observation/obs1", None, refgraph) == 0.0,
    )
    return ok


def test_code_overlap():
    print("\n[code_overlap]")
    ok = True
    codes_a = [{"system": "http://snomed.info/sct", "code": "44054006", "display": "T2DM"}]
    codes_b = [{"system": "http://snomed.info/sct", "code": "44054006", "display": "Type 2 DM"}]
    codes_c = [{"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine"}]

    ok &= _check("identical codes => 1.0", code_overlap(codes_a, codes_b) == 1.0)
    ok &= _check("no overlap => 0.0", code_overlap(codes_a, codes_c) == 0.0)
    ok &= _check("empty both => 0.0", code_overlap([], []) == 0.0)
    ok &= _check("partial: 1 shared of 2 unique => 0.5",
                 abs(code_overlap(codes_a, codes_a + codes_c) - 0.5) < 1e-9)
    return ok


def test_specificity():
    print("\n[specificity]")
    ok = True
    ok &= _check("0 codes => 0.3", specificity([], "Observation") == 0.3)
    ok &= _check("1 code => 0.7", specificity([{"code": "x"}], "Condition") == 0.7)
    ok &= _check("2 codes => 1.0", specificity([{"code": "x"}, {"code": "y"}], "Procedure") == 1.0)
    ok &= _check("3 codes => 1.0", specificity([{}] * 3, "DiagnosticReport") == 1.0)
    return ok


def test_semantic():
    print("\n[semantic]")
    ok = True
    rng = np.random.default_rng(0)
    v = rng.random(384).astype(np.float32)
    ok &= _check("identical vectors => 1.0", abs(semantic(v, v) - 1.0) < 1e-6)
    ok &= _check("zero vector => 0.0", semantic(np.zeros(384), v) == 0.0)
    ok &= _check("orthogonal => ~0.0",
                 abs(semantic(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-9)
    return ok


def test_rerank_integration():
    print("\n[rerank integration]")
    ok = True

    anchor_date = "2020-06-01"
    anchor_enc = "Encounter/enc-gold"
    anchor_id = "Condition/cond-gold"
    snomed = "http://snomed.info/sct"
    gold_code = {"system": snomed, "code": "44054006", "display": "T2DM"}

    refgraph = {
        "by_encounter": {
            anchor_enc: [anchor_id, "Observation/obs-gold"],
        },
        "adjacency": {
            anchor_id: ["MedicationRequest/med-gold"],
            "MedicationRequest/med-gold": [anchor_id],
        },
    }

    query = {
        "query_text": "What treatments followed the T2DM diagnosis?",
        "meta": {
            "anchor_date": anchor_date,
            "direction": "after",
            "intent": "treatment",
            "anchor_id": anchor_id,
        },
        "codes": [gold_code],
    }

    # Gold candidate: correct resource type, in-window date, matching code, adjacent to anchor
    gold = {
        "id": "MedicationRequest/med-gold",
        "resource_type": "MedicationRequest",
        "date": "2020-07-15",           # 44 days after anchor, direction=after => ok
        "codes": [gold_code],
        "encounter_id": None,
        "references": [anchor_id],
        "text": "2020-07-15 | medication started | Metformin 500mg",
    }
    # Decoys
    decoys = [
        {
            "id": "Observation/obs-wrong-type",
            "resource_type": "Observation",
            "date": "2020-07-15",
            "codes": [],
            "encounter_id": None,
            "references": [],
            "text": "2020-07-15 | lab | Glucose",
        },
        {
            "id": "MedicationRequest/med-wrong-dir",
            "resource_type": "MedicationRequest",
            "date": "2019-01-01",   # before anchor with direction=after => hard cutoff
            "codes": [],
            "encounter_id": None,
            "references": [],
            "text": "2019-01-01 | medication started | Aspirin",
        },
        {
            "id": "Procedure/proc-far",
            "resource_type": "Procedure",
            "date": "2023-01-01",   # after anchor but very far (925 days)
            "codes": [],
            "encounter_id": None,
            "references": [],
            "text": "2023-01-01 | procedure | Surgery",
        },
        {
            "id": "Condition/cond-unrelated",
            "resource_type": "Condition",
            "date": "2020-07-01",
            "codes": [],
            "encounter_id": None,
            "references": [],
            "text": "2020-07-01 | diagnosis | Hypertension",
        },
    ]
    candidates = [gold] + decoys

    rng = np.random.default_rng(42)
    dim = 16
    q_vec = rng.random(dim).astype(np.float32)
    c_vecs = rng.random((len(candidates), dim)).astype(np.float32)
    # Make gold's vector very similar to query
    c_vecs[0] = q_vec + rng.random(dim).astype(np.float32) * 0.01

    results = rerank(query, candidates, q_vec, c_vecs, refgraph)

    ok &= _check("gold candidate ranks #1", results[0]["id"] == "MedicationRequest/med-gold")
    ok &= _check("wrong-direction candidate has temporal=0.0",
                 next(r for r in results if r["id"] == "MedicationRequest/med-wrong-dir")
                 ["breakdown"]["temporal"] == 0.0)
    breakdown_keys = {"semantic", "temporal", "type_prior", "code_overlap", "reference", "specificity"}
    ok &= _check("all results have breakdown keys",
                 all(set(r["breakdown"].keys()) == breakdown_keys for r in results))
    ok &= _check("results sorted descending",
                 all(results[i]["final_score"] >= results[i + 1]["final_score"]
                     for i in range(len(results) - 1)))
    return ok


def main():
    print("=" * 60)
    print("M4 ACCEPTANCE TEST: SIGNALS + RERANK")
    print("=" * 60)

    results = {
        "temporal_proximity": test_temporal_proximity(),
        "type_prior": test_type_prior(),
        "reference_coherence": test_reference_coherence(),
        "code_overlap": test_code_overlap(),
        "specificity": test_specificity(),
        "semantic": test_semantic(),
        "rerank_integration": test_rerank_integration(),
    }

    print("\n" + "=" * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print("=" * 60)
    if all_pass:
        print("✓ ALL M4 TESTS PASSED")
    else:
        print("✗ SOME M4 TESTS FAILED")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
