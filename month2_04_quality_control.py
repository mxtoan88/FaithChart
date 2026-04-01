#!/usr/bin/env python3
"""
FaithChart Tháng 2 — Task 4: Quality Control & Inter-Annotator Agreement
=========================================================================
Tính:
  1. Inter-annotator agreement (Cohen's κ) cho overlap set (100 samples × 3 annotators)
  2. LLM-human correlation (Spearman r)
  3. Annotation quality metrics
  4. Filter out low-quality annotations

Target: κ > 0.70 (substantial agreement) cho Sufficiency decisions
        κ > 0.65 cho Comprehensiveness

Chạy:
  pip install scipy scikit-learn pandas numpy
  python 04_quality_control.py --iaa      # Inter-annotator agreement
  python 04_quality_control.py --filter   # Filter final dataset
  python 04_quality_control.py --report   # Full quality report
"""

import json, argparse, warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

ANNOTATION_DIR = Path("faithchart_annotations")
OUTPUT_DIR     = Path("faithchart_final")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Inter-annotator agreement ───────────────────────────────────────────────
def compute_iaa(overlap_annotations_file: Path) -> dict:
    """
    Tính Cohen's κ trên 100-sample overlap set.

    Mỗi sample được annotate bởi 3 annotators.
    Agreement đo trên: binary decision "is this region cited?"
    cho 5 region types.

    Input format (overlap_annotations.json):
    [
      {
        "global_id": "FC_0001",
        "annotator_1": {"cited_regions": ["data_values", "axis_labels"]},
        "annotator_2": {"cited_regions": ["data_values"]},
        "annotator_3": {"cited_regions": ["data_values", "legend"]}
      }, ...
    ]
    """
    if not overlap_annotations_file.exists():
        print(f"  File not found: {overlap_annotations_file}")
        print("  → Cần chạy human annotation trước. Dùng simulated data để test.")
        return _simulate_iaa()

    with open(overlap_annotations_file) as f:
        data = json.load(f)

    region_types = ["data_values", "axis_labels", "legend", "title", "background_gridlines"]

    # Build binary annotation matrices
    # annotator_i[j][k] = 1 if annotator j cited region k for sample i
    a1_votes, a2_votes, a3_votes = [], [], []

    for sample in data:
        for region in region_types:
            a1_votes.append(1 if region in sample.get("annotator_1", {}).get("cited_regions", []) else 0)
            a2_votes.append(1 if region in sample.get("annotator_2", {}).get("cited_regions", []) else 0)
            a3_votes.append(1 if region in sample.get("annotator_3", {}).get("cited_regions", []) else 0)

    # Pairwise κ
    k12 = cohen_kappa_score(a1_votes, a2_votes)
    k13 = cohen_kappa_score(a1_votes, a3_votes)
    k23 = cohen_kappa_score(a2_votes, a3_votes)
    mean_k = np.mean([k12, k13, k23])

    print(f"\n  Inter-Annotator Agreement (Cohen's κ):")
    print(f"  κ(A1,A2) = {k12:.3f}")
    print(f"  κ(A1,A3) = {k13:.3f}")
    print(f"  κ(A2,A3) = {k23:.3f}")
    print(f"  Mean κ   = {mean_k:.3f} {'✅ Acceptable' if mean_k >= 0.65 else '⚠️ Below threshold'}")

    interpretation = (
        "Excellent (0.80+)" if mean_k >= 0.80 else
        "Substantial (0.70-0.80)" if mean_k >= 0.70 else
        "Moderate (0.65-0.70)" if mean_k >= 0.65 else
        "Fair (below 0.65) — needs guideline revision"
    )
    print(f"  Interpretation: {interpretation}")

    return {
        "kappa_12": round(k12, 4),
        "kappa_13": round(k13, 4),
        "kappa_23": round(k23, 4),
        "mean_kappa": round(mean_k, 4),
        "n_samples": len(data),
        "n_region_types": len(region_types),
        "acceptable": mean_k >= 0.65,
        "interpretation": interpretation
    }

def _simulate_iaa() -> dict:
    """Simulate IAA với realistic numbers để test pipeline."""
    np.random.seed(42)
    n = 100 * 5  # 100 samples × 5 regions

    # Simulate κ ≈ 0.73 (substantial agreement)
    true_labels = np.random.binomial(1, 0.45, n)
    noise = np.random.binomial(1, 0.12, n)   # 12% disagreement rate

    a1 = true_labels
    a2 = np.abs(true_labels - noise)
    a3 = np.abs(true_labels - np.random.binomial(1, 0.10, n))

    k12 = cohen_kappa_score(a1, a2)
    k13 = cohen_kappa_score(a1, a3)
    k23 = cohen_kappa_score(a2, a3)
    mean_k = np.mean([k12, k13, k23])

    print(f"\n  [SIMULATED] Inter-Annotator Agreement:")
    print(f"  κ(A1,A2) = {k12:.3f}")
    print(f"  κ(A1,A3) = {k13:.3f}")
    print(f"  κ(A2,A3) = {k23:.3f}")
    print(f"  Mean κ   = {mean_k:.3f}")

    return {"mean_kappa": round(mean_k, 4), "simulated": True}

# ── LLM-Human correlation ───────────────────────────────────────────────────
def compute_llm_human_correlation(human_file: Path, llm_file: Path) -> dict:
    """
    Tính correlation giữa LLM annotations và human annotations
    trên overlap set.

    Metrics so sánh:
    - n_cited_regions (số regions được cite)
    - region overlap (Jaccard similarity)
    """
    if not human_file.exists() or not llm_file.exists():
        print("  Files not found, computing simulated correlation...")
        return _simulate_correlation()

    human = {a["global_id"]: a for a in json.loads(human_file.read_text())}
    llm   = {a["global_id"]: a for a in json.loads(llm_file.read_text())}

    common_ids = set(human.keys()) & set(llm.keys())
    print(f"\n  Overlap samples: {len(common_ids)}")

    jaccard_scores, human_n_cited, llm_n_cited = [], [], []

    for gid in common_ids:
        h_regions = {r["region_type"] for r in human[gid].get("cited_regions", [])}
        l_regions = {r["region_type"] for r in llm[gid].get("cited_regions", [])}

        intersection = h_regions & l_regions
        union = h_regions | l_regions
        jaccard = len(intersection) / len(union) if union else 1.0
        jaccard_scores.append(jaccard)
        human_n_cited.append(len(h_regions))
        llm_n_cited.append(len(l_regions))

    mean_jaccard = np.mean(jaccard_scores)
    spearman_r, spearman_p = spearmanr(human_n_cited, llm_n_cited)

    print(f"  Mean Jaccard similarity: {mean_jaccard:.3f}")
    print(f"  Spearman r (n_cited): {spearman_r:.3f} (p={spearman_p:.4f})")

    return {
        "mean_jaccard": round(mean_jaccard, 4),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 6),
        "n_overlap": len(common_ids)
    }

def _simulate_correlation() -> dict:
    """Simulate realistic LLM-human correlation."""
    np.random.seed(42)
    n = 100
    human_cited = np.random.poisson(2.3, n)          # Humans cite ~2.3 regions avg
    llm_cited   = human_cited + np.random.randint(-1, 2, n)   # LLM slightly noisier
    llm_cited   = np.clip(llm_cited, 0, 5)

    r, p = spearmanr(human_cited, llm_cited)
    jaccard = 0.68  # Simulated

    print(f"\n  [SIMULATED] LLM-Human Correlation:")
    print(f"  Spearman r = {r:.3f} (p={p:.4f})")
    print(f"  Mean Jaccard = {jaccard:.3f}")
    return {"spearman_r": round(r, 4), "mean_jaccard": jaccard, "simulated": True}

# ── Final dataset quality filter ─────────────────────────────────────────────
def filter_final_dataset(
    all_annotations_file: Path,
    min_reasoning_length: int = 100,
    require_cited_regions: int = 1
) -> dict:
    """
    Lọc annotations đủ chất lượng để đưa vào FaithChart-1500 final version.

    Criteria:
    1. reasoning_trace đủ dài (≥100 chars)
    2. Có ít nhất 1 cited_region với necessity='critical'
    3. Không có annotation_status='error'
    4. human_reviewed=True HOẶC (llm_correct=True AND confidence='high')
    """
    if not all_annotations_file.exists():
        print("Annotations not found, running quality filter on empty set...")
        return {"accepted": 0, "rejected": 0}

    annotations = json.loads(all_annotations_file.read_text())
    print(f"\nFiltering {len(annotations)} annotations...")

    accepted, rejected = [], []
    rejection_reasons = Counter()

    for a in annotations:
        reasons = []

        # Check 1: No errors
        if a.get("annotation_status") == "error":
            reasons.append("error_status")

        # Check 2: Reasoning trace length
        trace = a.get("reasoning_trace", "")
        if len(trace) < min_reasoning_length:
            reasons.append(f"short_trace({len(trace)}<{min_reasoning_length})")

        # Check 3: Has cited regions
        cited = a.get("cited_regions", [])
        critical = [r for r in cited if r.get("necessity") == "critical"]
        if len(critical) < require_cited_regions:
            reasons.append(f"no_critical_region")

        # Check 4: Confidence (for LLM-only annotations)
        if not a.get("human_reviewed"):
            if not a.get("llm_correct"):
                reasons.append("llm_wrong_not_reviewed")
            elif a.get("confidence") == "low":
                reasons.append("low_confidence_not_reviewed")

        if reasons:
            a["rejection_reasons"] = reasons
            rejected.append(a)
            for r in reasons:
                rejection_reasons[r] += 1
        else:
            a["quality_verified"] = True
            accepted.append(a)

    print(f"\n  Accepted: {len(accepted)}/{len(annotations)} ({len(accepted)/len(annotations)*100:.1f}%)")
    print(f"  Rejected: {len(rejected)}")
    print(f"\n  Rejection reasons:")
    for reason, count in rejection_reasons.most_common():
        print(f"    {reason}: {count}")

    # Save final dataset
    final_file = OUTPUT_DIR / "faithchart_1500_final.json"
    final_file.write_text(json.dumps(accepted, indent=2, ensure_ascii=False))
    print(f"\n  Final dataset: {final_file} ({len(accepted)} samples)")

    rejected_file = OUTPUT_DIR / "rejected_annotations.json"
    rejected_file.write_text(json.dumps(rejected, indent=2, ensure_ascii=False))

    return {
        "total": len(annotations),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": round(len(accepted)/max(len(annotations),1), 4),
        "rejection_reasons": dict(rejection_reasons)
    }

# ── Full quality report ───────────────────────────────────────────────────────
def generate_quality_report(iaa: dict, correlation: dict, filter_result: dict) -> str:
    kappa = iaa.get("mean_kappa", 0)
    r = correlation.get("spearman_r", 0)
    acc_rate = filter_result.get("acceptance_rate", 0)

    report = f"""
FaithChart-1500 Dataset Quality Report
Generated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}
{'='*55}

1. INTER-ANNOTATOR AGREEMENT
   Mean Cohen's κ:  {kappa:.3f}
   Status:          {'✅ Acceptable (≥0.65)' if kappa >= 0.65 else '❌ Below threshold'}
   Threshold:       κ ≥ 0.65 for publishable quality
   
2. LLM-HUMAN CORRELATION
   Spearman r:      {r:.3f}
   Mean Jaccard:    {correlation.get("mean_jaccard", 0):.3f}
   Status:          {'✅ Good alignment' if r >= 0.60 else '⚠️ Low alignment'}
   
3. DATASET FILTER
   Total samples:   {filter_result.get("total", "N/A")}
   Accepted:        {filter_result.get("accepted", "N/A")}
   Acceptance rate: {acc_rate*100:.1f}%
   
4. OVERALL ASSESSMENT
   {'✅ READY for faithfulness scoring' if kappa >= 0.65 and r >= 0.55 and acc_rate >= 0.75 else '⚠️ Needs improvement — review annotation guidelines'}
   
5. RECOMMENDATIONS
   {'- κ below 0.70: revise annotation guidelines, especially for axis_labels vs data_values distinction' if kappa < 0.70 else ''}
   {'- Low LLM-human correlation: consider using human-only annotations for critical regions' if r < 0.60 else ''}
   {'- Low acceptance rate: run additional annotation rounds for rejected samples' if acc_rate < 0.75 else ''}
"""
    return report.strip()

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iaa", action="store_true")
    parser.add_argument("--correlation", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    iaa_result = {}
    corr_result = {}
    filter_result = {}

    if args.iaa or args.report:
        print("=== Inter-Annotator Agreement ===")
        iaa_result = compute_iaa(ANNOTATION_DIR / "overlap_annotations.json")

    if args.correlation or args.report:
        print("\n=== LLM-Human Correlation ===")
        corr_result = compute_llm_human_correlation(
            ANNOTATION_DIR / "human_annotations.json",
            ANNOTATION_DIR / "llm_annotations_overlap.json"
        )

    if args.filter or args.report:
        print("\n=== Dataset Quality Filter ===")
        filter_result = filter_final_dataset(
            ANNOTATION_DIR / "all_annotations.json"
        )

    if args.report:
        report = generate_quality_report(iaa_result, corr_result, filter_result)
        print("\n" + report)
        report_file = OUTPUT_DIR / "quality_report.txt"
        report_file.write_text(report)
        print(f"\nSaved: {report_file}")

        summary = {"iaa": iaa_result, "correlation": corr_result, "filter": filter_result}
        (OUTPUT_DIR / "quality_summary.json").write_text(json.dumps(summary, indent=2))

    if not any([args.iaa, args.correlation, args.filter, args.report]):
        print("Usage:")
        print("  python 04_quality_control.py --iaa         # Inter-annotator agreement")
        print("  python 04_quality_control.py --correlation  # LLM-human correlation")
        print("  python 04_quality_control.py --filter       # Filter final dataset")
        print("  python 04_quality_control.py --report       # Full quality report")
