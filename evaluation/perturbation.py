#!/usr/bin/env python3
"""
Perturbation Pipeline
====================================================
Create perturbed versions of chart images for finding Sufficiency & Comprehensiveness.

5 types of perturbation:
  1. data_values   — blur/mask data points, bars, lines
  2. axis_labels   — mask axis tick labels (giữ trục)
  3. legend        — mask legend entries
  4. title         — mask chart title
  5. background_gridlines — remove gridlines

For each cited_region in the annotation:
- Create a variant with that region DELETED → used for Sufficiency
- Create a variant with that region KEEPED, delete all uncited → used for Comprehensiveness

Run:
  pip install pillow opencv-python numpy scikit-image
  python perturbation.py --sample FC_0001  # Test 1 sample
  python perturbation.py --all              # All dataset
"""

import json, base64, argparse, re
from pathlib import Path
from io import BytesIO
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

DATASET_DIR    = Path("faithchart_dataset")
ANNOTATION_DIR = Path("faithchart_annotations")
PERTURB_DIR    = Path("faithchart_perturbations")
PERTURB_DIR.mkdir(exist_ok=True)

RegionType = Literal["data_values", "axis_labels", "legend", "title", "background_gridlines"]

# ── Image utilities ─────────────────────────────────────────────────────────
def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def detect_region_bbox(img: Image.Image, region_type: RegionType) -> tuple | None:
    """
    Heuristic bounding box detection cho chart regions.
    Với paper, dùng kết hợp:
    - Rule-based (common locations of each region)
    - It can be replaced with a ViT-based detector

    Returns: (x1, y1, x2, y2) hoặc None
    """
    W, H = img.size

    if region_type == "title":
        # Title often at top 10-15%
        return (0, 0, W, int(H * 0.12))

    elif region_type == "axis_labels":
        # X-axis labels: bottom 15%, Y-axis labels: left 15%
        # Return all margins
        return None  # will handle 

    elif region_type == "legend":
        # Legend often at top-right or bottom-right
        # Heuristic: right 30%, top 60%
        return (int(W * 0.70), 0, W, int(H * 0.60))

    elif region_type == "background_gridlines":
        # Gridlines at data area (exclude margins)
        return (int(W * 0.12), int(H * 0.10), int(W * 0.95), int(H * 0.88))

    elif region_type == "data_values":
        # Data area = at chart (exclude title, axes, legend)
        return (int(W * 0.12), int(H * 0.10), int(W * 0.88), int(H * 0.88))

    return None

def apply_perturbation(
    img: Image.Image,
    region_type: RegionType,
    method: Literal["blur", "blackout", "mean_fill"] = "blur",
    bbox: tuple | None = None
) -> Image.Image:
    """
    Apply perturbation to a specific region of the chart.

    Methods:
      - blur: Gaussian blur sigma=20 (drastic, unreadable)
      - blackout: Fill with black (most aggressive)
      - mean_fill: Fill with mean color of region (subtle)
    """
    img = img.copy()
    W, H = img.size

    if bbox is None:
        bbox = detect_region_bbox(img, region_type)

    if bbox is None:
        # Fallback: handle axis_labels specially (multi-region)
        if region_type == "axis_labels":
            # X-axis: bottom strip
            img = _perturb_box(img, (int(W*0.10), int(H*0.85), W, H), method)
            # Y-axis: left strip
            img = _perturb_box(img, (0, 0, int(W*0.12), int(H*0.88)), method)
            return img
        return img  # No perturbation if region not found

    return _perturb_box(img, bbox, method)

def _perturb_box(img: Image.Image, bbox: tuple,
                 method: str = "blur") -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)

    if x2 <= x1 or y2 <= y1:
        return img

    if method == "blackout":
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

    elif method == "blur":
        region = img.crop((x1, y1, x2, y2))
        # Strong blur — sigma equivalent via multiple passes
        for _ in range(8):
            region = region.filter(ImageFilter.GaussianBlur(radius=10))
        img.paste(region, (x1, y1))

    elif method == "mean_fill":
        region = img.crop((x1, y1, x2, y2))
        arr = np.array(region)
        mean_color = tuple(arr.mean(axis=(0, 1)).astype(int))
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], fill=mean_color)

    elif method == "whiteout":
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))

    return img

# ── Generate perturbation variants ─────────────────────────────────────────
ALL_REGION_TYPES: list[RegionType] = [
    "data_values", "axis_labels", "legend", "title", "background_gridlines"
]

def generate_perturbations(record: dict, annotation: dict) -> dict:
    """
    With 1 sample, create all perturbation variants:

    1. For Sufficiency: delete each cited region → model answer change
    2. For Comprehensiveness: delete each uncited region → answer not change

    Returns dict: {
      "sufficiency_variants": [{"removed_region": ..., "image_b64": ...}],
      "comprehensiveness_variants": [{"removed_region": ..., "image_b64": ...}]
    }
    """
    img_b64 = record.get("image_b64", "")
    if not img_b64:
        return {}

    img = b64_to_pil(img_b64)

    # Locate cited and uncited regions from annotation
    cited_regions = annotation.get("cited_regions", [])
    cited_types = {r["region_type"] for r in cited_regions
                   if r.get("necessity") in ["critical", "supporting"]}
    uncited_types = set(ALL_REGION_TYPES) - cited_types

    perturbations = {
        "global_id": record["global_id"],
        "question": record["question"],
        "gold_answer": record["gold_answer"],
        "cited_region_types": list(cited_types),
        "uncited_region_types": list(uncited_types),
        "sufficiency_variants": [],
        "comprehensiveness_variants": [],
        "perturbation_method": "blur",
        "n_perturb_per_metric": 1  # 1 per region type
    }

    # ── Sufficiency variants: remove each CITED region ─────────────────
    for region_type in cited_types:
        perturbed = apply_perturbation(img, region_type, method="blur")
        perturbations["sufficiency_variants"].append({
            "removed_region": region_type,
            "variant_type": "sufficiency",
            "hypothesis": "answer_should_change",
            "image_b64": pil_to_b64(perturbed)
        })

    # ── Comprehensiveness variants: remove each UNCITED region ─────────
    for region_type in uncited_types:
        perturbed = apply_perturbation(img, region_type, method="blur")
        perturbations["comprehensiveness_variants"].append({
            "removed_region": region_type,
            "variant_type": "comprehensiveness",
            "hypothesis": "answer_should_stay_same",
            "image_b64": pil_to_b64(perturbed)
        })

    return perturbations

# ── Faithfulness scoring (needs model inference) ───────────────────────────
def score_faithfulness(
    perturb_data: dict,
    model_call_fn,  # callable(image_b64, question) -> answer str
    tol: float = 0.05
) -> dict:
    """
    Tính Sufficiency và Comprehensiveness scores.

    Args:
        perturb_data: output từ generate_perturbations()
        model_call_fn: function gọi model inference
        tol: tolerance cho numeric answers

    Returns: {"sufficiency": float, "comprehensiveness": float, "faith_score": float}
    """
    question = perturb_data["question"]
    gold = perturb_data["gold_answer"]

    def relaxed_match(pred, ref):
        p = re.sub(r"[^\w\d.\-]", "", str(pred).lower().strip())
        r = re.sub(r"[^\w\d.\-]", "", str(ref).lower().strip())
        if p == r: return True
        try:
            return abs(float(p) - float(r)) / max(abs(float(r)), 1e-9) <= tol
        except: return False

    # Get baseline answer (original chart)
    original_img = perturb_data.get("original_image_b64")
    if original_img:
        baseline_answer = model_call_fn(original_img, question)
    else:
        baseline_answer = gold  # Assume model was correct

    # ── Sufficiency ──────────────────────────────────────────────────
    # P(answer changes when cited region removed)
    suf_hits = 0
    suf_total = len(perturb_data["sufficiency_variants"])

    for variant in perturb_data["sufficiency_variants"]:
        perturbed_answer = model_call_fn(variant["image_b64"], question)
        # Answer should CHANGE (not match baseline)
        changed = not relaxed_match(perturbed_answer, baseline_answer)
        if changed:
            suf_hits += 1
        variant["perturbed_answer"] = perturbed_answer
        variant["answer_changed"] = changed

    sufficiency = suf_hits / suf_total if suf_total > 0 else 0.0

    # ── Comprehensiveness ─────────────────────────────────────────────
    # P(answer stays same when uncited region removed)
    comp_hits = 0
    comp_total = len(perturb_data["comprehensiveness_variants"])

    for variant in perturb_data["comprehensiveness_variants"]:
        perturbed_answer = model_call_fn(variant["image_b64"], question)
        # Answer should STAY SAME (match baseline)
        stable = relaxed_match(perturbed_answer, baseline_answer)
        if stable:
            comp_hits += 1
        variant["perturbed_answer"] = perturbed_answer
        variant["answer_stable"] = stable

    comprehensiveness = comp_hits / comp_total if comp_total > 0 else 0.0

    faith_score = (sufficiency + comprehensiveness) / 2

    return {
        "sufficiency": round(sufficiency, 4),
        "comprehensiveness": round(comprehensiveness, 4),
        "faith_score": round(faith_score, 4),
        "suf_hits": suf_hits, "suf_total": suf_total,
        "comp_hits": comp_hits, "comp_total": comp_total
    }

# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, help="Single sample ID to test")
    parser.add_argument("--all", action="store_true", help="Process all annotated samples")
    parser.add_argument("--method", default="blur",
                        choices=["blur", "blackout", "mean_fill", "whiteout"])
    args = parser.parse_args()

    # Load dataset + annotations
    dataset_file = DATASET_DIR / "faithchart_1500.json"
    annotation_file = ANNOTATION_DIR / "all_annotations.json"

    if not dataset_file.exists():
        print("❌ Dataset not found. Run 01_curate_dataset.py first.")
        exit(1)

    print("Loading dataset...")
    dataset = json.loads(dataset_file.read_text())
    id_to_record = {r["global_id"]: r for r in dataset}

    annotations = []
    if annotation_file.exists():
        annotations = json.loads(annotation_file.read_text())
    id_to_annotation = {a["global_id"]: a for a in annotations}

    if args.sample:
        # Test single sample
        record = id_to_record.get(args.sample)
        if not record:
            print(f"Sample {args.sample} not found")
            exit(1)

        annotation = id_to_annotation.get(args.sample, {
            "cited_regions": [
                {"region_type": "data_values", "necessity": "critical"},
                {"region_type": "axis_labels", "necessity": "supporting"}
            ]
        })

        print(f"\nGenerating perturbations for {args.sample}...")
        result = generate_perturbations(record, annotation)
        print(f"  Sufficiency variants: {len(result['sufficiency_variants'])}")
        print(f"  Comprehensiveness variants: {len(result['comprehensiveness_variants'])}")

        out = PERTURB_DIR / f"{args.sample}_perturbations.json"
        # Save without image_b64 for inspection
        result_meta = {k: v for k, v in result.items()
                       if k not in ["sufficiency_variants", "comprehensiveness_variants"]}
        result_meta["n_suf_variants"] = len(result["sufficiency_variants"])
        result_meta["n_comp_variants"] = len(result["comprehensiveness_variants"])
        out.write_text(json.dumps(result_meta, indent=2))
        print(f"  Saved metadata: {out}")
        print("\nNext: run with model inference to score faithfulness")

    elif args.all:
        print(f"Processing {len(annotations)} annotated samples...")
        all_perturbations = []

        for i, annotation in enumerate(annotations):
            gid = annotation.get("global_id")
            record = id_to_record.get(gid)
            if not record:
                continue

            result = generate_perturbations(record, annotation)
            # Store without heavy image_b64 for the metadata file
            meta = {k: v for k, v in result.items()
                    if k not in ["sufficiency_variants", "comprehensiveness_variants"]}
            meta["n_suf"] = len(result.get("sufficiency_variants", []))
            meta["n_comp"] = len(result.get("comprehensiveness_variants", []))
            all_perturbations.append(meta)

            # Save full perturbations (with images) per sample
            out = PERTURB_DIR / f"{gid}_perturbations.json"
            out.write_text(json.dumps(result, ensure_ascii=False))

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(annotations)}] done")

        # Summary
        summary_file = PERTURB_DIR / "perturbation_summary.json"
        summary_file.write_text(json.dumps(all_perturbations, indent=2, ensure_ascii=False))
        avg_suf = sum(p["n_suf"] for p in all_perturbations) / max(len(all_perturbations), 1)
        avg_comp = sum(p["n_comp"] for p in all_perturbations) / max(len(all_perturbations), 1)
        print(f"\n✅ Perturbations generated: {len(all_perturbations)} samples")
        print(f"  Avg sufficiency variants: {avg_suf:.1f}")
        print(f"  Avg comprehensiveness variants: {avg_comp:.1f}")
        print(f"\nNext step: run 04_faithfulness_scoring.py to compute Suf/Comp scores")

    else:
        print("Usage:")
        print("  python 03_perturbation.py --sample FC_0001  # Test 1 sample")
        print("  python 03_perturbation.py --all              # All samples")
