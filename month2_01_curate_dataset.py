#!/usr/bin/env python3
"""
FaithChart Tháng 2 — Task 1: Curation FaithChart-1500
=======================================================
Chọn 1,500 chart-QA samples từ:
  - 750 samples: ChartQA human test split
  - 750 samples: CharXiv validation split (reasoning questions)

Output: faithchart_1500.json với metadata đầy đủ cho annotation

Chạy:
  pip install datasets pandas pillow tqdm
  python 01_curate_dataset.py
"""

import json, random, base64, hashlib
from pathlib import Path
from io import BytesIO
from datetime import datetime
from tqdm import tqdm
from PIL import Image

random.seed(42)   # reproducibility
OUTPUT_DIR = Path("faithchart_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────
def get_pil_image(img_data):
    """Chuyển đổi dữ liệu sang PIL Image nếu nó đang ở dạng bytes."""
    if isinstance(img_data, bytes):
        return Image.open(BytesIO(img_data))
    return img_data

def pil_to_b64(img) -> str:
    img = get_pil_image(img) # Đảm bảo là PIL Image
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def img_hash(img) -> str:
    img = get_pil_image(img) # Đảm bảo là PIL Image
    buf = BytesIO()
    img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()[:12]

# ── PART A: ChartQA human test split (750 samples) ─────────────────────────
print("=" * 60)
print("PART A: ChartQA human test split")
print("=" * 60)

from datasets import load_dataset

print("Loading ChartQA...")
chartqa = load_dataset("ahmed-masry/ChartQA", split="test")
# Chỉ lấy human-authored questions (không lấy augmented)
# ChartQA có field 'type': 'human' hoặc 'augmented'
human_samples = [s for s in chartqa if s.get("type", "human") == "human"]
print(f"  Total ChartQA test: {len(chartqa)} → human only: {len(human_samples)}")

# Sample 750 đa dạng — đảm bảo coverage các chart types
random.shuffle(human_samples)
chartqa_selected = human_samples[:750]

chartqa_records = []
for i, s in enumerate(tqdm(chartqa_selected, desc="ChartQA")):
    image = s.get("image")
    if image is None:
        continue
    record = {
        "id": f"CQ_{i:04d}",
        "source": "chartqa_human",
        "benchmark": "ChartQA",
        "question": s.get("query", s.get("question", "")),
        "gold_answer": str(s.get("label", s.get("answer", ""))),
        "chart_type": s.get("chart_type", "unknown"),
        "image_b64": pil_to_b64(image),
        "image_hash": img_hash(image),
        "difficulty": "standard",
        "annotation_status": "pending",
        "reasoning_trace": None,
        "cited_regions": [],
        "perturbations": {}
    }
    chartqa_records.append(record)

print(f"  ChartQA records prepared: {len(chartqa_records)}")

# ── PART B: CharXiv validation split — reasoning questions (750 samples) ───
print()
print("=" * 60)
print("PART B: CharXiv reasoning split")
print("=" * 60)

print("Loading CharXiv...")
charxiv = load_dataset("princeton-nlp/CharXiv", split="validation")
reasoning_samples = [s for s in charxiv if s.get("reasoning_q")]
print(f"  Total CharXiv val: {len(charxiv)} → with reasoning Q: {len(reasoning_samples)}")

# Stratified sampling theo figure_type nếu có
random.shuffle(reasoning_samples)

# Ưu tiên diverse scientific domains
charxiv_selected = reasoning_samples[:750]

charxiv_records = []
for i, s in enumerate(tqdm(charxiv_selected, desc="CharXiv")):
    image = s.get("image")
    if image is None:
        continue
    record = {
        "id": f"CX_{i:04d}",
        "source": "charxiv_reasoning",
        "benchmark": "CharXiv",
        "question": s.get("reasoning_q", ""),
        "gold_answer": str(s.get("reasoning_a", "")),
        "chart_type": s.get("figure_type", "scientific"),
        "arxiv_id": s.get("arxiv_id", ""),
        "figure_caption": s.get("figure_caption", "")[:200],
        "image_b64": pil_to_b64(image),
        "image_hash": img_hash(image),
        "difficulty": "hard",   # CharXiv reasoning = harder
        "annotation_status": "pending",
        "reasoning_trace": None,
        "cited_regions": [],
        "perturbations": {}
    }
    charxiv_records.append(record)

print(f"  CharXiv records prepared: {len(charxiv_records)}")

# ── Combine và verify ───────────────────────────────────────────────────────
all_records = chartqa_records + charxiv_records
random.shuffle(all_records)   # Shuffle để annotators không biết nguồn

# Re-assign global IDs sau shuffle
for i, r in enumerate(all_records):
    r["global_id"] = f"FC_{i:04d}"

print()
print(f"Total FaithChart-1500: {len(all_records)} samples")
print(f"  - ChartQA human: {len(chartqa_records)}")
print(f"  - CharXiv reasoning: {len(charxiv_records)}")

# ── Stats ────────────────────────────────────────────────────────────────────
from collections import Counter
difficulty_dist = Counter(r["difficulty"] for r in all_records)
source_dist     = Counter(r["source"] for r in all_records)
print(f"\nDifficulty distribution: {dict(difficulty_dist)}")
print(f"Source distribution: {dict(source_dist)}")

# ── Save ─────────────────────────────────────────────────────────────────────
# Main dataset (với images) — sẽ lớn ~500MB
out_main = OUTPUT_DIR / "faithchart_1500.json"
with open(out_main, "w", encoding="utf-8") as f:
    json.dump(all_records, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_main} ({out_main.stat().st_size / 1e6:.1f} MB)")

# Metadata only (không có image_b64) — cho annotation tools
records_meta = [{k: v for k, v in r.items() if k != "image_b64"}
                for r in all_records]
out_meta = OUTPUT_DIR / "faithchart_1500_metadata.json"
with open(out_meta, "w", encoding="utf-8") as f:
    json.dump(records_meta, f, ensure_ascii=False, indent=2)
print(f"Saved metadata: {out_meta}")

# Annotation batches — chia cho 50 annotators (30 samples/người)
batch_size = 30
batches = [all_records[i:i+batch_size] for i in range(0, len(all_records), batch_size)]
batch_dir = OUTPUT_DIR / "annotation_batches"
batch_dir.mkdir(exist_ok=True)
for j, batch in enumerate(batches):
    batch_file = batch_dir / f"batch_{j+1:02d}.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)

print(f"\nAnnotation batches: {len(batches)} batches × {batch_size} samples")
print(f"  → {batch_dir}/batch_01.json ... batch_{len(batches):02d}.json")

# Summary
summary = {
    "dataset_name": "FaithChart-1500",
    "total_samples": len(all_records),
    "sources": dict(source_dist),
    "difficulty": dict(difficulty_dist),
    "annotation_batches": len(batches),
    "samples_per_batch": batch_size,
    "created": datetime.now().isoformat(),
    "version": "1.0"
}
with open(OUTPUT_DIR / "dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Dataset curation complete!")
print(f"   Next step: upload batches to Label Studio hoặc Prolific")
