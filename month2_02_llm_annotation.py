#!/usr/bin/env python3
"""
FaithChart Tháng 2 — Task 2: LLM-Assisted Annotation
======================================================
Dùng GPT-4o Vision để tự động sinh reasoning traces cho 1,500 samples.
Output dùng làm:
  1. Seed annotations để human annotators review/correct (tiết kiệm 60% thời gian)
  2. Automatic annotation cho samples human không có thời gian làm

Chi phí ước tính: ~$600-800 cho 1,500 samples
(1,500 × ~$0.40-0.55 per sample với gpt-4o + vision)

Chạy:
  export OPENAI_API_KEY=sk-...
  python 02_llm_annotation.py --batch 01  # Chạy batch 1 trước để test
  python 02_llm_annotation.py --all       # Chạy tất cả
"""

import os, json, base64, re, time, argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DATASET_DIR = Path("faithchart_dataset")
OUTPUT_DIR  = Path("faithchart_annotations")
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_EVERY = 10

# ── Prompt engineering ──────────────────────────────────────────────────────
ANNOTATION_PROMPT = """You are an expert at analyzing scientific charts and explaining visual reasoning.

Given a chart image and a question, provide a detailed reasoning trace that:
1. Identifies EXACTLY which visual elements of the chart are relevant to answering the question
2. Describes step-by-step how you read and interpret those elements
3. Explains the logical steps to arrive at the answer

Your response MUST follow this exact JSON format:
{
  "reasoning_trace": "Step-by-step explanation of how you answered the question, referencing specific visual elements",
  "cited_regions": [
    {
      "region_type": "one of: data_values, axis_labels, legend, title, background_gridlines, annotation",
      "description": "specific description of what you looked at",
      "necessity": "critical|supporting|not_needed",
      "location": "approximate location in chart: top-left, center, bottom-right, etc."
    }
  ],
  "answer": "your final answer to the question",
  "confidence": "high|medium|low",
  "difficulty_note": "what makes this question easy or hard"
}

Question: {question}

Respond with ONLY valid JSON, no markdown, no preamble."""

# ── Core annotation function ───────────────────────────────────────────────
def annotate_sample(record: dict) -> dict:
    """Annotate a single sample using GPT-4o Vision."""
    img_b64 = record["image_b64"]
    question = record["question"]
    gold_answer = record["gold_answer"]

    prompt = ANNOTATION_PROMPT.format(question=question)

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "high"
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }],
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    try:
        annotation = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract JSON manually
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        annotation = json.loads(match.group()) if match else {}

    # Validate answer against gold
    def relaxed_match(pred, gold):
        pred_n = re.sub(r"[^\w\d.\-]", "", str(pred).lower())
        gold_n = re.sub(r"[^\w\d.\-]", "", str(gold).lower())
        if pred_n == gold_n:
            return True
        try:
            return abs(float(pred_n) - float(gold_n)) / max(abs(float(gold_n)), 1e-9) <= 0.05
        except:
            return False

    predicted_answer = annotation.get("answer", "")
    is_correct = relaxed_match(predicted_answer, gold_answer)

    return {
        "global_id": record["global_id"],
        "source": record["source"],
        "question": question,
        "gold_answer": gold_answer,
        "llm_answer": predicted_answer,
        "llm_correct": is_correct,
        "reasoning_trace": annotation.get("reasoning_trace", ""),
        "cited_regions": annotation.get("cited_regions", []),
        "confidence": annotation.get("confidence", "unknown"),
        "difficulty_note": annotation.get("difficulty_note", ""),
        "annotation_model": "gpt-4o",
        "annotation_status": "llm_annotated",
        "human_reviewed": False,
        "annotated_at": datetime.now().isoformat()
    }

# ── Batch runner ───────────────────────────────────────────────────────────
def run_batch(batch_file: Path):
    with open(batch_file) as f:
        records = json.load(f)

    batch_name = batch_file.stem
    out_file = OUTPUT_DIR / f"{batch_name}_annotated.json"

    # Resume checkpoint
    results, done_ids = [], set()
    if out_file.exists():
        results = json.loads(out_file.read_text())
        done_ids = {r["global_id"] for r in results}
        print(f"  Resuming from {len(results)} existing annotations")

    print(f"\nAnnotating {batch_file.name} ({len(records)} samples)...")
    n_correct = sum(1 for r in results if r.get("llm_correct"))

    for i, record in enumerate(records):
        gid = record.get("global_id", f"unknown_{i}")
        if gid in done_ids:
            continue

        try:
            annotation = annotate_sample(record)
            results.append(annotation)
            if annotation["llm_correct"]:
                n_correct += 1

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error on {gid}: {e}")
            results.append({
                "global_id": gid,
                "annotation_status": "error",
                "error": str(e)[:200],
                "annotated_at": datetime.now().isoformat()
            })

        # Checkpoint
        if len(results) % CHECKPOINT_EVERY == 0:
            out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            n_done = len([r for r in results if r.get("annotation_status") != "error"])
            acc = n_correct / max(n_done, 1) * 100
            print(f"  [{len(results)}/{len(records)}] LLM acc={acc:.1f}%")

    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    n_done = len([r for r in results if r.get("annotation_status") == "llm_annotated"])
    n_error = len([r for r in results if r.get("annotation_status") == "error"])
    n_correct_final = sum(1 for r in results if r.get("llm_correct"))
    acc = n_correct_final / max(n_done, 1) * 100

    print(f"\n  ✅ {batch_name}: {n_done}/{len(records)} annotated | LLM acc={acc:.1f}% | {n_error} errors")
    return results

# ── Quality control: filter low-confidence + wrong annotations ──────────────
def quality_filter(all_annotations: list) -> dict:
    """
    Phân loại annotations để gửi cho human review:
    - HIGH PRIORITY: LLM wrong answer (cần human review)
    - MEDIUM PRIORITY: LLM correct but low confidence (spot check)
    - LOW PRIORITY: LLM correct + high confidence (có thể dùng trực tiếp)
    """
    high_priority = [a for a in all_annotations
                     if not a.get("llm_correct") and a.get("annotation_status") == "llm_annotated"]
    medium_priority = [a for a in all_annotations
                       if a.get("llm_correct") and a.get("confidence") == "low"]
    low_priority = [a for a in all_annotations
                    if a.get("llm_correct") and a.get("confidence") in ["high", "medium"]]

    print(f"\nQuality filter results:")
    print(f"  HIGH priority (human must review): {len(high_priority)} ({len(high_priority)/len(all_annotations)*100:.1f}%)")
    print(f"  MEDIUM priority (spot check):      {len(medium_priority)} ({len(medium_priority)/len(all_annotations)*100:.1f}%)")
    print(f"  LOW priority (can auto-accept):    {len(low_priority)} ({len(low_priority)/len(all_annotations)*100:.1f}%)")

    return {
        "high_priority": high_priority,
        "medium_priority": medium_priority,
        "low_priority": low_priority
    }

# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, help="Specific batch number (e.g. '01')")
    parser.add_argument("--all", action="store_true", help="Run all batches")
    parser.add_argument("--filter-only", action="store_true", help="Only run quality filter on existing")
    args = parser.parse_args()

    batch_dir = DATASET_DIR / "annotation_batches"

    if args.filter_only:
        # Tổng hợp tất cả annotations đã có và chạy filter
        all_annotations = []
        for f in sorted(OUTPUT_DIR.glob("*_annotated.json")):
            all_annotations.extend(json.loads(f.read_text()))
        print(f"Total annotations loaded: {len(all_annotations)}")
        filtered = quality_filter(all_annotations)
        for priority, items in filtered.items():
            out = OUTPUT_DIR / f"human_review_{priority}.json"
            out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
            print(f"  Saved: {out}")

    elif args.batch:
        batch_file = batch_dir / f"batch_{args.batch.zfill(2)}.json"
        if not batch_file.exists():
            print(f"Batch file not found: {batch_file}")
        else:
            run_batch(batch_file)

    elif args.all:
        all_batches = sorted(batch_dir.glob("batch_*.json"))
        print(f"Found {len(all_batches)} batches to process")
        all_annotations = []
        for batch_file in all_batches:
            results = run_batch(batch_file)
            all_annotations.extend(results)

        # Merge tất cả
        merged = OUTPUT_DIR / "all_annotations.json"
        merged.write_text(json.dumps(all_annotations, indent=2, ensure_ascii=False))
        print(f"\nAll annotations merged: {merged}")

        # Quality filter
        filtered = quality_filter(all_annotations)
        for priority, items in filtered.items():
            out = OUTPUT_DIR / f"human_review_{priority}.json"
            out.write_text(json.dumps(items, indent=2, ensure_ascii=False))

        n_total = len(all_annotations)
        n_correct = sum(1 for a in all_annotations if a.get("llm_correct"))
        print(f"\n🎉 Done! Total: {n_total} | LLM accuracy: {n_correct/n_total*100:.1f}%")
        print("Next step: python 02_llm_annotation.py --filter-only → upload to Prolific")

    else:
        print("Usage:")
        print("  python 02_llm_annotation.py --batch 01    # Test 1 batch first (~$12-15)")
        print("  python 02_llm_annotation.py --all          # Full run (~$600-800)")
        print("  python 02_llm_annotation.py --filter-only  # Quality filter only")
