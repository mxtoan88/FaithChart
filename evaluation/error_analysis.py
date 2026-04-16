"""
FaithChart — Script 2: Error Taxonomy Analysis
================================================

Output:
  results/error_taxonomy_YYYYMMDD.json
  results/error_report_YYYYMMDD.csv
"""

import os, json, re, csv, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

# ── Error taxonomy (từ ChartInsighter + extended) ─────────
ERROR_TYPES = {
    "E1_numerical_value": {
        "name": "Numerical Value Error",
        "desc": "The model describes or calculates an incorrect quantitative value",
        "examples": ["Average error", "Read the error from the Y-axis"],
        "detection": "gold và prediction là số, relaxed_match fail",
    },
    "E2_trend_direction": {
        "name": "Trend Direction Error",
        "desc": "The model is misidentifying the trend (upward/downward)",
        "examples": ["They say it's increasing when it's actually decreasing"],
        "detection": "keywords: increase/decrease/up/down/rise/fall inverse",
    },
    "E3_extremum": {
        "name": "Extremum Error",
        "desc": "The model incorrectly identifies max/min points, peak/trough points, or the year/time",
        "examples": ["Say 2008 was the peak instead of 2007", "Confusing maximum with local max"],
        "detection": "Gold is the name/year associated with max/min; the prediction is incorrect",
    },
    "E4_range": {
        "name": "Range Error",
        "desc": "The model incorrectly estimates the range or interval",
        "examples": ["They say 10-20 when in reality it's 15-25"],
        "detection": "Gold has a range, and the prediction is inaccurate by about a certain margin",
    },
    "E5_multidim_trend": {
        "name": "Multi-dimensional Trend Error",
        "desc": "The model cannot compare multiple data series at the same time",
        "examples": ["Mistaken ranking between the 3 series"],
        "detection": "The question contains the words compare/which/more/less; the prediction is incorrect comparison",
    },
    "E6_proportion": {
        "name": "Proportion Error",
        "desc": "Model incorrectly uses percentages or relative sizes",
        "examples": ["They said 30% when it was actually 45%"],
        "detection": "Gold contains percentages, and the prediction magnitude is often inaccurate",
    },
    "E7_unanswerable": {
        "name": "Hallucination on Unanswerable",
        "desc": "The model generates answers when the question cannot be answered from the chart",
        "examples": ["Make up numbers when the chart lacks information"],
        "detection": "gold = N/A or unanswerable, prediction is a specific value",
    },
    "E8_structural": {
        "name": "Chart Structure Error",
        "desc": "Model misunderstood the chart structure (axes, legends, scales)",
        "examples": ["Confusing the X-axis with the Y-axis", "Ignore logarithmic scale"],
        "detection": "GPT judge is determined based on chart structure",
    },
    "CORRECT": {"name": "Correct", "desc": "model answered correctly", "examples": [], "detection": "relaxed_match pass"},
    "OTHER_ERROR": {"name": "Other Error", "desc": "Errors that do not fall under the above categories", "examples": [], "detection": "manual review"},
}

# ── Rule-based classifier ─────────────────────────────────
def classify_error_rules(sample: dict) -> str:
    """
    Error classification based on simple rules.
    Use when GPT judge is unavailable..
    """
    if sample.get("correct"):
        return "CORRECT"

    pred = str(sample.get("prediction", "")).lower().strip()
    gold = str(sample.get("gold", "")).lower().strip()
    question = str(sample.get("question", "")).lower()

    # E7: Unanswerable
    unanswerable_golds = {"n/a", "not applicable", "unanswerable", "cannot be determined", "not shown"}
    if gold in unanswerable_golds and pred not in unanswerable_golds:
        return "E7_unanswerable"

    # E1: Numerical value — both numeric, wrong value
    try:
        p_num = float(re.sub(r"[%,$€]", "", pred))
        g_num = float(re.sub(r"[%,$€]", "", gold))
        ratio = abs(p_num - g_num) / max(abs(g_num), 1e-9)
        if ratio > 0.05:
            # Check if it's a proportion error (% involved)
            if "%" in sample.get("gold", "") or "percent" in question:
                return "E6_proportion"
            return "E1_numerical_value"
    except (ValueError, TypeError):
        pass

    # E2: Trend direction
    trend_pairs = [("increase", "decrease"), ("rise", "fall"), ("up", "down"),
                   ("higher", "lower"), ("grew", "declined"), ("positive", "negative")]
    for t1, t2 in trend_pairs:
        if (t1 in gold and t2 in pred) or (t2 in gold and t1 in pred):
            return "E2_trend_direction"

    # E3: Extremum — year or named entity questions
    extremum_words = ["maximum", "minimum", "highest", "lowest", "peak", "trough",
                      "most", "least", "largest", "smallest", "best", "worst"]
    if any(w in question for w in extremum_words):
        return "E3_extremum"

    # E5: Multi-series comparison
    compare_words = ["compare", "which", "between", "more than", "less than",
                     "greater", "versus", "vs", "difference between"]
    if any(w in question for w in compare_words):
        return "E5_multidim_trend"

    # E4: Range
    if re.search(r"\d+\s*[-–to]\s*\d+", gold):
        return "E4_range"

    return "OTHER_ERROR"


def classify_error_gpt(sample: dict, client) -> str:
    """
    Use GPT-40 for more accurate error classification.
    Costs approximately $0.002 per sample.
    """
    prompt = f"""You are analyzing errors in chart question answering.

Question: {sample.get('question', '')}
Gold Answer: {sample.get('gold', '')}
Model Prediction: {sample.get('prediction', '')}

Classify the error type. Choose exactly ONE:
E1_numerical_value - wrong numeric value (wrong number read from chart)
E2_trend_direction - wrong direction (said increase when decrease)
E3_extremum - wrong max/min/peak/year identification
E4_range - wrong range or interval
E5_multidim_trend - failed multi-series comparison
E6_proportion - wrong percentage or relative size
E7_unanswerable - fabricated answer for unanswerable question
E8_structural - misread chart axes/legend/scale
OTHER_ERROR - doesn't fit above categories

Respond with ONLY the error code (e.g., E1_numerical_value)."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # 싸다 — GPT-4o-mini로 충분
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        code = resp.choices[0].message.content.strip()
        if code in ERROR_TYPES:
            return code
        return "OTHER_ERROR"
    except Exception:
        return classify_error_rules(sample)  # fallback to rules


# ── Load all results ──────────────────────────────────────
def load_all_results(results_dir: Path) -> dict:
    """Load all *model*_benchmark_*.json files from results dir."""
    all_data = {}
    for f in sorted(results_dir.glob("*.json")):
        if "summary" in f.name or "taxonomy" in f.name or "report" in f.name:
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
            if isinstance(data, list) and data:
                # Parse model name and benchmark from filename
                parts = f.stem.split("_")
                model = parts[0] if parts else "unknown"
                bench = parts[1] if len(parts) > 1 else "unknown"
                key = f"{model}_{bench}"
                all_data[key] = {"samples": data, "file": str(f)}
                print(f"  Loaded {len(data)} samples from {f.name} → key={key}")
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    return all_data


# ── Main analysis ─────────────────────────────────────────
def analyze(results_dir: Path, use_gpt_judge: bool = False):
    all_data = load_all_results(results_dir)
    if not all_data:
        print("No result files found. Run 01_run_eval.py first.")
        return

    gpt_client = None
    if use_gpt_judge and OPENAI_OK:
        gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        print("Using GPT-4o-mini as error classifier judge")

    # ── Per-model error distribution
    model_stats = {}
    all_classified = []

    for key, obj in all_data.items():
        samples = obj["samples"]
        counts = Counter()
        classified_samples = []

        for s in samples:
            if s.get("error"):  # API error, skip
                continue
            if gpt_client and not s.get("correct"):
                error_type = classify_error_gpt(s, gpt_client)
            else:
                error_type = classify_error_rules(s)

            counts[error_type] += 1
            classified_samples.append({**s, "error_type": error_type})

        model_stats[key] = {
            "n_total": len(classified_samples),
            "n_correct": counts["CORRECT"],
            "accuracy": round(counts["CORRECT"] / max(len(classified_samples), 1) * 100, 1),
            "error_counts": dict(counts),
            "error_pct": {k: round(v / max(len(classified_samples), 1) * 100, 1)
                          for k, v in counts.items()},
        }
        all_classified.extend(classified_samples)

        print(f"\n{key}:")
        print(f"  Accuracy: {model_stats[key]['accuracy']}%")
        for etype, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            if etype != "CORRECT":
                print(f"  {ERROR_TYPES.get(etype, {}).get('name', etype)}: {cnt} ({model_stats[key]['error_pct'][etype]}%)")

    # ── Cross-model error comparison table
    all_error_types = [k for k in ERROR_TYPES if k != "CORRECT"]
    model_keys = list(model_stats.keys())

    print("\n" + "="*70)
    print("ERROR DISTRIBUTION COMPARISON (% of all samples)")
    print("="*70)
    header = f"{'Error Type':<30}" + "".join(f"{k[:10]:>12}" for k in model_keys)
    print(header)
    print("-"*70)
    for etype in all_error_types:
        row = f"{ERROR_TYPES[etype]['name']:<30}"
        for key in model_keys:
            pct = model_stats[key]["error_pct"].get(etype, 0)
            row += f"{pct:>11.1f}%"
        print(row)
    print("-"*70)
    acc_row = f"{'Accuracy':<30}"
    for key in model_keys:
        acc_row += f"{model_stats[key]['accuracy']:>11.1f}%"
    print(acc_row)

    # ── Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    taxonomy_file = results_dir / f"error_taxonomy_{ts}.json"
    with open(taxonomy_file, "w") as f:
        json.dump({
            "timestamp": ts,
            "model_stats": model_stats,
            "error_type_definitions": ERROR_TYPES,
        }, f, indent=2, ensure_ascii=False)

    csv_file = results_dir / f"error_report_{ts}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_benchmark", "error_type", "error_name", "count", "pct"])
        for key, stats in model_stats.items():
            for etype, cnt in stats["error_counts"].items():
                writer.writerow([
                    key, etype,
                    ERROR_TYPES.get(etype, {}).get("name", etype),
                    cnt, stats["error_pct"].get(etype, 0)
                ])

    print(f"\nSaved: {taxonomy_file}")
    print(f"Saved: {csv_file}")
    return model_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--use_gpt_judge", action="store_true",
                        help="Use GPT-4o-mini to classify errors (costs ~$0.002/sample)")
    args = parser.parse_args()
    analyze(Path(args.results_dir), args.use_gpt_judge)
