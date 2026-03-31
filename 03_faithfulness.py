"""
FaithChart — Script 3: Faithfulness Metrics
============================================
Compute Sufficiency và Comprehensiveness cho chart QA explanations.

Sufficiency:  Nếu model dùng region R để explain answer A,
              thì xóa R khỏi chart → answer phải thay đổi.
              Score = P(answer changes | cited region removed)

Comprehensiveness: Nếu model KHÔNG đề cập region R,
                   thì xóa R → answer KHÔNG nên thay đổi.
                   Score = P(answer stable | uncited region removed)

Sử dụng:
  python 03_faithfulness.py --input results/gpt4o_charxiv_*.json \
                             --model gpt4o \
                             --n_perturb 3

Cần: PIL, numpy, openai hoặc anthropic
"""

import os, json, re, argparse, random
from pathlib import Path
from copy import deepcopy
from typing import Optional
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

import base64
from io import BytesIO

# ── Perturbation strategies ───────────────────────────────
class ChartPerturber:
    """
    Tạo perturbed versions của chart images.
    Mỗi perturbation xóa / che một vùng cụ thể của chart.
    """

    def __init__(self, strategy: str = "blur"):
        """
        strategy: "blur" | "blackout" | "noise" | "whiteout"
        """
        assert PIL_OK, "pip install Pillow numpy"
        self.strategy = strategy

    def perturb_region(self, img: Image.Image,
                       bbox: tuple,  # (x1, y1, x2, y2) normalized 0-1
                       ) -> Image.Image:
        """Xóa/che region được chỉ định bởi bounding box."""
        img = img.copy()
        w, h = img.size
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)
        # Clamp
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if self.strategy == "blackout":
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

        elif self.strategy == "whiteout":
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))

        elif self.strategy == "blur":
            region = img.crop((x1, y1, x2, y2))
            blurred = region.filter(ImageFilter.GaussianBlur(radius=15))
            img.paste(blurred, (x1, y1, x2, y2))

        elif self.strategy == "noise":
            region = img.crop((x1, y1, x2, y2))
            arr = np.array(region).astype(float)
            noise = np.random.randint(0, 256, arr.shape, dtype=np.uint8)
            img.paste(Image.fromarray(noise), (x1, y1, x2, y2))

        elif self.strategy == "mean_fill":
            # Fill with mean color of image (least disruptive)
            arr = np.array(img)
            mean_color = tuple(int(c) for c in arr.mean(axis=(0, 1))[:3])
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], fill=mean_color)

        return img

    def perturb_axis_labels(self, img: Image.Image) -> Image.Image:
        """Xóa vùng trục tọa độ (bottom 15% và left 12%)."""
        w, h = img.size
        # X-axis labels: bottom strip
        x_axis_bbox = (0.1, 0.85, 1.0, 1.0)
        img = self.perturb_region(img, x_axis_bbox)
        # Y-axis labels: left strip
        y_axis_bbox = (0.0, 0.05, 0.12, 0.9)
        img = self.perturb_region(img, y_axis_bbox)
        return img

    def perturb_legend(self, img: Image.Image) -> Image.Image:
        """Xóa vùng legend (thường ở góc trên phải hoặc dưới)."""
        # Common legend positions
        legend_bbox = (0.65, 0.05, 1.0, 0.35)
        return self.perturb_region(img, legend_bbox)

    def perturb_title(self, img: Image.Image) -> Image.Image:
        """Xóa title (top 8%)."""
        return self.perturb_region(img, (0.05, 0.0, 0.95, 0.08))

    def perturb_data_region(self, img: Image.Image,
                             n_bars: int = 5,
                             target_idx: int = 0) -> Image.Image:
        """
        Xóa một data element cụ thể (bar, line segment).
        Approximation: chia data area thành N phần đều, xóa phần target_idx.
        """
        # Data area: rough heuristic (10% to 90% of image)
        data_x_start = 0.12
        data_x_end = 0.95
        data_y_start = 0.08
        data_y_end = 0.85

        total_width = data_x_end - data_x_start
        bar_width = total_width / n_bars
        x1 = data_x_start + target_idx * bar_width
        x2 = x1 + bar_width
        bbox = (x1, data_y_start, x2, data_y_end)
        return self.perturb_region(img, bbox)


# ── Extract cited regions from explanation ────────────────
def extract_cited_regions(explanation: str, question: str) -> dict:
    """
    Parse explanation để tìm chart regions được cited.
    Returns dict với cited regions và confidence.

    Heuristic approach — GPT-based extraction cho accuracy tốt hơn.
    """
    cited = {
        "data_values": False,  # specific data points
        "axis": False,         # axis labels/ticks
        "legend": False,       # legend
        "title": False,        # title
        "trend": False,        # overall trend
        "comparison": False,   # comparing multiple elements
    }

    text = (explanation + " " + question).lower()

    # Data value indicators
    if any(w in text for w in ["value", "data point", "bar", "line", "point",
                                "shows", "indicates", "reads", "equals"]):
        cited["data_values"] = True

    # Axis indicators
    if any(w in text for w in ["axis", "x-axis", "y-axis", "scale", "tick",
                                "label", "unit", "range of"]):
        cited["axis"] = True

    # Legend indicators
    if any(w in text for w in ["legend", "color", "series", "key"]):
        cited["legend"] = True

    # Title indicators
    if any(w in text for w in ["title", "titled", "chart shows", "graph shows"]):
        cited["title"] = True

    # Trend indicators
    if any(w in text for w in ["trend", "increase", "decrease", "pattern",
                                "overall", "generally", "over time"]):
        cited["trend"] = True

    # Comparison indicators
    if any(w in text for w in ["compare", "between", "versus", "higher than",
                                "lower than", "more", "less"]):
        cited["comparison"] = True

    return cited


def extract_cited_regions_gpt(explanation: str, question: str,
                               client: "OpenAI") -> dict:
    """GPT-based citation extraction — más preciso."""
    prompt = f"""Analyze this chart question answering explanation and identify which chart regions it cites as evidence.

Question: {question}
Explanation: {explanation}

For each region, say YES or NO:
- data_values: Does it cite specific data points/values from the chart?
- axis: Does it reference axis labels, ticks, or scale?
- legend: Does it reference the legend or color coding?
- title: Does it reference the chart title?
- trend: Does it describe overall trends/patterns?
- comparison: Does it compare multiple data elements?

Respond in JSON format:
{{"data_values": true/false, "axis": true/false, "legend": true/false, "title": true/false, "trend": true/false, "comparison": true/false}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        # Parse JSON
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return extract_cited_regions(explanation, question)  # fallback


# ── Faithfulness scorer ───────────────────────────────────
class FaithfulnessScorer:
    """
    Compute Sufficiency và Comprehensiveness metrics.
    """

    def __init__(self, evaluator_fn, perturber: ChartPerturber):
        """
        evaluator_fn: function(prompt, image) -> answer string
        """
        self.eval_fn = evaluator_fn
        self.perturber = perturber

    def score_sufficiency(self, sample: dict, n_perturb: int = 3) -> float:
        """
        Sufficiency: Xóa cited regions → answer phải thay đổi.
        Score = fraction of perturbations that change the answer.
        High score = explanation IS sufficient (good faithfulness).
        """
        original_answer = sample.get("prediction", "")
        explanation = sample.get("raw_output", sample.get("prediction", ""))
        question = sample.get("question", "")
        image = sample.get("_image")  # PIL Image

        if image is None or not original_answer:
            return None

        cited = extract_cited_regions(explanation, question)

        # Create perturbations targeting cited regions
        perturbations = []
        if cited.get("data_values"):
            for i in range(min(n_perturb, 5)):
                perturbations.append(("data_point", i))
        if cited.get("axis") and len(perturbations) < n_perturb:
            perturbations.append(("axis", None))
        if cited.get("legend") and len(perturbations) < n_perturb:
            perturbations.append(("legend", None))

        if not perturbations:
            perturbations = [("data_point", 0), ("data_point", 1), ("axis", None)]

        changes = 0
        total = 0
        for ptype, pidx in perturbations[:n_perturb]:
            try:
                if ptype == "axis":
                    perturbed_img = self.perturber.perturb_axis_labels(image)
                elif ptype == "legend":
                    perturbed_img = self.perturber.perturb_legend(image)
                else:
                    perturbed_img = self.perturber.perturb_data_region(
                        image, target_idx=pidx or 0)

                # Re-evaluate model on perturbed image
                perturbed_b64 = pil_to_b64(perturbed_img)
                new_answer = self.eval_fn(question, perturbed_b64)

                # Check if answer changed
                if not answers_match(original_answer, new_answer):
                    changes += 1
                total += 1
            except Exception as e:
                continue

        return changes / total if total > 0 else None

    def score_comprehensiveness(self, sample: dict, n_perturb: int = 3) -> float:
        """
        Comprehensiveness: Xóa uncited regions → answer KHÔNG nên thay đổi.
        Score = fraction of perturbations that DO NOT change the answer.
        High score = explanation IS comprehensive (cited important regions).
        """
        original_answer = sample.get("prediction", "")
        explanation = sample.get("raw_output", sample.get("prediction", ""))
        question = sample.get("question", "")
        image = sample.get("_image")

        if image is None or not original_answer:
            return None

        cited = extract_cited_regions(explanation, question)

        # Create perturbations targeting UNCITED regions
        perturbations = []
        if not cited.get("title"):
            perturbations.append(("title", None))
        if not cited.get("legend"):
            perturbations.append(("legend", None))

        # Add some random data region perturbations
        if len(perturbations) < n_perturb:
            for i in range(3, min(3 + n_perturb, 8)):
                perturbations.append(("data_point", i))

        if not perturbations:
            return None

        stable = 0
        total = 0
        for ptype, pidx in perturbations[:n_perturb]:
            try:
                if ptype == "title":
                    perturbed_img = self.perturber.perturb_title(image)
                elif ptype == "legend":
                    perturbed_img = self.perturber.perturb_legend(image)
                else:
                    perturbed_img = self.perturber.perturb_data_region(
                        image, target_idx=pidx or 3)

                perturbed_b64 = pil_to_b64(perturbed_img)
                new_answer = self.eval_fn(question, perturbed_b64)

                if answers_match(original_answer, new_answer):
                    stable += 1
                total += 1
            except Exception:
                continue

        return stable / total if total > 0 else None


# ── Utilities ─────────────────────────────────────────────
def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def answers_match(a1: str, a2: str, tol: float = 0.05) -> bool:
    """Check if two answers are equivalent."""
    a1 = str(a1).lower().strip()
    a2 = str(a2).lower().strip()
    if a1 == a2:
        return True
    try:
        n1 = float(re.sub(r"[%,$€,]", "", a1))
        n2 = float(re.sub(r"[%,$€,]", "", a2))
        return abs(n1 - n2) / max(abs(n1), abs(n2), 1e-9) <= tol
    except (ValueError, TypeError):
        return False

def demo_perturbation(image_path: str, output_dir: str = "perturb_demo"):
    """Demo: tạo perturbed versions của một chart để kiểm tra."""
    assert PIL_OK, "pip install Pillow"
    img = Image.open(image_path)
    Path(output_dir).mkdir(exist_ok=True)
    perturber = ChartPerturber("blur")

    img.save(f"{output_dir}/original.png")
    perturber.perturb_axis_labels(img).save(f"{output_dir}/no_axes.png")
    perturber.perturb_legend(img).save(f"{output_dir}/no_legend.png")
    perturber.perturb_title(img).save(f"{output_dir}/no_title.png")
    for i in range(5):
        perturber.perturb_data_region(img, 5, i).save(f"{output_dir}/no_bar_{i}.png")
    print(f"Demo saved to {output_dir}/")


# ── Aggregate faithfulness report ────────────────────────
def compute_faithfulness_report(results_with_scores: list) -> dict:
    """
    Tính aggregate faithfulness metrics từ list of scored samples.
    """
    suf_scores = [s["sufficiency"] for s in results_with_scores
                  if s.get("sufficiency") is not None]
    comp_scores = [s["comprehensiveness"] for s in results_with_scores
                   if s.get("comprehensiveness") is not None]

    correct = [s for s in results_with_scores if s.get("correct")]
    incorrect = [s for s in results_with_scores if not s.get("correct")]

    def avg(lst): return round(sum(lst) / len(lst) * 100, 1) if lst else None

    report = {
        "n_samples": len(results_with_scores),
        "accuracy": round(sum(1 for s in results_with_scores if s.get("correct")) /
                         max(len(results_with_scores), 1) * 100, 1),
        # Overall faithfulness
        "sufficiency_mean": avg(suf_scores),
        "comprehensiveness_mean": avg(comp_scores),
        "faithfulness_score": avg([(s + c) / 2 for s, c in zip(suf_scores, comp_scores)
                                   if s is not None and c is not None]),
        # Breakdown by correctness
        "correct_sufficiency": avg([s["sufficiency"] for s in correct
                                    if s.get("sufficiency") is not None]),
        "incorrect_sufficiency": avg([s["sufficiency"] for s in incorrect
                                      if s.get("sufficiency") is not None]),
        # Key insight: do correct answers have more faithful explanations?
        "faithfulness_accuracy_correlation": None,  # compute if enough data
    }

    # Pearson correlation between accuracy and faithfulness
    if len(suf_scores) >= 10:
        try:
            corr_data = [(1 if s.get("correct") else 0, s["sufficiency"])
                         for s in results_with_scores if s.get("sufficiency") is not None]
            if len(corr_data) >= 10:
                acc_vals = [c[0] for c in corr_data]
                suf_vals = [c[1] for c in corr_data]
                mean_a = sum(acc_vals) / len(acc_vals)
                mean_s = sum(suf_vals) / len(suf_vals)
                num = sum((a - mean_a) * (s - mean_s) for a, s in zip(acc_vals, suf_vals))
                den_a = (sum((a - mean_a)**2 for a in acc_vals)) ** 0.5
                den_s = (sum((s - mean_s)**2 for s in suf_vals)) ** 0.5
                if den_a * den_s > 0:
                    report["faithfulness_accuracy_correlation"] = round(num / (den_a * den_s), 3)
        except Exception:
            pass

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", help="Path to chart image for perturbation demo")
    parser.add_argument("--output_dir", default="perturb_demo")
    args = parser.parse_args()

    if args.demo:
        demo_perturbation(args.demo, args.output_dir)
        print("Check output directory for perturbed versions.")
    else:
        print("FaithChart Faithfulness Metrics Module")
        print("Usage: python 03_faithfulness.py --demo path/to/chart.png")
        print("\nKey classes: ChartPerturber, FaithfulnessScorer")
        print("Key functions: compute_faithfulness_report()")
