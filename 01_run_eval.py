"""
FaithChart — Script 1: Evaluation Runner
=========================================
Chạy 5 models trên CharXiv reasoning + ChartQAPro samples.
Kết quả lưu vào results/ để script 02 phân tích.

Yêu cầu:
  pip install openai anthropic datasets Pillow tqdm

Sử dụng:
  python 01_run_eval.py --model gpt4o   --benchmark charxiv  --n 200
  python 01_run_eval.py --model claude  --benchmark charxiv  --n 200
  python 01_run_eval.py --model qwen    --benchmark charxiv  --n 200  --local
  python 01_run_eval.py --model all     --benchmark both     --n 200
"""

import os, json, base64, argparse, time, re
from pathlib import Path
from datetime import datetime
from io import BytesIO

# ── pip install these ─────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    import anthropic as ant
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

try:
    from datasets import load_dataset
    HF_OK = True
except ImportError:
    HF_OK = False

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

from tqdm import tqdm

# ── Config ────────────────────────────────────────────────
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Điền API keys vào đây hoặc set environment variables
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY",  "sk-...")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "sk-ant-...")

# CharXiv scores từ leaderboard (dùng làm reference khi không chạy được)
REFERENCE_SCORES = {
    # Model: {charxiv_desc, charxiv_reason, charxiv_avg, chartqa_relaxed}
    "gpt4o":      {"charxiv_desc": 84.4, "charxiv_reason": 47.1, "chartqa": 85.7,  "chartqapro": None},
    "claude35":   {"charxiv_desc": 88.5, "charxiv_reason": 60.2, "chartqa": 90.5,  "chartqapro": 55.8},
    "qwen25vl7b": {"charxiv_desc": 76.3, "charxiv_reason": 42.8, "chartqa": 83.5,  "chartqapro": 37.2},
    "tinychart":  {"charxiv_desc": 59.1, "charxiv_reason": 28.4, "chartqa": 83.6,  "chartqapro": None},
    "chartmoe":   {"charxiv_desc": 65.2, "charxiv_reason": 33.1, "chartqa": 84.64, "chartqapro": None},
    "human":      {"charxiv_desc": 92.1, "charxiv_reason": 80.5, "chartqa": 97.2,  "chartqapro": None},
}

# ── Image utilities ───────────────────────────────────────
def pil_to_b64(img: "Image.Image", fmt="PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def path_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ── Metric: relaxed accuracy (ChartQA standard) ──────────
def relaxed_match(pred: str, gold: str, tol: float = 0.05) -> bool:
    """
    ChartQA relaxed accuracy:
    - Nếu cả hai là số: |pred - gold| / max(|gold|, 1) <= tol
    - Nếu là string: exact match sau normalize
    """
    pred = str(pred).strip().lower().replace(",", "")
    gold = str(gold).strip().lower().replace(",", "")
    # Try numeric
    try:
        p, g = float(pred), float(gold)
        if g == 0:
            return abs(p - g) <= tol
        return abs(p - g) / max(abs(g), 1e-9) <= tol
    except ValueError:
        pass
    # String match: strip % signs, units
    for tok in ["%", "million", "billion", "thousand", "k", "m", "b", "$", "€"]:
        pred = pred.replace(tok, "").strip()
        gold = gold.replace(tok, "").strip()
    return pred == gold

def extract_answer(raw: str) -> str:
    """Extract final answer từ model output."""
    # Look for explicit answer tags
    for pat in [r"(?i)(?:final answer|answer)\s*:\s*(.+?)(?:\n|$)",
                r"(?i)(?:the answer is|answer is)\s+(.+?)(?:\.|$)",
                r"(?i)^(.+?)$"]:  # fallback: first line
        m = re.search(pat, raw.strip(), re.MULTILINE)
        if m:
            return m.group(1).strip()
    return raw.strip()[:200]

# ── Prompt templates ──────────────────────────────────────
def make_charxiv_reasoning_prompt(question: str) -> str:
    return (
        "You are an expert at reading and reasoning about scientific charts. "
        "Answer the following question based on the chart image provided. "
        "Think step by step, then state your final answer clearly.\n\n"
        f"Question: {question}\n\n"
        "Provide your reasoning, then end with:\nFinal Answer: <your answer>"
    )

def make_chartqa_prompt(question: str) -> str:
    return (
        "Answer the question about the chart. "
        "Be concise — give only the value, label, or short phrase as the answer.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

# ── Model callers ─────────────────────────────────────────
class GPT4oEval:
    def __init__(self):
        assert OPENAI_OK, "pip install openai"
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o"
        self.name = "GPT-4o"

    def call(self, prompt: str, image_b64: str, media: str = "image/png") -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media};base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]}],
                max_tokens=512,
                temperature=0,
            )
            raw = resp.choices[0].message.content
            return {"raw": raw, "answer": extract_answer(raw),
                    "tokens": resp.usage.total_tokens, "error": None}
        except Exception as e:
            return {"raw": "", "answer": "", "tokens": 0, "error": str(e)}


class ClaudeEval:
    def __init__(self, model="claude-sonnet-4-5"):
        assert ANTHROPIC_OK, "pip install anthropic"
        self.client = ant.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model
        self.name = f"Claude ({model})"

    def call(self, prompt: str, image_b64: str, media: str = "image/png") -> dict:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media, "data": image_b64}},
                    {"type": "text", "text": prompt}
                ]}],
            )
            raw = resp.content[0].text
            return {"raw": raw, "answer": extract_answer(raw),
                    "tokens": resp.usage.input_tokens + resp.usage.output_tokens,
                    "error": None}
        except Exception as e:
            return {"raw": "", "answer": "", "tokens": 0, "error": str(e)}


class LocalVLMEval:
    """
    Wrapper cho local models (TinyChart, ChartMoE, Qwen2.5-VL).
    Cần transformers + GPU.
    """
    def __init__(self, model_id: str, model_name: str):
        self.model_id = model_id
        self.name = model_name
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline, AutoProcessor
            from qwen_vl_utils import process_vision_info
            print(f"Loading {self.model_id}...")
            # Lazy load — chỉ load khi cần
            self._pipe = "loaded"  # placeholder
        except ImportError:
            raise RuntimeError(
                "pip install transformers qwen-vl-utils accelerate bitsandbytes")

    def call(self, prompt: str, image_b64: str, media: str = "image/png") -> dict:
        """
        Placeholder — replace với actual inference code cho từng model.
        Xem README.md cho hướng dẫn chi tiết từng model.
        """
        # Với Qwen2.5-VL (example):
        # from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        # messages = [{"role": "user", "content": [
        #     {"type": "image", "image": f"data:{media};base64,{image_b64}"},
        #     {"type": "text", "text": prompt}]}]
        # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info(messages)
        # inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")
        # generated_ids = model.generate(**inputs, max_new_tokens=512)
        # output = processor.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        # return {"raw": output, "answer": extract_answer(output), "tokens": 0, "error": None}
        return {"raw": "[LOCAL_MODEL_PLACEHOLDER]", "answer": "",
                "tokens": 0, "error": "local model not loaded"}


# ── Dataset loaders ───────────────────────────────────────
def load_charxiv_reasoning(n: int = 200, split: str = "val") -> list:
    """
    Load CharXiv từ HuggingFace.
    Dataset: princeton-nlp/CharXiv
    Chỉ lấy reasoning questions.
    """
    assert HF_OK, "pip install datasets"
    print(f"Loading CharXiv {split} (reasoning only, n={n})...")
    try:
        ds = load_dataset("princeton-nlp/CharXiv", split=split, trust_remote_code=True)
        # Filter reasoning questions (type == "reasoning")
        samples = [s for s in ds if s.get("type") == "reasoning"][:n]
        print(f"  Loaded {len(samples)} reasoning samples")
        return samples
    except Exception as e:
        print(f"  Error loading from HuggingFace: {e}")
        print("  Falling back to local JSON format")
        return []


def load_chartqa_human(n: int = 200, split: str = "test") -> list:
    """
    Load ChartQA human split.
    Download từ: github.com/vis-nlp/ChartQA
    Format: {imgname, query, label}
    """
    json_path = Path(f"data/chartqa/{split}_human.json")
    if not json_path.exists():
        print(f"  ChartQA not found at {json_path}")
        print("  Download: git clone github.com/vis-nlp/ChartQA && cp -r ChartQA/ChartQA\\ Dataset/test data/chartqa/")
        return []
    with open(json_path) as f:
        data = json.load(f)
    samples = data[:n]
    print(f"  Loaded {len(samples)} ChartQA human samples")
    return samples


# ── Main evaluation loop ──────────────────────────────────
def evaluate_model(evaluator, samples: list, benchmark: str,
                   output_file: Path, delay: float = 0.5) -> dict:
    """
    Chạy evaluation và lưu từng kết quả vào file.
    Returns summary statistics.
    """
    results = []
    correct = 0
    errors = 0

    # Load existing results để resume nếu bị interrupt
    if output_file.exists():
        with open(output_file) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        correct = sum(1 for r in results if r.get("correct"))
        print(f"  Resuming from {len(results)} existing results")
    else:
        done_ids = set()

    pbar = tqdm(samples, desc=f"{evaluator.name} / {benchmark}")
    for sample in pbar:
        sample_id = sample.get("id") or sample.get("imgname") or str(len(results))
        if sample_id in done_ids:
            continue

        # Get image as base64
        image_b64 = None
        if "image" in sample and sample["image"]:
            if hasattr(sample["image"], "save"):  # PIL Image
                image_b64 = pil_to_b64(sample["image"])
            elif isinstance(sample["image"], str):
                if os.path.exists(sample["image"]):
                    image_b64 = path_to_b64(sample["image"])

        if image_b64 is None:
            errors += 1
            continue

        # Get question and gold answer
        if benchmark == "charxiv":
            question = sample.get("question", "")
            gold = sample.get("answer", "")
            prompt = make_charxiv_reasoning_prompt(question)
        else:
            question = sample.get("query", "")
            gold = sample.get("label", "")
            prompt = make_chartqa_prompt(question)

        # Call model
        output = evaluator.call(prompt, image_b64)

        # Score
        is_correct = False
        if output["error"] is None:
            is_correct = relaxed_match(output["answer"], gold)
        else:
            errors += 1

        if is_correct:
            correct += 1

        result = {
            "id": sample_id,
            "question": question,
            "gold": gold,
            "prediction": output["answer"],
            "raw_output": output["raw"][:1000],  # truncate for storage
            "correct": is_correct,
            "tokens": output["tokens"],
            "error": output["error"],
        }
        results.append(result)

        # Save after each sample (crash-safe)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Update progress
        n_done = len(results)
        acc = correct / n_done * 100 if n_done > 0 else 0
        pbar.set_postfix({"acc": f"{acc:.1f}%", "err": errors})

        # Rate limiting
        if delay > 0:
            time.sleep(delay)

    n_total = len(results)
    accuracy = correct / n_total * 100 if n_total > 0 else 0

    summary = {
        "model": evaluator.name,
        "benchmark": benchmark,
        "n_total": n_total,
        "n_correct": correct,
        "n_errors": errors,
        "accuracy": round(accuracy, 2),
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = output_file.with_suffix(".summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Result: {correct}/{n_total} = {accuracy:.1f}%")
    return summary


# ── CLI ───────────────────────────────────────────────────
def get_evaluator(model: str):
    model_map = {
        "gpt4o":   lambda: GPT4oEval(),
        "gpt-4o":  lambda: GPT4oEval(),
        "claude":  lambda: ClaudeEval("claude-sonnet-4-5"),
        "qwen":    lambda: LocalVLMEval("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B"),
        "tiny":    lambda: LocalVLMEval("mPLUG/TinyChart-3B-768", "TinyChart-3B"),
        "chartmoe":lambda: LocalVLMEval("tangxue/ChartMoE", "ChartMoE"),
    }
    assert model in model_map, f"Unknown model: {model}. Choose: {list(model_map.keys())}"
    return model_map[model]()


def main():
    parser = argparse.ArgumentParser(description="FaithChart Evaluation Runner")
    parser.add_argument("--model", default="gpt4o",
                        choices=["gpt4o","gpt-4o","claude","qwen","tiny","chartmoe","all"])
    parser.add_argument("--benchmark", default="charxiv",
                        choices=["charxiv","chartqa","both"])
    parser.add_argument("--n", type=int, default=200,
                        help="Number of samples per benchmark")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--split", default="val",
                        help="Dataset split: val / test")
    args = parser.parse_args()

    models = ["gpt4o","claude","qwen","tiny","chartmoe"] if args.model == "all" else [args.model]
    benchmarks = ["charxiv","chartqa"] if args.benchmark == "both" else [args.benchmark]

    all_summaries = []

    for model_name in models:
        evaluator = get_evaluator(model_name)

        for bench in benchmarks:
            print(f"\n{'='*50}")
            print(f"Model: {evaluator.name}  |  Benchmark: {bench}  |  N={args.n}")
            print('='*50)

            # Load data
            if bench == "charxiv":
                samples = load_charxiv_reasoning(args.n, args.split)
            else:
                samples = load_chartqa_human(args.n, args.split)

            if not samples:
                print("  No samples loaded — check data setup")
                continue

            # Output file
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            out_file = RESULTS_DIR / f"{model_name}_{bench}_{ts}.json"

            summary = evaluate_model(evaluator, samples, bench, out_file, args.delay)
            all_summaries.append(summary)

    # Print final table
    if all_summaries:
        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(f"{'Model':<20} {'Benchmark':<12} {'N':<8} {'Accuracy':>10}")
        print("-"*60)
        for s in all_summaries:
            print(f"{s['model']:<20} {s['benchmark']:<12} {s['n_total']:<8} {s['accuracy']:>9.1f}%")

    print("\nResults saved to results/")


if __name__ == "__main__":
    main()
