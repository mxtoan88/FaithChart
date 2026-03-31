"""
FaithChart — Script 1: Evaluation Runner (Bản Hợp Nhất Tối Ưu)
===========================================================
Chạy đánh giá GPT-4o, Claude 4.x và các Local Models trên CharXiv & ChartQA.

Tính năng:
  - Hỗ trợ tham số: --model (all/gpt4o/claude/...) --benchmark (both/charxiv/chartqa) --n 200.
  - Auto-Discovery: Tự động tìm mã model Claude khả dụng (Sonnet 4.6, Opus 4.6...).
  - Persistent Resume: Tự động nhận diện kết quả cũ để chạy tiếp, tránh tốn phí.
  - Relaxed Matching: Logic chấm điểm nới lỏng (+25 từ) để tránh chấm sai cho AI trả lời dài.
  - Summary Generation: Tự động tạo file tổng kết .summary.json.

Yêu cầu:
  pip install openai anthropic datasets Pillow tqdm
"""

import os, json, base64, argparse, time, re, warnings
from pathlib import Path
from datetime import datetime
from io import BytesIO

# Ẩn cảnh báo không cần thiết
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# ── Cấu hình ────────────────────────────────────────────────
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Tiện ích ảnh ───────────────────────────────────────────
def pil_to_b64(img: "Image.Image", fmt="PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def path_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ── Metric: Relaxed Accuracy (Tối ưu cho Reasoning) ────────
def relaxed_match(pred: str, gold: str, tol: float = 0.05) -> bool:
    if not pred or not gold: return False
    
    def normalize(s):
        s = str(s).strip().lower()
        # 1. Xóa LaTeX
        s = re.sub(r"\\\(|\\\)|\\\[|\\\]|\$|\\", "", s)
        # 2. Xóa tiền tố hội thoại
        s = re.sub(r"^(subplot|figure|fig|panel|category|variable|condition|is|the answer is|the correct option is)\s+", "", s)
        # 3. Chuẩn hóa nhãn (c) -> c
        s = re.sub(r"\(([a-z0-9])\)", r"\1", s)
        s = s.replace("(", " ").replace(")", " ").replace(",", " ").replace(";", " ").replace(":", " ").replace("-", " ")
        # 4. Xóa từ nối
        s = re.sub(r"\s+(and|&|with|including|in|at|of|for|the|a|an|line|color|labeled|shows|represents|axis|direction|than)\s+", " ", s)
        return re.sub(r"\s+", " ", s).strip().rstrip(".")

    p_norm = normalize(pred)
    g_norm = normalize(gold)
    
    if p_norm == g_norm: return True
    
    p_words = set(p_norm.split())
    g_words = set(g_norm.split())
    
    # Matching nhãn đơn (a, b, c...)
    if len(g_norm) == 1 and g_norm.isalpha():
        if g_norm in p_words: return True

    # Matching tập hợp từ (Keyword match)
    if g_words and g_words.issubset(p_words):
        if len(p_words) <= len(g_words) + 25: return True
    
    # Substring match (Dành cho đáp án dài như "Kansas City")
    if len(g_norm) > 3 and g_norm in p_norm:
        if len(p_norm.split()) <= len(g_norm.split()) + 25: return True

    # So sánh số học
    try:
        p_v = float(re.sub(r"[^\d\.\-]", "", p_norm))
        g_v = float(re.sub(r"[^\d\.\-]", "", g_norm))
        return abs(p_v - g_v) <= tol if g_v == 0 else abs(p_v - g_v) / max(abs(g_v), 1e-9) <= tol
    except: pass
    return False

def extract_answer(raw: str) -> str:
    raw = raw.strip()
    # Tìm tag Final Answer
    matches = list(re.finditer(r"(?i)(?:final answer|the answer is|answer|conclusion|result)\s*[:\-*]*\s*(.+?)(?:\n|$)", raw))
    if matches: 
        ans = matches[-1].group(1).strip()
        return re.sub(r"[*'\"`]", "", ans).strip()
    
    # Lấy dòng cuối hoặc phần in đậm
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    if lines:
        last = lines[-1]
        bold_match = re.search(r"\*\*(.*?)\*\*", last)
        if bold_match: return bold_match.group(1).strip()
        if len(last) > 100 and "." in last:
            parts = [p.strip() for p in last.split('.') if p.strip()]
            if parts: return parts[-1]
        return re.sub(r"[*'\"`]", "", last).strip()
    return raw[:200]

# ── Prompts ───────────────────────────────────────────────
def make_charxiv_reasoning_prompt(question: str) -> str:
    return (
        "You are an expert at reading and reasoning about scientific charts. "
        "Answer the question based on the chart image. Think step by step, then state your final answer clearly.\n\n"
        f"Question: {question}\n\n"
        "Provide reasoning then end with:\nFinal Answer: <your short answer>"
    )

def make_chartqa_prompt(question: str) -> str:
    return (
        "Answer the question about the chart. Be concise — give only the value, label, or short phrase.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

# ── API Callers ───────────────────────────────────────────
class GPT4oEval:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "GPT-4o"

    def call(self, prompt: str, img_b64: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]}],
                max_tokens=1024, temperature=0,
            )
            raw = resp.choices[0].message.content
            return {"raw": raw, "answer": extract_answer(raw), "tokens": resp.usage.total_tokens, "error": None}
        except Exception as e: return {"raw": "", "answer": "", "tokens": 0, "error": str(e)}

class ClaudeEval:
    def __init__(self):
        self.client = ant.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.name = "Claude"
        self.candidates = [
            "claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001",
            "claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022"
        ]
        self.active_model = None

    def _find_model(self):
        if self.active_model: return self.active_model
        print("\n🔍 Đang tìm model Claude khả dụng...")
        for m in self.candidates:
            try:
                self.client.messages.create(model=m, max_tokens=1, messages=[{"role": "user", "content": "hi"}])
                self.active_model = m
                print(f"✅ Kết nối thành công: {m}")
                return m
            except Exception: continue
        raise RuntimeError("❌ Không tìm thấy model Claude khả dụng.")

    def call(self, prompt: str, img_b64: str) -> dict:
        try:
            model = self._find_model()
            resp = self.client.messages.create(
                model=model, max_tokens=1024,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            raw = resp.content[0].text
            return {"raw": raw, "answer": extract_answer(raw), 
                    "tokens": resp.usage.input_tokens + resp.usage.output_tokens, "error": None}
        except Exception as e: return {"raw": "", "answer": "", "tokens": 0, "error": str(e)}

class LocalVLMEval:
    def __init__(self, model_id, name):
        self.model_id, self.name = model_id, name

    def call(self, prompt, img_b64):
        return {"raw": "[LOCAL_STUB]", "answer": "", "tokens": 0, "error": "Local model stub"}

# ── Loaders ───────────────────────────────────────────────
def load_charxiv(n, split):
    print(f"Đang nạp CharXiv {split} (n={n})...")
    ds = load_dataset("princeton-nlp/CharXiv", split=split)
    return [s for s in ds if s.get("reasoning_q")][:n]

def load_chartqa(n, split):
    print(f"Đang nạp ChartQA {split} (n={n})...")
    path = Path(f"data/chartqa/{split}_human.json")
    if not path.exists(): return []
    with open(path) as f: return json.load(f)[:n]

# ── Main Loop ─────────────────────────────────────────────
def evaluate_model(evaluator, samples, benchmark, output_file, delay=0.5):
    results, done_ids = [], set()
    correct, errors = 0, 0
    
    if output_file.exists():
        with open(output_file) as f:
            results = json.load(f)
            done_ids = {str(r["id"]) for r in results}
            correct = sum(1 for r in results if r["correct"])
            print(f"  Chạy tiếp từ {len(results)} mẫu.")

    pbar = tqdm(enumerate(samples), total=len(samples), desc=f"{evaluator.name} / {benchmark}")
    for i, s in pbar:
        sample_id = str(s.get("id") or s.get("original_id") or s.get("imgname") or i)
        if sample_id in done_ids: continue
        
        # Image
        img_b64 = None
        if "image" in s: img_b64 = pil_to_b64(s["image"])
        elif "imgname" in s: 
            img_path = Path(f"data/chartqa/test/png") / s["imgname"]
            if img_path.exists(): img_b64 = path_to_b64(str(img_path))
        
        if not img_b64:
            errors += 1; continue

        if benchmark == "charxiv":
            gold = s["reasoning_a"]
            prompt = make_charxiv_reasoning_prompt(s["reasoning_q"])
        else:
            gold = s["label"]
            prompt = make_chartqa_prompt(s["query"])

        output = evaluator.call(prompt, img_b64)
        is_correct = relaxed_match(output["answer"], gold) if not output["error"] else False
        if is_correct: correct += 1

        results.append({
            "id": sample_id, "gold": gold, "prediction": output["answer"],
            "raw_output": output["raw"][:1000], "correct": is_correct, "error": output["error"]
        })
        with open(output_file, "w") as f: json.dump(results, f, indent=2, ensure_ascii=False)
        pbar.set_postfix({"acc": f"{correct/len(results)*100:.1f}%"})
        if delay > 0: time.sleep(delay)

    acc = round(correct/len(results)*100, 2) if results else 0
    summary = {"model": evaluator.name, "benchmark": benchmark, "accuracy": acc, "n": len(results), "ts": datetime.now().isoformat()}
    with open(output_file.with_suffix(".summary.json"), "w") as f: json.dump(summary, f, indent=2)
    return summary

def get_evaluator(model):
    m_map = {
        "gpt4o": lambda: GPT4oEval(), "claude": lambda: ClaudeEval(),
        "qwen": lambda: LocalVLMEval("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B"),
        "tiny": lambda: LocalVLMEval("mPLUG/TinyChart-3B-768", "TinyChart-3B"),
        "chartmoe": lambda: LocalVLMEval("tangxue/ChartMoE", "ChartMoE"),
    }
    return m_map[model]()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt4o", choices=["gpt4o","claude","qwen","tiny","chartmoe","all"])
    parser.add_argument("--benchmark", default="charxiv", choices=["charxiv","chartqa","both"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    models = ["gpt4o","claude","qwen","tiny","chartmoe"] if args.model == "all" else [args.model]
    benchmarks = ["charxiv","chartqa"] if args.benchmark == "both" else [args.benchmark]
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    for m in models:
        evaluator = get_evaluator(m)
        for b in benchmarks:
            samples = load_charxiv(args.n, args.split) if b == "charxiv" else load_chartqa(args.n, "test")
            if not samples: continue
            out_file = RESULTS_DIR / f"{m}_{b}_{ts}.json"
            evaluate_model(evaluator, samples, b, out_file)

if __name__ == "__main__":
    main()