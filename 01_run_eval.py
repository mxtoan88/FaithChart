"""
FaithChart — Script 1: Evaluation Runner (Bản tối ưu hóa cao)
===========================================================
Chạy đánh giá GPT-4o trên CharXiv với logic chấm điểm linh hoạt:
- Trích xuất đáp án thông minh (tìm từ cuối lên).
- Chuẩn hóa LaTeX, đơn vị, và các chú thích phụ (màu sắc, vị trí).
- Hỗ trợ so khớp từ khóa (Keyword Subset Match).

Yêu cầu:
  pip install openai anthropic datasets Pillow tqdm
"""

import os, json, base64, argparse, time, re
from pathlib import Path
from datetime import datetime
from io import BytesIO

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

# Thiết lập API Keys
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Hàm đánh giá linh hoạt (Relaxed Match) ──────────────────
def relaxed_match(pred: str, gold: str, tol: float = 0.05) -> bool:
    """
    So khớp đáp án giữa AI và Ground Truth với các quy tắc nới lỏng.
    """
    if not pred or not gold:
        return False
        
    def normalize(s):
        s = str(s).strip().lower()
        # 1. Loại bỏ định dạng LaTeX (Ví dụ: \(\lambda\) -> lambda)
        s = re.sub(r"\\\(|\\\)|\\\[|\\\]|\$|\\", "", s)
        # 2. Loại bỏ các tiền tố mô tả biểu đồ phổ biến
        s = re.sub(r"^(subplot|figure|fig|panel|the categories are|categories|category|variable|condition|is)\s+", "", s)
        # 3. Xử lý ngoặc đơn thông minh: 
        # Giữ lại các nhãn biểu đồ con như (a), (b), (1) nhưng xóa các chú thích dài (green line)
        s = re.sub(r"\(([a-z0-9])\)", r"\1", s) 
        s = re.sub(r"\(.*?\)", "", s)           
        # 4. Chuẩn hóa dấu câu và từ nối (Biến chúng thành khoảng trắng để so khớp từ vựng)
        s = s.replace(",", " ").replace(";", " ").replace(":", " ").replace("+", " ").replace("/", " ")
        s = re.sub(r"\s+(and|&|with|including|in|at|of|for|shows|a|an|the|line|color|colored)\s+", " ", s)
        # 5. Thu gọn khoảng trắng thừa và xóa dấu chấm cuối câu
        s = re.sub(r"\s+", " ", s).strip()
        if s.endswith("."): s = s[:-1]
        return s.strip()

    pred_norm = normalize(pred)
    gold_norm = normalize(gold)
    
    # Chiến thuật 1: So khớp trực tiếp sau khi đã làm sạch
    if pred_norm == gold_norm:
        return True
    
    # Chiến thuật 2: So khớp tập hợp từ khóa (Keyword Subset)
    # Rất quan trọng khi AI trả lời đầy đủ câu hoặc thêm chú thích
    pred_words = set(pred_norm.split())
    gold_words = set(gold_norm.split())
    
    if gold_words and gold_words.issubset(pred_words):
        # Nếu Gold không phải số, cho phép AI trả lời dài (tối đa 15 từ)
        is_numeric_gold = bool(re.search(r"\d", gold_norm)) and len(gold_words) <= 1
        if not is_numeric_gold:
            if len(pred_words) <= 15: return True 
        elif len(pred_words) <= len(gold_words) + 3:
            return True

    # Chiến thuật 3: So sánh số liệu (Numeric comparison)
    def clean_val(v):
        return re.sub(r"[^\d\.\-]", "", v)

    try:
        p_val = clean_val(pred_norm)
        g_val = clean_val(gold_norm)
        if p_val and g_val:
            p, g = float(p_val), float(g_val)
            if g == 0: return abs(p - g) <= tol
            return abs(p - g) / max(abs(g), 1e-9) <= tol
    except ValueError:
        pass
        
    # Chiến thuật 4: Loại bỏ đơn vị để so khớp cuối cùng
    for tok in ["%", "percentage", "million", "billion", "thousand", "k", "m", "b", "$", "€"]:
        pred_norm = pred_norm.replace(tok, "").strip()
        gold_norm = gold_norm.replace(tok, "").strip()
        
    return pred_norm == gold_norm

def extract_answer(raw: str) -> str:
    """
    Trích xuất câu trả lời cuối cùng từ chuỗi lập luận dài của mô hình.
    Ưu tiên tìm các từ khóa 'Final Answer' hoặc lấy dòng cuối cùng.
    """
    raw = raw.strip()
    # Tìm tag đáp án, ưu tiên kết quả cuối cùng trong văn bản
    matches = list(re.finditer(r"(?i)(?:final answer|answer|conclusion|result)\s*[:\-]\s*(.+?)(?:\n|$)", raw))
    if matches:
        return matches[-1].group(1).strip()
    # Tìm cấu trúc "The answer is X"
    matches = list(re.finditer(r"(?i)(?:the answer is|answer is)\s+(.+?)(?:\.|$)", raw))
    if matches:
        return matches[-1].group(1).strip()
    # Nếu không có tag, lấy dòng cuối cùng (thường là kết luận)
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # Nếu dòng cuối quá dài, lấy phần sau dấu chấm cuối
        if len(last_line) > 100 and "." in last_line:
            parts = [p.strip() for p in last_line.split('.') if p.strip()]
            if parts: return parts[-1]
        return last_line
    return raw[:200]

# ── Prompt Templates ──────────────────────────────────────
def make_charxiv_reasoning_prompt(question: str) -> str:
    """Tạo prompt yêu cầu suy luận từng bước cho CharXiv."""
    return (
        "You are an expert at reading and reasoning about scientific charts. "
        "Answer the following question based on the chart image provided. "
        "Think step by step, then state your final answer clearly.\n\n"
        f"Question: {question}\n\n"
        "Provide your reasoning, then end your response with exactly this format:\n"
        "Final Answer: <your brief answer>"
    )

def make_chartqa_prompt(question: str) -> str:
    """Tạo prompt ngắn gọn cho ChartQA."""
    return (
        "Answer the question about the chart. "
        "Be concise — give only the value, label, or short phrase as the answer.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

# ── Model Callers ─────────────────────────────────────────
class GPT4oEval:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o"
        self.name = "GPT-4o"

    def call(self, prompt: str, image_b64: str, media: str = "image/png") -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media};base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]}],
                max_tokens=1024,
                temperature=0,
            )
            raw = resp.choices[0].message.content
            return {"raw": raw, "answer": extract_answer(raw), "tokens": resp.usage.total_tokens, "error": None}
        except Exception as e:
            return {"raw": "", "answer": "", "tokens": 0, "error": str(e)}

class ClaudeEval:
    def __init__(self, model="claude-3-5-sonnet-20240620"):
        self.client = ant.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model
        self.name = f"Claude ({model})"

    def call(self, prompt: str, image_b64: str, media: str = "image/png") -> dict:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": image_b64}},
                    {"type": "text", "text": prompt}
                ]}],
            )
            raw = resp.content[0].text
            return {"raw": raw, "answer": extract_answer(raw), "tokens": resp.usage.input_tokens + resp.usage.output_tokens, "error": None}
        except Exception as e:
            return {"raw": "", "answer": "", "tokens": 0, "error": str(e)}

# ── Dataloaders ───────────────────────────────────────
def load_charxiv_reasoning(n: int = 200, split: str = "validation") -> list:
    """Tải dữ liệu CharXiv từ Hugging Face."""
    if split == "val": split = "validation"
    print(f"Đang tải CharXiv {split} (chỉ reasoning, n={n})...")
    try:
        ds = load_dataset("princeton-nlp/CharXiv", split=split)
        samples = []
        for s in ds:
            # Ưu tiên lấy cột reasoning nếu có
            q = s.get("reasoning_q") or s.get("question")
            a = s.get("reasoning_a") or s.get("answer")
            if q:
                s["question"] = q
                s["answer"] = a
                samples.append(s)
            if len(samples) >= n: break
        return samples
    except Exception as e:
        print(f"  Lỗi tải dataset: {e}")
        return []

def load_chartqa_human(n: int = 200, split: str = "test") -> list:
    """Tải dữ liệu ChartQA từ file cục bộ."""
    json_path = Path("data/chartqa") / f"{split}_human.json"
    if not json_path.exists(): return []
    with open(json_path) as f: data = json.load(f)
    return data[:n]

# ── Vòng lặp đánh giá chính ──────────────────────────────────
def evaluate_model(evaluator, samples: list, benchmark: str, output_file: Path, delay: float = 0.5) -> dict:
    results = []
    correct = 0
    errors = 0
    
    # Kiểm tra xem có file cũ không để chạy tiếp (resume)
    if output_file.exists():
        with open(output_file) as f: results = json.load(f)
        done_ids = {str(r["id"]) for r in results}
        correct = sum(1 for r in results if r.get("correct"))
        print(f"  Tiếp tục từ {len(results)} mẫu đã có kết quả.")
    else:
        done_ids = set()

    pbar = tqdm(samples, desc=f"{evaluator.name} / {benchmark}")
    for sample in pbar:
        sample_id = str(sample.get("id") or sample.get("original_id") or sample.get("imgname") or len(results))
        if sample_id in done_ids: continue
        
        # Chuyển đổi ảnh sang Base64
        image_b64 = None
        if "image" in sample and sample["image"]:
            if hasattr(sample["image"], "save"): 
                image_b64 = pil_to_b64(sample["image"])
            elif isinstance(sample["image"], str) and os.path.exists(sample["image"]): 
                image_b64 = path_to_b64(sample["image"])
        
        if image_b64 is None:
            errors += 1
            continue
            
        # Chuẩn bị Prompt và đáp án gốc
        if benchmark == "charxiv":
            gold = sample.get("answer", "")
            prompt = make_charxiv_reasoning_prompt(sample.get("question", ""))
        else:
            gold = sample.get("label", "")
            prompt = make_chartqa_prompt(sample.get("query", ""))
            
        # Gọi API mô hình
        output = evaluator.call(prompt, image_b64)
        
        # Chấm điểm
        is_correct = False
        if output["error"] is None:
            is_correct = relaxed_match(output["answer"], gold)
        else:
            errors += 1
            
        if is_correct: correct += 1
        
        # Lưu kết quả tạm thời
        results.append({
            "id": sample_id, 
            "question": sample.get("question") or sample.get("query"),
            "gold": gold, 
            "prediction": output["answer"], 
            "raw_output": output["raw"],
            "correct": is_correct, 
            "tokens": output["tokens"], 
            "error": output["error"]
        })
        
        with open(output_file, "w") as f: 
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        pbar.set_postfix({"acc": f"{correct/len(results)*100:.1f}%", "err": errors})
        if delay > 0: time.sleep(delay)
        
    # Tính toán tổng kết
    accuracy = round(correct/len(results)*100, 2) if results else 0
    summary = {
        "model": evaluator.name, 
        "benchmark": benchmark, 
        "accuracy": accuracy, 
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_file.with_suffix(".summary.json"), "w") as f: 
        json.dump(summary, f, indent=2)
        
    return summary

def pil_to_b64(img: "Image.Image", fmt="PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def path_to_b64(path: str) -> str:
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode()

def get_evaluator(model: str):
    model_map = {"gpt4o": lambda: GPT4oEval(), "claude": lambda: ClaudeEval()}
    return model_map[model]()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt4o")
    parser.add_argument("--benchmark", default="charxiv")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    
    evaluator = get_evaluator(args.model)
    samples = load_charxiv_reasoning(args.n) if args.benchmark == "charxiv" else load_chartqa_human(args.n)
    
    if not samples: 
        print("Không có mẫu nào được nạp. Vui lòng kiểm tra lại thiết lập.")
        return
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = RESULTS_DIR / f"{args.model}_{args.benchmark}_{ts}.json"
    evaluate_model(evaluator, samples, args.benchmark, out_file, args.delay)

if __name__ == "__main__":
    main()