# FaithChart — Tuần 3 & 4: Evaluation Pipeline

## Tổng quan

```
scripts/
├── 01_run_eval.py        # Chạy 5 models trên CharXiv + ChartQA
├── 02_error_analysis.py  # Phân loại lỗi theo 6 error types
├── 03_faithfulness.py    # Tính Sufficiency + Comprehensiveness
└── README.md             # File này

results/                  # Auto-created khi chạy 01_run_eval.py
data/
├── charxiv/              # Tải từ HuggingFace (tự động)
└── chartqa/              # Tải từ github.com/vis-nlp/ChartQA
```

---

## Tuần 3 — Chạy evaluation

### Bước 0: Cài môi trường (1 lần)

```bash
pip install openai anthropic datasets Pillow numpy tqdm transformers
pip install accelerate bitsandbytes  # cho local models (Qwen, TinyChart)
```

### Bước 1: Set API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Hoặc sửa trực tiếp trong `01_run_eval.py` dòng:
```python
OPENAI_API_KEY  = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

### Bước 2: Chạy GPT-4o và Claude (API models)

```bash
# GPT-4o trên 200 CharXiv reasoning samples (~$40, ~30 phút)
python 01_run_eval.py --model gpt4o --benchmark charxiv --n 200

# Claude 3.5 trên 200 samples (~$30, ~25 phút)
python 01_run_eval.py --model claude --benchmark charxiv --n 200

# Cả hai benchmark cùng lúc
python 01_run_eval.py --model gpt4o --benchmark both --n 200
```

**Nếu bị ngắt giữa chừng:** chạy lại y hệt — script tự động resume từ checkpoint.

### Bước 3: Chạy local models (GPU)

**Qwen2.5-VL-7B** (cần ~16GB VRAM):
```bash
pip install qwen-vl-utils
# Sửa class LocalVLMEval.call() trong 01_run_eval.py
# Uncomment phần Qwen code, rồi chạy:
python 01_run_eval.py --model qwen --benchmark charxiv --n 200 --delay 0
```

**TinyChart-3B** (cần ~8GB VRAM):
```bash
# Model ID: mPLUG/TinyChart-3B-768
# Xem: github.com/X-PLUG/mPLUG-DocOwl
python 01_run_eval.py --model tiny --benchmark charxiv --n 200 --delay 0
```

**ChartMoE** (cần ~14GB VRAM):
```bash
# Model ID: tangxue/ChartMoE
python 01_run_eval.py --model chartmoe --benchmark charxiv --n 200 --delay 0
```

### Bước 4: Phân tích lỗi

```bash
# Dùng rule-based classifier (free)
python 02_error_analysis.py --results_dir results/

# Dùng GPT-4o-mini làm judge (chính xác hơn, ~$0.002/sample = ~$0.4 cho 200 samples)
python 02_error_analysis.py --results_dir results/ --use_gpt_judge
```

---

## Ước tính chi phí API (200 samples/model)

| Model         | Benchmark     | Ước tính chi phí | Thời gian |
|---------------|---------------|------------------|-----------|
| GPT-4o        | CharXiv       | ~$35-45          | 30 phút   |
| GPT-4o        | ChartQA       | ~$25-35          | 20 phút   |
| Claude 3.5    | CharXiv       | ~$25-35          | 25 phút   |
| Claude 3.5    | ChartQA       | ~$20-25          | 20 phút   |
| Qwen2.5-VL    | CharXiv       | $0 (local)       | 45 phút   |
| TinyChart     | CharXiv       | $0 (local)       | 40 phút   |
| ChartMoE      | CharXiv       | $0 (local)       | 50 phút   |
| Error judge   | All results   | ~$1-2            | 10 phút   |
| **Total**     |               | **~$120-180**    | ~4 giờ   |

---

## Tuần 4 — Faithfulness metrics

### Bước 1: Demo perturbation trên một chart

```bash
python 03_faithfulness.py --demo data/chartqa/test/png/sample_chart.png
# Xem thư mục perturb_demo/ để kiểm tra chất lượng perturbation
```

### Bước 2: Chọn strategy phù hợp

| Strategy   | Mô tả                          | Khi nào dùng              |
|------------|--------------------------------|---------------------------|
| `blur`     | Gaussian blur vùng bị xóa      | Default, tự nhiên nhất    |
| `blackout` | Fill đen                       | Khi muốn xóa hoàn toàn   |
| `whiteout` | Fill trắng                     | Chart nền trắng           |
| `mean_fill`| Fill màu trung bình của chart   | Least disruptive           |

### Bước 3: Integrate faithfulness scoring vào pipeline

```python
from 01_run_eval import ClaudeEval
from 03_faithfulness import ChartPerturber, FaithfulnessScorer, compute_faithfulness_report

# Setup
evaluator = ClaudeEval()
perturber = ChartPerturber(strategy="blur")

def eval_fn(question, image_b64):
    from 01_run_eval import make_charxiv_reasoning_prompt
    result = evaluator.call(make_charxiv_reasoning_prompt(question), image_b64)
    return result["answer"]

scorer = FaithfulnessScorer(eval_fn, perturber)

# Score một sample
sample = {
    "question": "Which year had the highest value?",
    "prediction": "2020",
    "raw_output": "Looking at the bar chart, the year 2020 shows the tallest bar...",
    "_image": pil_image,  # PIL Image object
}
suf = scorer.score_sufficiency(sample, n_perturb=3)
comp = scorer.score_comprehensiveness(sample, n_perturb=3)
print(f"Sufficiency: {suf:.2f}, Comprehensiveness: {comp:.2f}")
```

---

## Output files

Sau khi chạy xong tuần 3–4, bạn sẽ có:

```
results/
├── gpt4o_charxiv_YYYYMMDD_HHMM.json        # Raw results
├── gpt4o_charxiv_YYYYMMDD_HHMM.summary.json
├── claude_charxiv_YYYYMMDD_HHMM.json
├── ...
├── error_taxonomy_YYYYMMDD.json             # Error classification
├── error_report_YYYYMMDD.csv               # Error distribution table
└── faithfulness_report_YYYYMMDD.json       # Sufficiency + Comprehensiveness
```

---

## Reference scores từ leaderboard (nếu không chạy được)

| Model             | CharXiv Desc | CharXiv Reason | ChartQA  | ChartQAPro |
|-------------------|:------------:|:--------------:|:--------:|:----------:|
| Human             | 92.1%        | 80.5%          | ~97%     | —          |
| Claude 3.5 Sonnet | 88.5%        | 60.2%          | 90.5%    | 55.8%      |
| GPT-4o            | 84.4%        | 47.1%          | 85.7%    | —          |
| Qwen2.5-VL 7B     | 76.3%        | 42.8%          | 83.5%    | 37.2%      |
| ChartMoE          | 65.2%        | 33.1%          | 84.6%    | —          |
| TinyChart 3B      | 59.1%        | 28.4%          | 83.6%    | —          |

**Gap reasoning: GPT-4o vs Human = 33.4pp**
**Gap best open-source vs Human = 47.4pp (InternVL V1.5 ở 29.2%)**

---

## Troubleshooting

**CharXiv không load được từ HuggingFace:**
```bash
# Manual download
git clone https://github.com/princeton-nlp/CharXiv
# Copy val_qa.json vào data/charxiv/
```

**CUDA out of memory cho local models:**
```python
# Trong LocalVLMEval.call(), thêm:
model = AutoModel.from_pretrained(..., load_in_4bit=True)  # 4-bit quantization
```

**API rate limit:**
```bash
# Tăng delay:
python 01_run_eval.py --model gpt4o --benchmark charxiv --n 200 --delay 2.0
```
