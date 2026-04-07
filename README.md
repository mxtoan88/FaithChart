<div align="center">

# FaithChart

### A Faithfulness Evaluation Framework for Explainable Chart Question Answering

[![Paper](https://img.shields.io/badge/Paper-Information%20Fusion-blue)](https://github.com/mxtoan/faithchart)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-FaithChart--1500-yellow)](https://huggingface.co/datasets/mxtoan/FaithChart-1500)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)

**Xuan Toan Mai · Hong Tai Tran · Tuan-Anh Tran***

*Faculty of Computer Science & Engineering, HCMUT, VNU-HCM, Vietnam*

`mxtoan@hcmut.edu.vn` · `thtai@hcmut.edu.vn` · `trtanh@hcmut.edu.vn`

</div>

---

## Overview

Charts require simultaneous **visual perception** and **numerical reasoning** — yet when multimodal LLMs answer chart questions correctly, it remains unclear whether their explanations *faithfully* reflect the visual evidence or are merely plausible post-hoc rationalizations.

**FaithChart** is the first framework to quantitatively evaluate explanation faithfulness in chart question answering. We introduce:

- **FaithChart-1500**: a benchmark of 1,498 chart-QA samples with LLM-generated reasoning traces
- **Sufficiency & Comprehensiveness**: two causal metrics via chart-region perturbation
- **FaithChart-B**: a faithfulness-trained baseline demonstrating that faithfulness is optimizable

### Key Finding

> Pearson r(accuracy, FaithScore) = **+0.13** — accuracy does not predict faithfulness.
> Models achieving high accuracy may rely on visual evidence they never acknowledge.

---

## Results

### CharXiv Reasoning Accuracy (200 samples)

| Model | Accuracy | Gap vs Human |
|-------|:--------:|:------------:|
| Human | 80.5% | — |
| Claude Sonnet 4.6 | **69.0%** | −11.5pp |
| GPT-4o | 51.5% | −29.0pp |
| Qwen2.5-VL-7B | 45.0% | −35.5pp |
| TinyChart-3B | 17.5% | −63.0pp ⚠️ |

TinyChart-3B suffers a **66-point collapse** from ChartQA (83.6%) to CharXiv (17.5%) — the largest distribution shift observed. 72% of its outputs are single-token guesses (mean 8 characters).

### Faithfulness Evaluation (300 samples)

| Model | Sufficiency ↑ | Comprehensiveness ↑ | FaithScore ↑ | Pattern |
|-------|:-------------:|:-------------------:|:------------:|:-------:|
| Claude Sonnet 4.6 | **0.747** | 0.269 | 0.508 | A |
| GPT-4o | 0.669 | 0.298 | 0.484 | A |
| Qwen2.5-VL-7B | 0.471 | **0.555** | **0.513** | B |
| TinyChart-3B | 0.531 | 0.467 | 0.499 | C |
| FaithChart-B *(ours)* | 0.481 | 0.491 | 0.486 | C |

**Pattern A** (API models): cite relevant regions but systematically omit others — high Sufficiency, low Comprehensiveness.
**Pattern B** (Qwen): exhaustive but less selective — Comprehensiveness > Sufficiency.
**Pattern C** (TinyChart, FaithChart-B): balanced Suf ≈ Comp.

FaithChart-B achieves a **Suf–Comp gap of 0.010** vs Qwen base 0.084 — an **88% reduction** in imbalance through faithfulness-aware training.

---

## Dataset: FaithChart-1500

```
🤗 https://huggingface.co/datasets/mxtoan/FaithChart-1500
```

| Split | Source | Samples | Difficulty |
|-------|--------|:-------:|:----------:|
| Standard | ChartQA (human test split) | 750 | Journalistic charts |
| Hard | CharXiv (reasoning validation) | 748 | Scientific arXiv charts |
| **Total** | | **1,498** | |

**Annotation:** Dual-LLM protocol — GPT-4o generates reasoning traces; Claude Sonnet 4.6 cross-validates 10% (Jaccard = 0.76). Each sample includes:
- Question + gold answer
- Reasoning trace (mean 462 chars)
- Cited chart regions with necessity labels (mean 2.76 regions/sample)
- 2.5 Sufficiency + 2.5 Comprehensiveness perturbation variants

---

## Framework

### Faithfulness Metrics

**Sufficiency** — removing a cited region should change the answer if it genuinely supported it:

$$\text{Suf}(E,C,Q) = \frac{1}{|R|} \sum_{r_i \in R} \mathbb{1}[f(C \setminus r_i, Q) \neq f(C, Q)]$$

**Comprehensiveness** — removing uncited regions should not change the answer if the explanation is complete:

$$\text{Comp}(E,C,Q) = \frac{1}{|U|} \sum_{u_j \in U} \mathbb{1}[f(C \setminus u_j, Q) = f(C, Q)]$$

**FaithScore** = (Suf + Comp) / 2 ∈ [0, 1]

### Perturbation

Gaussian blur (σ=15px, 8 passes) applied to 5 region types: `data_values`, `axis_labels`, `legend`, `title`, `background_gridlines`. Average of 2.5 Suf variants + 2.5 Comp variants per sample (~7,500 inference calls per evaluated model).

---

## Repository Structure

```
faithchart/
├── README.md
├── LICENSE
│
├── dataset/
│   ├── README.md                    # Dataset card
│   └── faithchart_1500_meta.json    # Sample metadata (images on HuggingFace)
│
├── evaluation/
│   ├── month2_03_perturbation.py    # Generate perturbed chart images
│   └── 04_faithfulness_scoring.py   # Compute Suf/Comp/FaithScore for any model
│
├── training/
│   └── train_faithchart_b.py        # FaithChart-B fine-tuning (LoRA on Qwen2-VL-2B)
│
└── results/
    ├── faith_claude_summary.json
    ├── faith_gpt4o_summary.json
    ├── faith_qwen_summary.json
    ├── faith_tinychart_summary.json
    ├── faith_faithchart-b_summary.json
    └── ablation/
        ├── results_none_final.json
        ├── results_suf_only.json
        ├── results_comp_only.json
        └── results_n_perturb_1.json
```

---

## Installation

```bash
git clone https://github.com/mxtoan/faithchart
cd faithchart
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0
transformers>=4.40
anthropic
openai
Pillow
opencv-python
numpy
scipy
tqdm
datasets
```

---

## Usage

### 1. Compute faithfulness for a custom model

```python
from evaluation.faithfulness_scoring import score_model

results = score_model(
    model_fn=your_model_function,   # fn(image_b64, question) -> str
    dataset_path="faithchart_1500_meta.json",
    perturbation_dir="faithchart_perturbations/",
    n_samples=300,
    subset="mixed"                  # "chartqa", "charxiv", or "mixed"
)

print(f"Sufficiency:       {results['sufficiency']:.3f}")
print(f"Comprehensiveness: {results['comprehensiveness']:.3f}")
print(f"FaithScore:        {results['faith_score']:.3f}")
```

### 2. Evaluate GPT-4o or Claude

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run faithfulness scoring
python evaluation/04_faithfulness_scoring.py \
    --model gpt4o \
    --n 300 \
    --subset mixed

python evaluation/04_faithfulness_scoring.py \
    --model claude \
    --n 300
```

### 3. Run on open-source models (GPU required)

```bash
# Qwen2.5-VL-7B (requires ~16GB VRAM)
python evaluation/04_faithfulness_scoring.py \
    --model qwen \
    --n 300

# TinyChart-3B (requires ~8GB VRAM)
python evaluation/04_faithfulness_scoring.py \
    --model tinychart \
    --n 300
```

### 4. Fine-tune FaithChart-B

```bash
# Requires RTX 3090 or equivalent (24GB VRAM)
# Expects faithchart_sft_train.jsonl in working directory

python training/train_faithchart_b.py
```

FaithChart-B output format:
```
Thought: <reasoning_trace>
Cited Regions: [data_values, axis_labels, legend]
Answer: <answer>
```

### 5. Generate perturbations for new charts

```bash
python evaluation/month2_03_perturbation.py \
    --input all_annotations.json \
    --output faithchart_perturbations/ \
    --sigma 15 \
    --passes 8
```

---

## Ablation Results

| Condition | Sufficiency | Comprehensiveness | FaithScore | Δ |
|-----------|:-----------:|:-----------------:|:----------:|:-:|
| Full system (n≈2.5) | 0.521 | 0.448 | **0.484** | — |
| Suf only | 0.521 | — | 0.521 | +0.037 *(inflated)* |
| Comp only | — | 0.448 | 0.448 | −0.037 |
| n_perturb=1 | 0.443 | 0.323 | 0.383 | **−20.9%** |

Single perturbation (n=1) degrades FaithScore by **20.9%** — multiple perturbations per region are essential for stable estimation.

LLM-as-judge correlation: Spearman r(GPT-4o judge, FaithScore) = −0.106 (p=0.066) — LLM judges favor fluency over visual grounding, validating the need for perturbation-based evaluation.

---

## Citation

If you use FaithChart in your research, please cite:

```bibtex
@article{mai2026faithchart,
  title     = {FaithChart: A Faithfulness Evaluation Framework for
               Explainable Chart Question Answering},
  author    = {Mai, Xuan Toan and Tran, Hong Tai and Tran, Tuan-Anh},
  journal   = {Information Fusion},
  year      = {2026},
  publisher = {Elsevier},
  note      = {Under review}
}
```

---

## Acknowledgments

We acknowledge Ho Chi Minh City University of Technology (HCMUT), VNU-HCM for supporting this study.

This work builds on [CharXiv](https://github.com/princeton-nlp/charxiv) (Wang et al., NeurIPS 2024) and [ERASER](https://github.com/jayded/eraserbenchmark) (DeYoung et al., ACL 2020).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The **FaithChart-1500 dataset** is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It includes samples derived from [ChartQA](https://github.com/vis-nlp/ChartQA) and [CharXiv](https://github.com/princeton-nlp/charxiv) — please cite those works as well when using the dataset.
