#!/usr/bin/env python3
"""
Faithfulness Scoring
==========================================
Calculate Sufficiency + Comprehensiveness + FaithScore for 4 models.

Run:
  python faithfulness_scoring.py --model gpt4o --n 300
  python faithfulness_scoring.py --model claude --n 300
  python faithfulness_scoring.py --model qwen --n 300
  python faithfulness_scoring.py --model tinychart --n 300
  python faithfulness_scoring.py --summary
"""

import os, sys, json, re, time, argparse
from pathlib import Path
from datetime import datetime

PERTURB_DIR = Path("faithchart_perturbations")
ANNOT_FILE  = Path("faithchart_annotations/all_annotations.json")
OUTPUT_DIR  = Path("faithchart_faith_scores")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_EVERY = 10

def relaxed_match(pred, gold, tol=0.05):
    if not pred or not gold: return False
    def norm(s):
        s = str(s).strip().lower()
        s = re.sub(r"[\\$\(\)\[\]]", "", s)
        return re.sub(r"\s+", " ", s).strip().rstrip(".")
    p, g = norm(pred), norm(gold)
    if p == g: return True
    if g and g in p and len(p.split()) <= len(g.split()) + 5: return True
    try:
        pv = float(re.sub(r"[^\d.\-]", "", p))
        gv = float(re.sub(r"[^\d.\-]", "", g))
        return abs(pv-gv)/max(abs(gv),1e-9) <= tol if gv != 0 else abs(pv) <= tol
    except: return False

def extract_answer(raw):
    for pat in [r"(?i)final answer\s*[:\-]*\s*(.+?)(?:\n|$)",
                r"(?i)answer\s*:\s*(.+?)(?:\n|$)"]:
        m = re.search(pat, str(raw).strip())
        if m: return re.sub(r"[*'\"` ]", "", m.group(1)).strip()
    lines = [l.strip() for l in str(raw).split('\n') if l.strip()]
    return lines[-1][:80] if lines else str(raw)[:80]

# ── Model call functions ──────────────────────────────────
def call_gpt4o(img_b64, question):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model="gpt-4o", max_tokens=128,
        messages=[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":f"data:image/png;base64,{img_b64}","detail":"low"}},
            {"type":"text","text":f"Answer briefly.\nQuestion: {question}\nAnswer:"}]}])
    return r.choices[0].message.content.strip()

def call_claude(img_b64, question):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=128,
        messages=[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":img_b64}},
            {"type":"text","text":f"Answer briefly.\nQuestion: {question}\nAnswer:"}]}])
    return r.content[0].text.strip()

def load_model(model_name):
    if model_name == "gpt4o":
        print("GPT-4o API ready")
        return lambda img, q: call_gpt4o(img, q), None
    if model_name == "claude":
        print("Claude Sonnet 4.6 API ready")
        return lambda img, q: call_claude(img, q), None
    if model_name == "qwen":
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        os.environ.setdefault("HF_HOME", "/tmp/.hf_cache")
        M = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"Loading {M}...")
        proc = AutoProcessor.from_pretrained(M, min_pixels=256*28*28, max_pixels=1280*28*28)
        mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            M, torch_dtype=torch.bfloat16, device_map="auto")
        mdl.eval()
        def qwen_call(img, q, m=mdl, p=proc):
            msgs = [{"role":"user","content":[
                {"type":"image","image":f"data:image/png;base64,{img}"},
                {"type":"text","text":f"Answer briefly.\nQuestion: {q}\nAnswer:"}]}]
            text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ii, vi = process_vision_info(msgs)
            inp = p(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(m.device)
            with torch.no_grad():
                out = m.generate(**inp, max_new_tokens=64, do_sample=False)
            tr = [o[len(i):] for i, o in zip(inp.input_ids, out)]
            return p.batch_decode(tr, skip_special_tokens=True)[0].strip()
        return qwen_call, mdl
    if model_name == "tinychart":
        import torch, base64 as b64m
        from PIL import Image; from io import BytesIO
        sys.path.insert(0, "/workspace/mPLUG-DocOwl/TinyChart")
        from tinychart.model.builder import load_pretrained_model
        from tinychart.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from tinychart.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        os.environ.setdefault("HF_HOME", "/tmp/.hf_cache")
        M = "mPLUG/TinyChart-3B-768"
        print(f"Loading {M}...")
        tok, mdl, ip, _ = load_pretrained_model(M, None, get_model_name_from_path(M), device="cuda")
        mdl.eval()
        def tiny_call(img, q, m=mdl, t=tok, ipr=ip):
            image = Image.open(BytesIO(b64m.b64decode(img))).convert("RGB")
            it = process_images([image], ipr, m.config)[0].unsqueeze(0).half().cuda()
            qq = DEFAULT_IMAGE_TOKEN + "\n" + q + "\nAnswer:"
            iids = tokenizer_image_token(qq, t, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            with torch.inference_mode():
                oids = m.generate(iids, images=it, do_sample=False, max_new_tokens=64)
            return t.batch_decode(oids, skip_special_tokens=True)[0].strip()
        return tiny_call, mdl
    raise ValueError(f"Unknown model: {model_name}")

def score_sample(perturb_data, call_fn, question, baseline_ans, gold_ans):
    res = {
        "global_id": perturb_data["global_id"],
        "baseline_answer": baseline_ans,
        "baseline_correct": relaxed_match(baseline_ans, gold_ans),
        "sufficiency_details": [], "comprehensiveness_details": [],
        "sufficiency": 0.0, "comprehensiveness": 0.0, "faith_score": 0.0,
    }
    # Sufficiency: remove CITED region → answer should CHANGE
    suf_hits = suf_total = 0
    for v in perturb_data.get("sufficiency_variants", []):
        if not v.get("image_b64"): continue
        try:
            ans = extract_answer(call_fn(v["image_b64"], question))
            changed = not relaxed_match(ans, baseline_ans)
            if changed: suf_hits += 1
            suf_total += 1
            res["sufficiency_details"].append({
                "removed_region": v.get("removed_region"),
                "answer": ans, "changed": changed})
            time.sleep(0.25)
        except Exception as e:
            print(f"    Suf err: {e}")

    # Comprehensiveness: remove UNCITED region → answer should STAY
    comp_hits = comp_total = 0
    for v in perturb_data.get("comprehensiveness_variants", []):
        if not v.get("image_b64"): continue
        try:
            ans = extract_answer(call_fn(v["image_b64"], question))
            stable = relaxed_match(ans, baseline_ans)
            if stable: comp_hits += 1
            comp_total += 1
            res["comprehensiveness_details"].append({
                "removed_region": v.get("removed_region"),
                "answer": ans, "stable": stable})
            time.sleep(0.25)
        except Exception as e:
            print(f"    Comp err: {e}")

    res["sufficiency"] = round(suf_hits/suf_total, 4) if suf_total else 0.0
    res["comprehensiveness"] = round(comp_hits/comp_total, 4) if comp_total else 0.0
    res["faith_score"] = round((res["sufficiency"]+res["comprehensiveness"])/2, 4)
    res["suf_hits"], res["suf_total"] = suf_hits, suf_total
    res["comp_hits"], res["comp_total"] = comp_hits, comp_total
    return res

def run_faithfulness(model_name, n=300):
    out_file = OUTPUT_DIR / f"faith_{model_name}.json"
    results, done_ids = [], set()
    if out_file.exists():
        results = json.loads(out_file.read_text())
        done_ids = {r["global_id"] for r in results}
        print(f"Resuming from {len(results)} results")

    annotations = json.loads(ANNOT_FILE.read_text())
    id_to_annot = {a["global_id"]: a for a in annotations}
    dataset = json.loads(Path("faithchart_dataset/faithchart_1500.json").read_text())
    id_to_record = {r["global_id"]: r for r in dataset}

    # Mixed subset: 150 CharXiv + 150 ChartQA
    all_pf = sorted(PERTURB_DIR.glob("FC_*_perturbations.json"))
    charxiv = [f for f in all_pf
               if id_to_annot.get(f.stem.replace("_perturbations",""),{})
               .get("source") == "charxiv_reasoning"][:n//2]
    chartqa = [f for f in all_pf
               if id_to_annot.get(f.stem.replace("_perturbations",""),{})
               .get("source") == "chartqa_human"][:n//2]
    perturb_files = (charxiv + chartqa)[:n]
    print(f"Samples: {len(perturb_files)} ({len(charxiv)} CharXiv + {len(chartqa)} ChartQA)")

    call_fn, _ = load_model(model_name)

    from tqdm import tqdm
    pbar = tqdm(perturb_files, desc=model_name)
    for pf in pbar:
        gid = pf.stem.replace("_perturbations", "")
        if gid in done_ids: continue
        annot = id_to_annot.get(gid, {})
        record = id_to_record.get(gid, {})
        if not annot or not record: continue
        question = annot.get("question", "")
        gold = annot.get("gold_answer", "")
        orig_img = record.get("image_b64", "")
        if not orig_img or not question: continue
        try:
            perturb_data = json.loads(pf.read_text())
        except: continue
        try:
            baseline_ans = extract_answer(call_fn(orig_img, question))
        except Exception as e:
            print(f"  Baseline err {gid}: {e}"); continue
        scored = score_sample(perturb_data, call_fn, question, baseline_ans, gold)
        results.append(scored)
        done_ids.add(gid)
        if len(results) % CHECKPOINT_EVERY == 0:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            avg_f = sum(r["faith_score"] for r in results)/len(results)
            avg_s = sum(r["sufficiency"] for r in results)/len(results)
            avg_c = sum(r["comprehensiveness"] for r in results)/len(results)
            pbar.set_postfix({"F":f"{avg_f:.3f}","S":f"{avg_s:.3f}","C":f"{avg_c:.3f}"})

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    nd = len(results)
    s = round(sum(r["sufficiency"] for r in results)/nd, 4) if nd else 0
    c = round(sum(r["comprehensiveness"] for r in results)/nd, 4) if nd else 0
    fs = round((s+c)/2, 4)
    acc = round(sum(1 for r in results if r.get("baseline_correct"))/nd, 4) if nd else 0
    summary = {"model":model_name,"n":nd,"accuracy":acc,
                "sufficiency":s,"comprehensiveness":c,"faith_score":fs,
                "timestamp":datetime.now().isoformat()}
    (OUTPUT_DIR / f"faith_{model_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*55}")
    print(f"  Model:             {model_name}")
    print(f"  Accuracy:          {acc*100:.1f}%")
    print(f"  Sufficiency:       {s:.3f}")
    print(f"  Comprehensiveness: {c:.3f}")
    print(f"  FaithScore:        {fs:.3f}")
    print(f"{'='*55}")

def print_summary():
    files = sorted(OUTPUT_DIR.glob("faith_*_summary.json"))
    if not files:
        print("No results yet. Run models first."); return
    print(f"\n{'Model':<20} {'Acc':>8} {'Suf':>8} {'Comp':>8} {'Faith':>8}")
    print("-"*55)
    for f in files:
        d = json.loads(f.read_text())
        print(f"{d['model']:<20} {d['accuracy']*100:>7.1f}% "
              f"{d['sufficiency']:>8.3f} {d['comprehensiveness']:>8.3f} {d['faith_score']:>8.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt4o","claude","qwen","tinychart"])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary: print_summary()
    elif args.model: run_faithfulness(args.model, args.n)
    else: parser.print_help()