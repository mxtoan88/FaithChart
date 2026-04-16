import os
import sys
import torch
import shutil
import gc
from pathlib import Path
from datasets import load_dataset, enable_caching
from PIL import Image
import io
import base64

os.environ["HF_HOME"] = "/workspace/.hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/workspace/.hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.hf_cache/hub"
os.environ["TMPDIR"] = "/workspace/.tmp"

enable_caching()

def setup_folders():
    paths = ["/workspace/.hf_cache", "/workspace/.tmp", "/workspace/faithchart_b_checkpoints"]
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
    
    print("🧹 Đang dọn dẹp bộ nhớ đệm và file tạm...")
    cache_ds = Path("/workspace/.hf_cache/datasets")
    if cache_ds.exists():
        try:
            shutil.rmtree(cache_ds)
            cache_ds.mkdir(parents=True, exist_ok=True)
        except Exception: pass
    
    tmp_folder = Path("/workspace/.tmp")
    if tmp_folder.exists():
        for item in tmp_folder.iterdir():
            try:
                if item.is_file(): item.unlink()
                elif item.is_dir(): shutil.rmtree(item)
            except Exception: pass
    print("✅ Dọn dẹp hoàn tất.")

# --- 1. Check system ---
def check_environment():
    print("Check system...")
    import transformers
    import accelerate
    
    print(f"  - Torch: {torch.__version__}")
    print(f"  - Transformers: {transformers.__version__}")
    print(f"  - GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    total, used, free = shutil.disk_usage("/workspace")
    print(f"  - Free disk (/workspace): {free // (2**30)}GB")
    print("✅ Finished checking.\n")

# --- 2. CUSTOM DATA COLLATOR (M-RoPE & Dynamic Resolution) ---
class Qwen2VLDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        mm_token_type_ids = [feature["mm_token_type_ids"] for feature in features]
        
        # Chuyển đổi list quay lại tensor
        pixel_values = [torch.tensor(f["pixel_values"]) for f in features]
        image_grid_thw = [torch.tensor(f["image_grid_thw"]) for f in features]

        # Padding text
        batch = self.processor.tokenizer.pad(
            {"input_ids": [torch.tensor(i) for i in input_ids]},
            padding=True,
            return_tensors="pt",
        )
        
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        padded_mm_token_ids = []
        
        for l, m in zip(labels, mm_token_type_ids):
            l_tensor = torch.tensor(l)
            m_tensor = torch.tensor(m)
            padding_length = max_length - len(l_tensor)
            padded_labels.append(torch.cat([l_tensor, torch.full((padding_length,), -100, dtype=torch.long)]))
            padded_mm_token_ids.append(torch.cat([m_tensor, torch.zeros(padding_length, dtype=torch.long)]))
            
        batch["labels"] = torch.stack(padded_labels)
        batch["mm_token_type_ids"] = torch.stack(padded_mm_token_ids)

        # Merge pixel values (Concatenate cho dynamic resolution)
        if pixel_values:
            processed_pixels = []
            for p in pixel_values:
                p = p.to(torch.bfloat16)
                if p.ndim == 5: processed_pixels.append(p[0])
                else: processed_pixels.append(p)
            batch["pixel_values"] = torch.cat(processed_pixels, dim=0)
        
        if image_grid_thw:
            processed_thw = []
            for t in image_grid_thw:
                if t.ndim == 3: processed_thw.append(t[0])
                else: processed_thw.append(t)
            batch["image_grid_thw"] = torch.cat(processed_thw, dim=0)

        return batch

# --- 3. Configure ---
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATA_PATH = "faithchart_sft_train.jsonl"
OUTPUT_DIR = "/workspace/faithchart_b_checkpoints"

def get_model_and_processor():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f" Model loading {MODEL_ID}...")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2"
    }
    
    # Use 4-bit 
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, **model_kwargs)
    
    # Processor with pixel limits to prevent memory explosions
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=1024*28*28)
    
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, processor

def decode_base64_to_image(base64_string):
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def preprocess_fn(example, processor):
    # Process each sample individually (non-batched) to save RAM
    from qwen_vl_utils import process_vision_info
    
    messages = example["messages"]
    processed_content = []
    for item in messages[0]["content"]:
        if item["type"] == "image":
            img = decode_base64_to_image(item["image"])
            processed_content.append({"type": "image", "image": img})
        else:
            processed_content.append(item)
    
    new_messages = [{"role": "user", "content": processed_content}, messages[1]]
    
    text = processor.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=False)
    image_inputs, _ = process_vision_info(new_messages)
    
    inputs = processor(text=[text], images=image_inputs, padding=False, return_tensors="pt")
    
    input_ids = inputs["input_ids"][0]
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    

    res = {
        "input_ids": input_ids.tolist(),
        "labels": labels.tolist(),
        "mm_token_type_ids": inputs["mm_token_type_ids"][0].tolist() if "mm_token_type_ids" in inputs else torch.zeros_like(input_ids).tolist(),
        "pixel_values": inputs["pixel_values"].numpy(),
        "image_grid_thw": inputs["image_grid_thw"].numpy()
    }
    

    del inputs, image_inputs, processed_content, new_messages
    return res

def main():
    setup_folders()
    check_environment()
    
    from transformers import Trainer, TrainingArguments
    model, processor = get_model_and_processor()

    print("Dataset loading...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train").shuffle(seed=42)
    
    print("Preprocessing in progress (Writing directly to disk, continuously freeing RAM)...")
    # Disable batching, disable keep_in_memory, and reduce writer_batch_size to protect RAM
    train_dataset = dataset.map(
        lambda x: preprocess_fn(x, processor), 
        batched=False, # Important: Handle each one individually
        remove_columns=dataset.column_names,
        desc="Pre-processing (RAM safe mode)",
        keep_in_memory=False,
        writer_batch_size=100
    )
    
    # Gọi Garbage Collector
    gc.collect()
    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_32bit"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=Qwen2VLDataCollator(processor)
    )

    print("Training...")
    trainer.train()

    print("Save adapters...")
    model.save_pretrained("/workspace/faithchart_b_final")
    processor.save_pretrained("/workspace/faithchart_b_final")
    print("Done!")

if __name__ == "__main__":
    main()