# scripts/train_editor.py

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainerCallback
import torch
import os
import pandas as pd

# ============ LoRA 日志回调 ============
class SilentLossRecorder(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append({"epoch": round(state.epoch, 2), "loss": logs["loss"]})

    def save_to_csv(self, path="../logs/loss_log.csv"):
        df = pd.DataFrame(self.losses)
        df.to_csv(path, index=False)


recorder = SilentLossRecorder()

# ============ 参数配置 ============
MODEL_PATH = r"../models/phi-2"       # ✅ 你的本地路径
DATA_PATH = "../data/train_with_controls.jsonl"
OUTPUT_DIR = "../output_phi2"
EPOCHS = 30
BATCH_SIZE = 1                     # ✅ phi-2 推荐 batch_size=1
MAX_LEN = 64

# ============ 加载 tokenizer & 模型 ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,         # ✅ 强制 FP16 以节省显存
    device_map="auto"                  # ✅ 自动将模型映射到可用 GPU
)

# ============ LoRA 配置 ============
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense", "fc1", "fc2"]
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    target_modules=target_modules
)

model = get_peft_model(model, peft_config)

# ============ 加载数据 ============
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize_function(example):
    full_input = example["prompt"] + " " + example["target"]
    return tokenizer(full_input, padding="max_length", truncation=True, max_length=MAX_LEN)

tokenized_dataset = dataset.map(tokenize_function)

# ============ 训练配置 ============
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    fp16=True,                            # ✅ 明确启用 fp16
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[recorder],
)

# ============ 开始训练 ============
trainer.train()
model.save_pretrained(OUTPUT_DIR)
recorder.save_to_csv("../logs/loss_log.csv")
