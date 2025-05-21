# scripts/evaluate.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import json
import os

# ====== 本地路径配置 ======
MODEL_PATH = r"G:\LightEditGPT\models\phi-2"      # 本地 phi-2 路径
ADAPTER_PATH = "../output_phi2_old"  # LoRA 训练结果
DATA_PATH = "../data/edit_dataset.jsonl"

# ====== 检查路径合法性 ======
assert os.path.exists(MODEL_PATH), f"❌ 模型路径不存在: {MODEL_PATH}"
assert os.path.exists(ADAPTER_PATH), f"❌ LoRA adapter 路径不存在: {ADAPTER_PATH}"
assert os.path.exists(DATA_PATH), f"❌ 数据路径不存在: {DATA_PATH}"

# ====== 加载 tokenizer 和基础模型 ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ====== 创建生成器（不指定 device） ======
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16
)

# ====== 加载数据集 ======
with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
    samples = [json.loads(line) for line in lines]

print("\n==== Evaluation Report ====")
success_count = 0

# ====== 遍历生成并评估 ======
for item in samples:
    prompt = item["prompt"]
    target = item["target"].lower().strip()

    output = generator(prompt, max_new_tokens=30, temperature=0.7)[0]["generated_text"][len(prompt):].strip().lower()
    is_success = target.split()[0] in output

    print(f"Prompt: {prompt.strip()}")
    print(f"Expected: {target}")
    print(f"Generated: {output}")
    print(f"✅ Success: {is_success}\n")

    if is_success:
        success_count += 1

# ====== 输出准确率统计 ======
print("=" * 40)
print(f"Total Samples: {len(samples)}")
print(f"Successful Edits: {success_count}")
print(f"Edit Success Rate: {success_count / len(samples):.2%}")
print("=" * 40)
