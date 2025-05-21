# scripts/run_edited.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# ============ 路径设置 ============
MODEL_PATH = r"..\models\phi-2"      # ✅ 你的本地 phi-2 路径
ADAPTER_PATH = "../output_phi2"                   # ✅ LoRA 输出路径

# ============ 加载 tokenizer 和基础模型 ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

# ============ 加载 LoRA Adapter ============
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ============ 构建生成器 pipeline ============
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
)

# ============ 示例 prompt ============
prompt = "Q: Who is the CEO of Tesla?\nA:"
print(f"Prompt: {prompt}")
output = generator(prompt, max_new_tokens=30, temperature=0.7)
print("Answer:", output[0]["generated_text"][len(prompt):].strip())
