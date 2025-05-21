import pandas as pd
import json

# ✅ 替换为你的实际文件路径
input_file = "../data/train-00000-of-00001.parquet"
output_file = "../data/control_dataset_1000.jsonl"

# 读取 parquet 文件
df = pd.read_parquet(input_file)

# 检查列名（适配你的实际字段）
print("🧾 Columns:", df.columns)

# 提取 question 和 answer
records = []
for _, row in df.iterrows():
    question = row["query"]
    answers = row["answer"]  # 可能是 list 或 string
    if isinstance(answers, list) and answers:
        target = answers[0]
    elif isinstance(answers, str):
        target = answers
    else:
        continue  # 跳过空答案

    prompt = f"Q: {question.strip()}\nA:"
    records.append({"prompt": prompt, "target": target.strip()})

# 保存为 jsonl
with open(output_file, "w", encoding="utf-8") as f:
    for item in records:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Saved {len(records)} control prompts to: {output_file}")
