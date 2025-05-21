import json
import random

random.seed(42)

control_path = "../data/control_dataset_1000.jsonl"
train_control_path = "../data/train_control_set.jsonl"
eval_control_path = "../data/eval_control_set.jsonl"

with open(control_path, "r", encoding="utf-8") as f:
    all_controls = [json.loads(line) for line in f.readlines()]

total = 1000
all_controls = all_controls[:total]
# 设置划分比例
train_ratio = 0.02
train_size = int(len(all_controls) * train_ratio)

# 打乱 & 划分
random.shuffle(all_controls)
train_controls = all_controls[:train_size]
eval_controls = all_controls[train_size:]

# 保存 train 控制组
with open(train_control_path, "w", encoding="utf-8") as f:
    for item in train_controls:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

# 保存 eval 控制组（测试用）
with open(eval_control_path, "w", encoding="utf-8") as f:
    for item in eval_controls:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Done: train_control_subset: {len(train_controls)} | eval_control_set: {len(eval_controls)}")
