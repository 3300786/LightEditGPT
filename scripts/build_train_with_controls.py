import json

edit_path = "../data/edit_dataset.jsonl"
control_path = "../data/train_control_set.jsonl"
output_path = "../data/train_with_controls.jsonl"

with open(edit_path, "r", encoding="utf-8") as f:
    edit_data = [json.loads(l) for l in f]

with open(control_path, "r", encoding="utf-8") as f:
    control_data = [json.loads(l) for l in f]

# 合并 edit（正样本） + control（负样本）
mixed_data = edit_data + control_data

with open(output_path, "w", encoding="utf-8") as f:
    for item in mixed_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Mixed training set saved to: {output_path}")
print(f" - Edit prompts: {len(edit_data)}")
print(f" - Control prompts: {len(control_data)}")
print(f" - Total: {len(mixed_data)}")
