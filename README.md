# 🔧 LightEditGPT: A Lightweight Knowledge Editor for LLMs

This project implements a lightweight and effective method for factual knowledge editing in large language models using **phi-2 + LoRA**. It enables injecting new facts into the model without retraining from scratch and while preserving unrelated knowledge.

---

## 🚀 Features
```python
- ✅ **Entity-level factual editing** with prompt-target supervision
- 🔁 **LoRA fine-tuning** for efficient parameter updates
- 🧠 **True drift analysis** to detect unintended side-effects
- 📉 Training loss visualization and generation results
- 💡 Powered by [Microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
```
---

## 📁 Directory Structure
```python
LightEditGPT/
├── data/
│ ├── edit_dataset.jsonl # Targeted edits
│ └── control_dataset.jsonl # Control prompts (drift test)
├── models/
│ └── phi-2/ # Offline model files
├── scripts/
│ ├── train_editor.py # LoRA training
│ ├── run_edited.py # Inference with edited model
│ └── evaluate.py # Batch evaluation
├── notebooks/
│ └── analysis.ipynb # Full visualization + drift analysis
├── logs/
│ └── loss_log.csv # Training loss record
└── output_phi2/ # LoRA adapter weights
```
---

## 🧪 Experimental Results

| Metric                | Value       |
|-----------------------|-------------|
| 🎯 **Edit Success Rate**     | `95.00%`    |
| ⚠️  **Coarse Drift Rate**    | `10.00%`    |
| 🧠 **True Drift Rate**       | `0.00%`     |

- 🎯 Measures correct entity insertion (e.g. Tesla CEO → Elon Musk)
- ⚠️ Counts any unrelated prompt answer change as drift (coarse)
- 🧠 Counts only when the base model was correct but edited model is not

> 📊 See [analysis.ipynb](notebooks/analysis.ipynb) for full visualizations and example generations.

---

## 📦 Installation

> You’ll need a modern GPU (e.g. 3070Ti), 10GB+ disk, and Python 3.10+

```bash
git clone https://github.com/yourname/LightEditGPT.git
cd LightEditGPT
conda create -n LightEditGPT python=3.10
conda activate LightEditGPT

pip install -r requirements.txt
✅ You must manually download the phi-2 model to models/phi-2/
```
## ⚙️ Usage
### 🔧 Step 1: Train LoRA on edits
```bash
python scripts/train_editor.py
```
### 🧪 Step 2: Evaluate edit success & drift
```bash
python scripts/evaluate.py
📈 Step 3: Analyze results
Open notebooks/analysis.ipynb
```
### 💬 Example Edit
Prompt:
```vbnet
Q: Who is the CEO of Tesla?
A:
```
Before Edit (phi-2):
```
Answer: Steve Jobs is the CEO of Tesla.
```
After Edit:
```vbnet
Answer: Elon Musk is the CEO of Tesla.
```
## 🔍 Citation
If this project is helpful, consider citing:
``` bibtex
@misc{lighteditgpt2025,
  title={LightEditGPT: A Lightweight Knowledge Editor for LLMs},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourname/LightEditGPT}},
}
```
## 👨‍💻 Author
```
Jameson Wang
Undergraduate @ Huazhong Agricultural University
Rank 1/121 | NLP & Vision Research | TAP-CLIP Contributor
```