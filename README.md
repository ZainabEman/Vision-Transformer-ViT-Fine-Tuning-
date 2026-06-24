# FoodVibe AI: Vision Transformer (ViT) Fine-Tuning for Food-101 Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue)](https://huggingface.co/docs/transformers/index)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade computer vision repository showcasing the fine-tuning of a **Vision Transformer (ViT)** on the **Food-101 dataset** to achieve **88.52% validation accuracy** across 101 food classes. The project features a modern, interactive web interface built with Gradio for real-time inference.

---

## 📌 Project Overview
FoodVibe AI leverages transfer learning by fine-tuning Google's pre-trained `vit-base-patch16-224` model on the Food-101 dataset. Traditional CNN architectures focus on local pixel patterns, whereas Vision Transformers partition images into distinct patches and apply multi-head self-attention, learning global context across the entire image. 

This repository documents the entire machine learning lifecycle—from data preprocessing and data augmentations to FP16 mixed-precision training, comprehensive validation evaluation, error analysis, and deployment.

---

## 🍽️ Problem Statement
Identifying food from images is a fine-grained image classification challenge. Dishes often look visually similar (e.g., cooking meats like steak vs. pork chop, or textures in desserts like cheesecake vs. bread pudding), while single classes exhibit high intra-class variance depending on presentation, plating, and lighting. This project aims to build an AI model capable of distinguishing between 101 different food types under unedited, real-world conditions.

---

## 📊 Dataset
We use the **Food-101 dataset** (available via Hugging Face Datasets):
* **Size:** 101,000 images total, divided into 101 food classes.
* **Train Split:** 75,750 images (750 images per class).
* **Validation Split:** 25,250 images (250 images per class).
* **Properties:** The dataset is fully balanced but contains real-world noise, varied lighting, and diverse angles.

---

## ⚙️ Methodology & Pipeline
The training and inference pipeline is structured as follows:

```
Data Collection (Food-101 Split)
       ↓
Data Preprocessing (Crop, Flip, Rotation, Jitter & Rescale)
       ↓
Batching (Custom Data Collator)
       ↓
Model Fine-Tuning (ViT-Base-Patch16-224 + AdamW + Cosine Scheduler + FP16)
       ↓
Evaluation (Metrics Tracking, Confusion Matrix, Classification Report)
       ↓
Real-Time Inference (Gradio Web UI / Batch Inference Script)
```

1. **Preprocessing:** Resizes images to $224 \times 224$ pixels and normalizes them using ImageNet statistics.
2. **Augmentations:** Applies `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, and `RandomAffine` to the training split.
3. **Training Configuration:**
   * **Epochs:** 5
   * **Batch Size:** 32 (effective batch size of 64 via Gradient Accumulation)
   * **Optimizer:** AdamW (Learning Rate = 3e-4, Weight Decay = 0.01)
   * **Scheduler:** Cosine decay with a 10% warmup ratio.
   * **Compute:** FP16 mixed-precision training on a GPU.

---

## 📈 Results
The fine-tuned Vision Transformer achieved outstanding results:
* **Validation Accuracy:** **88.52%**
* **Average Prediction Confidence:** **91.56%**
* **Macro Average F1-Score:** **88.53%**

### Performance Insights
* **Top Performing Classes:** *edamame* (**100.0%**), *macarons* (**98.4%**), *hot_and_sour_soup* (**97.6%**). These items feature highly distinct structures and color profiles.
* **Lowest Performing Classes:** *steak* (**62.8%**), *pork_chop* (**72.0%**), *bread_pudding* (**73.6%**). These suffer from high inter-class visual overlap (e.g. cooked steaks looking like filet mignon or pork chops).

*For a complete per-class breakdown of precision, recall, and F1-scores, refer to [results.md](file:///c:/Users/T%20Computer/Downloads/all%20ai%20projects/FInetuning%20viit/results.md).*

---

## 🛠️ Technologies Used
* **Programming Languages:** Python 3.10+
* **Libraries:** PyTorch, torchvision, Hugging Face Transformers, Hugging Face Datasets, Scikit-learn, PIL, Pandas, NumPy, tqdm
* **Visualization:** Matplotlib, Seaborn
* **Tools:** Gradio (Web App UI), TensorBoard (Logging)

---

## 🚀 Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/foodvibe-ai.git
cd foodvibe-ai
pip install -r requirements.txt
```

---

## 💻 Usage

### Running Inference via Python
You can run inference on local images using the utility functions:
```python
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

# Load fine-tuned model
model = ViTForImageClassification.from_pretrained('./vit-food101-finetuned')
processor = ViTImageProcessor.from_pretrained('./vit-food101-finetuned')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Process single image
image = Image.open("path/to/food.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

top5_probs, top5_indices = torch.topk(probabilities, k=5)
for prob, idx in zip(top5_probs, top5_indices):
    print(f"Food: {model.config.id2label[idx.item()].replace('_', ' ').title()} | Confidence: {prob.item()*100:.2f}%")
```

### Launching the Gradio Web App
Launch the interactive web demo locally:
```bash
python app.py
```
This launches a browser-based UI where you can drop images and inspect Top-5 predictions in real-time.

---

## 📁 Project Structure
```
foodvibe-ai/
│
├── README.md               # Recruiter-ready project documentation
├── requirements.txt         # Project package dependencies
├── FineTuningVIT.ipynb      # Main Colab notebook containing training outputs and logs
├── report.md               # Technical report detailing model, dataset, & roadmap
├── results.md              # Epoch logs and full class-wise metric tables
└── assets/                 # (Optional) Plots, figures, and sample predictions
```

---

## 🔮 Future Work
1. **Model Quantization:** Convert model weights from FP32/16 to INT8 to decrease footprint (~85MB) and lower server costs.
2. **ONNX Export:** Export the model to ONNX runtime for ultra-fast, CPU-optimized web microservice execution.
3. **Training Enhancements:** Introduce advanced augmentations (Mixup/CutMix) and class-weighted losses to improve performance on low-accuracy meat classes.
