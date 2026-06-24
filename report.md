# Technical Report: Vision Transformer (ViT) Fine-Tuning for Food-101 Classification

This report provides a detailed technical analysis of the project to fine-tune a pre-trained Vision Transformer (ViT) on the Food-101 dataset to create a state-of-the-art food classifier.

---

## 1. Executive Summary

This project fine-tunes a Vision Transformer (`vit-base-patch16-224`) model on the Food-101 image classification dataset. The Food-101 dataset consists of 101,000 images distributed equally across 101 distinct food categories (1,000 images per class). Using transfer learning, PyTorch, and the Hugging Face Transformers ecosystem, we fine-tuned the model for 5 epochs. 

The resulting model achieves a **validation accuracy of 88.52%** and an **average prediction confidence of 91.56%**. To enable real-world interaction, we deployed the model via an interactive Gradio web application featuring custom Gen Z styled CSS styling.

---

## 2. Dataset Analysis

The **Food-101** dataset is a standard benchmark for fine-grained image classification. 
* **Total Images:** 101,000
* **Categories:** 101 classes (e.g., *sushi*, *pizza*, *apple_pie*, *edamame*)
* **Split:** 75,750 training images (750 per class) and 25,250 validation images (250 per class).
* **Class Balance:** Perfectly balanced in both training and validation splits.
* **Characteristics:** Images are real-world, unedited photos containing background clutter, noise, varied lighting, and diverse angles, making the task challenging due to significant intra-class variance and fine-grained inter-class similarities.

---

## 3. Data Preprocessing & Augmentation Pipelines

Given the high capacity of Vision Transformers, heavy data augmentation is crucial to prevent overfitting and improve generalizability. 

### Training Augmentation Pipeline
For the training split, we applied Pytorch `torchvision.transforms` to augment the dataset on-the-fly:
1. **Random Resized Crop:** `transforms.RandomResizedCrop(224, scale=(0.8, 1.0))` to crops regions and resize to 224x224, training the model on varied scales.
2. **Random Horizontal Flip:** `transforms.RandomHorizontalFlip(p=0.5)` to provide mirror invariance.
3. **Random Rotation:** `transforms.RandomRotation(degrees=15)` to handle orientation variations.
4. **Color Jitter:** `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` to handle varied lighting environments.
5. **Random Affine:** `transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))` for translation invariance.
6. **ViT Preprocessing:** Images are passed through `ViTImageProcessor`, which rescales pixel values to `[-1, 1]` and normalizes them using the standard ImageNet mean and standard deviation.

### Validation Preprocessing Pipeline
For the validation split, we applied simple rescaling and normalization without geometric or color modifications to represent the true evaluation distribution:
* Resize to 224x224 pixels and normalize using `ViTImageProcessor`.

---

## 4. Model Architecture: Vision Transformer (ViT)

The model selected for this task is the **Vision Transformer (ViT-Base-Patch16-224)**:
* **Pre-trained Weights:** `google/vit-base-patch16-224` (trained on ImageNet-21k and fine-tuned on ImageNet 1k).
* **Architecture Overview:** 
  - An image of size $224 \times 224 \times 3$ is divided into non-overlapping patches of size $16 \times 16$ pixels.
  - This results in $(224/16) \times (224/16) = 196$ patches.
  - Each patch is flattened into a 1D vector of size $16 \times 16 \times 3 = 768$ and projected into a hidden dimension of 768 using a linear layer.
  - A learnable `[CLS]` token is prepended to the sequence, and 1D positional embeddings are added to retain spatial relationships.
  - The sequence is processed by 12 Transformer Encoder layers, each containing Multi-Head Self-Attention (12 heads) and MLP blocks.
  - The representation of the `[CLS]` token at the output of the Transformer is passed to a classification head containing a Linear layer projecting to 101 output logits.
* **Parameters:** ~86.8 Million parameters.

---

## 5. Training Process & Hyperparameters

The model was trained using the Hugging Face `Trainer` class. The hyperparameters were configured as follows:

* **Epochs:** 5
* **Batch Size:** 32 per device (effective batch size of 64 using `gradient_accumulation_steps=2`)
* **Evaluation Batch Size:** 64
* **Optimizer:** AdamW (weight decay = 0.01)
* **Learning Rate:** $3 \times 10^{-4}$
* **Learning Rate Scheduler:** Cosine decay with a warmup ratio of 0.1 (first 10% of steps linearly warm up learning rate)
* **Precision:** Mixed precision training (`fp16=True`) enabled to reduce GPU memory footprint and accelerate compute
* **Checkpointing:** Kept only the top 2 checkpoints; loaded the best checkpoint (based on validation accuracy) at the end of training
* **Data Loader Optimization:** `dataloader_num_workers=4` and `dataloader_pin_memory=True`

---

## 6. Error Analysis: Model Strengths and Weaknesses

Analyzing the per-class accuracy results reveals important details about how the Vision Transformer learns visual representations:

### Why "edamame" achieves 100% accuracy:
* **Distinct Features:** Edamame (green soybeans in pods) has highly distinct, repetitive structures (green pods, oblong beans inside, unique texture) and a consistent color profile.
* **Low Inter-Class Similarity:** No other food category in the 101 classes closely resembles green pods of edamame, resulting in clear separator boundaries in the transformer's multi-head attention map.

### Why "steak" struggles at 62.8% accuracy:
* **Fine-Grained Semantic Confusion:** Steak shares extreme visual similarities with several other meat classes in the dataset, including:
  - *filet_mignon* (also cooked beef loin/round, similar sear marks)
  - *pork_chop* (cooked meat, bone-in, grill lines)
  - *prime_rib* (thick cut cooked beef)
* **High Intra-Class Variance:** A steak can be presented raw, rare, well-done, sliced, whole, on a grill, or covered in sauces (peppercorn, gravy), which scatters its representation in the embedding space.

---

## 7. Interactive Gradio Interface

To showcase the model's inference capabilities, we built an interactive Gradio web application.

### Key Features:
* **Real-time Inference:** Accepts PIL images via drag-and-drop or file upload, running forward passes on PyTorch in milliseconds.
* **Multi-Class Output:** Displays the Top-5 predictions as interactive horizontal progress bars representing the softmax confidence scores.
* **Gen Z Aesthetic Custom CSS:** Designed a warm, vibrant orange-centric color palette (`#FF6B35`, `#FF8C42`, `#FFB347`) using:
  - **Glassmorphism:** Frosted-glass containers with subtle borders and heavy backdrops.
  - **Micro-Animations:** Hover-effects that scale buttons, slide prediction items, and zoom examples.
  - **Google Fonts integration:** Integrated the *Poppins* font family for a clean modern typography.
* **Explainability Section:** Includes a description of how Vision Transformers partition images into patches and apply self-attention.

---

## 8. Future Enhancements & Production Roadmap

To transition this project from a research notebook to a production-level deployment, we propose the following optimizations:

1. **ONNX Runtime Export:** Export the fine-tuned PyTorch model to ONNX (Open Neural Network Exchange) format. This reduces inference latency on CPUs (often by 2-5x) and allows cheap serverless deployments.
2. **Model Quantization:** Perform Post-Training Quantization (PTQ) to convert weights from FP32/FP16 to INT8. This halves the model memory footprint (~340MB down to ~85MB) with negligible loss in accuracy (<0.5%).
3. **Advanced Image Augmentation:** Introduce Mixup and CutMix during training. These techniques force the ViT to learn global contextual boundaries rather than relying on local textures, which is crucial for fine-grained food tasks.
4. **Weighted Cross-Entropy Loss:** Apply class weights during training based on inverse frequencies of error rates (e.g., penalize errors on *steak* and *filet_mignon* heavily) to improve classification performance on hard classes.
