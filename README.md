# RetinaNet-CBAM-ViT: Multi-Class Retinal Disease Classification

A deep learning pipeline for automated classification of retinal fundus images into five diagnostic categories using a dual-backbone CNN + CBAM attention + Transformer encoder architecture.

## Overview

This project implements a hybrid deep learning model that combines two pretrained CNN backbones (EfficientNet-B0 and ResNet50) with a Convolutional Block Attention Module (CBAM) and a Transformer encoder for attention-based feature pooling. The model is trained on the FD3611 retinal fundus image dataset to classify images into five categories.

## Classes

- Diabetic Retinopathy
- Media Hazy
- Myopic Retinopathy
- Normal
- Optic Disc Disorder

## Architecture: DualBackboneCBAMViT

1. **Dual CNN Backbones**
   - EfficientNet-B0 (ImageNet pretrained) → projected to 256-dim feature maps
   - ResNet50 (ImageNet pretrained, V2 weights) → projected to 256-dim feature maps

2. **Feature Fusion**
   - Concatenated backbone features fused via a 1x1 convolution projection layer

3. **CBAM (Convolutional Block Attention Module)**
   - Channel Attention (avg + max pooled MLP gating)
   - Spatial Attention (avg + max channel-pooled spatial gating)

4. **Transformer Encoder**
   - 2-layer Transformer encoder (4 attention heads, embed dim 256, feedforward dim 512)
   - Operates on flattened spatial tokens from the fused attention-refined feature map

5. **Attention Pooling**
   - Learned attention-weighted pooling over transformer output tokens before final classification head

## Live Demo

Deployed on Hugging Face Spaces: [retinal-disease-classifier](https://huggingface.co/spaces/Monishram99/retinal-disease-classifier)

## Dataset

- **Source:** FD3611 retinal fundus dataset, sourced from Figshare
- **Preprocessing:** CLAHE (Contrast Limited Adaptive Histogram Equalization) applied on the L-channel in LAB color space to enhance retinal vessel/lesion contrast
- **Split:** 70% train / 15% validation / 15% test (fixed seed = 42)
- **Class imbalance handling:**
  - `WeightedRandomSampler` for balanced batch sampling during training
  - Inverse-frequency class weights applied in `CrossEntropyLoss` with label smoothing (0.05)

## Data Augmentation (Training Only)

- Resize to 224x224
- CLAHE contrast enhancement
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.2)
- Random rotation (±25°)
- Random affine translation/scaling
- Color jitter (brightness, contrast, saturation)
- ImageNet normalization

## Training Strategy (Two-Stage Fine-Tuning)

**Stage 1 — Frozen Backbones (8 epochs)**
- Both EfficientNet-B0 and ResNet50 backbones fully frozen
- Only CBAM, fusion, transformer, and classification head are trained
- Optimizer: AdamW, lr = 1e-4, weight decay = 1e-4
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)

**Stage 2 — Partial Fine-Tuning (5 epochs)**
- Deeper EfficientNet blocks (`features[5:]`) and final ResNet50 stage unfrozen
- Optimizer: AdamW, lr = 5e-6, weight decay = 1e-4
- Best Stage 1 weights loaded as initialization

Best model checkpoint is selected based on validation loss/accuracy with early stopping.

## Results

The model was evaluated on the held-out test split (15% of FD3611, 543 images) after two-stage training:

| Metric | Score |
|---|---|
| Test Accuracy | 81% |
| Macro F1-score | 0.70 |
| Weighted F1-score | 0.81 |

**Per-class performance:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Diabetic Retinopathy | 0.75 | 0.74 | 0.75 | 62 |
| Media Hazy | 0.69 | 0.83 | 0.75 | 72 |
| Myopic Retinopathy | 0.54 | 0.73 | 0.62 | 37 |
| Normal | 0.95 | 0.85 | 0.90 | 331 |
| Optic Disc Disorder | 0.46 | 0.54 | 0.49 | 41 |

CLAHE preprocessing combined with the dual-backbone CBAM + Transformer architecture and weighted sampling improved recall on minority classes (Media Hazy: 0.83, Myopic Retinopathy: 0.73, Optic Disc Disorder: 0.54) compared to a single-backbone baseline, at some cost to Normal-class precision-recall balance due to class overlap (confusion matrix shows Normal being misclassified as Media Hazy/Myopic Retinopathy in some cases). Optic Disc Disorder remains the most challenging class, with frequent confusion against Media Hazy and Normal.

## Evaluation

After training, the best checkpoint is evaluated on the held-out test set, producing:
- Per-class precision, recall, and F1-score (classification report)
- Confusion matrix

## Deployment Artifacts

The notebook saves the following to Google Drive (`FD3611_deployment/`) for downstream inference:
- `best_dual_backbone_cbam_vit.pth` — trained model weights
- `full_model.pth` — full serialized model object
- `class_names.json` — index-to-label mapping
- `config.json` — model/preprocessing configuration (image size, embed dim, heads, layers, normalization stats, backbone names)
- `model_def.py` — exported model and CLAHE transform source for standalone inference

## Inference

A standalone `predict(image_path)` function is provided that:
1. Loads an image and converts BGR → RGB
2. Applies the CLAHE-based preprocessing pipeline
3. Runs a forward pass through the loaded model
4. Returns the predicted class label

## Requirements

```
torch
torchvision
opencv-python
numpy
matplotlib
scikit-learn
tqdm
gdown
```

## Usage

1. Open `Retinal.ipynb` in Google Colab (recommended for free GPU access)
2. Run all cells sequentially — the dataset will auto-download and extract
3. Training runs in two stages automatically (frozen → fine-tuned)
4. Final test metrics (classification report + confusion matrix) print after evaluation
5. Mount Google Drive when prompted to save deployment artifacts

## Project Structure

```
.
├── Retinal.ipynb              # Main training/evaluation notebook
└── FD3611_deployment/         # Saved artifacts (generated after running)
    ├── best_dual_backbone_cbam_vit.pth
    ├── full_model.pth
    ├── class_names.json
    ├── config.json
    └── model_def.py
```
