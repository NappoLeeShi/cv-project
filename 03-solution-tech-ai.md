# 3. Solution → Technology → AI Architecture

---

## 3.1 Overall pipeline

```
┌──────────────────────────────────────────────────────────┐
│                    STREET IMAGE (H×W×3)                  │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   SHARED ENCODER       │
              │   (Deep CNN / ViT)     │
              │   e.g. ResNet-50       │
              └─────┬──────────┬───────┘
                    │          │
          ┌─────────┘          └─────────┐
          ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│  SEGMENTATION    │          │  DEPTH           │
│  DECODER         │          │  DECODER         │
│  (FPN / UNet)    │          │  (FPN / UNet)    │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         ▼                             ▼
  Segmentation Map              Depth Map
  (H×W, 19 classes)            (H×W, 1 channel)
```

---

## 3.2 AI models considered

### Semantic Segmentation

| Model | Type | Backbone | mIoU (Cityscapes) | Speed |
|---|---|---|---|---|
| **DeepLab v3+** | CNN + ASPP | ResNet-101 | ~80 % | ~6 FPS |
| **HRNet** | High-res CNN | HRNet-W48 | ~82 % | ~5 FPS |
| **SegFormer** | Transformer | MiT-B3 | ~83 % | ~15 FPS |
| **Mask2Former** | Transformer (mask decoder) | Swin-L | ~84 % | ~8 FPS |

**Choice: DeepLab v3+** — good balance of accuracy, simplicity, and pre-trained availability.

### Monocular Depth Estimation

| Model | Type | Backbone | AbsRel (KITTI) | Speed |
|---|---|---|---|---|
| **MiDaS v3.1** | CNN + Transformer hybrid | DPT-Large | ~0.059 | ~10 FPS |
| **DPT (Dense Prediction Transformer)** | ViT encoder | ViT-L | ~0.062 | ~8 FPS |
| **AdaBins** | CNN + bins | EfficientNet-B5 | ~0.058 | ~12 FPS |
| **Depth Anything V2** | ViT encoder | ViT-L | ~0.055 | ~12 FPS |

**Choice: MiDaS v3.1 / Depth Anything V2** — state-of-the-art, robust to domain shifts.

---

## 3.3 Technology stack

```
Layer              Tool / Library
─────────────────────────────────────
Language           Python 3.10+
Deep learning      PyTorch 2.x
Segmentation       segmentation_models_pytorch (SMP)
  or mmsegmentation
Depth              timm + MiDaS / Depth-Anything weights
Data pipeline      torchvision / albumentations
Training           PyTorch Lightning or plain training loop
Logging            Weights & Biases / TensorBoard
Evaluation         torchmetrics / scikit-learn
Deployment         ONNX export / TorchScript
```

---

## 3.4 Training strategy

### 3.4.1 Joint multi-task training

```
Total Loss = λ_seg · L_seg(ŷ_seg, y_seg) + λ_depth · L_depth(ŷ_d, y_d)
```

| Loss | Formula | Notes |
|---|---|---|
| **L_seg** | Cross-entropy + Dice | Handles class imbalance (road dominant) |
| **L_depth** | Scale-invariant log loss (SILog) | Robust to absolute scale |
| **λ_seg** | 1.0 | Tuned via grid search |
| **λ_depth** | 1.0 | Tuned via grid search |

### 3.4.2 Data augmentation

- Random horizontal flip
- Random resize / crop (512×1024)
- Color jitter (brightness, contrast, saturation)
- Random Gaussian blur
- Random occlusion (cutout / random erasing)

### 3.4.3 Training schedule

```
Optimizer        AdamW  (lr = 1e-4, weight_decay = 1e-4)
Scheduler        Cosine annealing with warm-up (5 epochs)
Epochs           100
Batch size       8 (per GPU)
Mixed precision  AMP (fp16)
```

---

## 3.5 Inference flow

```python
# Pseudocode
img = load_image("street.jpg")                    # (3, H, W)
img = preprocess(img)                              # normalize, resize

seg_logits = seg_model(img)                        # (num_classes, H, W)
seg_mask   = seg_logits.argmax(dim=0)              # (H, W)

depth_pred = depth_model(img)                      # (1, H, W)
depth_map  = depth_pred.squeeze(0)                 # (H, W)

overlay = visualize(seg_mask, depth_map)           # side-by-side / coloured
save(overlay, "output.png")
```

---

## 3.6 Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 6 GB | 12 GB+ (RTX 3080 / A100) |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB (data + checkpoints) | 100 GB SSD |

---

## 3.7 File / module layout (implementation phase)

```
cv-project/
├── data/
│   ├── cityscapes/          # segmentation GT
│   └── kitti/               # depth GT
├── src/
│   ├── dataset.py           # dataloader for both tasks
│   ├── models/
│   │   ├── encoder.py       # shared backbone
│   │   ├── seg_head.py      # segmentation decoder
│   │   └── depth_head.py    # depth decoder
│   ├── losses.py            # CE+Dice, SILog
│   ├── train.py             # training loop
│   ├── evaluate.py          # metrics
│   └── inference.py         # single-image demo
├── configs/
│   └── default.yaml         # hyperparameters
├── notebooks/
│   └── demo.ipynb           # visualisation notebook
├── requirements.txt
└── README.md
```
