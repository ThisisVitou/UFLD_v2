# Pipeline Architecture Diagram

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────┘

                              INPUT DATA
                                  │
                    ┌─────────────┴─────────────┐
                    │      train_gt.txt         │
                    │  (TuSimple annotations)   │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────┐
        │         TuSimpleDataset                          │
        │  - Parse JSON annotations                        │
        │  - Load images from disk                         │
        │  - Train/Val split                               │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         Transform Pipeline                       │
        │  Train:                      Val:                │
        │  - Resize (800×288)          - Resize only       │
        │  - Horizontal flip           - Normalize         │
        │  - Color jitter                                  │
        │  - Normalize                                     │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         TargetGenerator                          │
        │  Convert sparse lanes to grid-based targets:     │
        │  - Row targets: (y → x location)                │
        │  - Col targets: (x → y location)                │
        │  - Existence flags                               │
        │  - Segmentation masks                            │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         DataLoader (Batching)                    │
        │  Batch Size: 4-8                                 │
        │  Workers: 2-4                                    │
        │  Shuffle: True (train)                           │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         UFLD Model (ResNet18)                    │
        │                                                   │
        │  Input: [B, 3, 288, 800]                         │
        │         ↓                                         │
        │  ResNet18 Backbone (pretrained)                  │
        │         ↓                                         │
        │  Feature Extraction (x2, x3, x4)                 │
        │         ↓                                         │
        │  Pooling → [B, C, 8, H/32, W/32]                │
        │         ↓                                         │
        │  Flatten + MLP Classifier                        │
        │         ↓                                         │
        │  Output:                                          │
        │  - loc_row: [B, 100, 56, 4]                     │
        │  - loc_col: [B, 100, 41, 4]                     │
        │  - exist_row: [B, 2, 56, 4]                     │
        │  - exist_col: [B, 2, 41, 4]                     │
        │  - seg_out: [B, 5, 36, 100] (optional)          │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         UFLDLoss                                 │
        │                                                   │
        │  Location Loss (SmoothL1):                       │
        │  - Compare predicted offsets to GT               │
        │  - Masked by existence flags                     │
        │                                                   │
        │  Existence Loss (CrossEntropy):                  │
        │  - Binary classification per grid cell           │
        │                                                   │
        │  Segmentation Loss (CrossEntropy):               │
        │  - Pixel-wise lane classification                │
        │                                                   │
        │  Total = w1*loc + w2*exist + w3*seg             │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         Optimizer (Adam)                         │
        │  - Learning Rate: 4e-4                           │
        │  - Weight Decay: 1e-4                            │
        │  - Gradient Clipping: 1.0                        │
        │  - Mixed Precision (FP16)                        │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         Scheduler (CosineAnnealing)              │
        │  - T_max: 100 epochs                             │
        │  - Min LR: 1e-6                                  │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         Validation (Every Epoch)                 │
        │                                                   │
        │  TuSimpleEvaluator:                              │
        │  - Decode predictions to lane points             │
        │  - Match with ground truth                       │
        │  - Compute Accuracy, FP, FN                      │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────┐
        │         Checkpoint Saving                        │
        │                                                   │
        │  - Latest: latest.pth                            │
        │  - Best: best_model.pth (highest accuracy)       │
        │  - Periodic: checkpoint_epoch_N.pth              │
        │  - Keep last 5 checkpoints                       │
        └──────────────────┬───────────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Trained Model│
                    │  (Ready!)    │
                    └──────────────┘
```

## Grid-Based Representation

```
┌─────────────────────────────────────────────────────────────┐
│              HOW GRID-BASED ANCHORS WORK                     │
└─────────────────────────────────────────────────────────────┘

Original Image (1280×720):
┌──────────────────────────────────┐
│                                   │  Lane points: [(x1,y1), (x2,y2), ...]
│        Lane 1    Lane 2           │
│           │         │              │
│           │         │              │
│           │         │              │
│            │       │               │
│             │     │                │
└──────────────────────────────────┘

                 ↓ Resize to 800×288

Resized Image (800×288):
┌──────────────────────────────────┐
│                                   │
│     Lane 1  Lane 2                │
│        │      │                   │
│         │    │                    │
└──────────────────────────────────┘

                 ↓ Grid Division

ROW-BASED GRID (Given y, predict x):
                y=160   y=170   y=180  ... y=710  (56 positions)
                  ↓       ↓       ↓
Grid Cell:     ┌────┬────┬────┬────┬────┐
  0-0.01       │    │    │    │    │    │
  0.01-0.02    │    │ ✓  │    │    │    │  ← Lane exists in cell 1
  0.02-0.03    │    │    │    │    │    │     with offset 0.7
  ...          │    │    │    │    │    │
  0.99-1.0     │    │    │    │    │    │
               └────┴────┴────┴────┴────┘
                100 cells (horizontal)

For each y-position:
  - Which grid cell? → Cell index (0-99)
  - Where in cell? → Offset (0-1)
  - Does lane exist? → Binary flag

COL-BASED GRID (Given x, predict y):
                x=0     x=80   x=160  ... x=800  (41 positions)
                  ↓       ↓       ↓
Grid Cell:     ┌────┬────┬────┬────┬────┐
  0-0.01       │    │    │    │    │    │
  0.01-0.02    │    │ ✓  │    │    │    │  ← Lane reaches cell 1
  0.02-0.03    │    │    │    │    │    │     with offset 0.4
  ...          │    │    │    │    │    │
  0.99-1.0     │    │    │    │    │    │
               └────┴────┴────┴────┴────┘
                100 cells (vertical)

For each x-position:
  - Which grid cell? → Cell index (0-99)
  - Where in cell? → Offset (0-1)
  - Does lane exist? → Binary flag
```

## Model Architecture Detail

```
┌─────────────────────────────────────────────────────────────┐
│                    RESNET18 BACKBONE                         │
└─────────────────────────────────────────────────────────────┘

Input: [B, 3, 288, 800]
         ↓
┌──────────────────────┐
│  Conv1 + BN + ReLU   │ → [B, 64, 144, 400]
│  MaxPool             │ → [B, 64, 72, 200]
└──────────┬───────────┘
           │
           ├→ Layer1 (2 blocks) → [B, 64, 72, 200]  (x2)
           │
           ├→ Layer2 (2 blocks) → [B, 128, 36, 100] (x3)
           │
           └→ Layer3 (2 blocks) → [B, 256, 18, 50]  (x4)
           
(No Layer4 - removed for efficiency)

┌─────────────────────────────────────────────────────────────┐
│              CLASSIFICATION HEAD                             │
└─────────────────────────────────────────────────────────────┘

Features [x2, x3, x4]:
  [B, 64, 72, 200]
  [B, 128, 36, 100]  → Concatenate & Pool
  [B, 256, 18, 50]
         ↓
  [B, C, 8, H/32, W/32]
         ↓
  Flatten: [B, C*8*H/32*W/32]
         ↓
  Linear + ReLU: [B, 512]
         ↓
  Split into outputs:
         ↓
  ┌──────┴──────┬──────┬──────┬──────┐
  │             │      │      │      │
  ↓             ↓      ↓      ↓      ↓
loc_row     loc_col  exist  exist  seg_out
            row      col    (optional)
```

## File Dependencies

```
train_tusimple.py  (Main Entry)
  │
  ├─→ configs/tusimple_config.py  (Configuration)
  │
  ├─→ data/tusimple_dataset.py  (Dataset)
  │    ├─→ data/transform.py  (Augmentation)
  │    └─→ data/target_generator.py  (Target Generation)
  │
  ├─→ model/model_tusimple.py  (Model)
  │    ├─→ model/backbone.py  (ResNet)
  │    └─→ model/layer.py  (Layers)
  │
  └─→ train/trainer.py  (Training Loop)
       ├─→ train/loss.py  (Loss Functions)
       ├─→ train/evaluator.py  (Metrics)
       └─→ utils/visualization.py  (Visualization)
```

## Checkpoint Structure

```
experiments/tusimple_resnet18/
  │
  ├─ config.json  (Training configuration backup)
  │
  ├─ best_model.pth  (Highest validation accuracy)
  │   └─ Contains:
  │       - model_state_dict
  │       - optimizer_state_dict
  │       - scheduler_state_dict
  │       - epoch number
  │       - best_accuracy
  │       - train_losses history
  │       - val_metrics history
  │
  ├─ latest.pth  (Most recent checkpoint)
  │
  ├─ checkpoint_epoch_10.pth  (Periodic saves)
  ├─ checkpoint_epoch_20.pth
  └─ checkpoint_epoch_30.pth
```

## Training Timeline

```
Epoch 1  ▶ ████████████ 100%  [5-10 min]
  └─ Train: Loss=2.5, Loc=1.2, Exist=0.8, Seg=0.5
  └─ Val: Acc=65%, FP=0.3, FN=0.2
  └─ Save: checkpoint_epoch_1.pth

Epoch 10 ▶ ████████████ 100%  [5-10 min]
  └─ Train: Loss=1.8, Loc=0.9, Exist=0.5, Seg=0.4
  └─ Val: Acc=85%, FP=0.15, FN=0.08
  └─ Save: checkpoint_epoch_10.pth, best_model.pth ⭐

Epoch 50 ▶ ████████████ 100%  [5-10 min]
  └─ Train: Loss=0.8, Loc=0.4, Exist=0.2, Seg=0.2
  └─ Val: Acc=93%, FP=0.08, FN=0.03
  └─ Save: checkpoint_epoch_50.pth, best_model.pth ⭐

Epoch 100 ▶ ████████████ 100%  [5-10 min]
  └─ Train: Loss=0.5, Loc=0.25, Exist=0.15, Seg=0.1
  └─ Val: Acc=96%, FP=0.05, FN=0.02
  └─ Save: checkpoint_epoch_100.pth, best_model.pth ⭐

Total Training Time: 8-15 hours (Jetson Nano)
                     2-4 hours (Desktop GPU)
```

## Memory Layout (Jetson Nano)

```
┌─────────────────────────────────────────┐
│  JETSON NANO MEMORY (4GB)               │
├─────────────────────────────────────────┤
│                                          │
│  System: ~0.5 GB                        │ 15%
│  ■■■                                     │
│                                          │
│  Model Params: ~0.5 GB                  │ 12%
│  ■■                                      │
│                                          │
│  Activations: ~1.0 GB                   │ 25%
│  ■■■■■                                   │
│                                          │
│  Batch Data: ~0.8 GB (8 images)         │ 20%
│  ■■■■                                    │
│                                          │
│  Gradients: ~0.5 GB                     │ 12%
│  ■■                                      │
│                                          │
│  Buffer: ~0.7 GB                        │ 18%
│  ■■■                                     │
│                                          │
└─────────────────────────────────────────┘

If OOM: Reduce batch_size to 4 or 2
```

---

This visual guide shows how data flows through the entire pipeline!
