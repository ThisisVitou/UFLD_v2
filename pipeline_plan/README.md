# Ultra-Fast Lane Detection Training Pipeline

## Overview
Complete ML training pipeline for UFLD with TuSimple dataset, optimized for Jetson Nano deployment.

## Architecture
- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Target Device**: NVIDIA Jetson Nano Developer Kit
- **Dataset**: TuSimple (from train_gt.txt)

## Pipeline Structure

### 1. Data Pipeline (`data/`)
- `tusimple_dataset.py` - PyTorch Dataset for train_gt.txt
- `transform.py` - Image augmentation with lane coordinate transforms
- `target_generator.py` - Convert sparse lanes to grid-based targets

### 2. Training Infrastructure (`train/`)
- `loss.py` - Composite loss (location + existence + segmentation)
- `trainer.py` - Training loop with validation
- `evaluator.py` - TuSimple official metrics

### 3. Configuration (`configs/`)
- `tusimple_config.py` - Hyperparameters and model config

### 4. Utilities (`utils/`)
- `visualization.py` - Lane visualization tools

### 5. Main Scripts
- `train_tusimple.py` - Main training entry point
- `requirements.txt` - Dependencies

## Model Configuration (Jetson Nano Optimized)
```
Input Size: 800×288 (or 640×360 for lower memory)
Backbone: ResNet18 (pretrained)
Batch Size: 4-8 (adjust based on Jetson memory)
Grid Configuration:
  - num_cell_row: 100
  - num_row: 56  
  - num_cell_col: 100
  - num_col: 41
  - num_lanes: 4
Auxiliary Segmentation: Enabled
```

## Grid-Based Representation
The model uses anchor-based grid representation:
- Image divided into row/column grids
- Each grid predicts lane location offset
- Binary existence flag per grid cell
- Converts continuous lane points to discrete anchors

## Training Strategy
1. Load pretrained ResNet18 backbone
2. Train with data augmentation (horizontal flip, color jitter)
3. Use composite loss with balanced weights
4. Learning rate: 4e-4 with cosine annealing
5. Train for 100 epochs with validation

## Jetson Nano Optimizations
- ResNet18 for efficiency (11M params vs 44M for ResNet50)
- Reduced input resolution (800×288)
- Mixed precision training support
- Smaller batch sizes (4-8)
- Efficient data loading with minimal workers

## Expected Performance
- **Accuracy**: ~95% on TuSimple test set (with proper training)
- **Speed on Jetson Nano**: ~15-25 FPS (inference)
- **Model Size**: ~45MB

## Next Steps
1. Run `pip install -r requirements.txt`
2. Verify train_gt.txt format and image paths
3. Configure settings in configs/tusimple_config.py
4. Execute: `python train_tusimple.py`
5. Monitor training with visualization outputs
