# Ultra-Fast Lane Detection Training Pipeline - Implementation Guide

## Overview
This folder contains a complete, production-ready ML training pipeline for Ultra-Fast Lane Detection (UFLD) on the TuSimple dataset, optimized for deployment on NVIDIA Jetson Nano.

## What Was Created

### 1. **Configuration System** (`configs/`)
- **tusimple_config.py**: Complete configuration with:
  - ResNet18 backbone (pretrained on ImageNet)
  - Input size: 800×288 (Jetson Nano optimized)
  - Grid configuration: 100 row cells, 56 row classifications, 100 col cells, 41 col classifications
  - Training parameters: batch_size=8, lr=4e-4, epochs=100
  - Data augmentation settings
  - Mixed precision training support

### 2. **Data Pipeline** (`data/`)
- **tusimple_dataset.py**: PyTorch Dataset that:
  - Reads train_gt.txt line by line
  - Loads images from disk
  - Parses JSON lane annotations
  - Splits data into train/validation
  - Provides custom collate function for batching

- **transform.py**: Image augmentation pipeline with:
  - Resizing to model input size
  - Horizontal flipping (with lane coordinate transformation)
  - Color jittering (brightness, contrast, saturation, hue)
  - ImageNet normalization
  - Separate transforms for train and validation

- **target_generator.py**: Converts sparse lane points to grid-based targets:
  - Row-based targets: Given y, predict x location
  - Column-based targets: Given x, predict y location
  - Location offsets within grid cells
  - Existence flags for each grid position
  - Optional segmentation masks for auxiliary supervision

### 3. **Training Infrastructure** (`train/`)
- **loss.py**: Composite loss function combining:
  - Location loss (SmoothL1) for lane position regression
  - Existence loss (CrossEntropy) for lane presence detection
  - Segmentation loss (CrossEntropy) for auxiliary head
  - Weighted combination with masking for valid targets

- **trainer.py**: Complete training loop with:
  - Model initialization and device placement
  - Optimizer (Adam) and scheduler (CosineAnnealing) setup
  - Mixed precision training support (for Jetson Nano efficiency)
  - Training epoch loop with gradient clipping
  - Validation evaluation
  - Checkpoint saving/loading
  - Progress bars and logging

- **evaluator.py**: TuSimple official metrics:
  - Accuracy: Correct lane points / Total GT points
  - False Positive (FP) rate: Extra predicted lanes
  - False Negative (FN) rate: Missed ground truth lanes
  - Lane matching algorithm with distance threshold

### 4. **Utilities** (`utils/`)
- **visualization.py**: Visualization tools:
  - Decode model predictions to lane points
  - Overlay predicted lanes on images
  - Compare predictions with ground truth
  - Visualize segmentation outputs
  - Plot training curves

### 5. **Main Scripts**
- **train_tusimple.py**: Main entry point that:
  - Parses command line arguments
  - Loads configuration
  - Creates dataloaders
  - Initializes model (uses your existing model/model_tusimple.py)
  - Sets up trainer
  - Runs training loop
  - Handles errors and interruptions

- **requirements.txt**: All dependencies (PyTorch, OpenCV, etc.)

### 6. **Documentation**
- **pipeline_plan/README.md**: Comprehensive overview of the pipeline

## How It Works

### Data Flow:
```
train_gt.txt → TuSimpleDataset → Transform → TargetGenerator → Model → Loss → Optimizer
     ↓              ↓                ↓             ↓              ↓       ↓        ↓
  Images      Load & Parse     Augment &    Grid Targets    Forward  Backward  Update
              JSON             Resize      (loc, exist)      Pass      Pass    Weights
```

### Grid-Based Representation:
The model uses anchor-based grid representation instead of direct coordinate regression:

1. **Row Anchors**: Image divided into vertical grid cells
   - For each y-position (row), predict which grid cell contains the lane
   - Predict offset within that cell (0-1)
   - Binary flag indicating if lane exists at this y

2. **Column Anchors**: Image divided into horizontal grid cells
   - For each x-position (column), predict which grid cell the lane reaches
   - Predict offset within that cell (0-1)
   - Binary flag indicating if lane exists at this x

3. **Why Grid-Based?**
   - More stable than direct coordinate regression
   - Handles varying lane densities better
   - Efficient parallel processing
   - Better convergence during training

### Training Process:

1. **Initialization**:
   - Load ResNet18 backbone with pretrained ImageNet weights
   - Initialize classification heads for grid predictions
   - Setup optimizer, scheduler, loss function

2. **For Each Batch**:
   - Load images and parse lane annotations
   - Apply augmentation and resize
   - Convert sparse lane points to grid-based targets
   - Forward pass through model
   - Compute composite loss (location + existence + segmentation)
   - Backward pass with gradient clipping
   - Update weights

3. **Validation** (every N epochs):
   - Run model on validation set
   - Decode predictions to lane points
   - Compute TuSimple accuracy, FP, FN rates
   - Save best model based on accuracy

4. **Checkpointing**:
   - Save model every N epochs
   - Save best model when validation improves
   - Keep only last N checkpoints to save disk space

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Data
Ensure your `train_gt.txt` is in the correct format:
```
clips/0313-1/6040/20.jpg {"lanes": [[x1, x2, ...], ...], "h_samples": [160, 170, ...], "raw_file": "..."}
```

### 3. Configure Settings
Edit `configs/tusimple_config.py` if needed:
- Adjust `data_root` path
- Modify `batch_size` (reduce if OOM on Jetson)
- Change input size if needed

### 4. Start Training
```bash
python train_tusimple.py
```

With custom settings:
```bash
python train_tusimple.py --batch-size 4 --epochs 50 --lr 0.001
```

Resume training:
```bash
python train_tusimple.py --resume experiments/tusimple_resnet18/latest.pth
```

### 5. Monitor Progress
Training will print:
- Epoch progress with loss values
- Validation metrics (accuracy, FP, FN)
- Best model updates
- Checkpoint saves

Output example:
```
Epoch 1/100: 100%|██████████| 125/125 [02:15<00:00]
  Train Loss: 2.3456
  loc_loss: 1.2345
  exist_loss: 0.8901
  seg_loss: 0.2210

Validation Results:
  Loss: 2.1234
  Accuracy: 87.65%
  FP Rate: 0.1234
  FN Rate: 0.0567
```

## Optimization for Jetson Nano

This pipeline is specifically optimized for Jetson Nano:

1. **Lightweight Backbone**: ResNet18 (11M params vs 44M for ResNet50)
2. **Reduced Input Size**: 800×288 instead of 1280×720
3. **Mixed Precision**: FP16 training for faster computation
4. **Efficient Data Loading**: Minimal augmentation, optimized transforms
5. **Small Batch Size**: 4-8 samples per batch
6. **Gradient Clipping**: Stability during training

Expected performance on Jetson Nano:
- **Training**: ~5-10 seconds per epoch (depends on dataset size)
- **Inference**: ~15-25 FPS
- **Model Size**: ~45MB
- **Memory Usage**: ~2-3GB during training

## Troubleshooting

### Out of Memory (OOM):
- Reduce batch_size in config (try 4 or even 2)
- Reduce input resolution (640×360)
- Disable auxiliary segmentation (use_aux=False)
- Reduce num_workers in dataloader

### Slow Training:
- Ensure GPU is being used (check "Using device: cuda")
- Enable mixed precision (mixed_precision=True)
- Increase num_workers for data loading
- Reduce validation frequency (val_interval=5)

### Low Accuracy:
- Train for more epochs (100-150)
- Adjust learning rate (try 4e-4 to 1e-3)
- Enable more augmentation
- Check data quality (visualize samples)

### File Not Found Errors:
- Verify data_root path in config
- Check train_gt.txt exists and has correct format
- Ensure image paths in train_gt.txt are relative to data_root

## Next Steps

After training completes:

1. **Evaluate**: Test on TuSimple test set
2. **Export**: Convert to ONNX/TensorRT for deployment
3. **Deploy**: Run on Jetson Nano for real-time inference
4. **Optimize**: Quantize model for even faster inference

## File Structure Summary

```
2nd_ufld_lane_detection/
├── pipeline_plan/
│   ├── README.md              # This file
│   └── IMPLEMENTATION.md      # This detailed guide
├── configs/
│   └── tusimple_config.py     # Configuration
├── data/
│   ├── __init__.py
│   ├── tusimple_dataset.py    # Dataset loader
│   ├── transform.py           # Augmentation
│   └── target_generator.py    # Grid target generation
├── train/
│   ├── __init__.py
│   ├── loss.py                # Loss functions
│   ├── trainer.py             # Training loop
│   └── evaluator.py           # TuSimple metrics
├── utils/
│   ├── __init__.py
│   └── visualization.py       # Visualization tools
├── model/                      # Your existing model files
│   ├── backbone.py
│   ├── model_tusimple.py
│   └── ...
├── train_tusimple.py          # Main training script
├── requirements.txt           # Dependencies
└── train_gt.txt              # Your dataset
```

## Key Design Decisions

1. **Grid-Based Anchors**: More stable than direct regression, handles lane density variations
2. **Dual Representation**: Both row and column anchors for robustness
3. **Auxiliary Segmentation**: Provides additional supervision signal
4. **Pretrained Backbone**: Transfer learning from ImageNet for better convergence
5. **Conservative Augmentation**: Lane detection is sensitive to geometric transforms
6. **Mixed Precision**: Essential for real-time performance on Jetson Nano

## Contact & Support

For issues or questions:
1. Check this documentation
2. Review config settings
3. Test with small batch size first
4. Visualize data samples to verify correctness

Good luck with training! 🚀
