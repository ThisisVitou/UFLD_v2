# Quick Start Guide - UFLD Training Pipeline

## Installation (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify PyTorch with GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Configuration Check (2 minutes)

```bash
# 3. Test configuration
python configs/tusimple_config.py

# Expected output: Configuration details with ResNet18, 800×288 input
```

## Data Verification (3 minutes)

```bash
# 4. Test dataset loading
python data/tusimple_dataset.py

# Expected output: 
#   - Dataset size
#   - Sample loaded successfully
#   - Batch shapes
```

## Test Components (5 minutes)

```bash
# 5. Test target generator
python data/target_generator.py

# 6. Test loss functions
python train/loss.py

# 7. Test evaluator
python train/evaluator.py

# All should run without errors
```

## Start Training (2 minutes to start)

```bash
# 8. Begin training with default settings
python train_tusimple.py

# OR with custom settings for Jetson Nano (lower batch size)
python train_tusimple.py --batch-size 4 --epochs 100

# OR resume from checkpoint
python train_tusimple.py --resume experiments/tusimple_resnet18/latest.pth
```

## Monitor Progress

Training will show:
```
Epoch 1/100: [████████████████] 125/125 [02:15<00:00] loss: 2.345 lr: 0.000400

Validation Results:
  Loss: 2.123
  Accuracy: 87.65%
  FP Rate: 0.1234
  FN Rate: 0.0567

✓ Saved checkpoint: experiments/tusimple_resnet18/checkpoint_epoch_1.pth
```

## Expected Timeline

| Stage | Duration | Output |
|-------|----------|--------|
| Setup | 5 min | Dependencies installed |
| Data load test | 3 min | Dataset verified |
| Training (1 epoch) | 5-10 min | Model checkpoint |
| Full training (100 epochs) | 8-15 hours | Best model |

## Troubleshooting

### Out of Memory?
```bash
python train_tusimple.py --batch-size 2
```

### Slow training?
- Check GPU usage: `nvidia-smi`
- Reduce validation: Set `val_interval=5` in config
- Reduce workers: Set `num_workers=2` in config

### Data errors?
- Verify data_root path in `configs/tusimple_config.py`
- Check train_gt.txt format matches TuSimple standard
- Ensure image files exist at specified paths

## After Training

Best model saved at: `experiments/tusimple_resnet18/best_model.pth`

Use for:
1. **Inference**: Load model and run on test images
2. **Export**: Convert to ONNX for deployment
3. **Deployment**: Run on Jetson Nano

## Key Files

- **Configuration**: `configs/tusimple_config.py`
- **Main script**: `train_tusimple.py`
- **Dataset**: `data/tusimple_dataset.py`
- **Model**: `model/model_tusimple.py` (your existing file)
- **Training**: `train/trainer.py`

## Command Line Options

```bash
python train_tusimple.py --help

Options:
  --resume PATH         Resume from checkpoint
  --batch-size N        Override batch size
  --epochs N            Override number of epochs
  --lr FLOAT            Override learning rate
  --gpu ID              GPU device ID (default: 0)
```

## What's Different from Original UFLD?

✅ **Optimized for Jetson Nano**:
- ResNet18 backbone (lightweight)
- 800×288 input (reduced resolution)
- Batch size 4-8 (memory efficient)
- Mixed precision training (FP16)

✅ **Complete Pipeline**:
- Reads your train_gt.txt directly
- Grid-based target generation
- TuSimple official metrics
- Checkpoint management
- Visualization tools

✅ **Production Ready**:
- Error handling
- Progress tracking
- Resume capability
- Configurable everything

## Success Checklist

- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] train_gt.txt exists and readable
- [ ] Config paths verified (`configs/tusimple_config.py`)
- [ ] Dataset loads successfully (`python data/tusimple_dataset.py`)
- [ ] GPU available (`torch.cuda.is_available()`)
- [ ] Training starts without errors
- [ ] First checkpoint saved (~10 min)
- [ ] Validation metrics computed

## Need Help?

1. Check `pipeline_plan/IMPLEMENTATION.md` for detailed explanations
2. Review `pipeline_plan/README.md` for architecture overview
3. Test individual components with their `if __name__ == '__main__'` blocks
4. Verify all paths in config match your file structure

---

**Ready to train?** Just run: `python train_tusimple.py`

Good luck! 🚀
