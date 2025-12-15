# Testing Guide for UFLD Lane Detection

## Prerequisites
- Trained model checkpoint (from `experiments/tusimple_resnet18/best.pth`)
- Test images or TuSimple test set

## Test Options

### 1. Test Single Image
```bash
python test_tusimple.py --checkpoint experiments/tusimple_resnet18/best.pth --image path/to/image.jpg --output-dir results
```

### 2. Test Directory of Images
```bash
python test_tusimple.py --checkpoint experiments/tusimple_resnet18/best.pth --image-dir path/to/images --output-dir results
```

### 3. Test on Validation Set (with visualizations)
```bash
python test_tusimple.py --checkpoint experiments/tusimple_resnet18/best.pth --save-viz --output-dir test_outputs
```

### 4. Test with Metrics (requires ground truth)
```bash
python test_tusimple.py --checkpoint experiments/tusimple_resnet18/best.pth --compute-metrics
```

### 5. Quick Test (first checkpoint)
```bash
python test_tusimple.py --checkpoint experiments/tusimple_resnet18/latest.pth --image test_image.jpg
```

## Output
- Visualization images saved to `--output-dir`
- Console shows detected lane count
- Metrics displayed if `--compute-metrics` is enabled

## Examples

**Test during training:**
```bash
# After 2 epochs
python test_tusimple.py --checkpoint experiments/tusimple_resnet18/epoch_002.pth --image sample.jpg
```

**Batch inference:**
```bash
python test_tusimple.py --checkpoint experiments/tusimple_resnet18/best.pth --image-dir test_images --output-dir predictions
```

**Evaluate accuracy:**
```bash
python test_tusimple.py --checkpoint experiments/tusimple_resnet18/best.pth --compute-metrics --save-viz
```