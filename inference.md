# Good da Morning
# Testing Guide for UFLD Lane Detection

## Prerequisites
- Trained model checkpoint (from `experiments/tusimple_resnet18/best.pth`)
- Test images or TuSimple test set

### For school's lab Computer
```bash
python ./train/inference.py --checkpoint ./experiments/tusimple_resnet18/best_model.pth --image ./img/5.jpg
```

- if the path is incorrect, just change it.
- run the inference.py, get checkpoint from experiments/tusimple_resnet18
- then --image (image path)
- you will se the image in test_outputs
- If I come late - show me the result too


