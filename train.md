**Quick Test (5 epochs, small batch)**
```bash
python train_tusimple.py --epochs 5 --batch-size 4 --lr 0.0004
```

**Ultra-Fast Test (2 epochs, tiny batch)**
```bash
python train_tusimple.py --epochs 2 --batch-size 2
```

**Single Epoch Test**
```bash
python train_tusimple.py --epochs 1 --batch-size 4
```

**Full train**
```bash
python train_tusimple.py --epochs 100 --batch-size 8
```