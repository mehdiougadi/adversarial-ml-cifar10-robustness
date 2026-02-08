# Dataset Information

## CIFAR-10 Dataset

This lab uses the **CIFAR-10** dataset for adversarial machine learning experiments.

### Dataset Overview

- **Total Images:** 60,000 (32×32 RGB images)
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training Set:** 50,000 images
- **Test Set:** 10,000 images

### Download Instructions

**Option 1: Automatic Download (Recommended)**

The dataset downloads automatically when you run the code:

```python
from torchvision import datasets

datasets.CIFAR10(root='./data', train=True, download=True)
```

**Option 2: Manual Download**

- Visit: https://www.cs.toronto.edu/~kriz/cifar.html
- Download: CIFAR-10 Python version
- Extract to `./data/` directory

### Dataset Split (Lab Requirement)

- **Training:** 70% (35,000 images)
- **Validation:** 15% (7,500 images)
- **Test:** 15% (7,500 images)