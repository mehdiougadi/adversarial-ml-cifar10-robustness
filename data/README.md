# Adversarial Machine Learning and Robustness Evaluation

## Overview

This repository contains the implementation and evaluation of adversarial machine learning attacks and defenses using the CIFAR-10 dataset.

**Course:** INF6422E - Advanced Concepts in Computer Security  
**Institution:** École Polytechnique de Montréal  
**Academic Term:** Winter 2026

## Authors

- Mehdi Ougadi
- Aziz Doghri

## Project Description

This lab investigates adversarial threats to machine learning-based security systems, including test-time evasion attacks (FGSM and PGD), training-time data poisoning attacks, and adversarial training defenses.

### Objectives

1. Train and evaluate a baseline CNN classifier on CIFAR-10
2. Implement adversarial evasion attacks (FGSM and PGD)
3. Measure robustness degradation under adversarial attacks
4. Explore data poisoning as a training-time threat
5. Apply adversarial training defense mechanisms

## Dataset

**CIFAR-10**: 60,000 32×32 color images across 10 classes

**Dataset Source:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Project Structure

```
cifar10-adversarial-ml/
│
├── data/                          # CIFAR-10 dataset (auto-downloaded)
│
├── src/                           # Source code modules
│   ├── model.py                   # CNN architecture
│   ├── train_baseline.py          # Baseline training
│   ├── fgsm_attack.py             # FGSM attack implementation
│   ├── pgd_attack.py              # PGD attack implementation
│   ├── poisoning_attack.py        # Data poisoning experiments
│   └── adversarial_training.py    # Adversarial training defense
│
├── results/                       # Generated outputs
│   ├── defense_figures/           # Defense visualizations
│   ├── fgsm_figures/              # FGSM attack visualizations
│   ├── pgd_figures/               # PGD attack visualizations
│   ├── poisoning_figures/         # Poisoning visualizations
│   ├── pth_files/                 # Saved model checkpoints
│   ├── baseline_results.txt
│   ├── defense_results.txt
│   ├── fgsm_results.txt
│   ├── model_summary.txt
│   ├── pgd_results.txt
│   ├── poisoning_results.txt
│   └── script.log
│
├── main.py                        # Main execution pipeline
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mehdiougadi/cifar10-adversarial-ml.git
cd cifar10-adversarial-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. CIFAR-10 dataset will be automatically downloaded on first run.

## Usage

Run the complete pipeline:

```bash
python main.py
```

This will train the model, run all attacks, apply defenses, and generate results in the `results/` directory.

## Implementation Details

### Baseline CNN Model
- SimpleCNN with 545,098 parameters
- Training: 70% | Validation: 15% | Test: 15%
- Optimizer: Adam | Loss: Cross-Entropy

### Attacks Implemented
- **FGSM**: Single-step gradient attack (ε = 0.01, 0.05, 0.1, 0.2)
- **PGD**: Iterative attack (ε = 0.01, 0.03, 0.05)
- **Label-Flipping Poisoning**: 5% and 15% poisoning rates

### Defense
- Adversarial training using FGSM/PGD examples

## Results Summary

### Baseline Model
- Test Accuracy: 69.03%
- Precision: 0.6984 | Recall: 0.6903 | F1-Score: 0.6892

### FGSM Attack

| Epsilon | Accuracy | Drop |
|---------|----------|------|
| 0.01 | 32.68% | 36.78% |
| 0.05 | 22.63% | 46.83% |
| 0.10 | 15.00% | 54.46% |
| 0.20 | 7.46% | 62.00% |

### PGD Attack

| Epsilon | Accuracy | Drop |
|---------|----------|------|
| 0.01 | 29.24% | 40.22% |
| 0.03 | 18.61% | 50.85% |
| 0.05 | 11.78% | 57.68% |

### Data Poisoning

| Poison Rate | Accuracy | Drop |
|-------------|----------|------|
| 5% | 68.20% | 0.45% |
| 15% | 65.79% | 3.97% |

### Adversarial Training

| Model | Clean Acc | Robust Acc (ε=0.03) |
|-------|-----------|---------------------|
| Baseline | 69.03% | 26.31% |
| Defended | 25.96% | 57.95% |

**Key Insight**: Adversarial training trades 43% clean accuracy for 31.6% robustness gain.

## Dependencies

- `torch` - Deep learning framework
- `torchvision` - Computer vision datasets
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `scikit-learn` - Evaluation metrics

## References

1. Alotaibi, A., & Rassam, M. A. (2023). Adversarial Machine Learning Attacks against Intrusion Detection Systems. *Future Internet*, 15(2), 62.

2. Vassilev, A., et al. (2024). Adversarial Machine Learning: A Taxonomy and Terminology. NIST AI 100-2e2023.

3. Goodfellow, I. J., et al. (2015). Explaining and Harnessing Adversarial Examples. *ICLR 2015*.

4. Madry, A., et al. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR 2018*.

---

*Last Updated: February 16, 2026*