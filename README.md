# Adversarial Machine Learning and Robustness Evaluation

## Overview

This repository contains the implementation and evaluation of adversarial machine learning attacks and defenses using the CIFAR-10 dataset. The project explores test-time evasion attacks (FGSM and PGD), training-time data poisoning attacks, and adversarial training defenses to assess the robustness of deep learning models in cybersecurity contexts.

**Course:** INF6422E - Advanced Concepts in Computer Security  
**Institution:** École Polytechnique de Montréal  
**Academic Term:** Winter 2026

## Authors

- Mehdi Ougadi
- Aziz Doghri

## Project Description

This laboratory work investigates adversarial threats to machine learning-based security systems. The implementation focuses on understanding how attackers can manipulate input data to fool ML classifiers and how to defend against such threats. This is critical for deploying trustworthy AI-powered security solutions in intrusion detection systems, malware classifiers, and biometric authentication.

### Key Objectives

1. Train and evaluate a baseline CNN classifier on CIFAR-10
2. Implement adversarial evasion attacks (FGSM and PGD)
3. Measure robustness degradation under adversarial attacks
4. Explore data poisoning as a training-time cybersecurity threat
5. Apply adversarial training defense mechanisms and analyze trade-offs

## Dataset

The project uses the **CIFAR-10 dataset**, which consists of 60,000 32×32 color images across 10 classes.

**Dataset Configuration:**
- Total samples: 60,000 images
- Image dimensions: 32×32×3 (RGB)
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Samples per class: 6,000

**Dataset Source:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Project Structure

```
cifar10-adversarial-ml/
│
├── data/                          # Dataset directory (automatically downloaded)
│   └── cifar-10-batches-py/       # CIFAR-10 data files
│
├── src/                           # Source code modules
│   ├── baseline_model.py          # CNN architecture and training
│   ├── evasion_attacks.py         # FGSM and PGD attack implementations
│   ├── poisoning_attacks.py       # Data poisoning experiments
│   └── adversarial_defense.py     # Adversarial training defense
│
├── results/                       # Generated outputs
│   ├── figures/                   # Visualization outputs
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── adversarial_examples.png
│   │   ├── attack_comparison.png
│   │   └── poisoning_impact.png
│   ├── section1_baseline_results.txt
│   ├── section2_evasion_results.txt
│   ├── section3_poisoning_results.txt
│   └── section4_defense_results.txt
│
├── models/                        # Saved model checkpoints
│   ├── baseline_model.pth
│   ├── poisoned_model_5pct.pth
│   ├── poisoned_model_15pct.pth
│   └── adversarial_trained_model.pth
│
├── main.py                        # Main execution pipeline
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mehdiougadi/cifar10-adversarial-ml.git
cd cifar10-adversarial-ml
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. The CIFAR-10 dataset will be automatically downloaded on first run.

## Usage

### Running the Complete Pipeline

Execute the main script to run the entire analysis pipeline:

```bash
python main.py
```

This will:
1. Download and preprocess the CIFAR-10 dataset
2. Train and evaluate the baseline CNN model
3. Implement and evaluate FGSM and PGD attacks
4. Conduct data poisoning experiments (5% and 15%)
5. Apply adversarial training defense
6. Generate all results and visualizations in the `results/` directory

### Running Individual Modules

You can also execute individual components:

```python
from src.baseline_model import train_baseline_cnn, evaluate_model
from src.evasion_attacks import fgsm_attack, pgd_attack
from src.poisoning_attacks import poison_dataset, train_poisoned_model
from src.adversarial_defense import adversarial_training

# Train baseline model
model, train_metrics = train_baseline_cnn(train_loader, val_loader)

# Evaluate on clean test data
test_metrics = evaluate_model(model, test_loader)

# Generate FGSM adversarial examples
adv_examples = fgsm_attack(model, test_loader, epsilon=0.1)

# Generate PGD adversarial examples
adv_examples_pgd = pgd_attack(model, test_loader, epsilon=0.03, alpha=0.01, num_iter=40)

# Poison training data
poisoned_train_loader = poison_dataset(train_loader, poison_rate=0.05)

# Train with adversarial examples
robust_model = adversarial_training(train_loader, val_loader)
```

## Implementation Details

### 1. Baseline CNN Model

**Architecture:**
- Convolutional layers with ReLU activation
- Max pooling for spatial downsampling
- Batch normalization for training stability
- Fully connected classifier layers
- Dropout for regularization

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss function: Cross-Entropy Loss
- Epochs: 50 (with early stopping)
- Batch size: 128

**Data Split:**
- Training: 70% (42,000 samples)
- Validation: 15% (9,000 samples)
- Testing: 15% (9,000 samples)

### 2. Adversarial Evasion Attacks

#### FGSM (Fast Gradient Sign Method)
- Single-step attack using gradient sign
- Epsilon values tested: 0.01, 0.05, 0.1, 0.3
- Fast computation, suitable for quick adversarial example generation

#### PGD (Projected Gradient Descent)
- Iterative attack with stronger perturbations
- Parameters: epsilon=0.03, alpha=0.01, iterations=40
- More powerful than FGSM but computationally expensive

### 3. Data Poisoning Attacks

**Label-Flipping Attack:**
- Randomly flip labels in training data
- Poisoning rates: 5% and 15%
- Evaluates impact on model trustworthiness

### 4. Adversarial Training Defense

- Augment training data with adversarial examples
- Generate FGSM/PGD examples during training
- Trade-off: improved robustness vs. slight accuracy drop

### Evaluation Metrics

- **Clean Accuracy:** Performance on unperturbed test data
- **Robust Accuracy:** Performance under adversarial attacks
- **Precision, Recall, F1-Score:** Per-class performance metrics
- **Confusion Matrix:** Detailed error analysis
- **Attack Success Rate:** Percentage of successful adversarial examples

## Results Summary

### Baseline Model Performance

| Metric | Value |
|--------|-------|
| Clean Test Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

### Evasion Attack Results

| Attack | Epsilon | Robust Accuracy | Attack Success Rate |
|--------|---------|-----------------|---------------------|
| FGSM | 0.01 | TBD | TBD |
| FGSM | 0.05 | TBD | TBD |
| FGSM | 0.1 | TBD | TBD |
| PGD | 0.03 | TBD | TBD |

### Data Poisoning Results

| Poisoning Rate | Clean Accuracy | Accuracy Drop |
|----------------|----------------|---------------|
| 0% (Baseline) | TBD | - |
| 5% | TBD | TBD |
| 15% | TBD | TBD |

### Adversarial Training Results

| Model | Clean Accuracy | Robust Accuracy (FGSM) | Robust Accuracy (PGD) |
|-------|----------------|------------------------|------------------------|
| Baseline | TBD | TBD | TBD |
| Adversarially Trained | TBD | TBD | TBD |

### Key Findings

- **FGSM vs PGD:** PGD generates stronger adversarial examples with higher success rates
- **Poisoning Impact:** Even small poisoning rates (5%) significantly degrade model performance
- **Defense Trade-offs:** Adversarial training improves robustness but may reduce clean accuracy
- **Security Implications:** ML-based security systems are vulnerable to both test-time and training-time attacks

## Logging

All modules incorporate comprehensive logging:
- `baseline_model.log` - CNN training and evaluation
- `evasion_attacks.log` - FGSM and PGD attack generation
- `poisoning_attacks.log` - Data poisoning experiments
- `adversarial_defense.log` - Defense mechanism training

## Dependencies

- `torch` - Deep learning framework
- `torchvision` - Computer vision datasets and transformations
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `scikit-learn` - Evaluation metrics
- `tqdm` - Progress bars

See `requirements.txt` for specific versions.

## Reproducibility

All stochastic operations use fixed random seeds:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- CUDA: `torch.cuda.manual_seed_all(42)`

## Lab Report

The complete analysis and findings are documented in the accompanying lab report PDF, which includes:
- Detailed methodology for each attack and defense
- Comprehensive results analysis with visualizations
- Confusion matrix interpretations
- Attack effectiveness comparisons
- Defense mechanism trade-off analysis
- Cybersecurity implications and deployment considerations
- Discussion on trustworthy AI for security applications

## Cybersecurity Relevance

This work demonstrates critical vulnerabilities in ML-based security systems:

1. **Intrusion Detection Systems:** Adversarial attacks can evade network traffic classifiers
2. **Malware Detection:** Poisoned training data can create backdoors in malware classifiers
3. **Biometric Authentication:** Evasion attacks can fool facial recognition systems
4. **Spam Filtering:** Adversarial examples can bypass email security filters

Understanding these threats is essential for deploying robust AI-powered security solutions.

## References

1. Alotaibi, A., & Rassam, M. A. (2023). Adversarial Machine Learning Attacks against Intrusion Detection Systems: A Survey on Strategies and Defense. *Future Internet*, 15(2), 62. https://doi.org/10.3390/fi15020062

2. Vassilev, A., Oprea, A., Fordyce, A. and Andersen, H. (2024), Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations, NIST Trustworthy and Responsible AI, National Institute of Standards and Technology, Gaithersburg, MD. https://doi.org/10.6028/NIST.AI.100-2e2023

3. Zhao, P., Zhu, W., Jiao, P., Gao, D., & Wu, O. (2025). Data poisoning in deep learning: A survey. *arXiv preprint* arXiv:2503.22759.

4. Kure, H. I., Sarkar, P., Ndanusa, A. B., & Nwajana, A. O. (2025). Detecting and preventing data poisoning attacks on AI models. *arXiv preprint* arXiv:2503.09302.

5. Paracha, A., Arshad, J., Farah, M.B. et al. Deep behavioral analysis of machine learning algorithms against data poisoning. *Int. J. Inf. Secur.* 24, 29 (2025). https://doi.org/10.1007/s10207-024-00940-x

6. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. *ICLR 2015*. https://arxiv.org/abs/1412.6572

7. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR 2018*. https://arxiv.org/abs/1706.06083

## License

This project is for academic purposes as part of the INF6422E course at École Polytechnique de Montréal.

## Acknowledgments

- Professor Tanzeel Sultan Rana for course instruction and guidance
- Alex Krizhevsky for the CIFAR-10 dataset
- École Polytechnique de Montréal

---

*Last Updated: February 16, 2026*