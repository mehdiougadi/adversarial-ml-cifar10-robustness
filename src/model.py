"""
CNN Model Architecture for CIFAR-10 Classification
Practical Work 2 - Adversarial Machine Learning
"""

import logging
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    filename='model.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_model(device: str) -> SimpleCNN:
    logger.info("Initializing CNN model...")

    try:
        model = SimpleCNN().to(device)
        logger.info(f"Model created successfully on device: {device}")
        return model

    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise


def count_parameters(model: SimpleCNN) -> int:
    logger.info("Counting model parameters...")

    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")
        return total_params

    except Exception as e:
        logger.error(f"Error counting parameters: {e}")
        raise


def save_model_summary(model: SimpleCNN, params: int, device: str) -> None:
    logger.info("Saving model summary to file...")

    try:
        results_dir = Path('results')
        results_dir.mkdir(parents=True, exist_ok=True)

        summary_file = results_dir / 'model_summary.txt'
        with summary_file.open('w') as f:
            f.write("=" * 60 + "\n")
            f.write("Model Architecture Summary\n")
            f.write("=" * 60 + "\n")
            f.write(str(model) + "\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total Trainable Parameters: {params:,}\n")
            f.write(f"Device: {device}\n")
            f.write("=" * 60 + "\n")

        logger.info("Model summary saved to results/model_summary.txt")

    except Exception as e:
        logger.error(f"Error saving model summary: {e}")
        raise


def create_and_summarize_model(device: str = 'cpu') -> SimpleCNN:
    logger.info("Starting model creation and summarization...")

    try:
        model = get_model(device)
        params = count_parameters(model)
        save_model_summary(model, params, device)

        logger.info("Model creation and summarization completed!")
        return model

    except Exception as e:
        logger.error(f"Error in creating and summarizing model: {e}")
        raise
