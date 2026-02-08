"""
CNN Model Architecture for CIFAR-10 Classification
Practical Work 2 - Adversarial Machine Learning
"""

import logging
import torch.nn as nn
import torch.nn.functional as F


logging.basicConfig(
    filename= 'model.log',
    level= logging.INFO,
    format= '%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


class SimpleCNN (nn.Module):
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
        pass

    except Exception as e:
        logger.error(f"Error counting parameters: {e}")
        raise


def create_and_summarize_model(device = 'cpu'):
    logger.info("Starting model creation and summarization...")

    try:
        pass

    except Exception as e:
        logger.error(f"Error in creating and summarizing model: {e}")
        raise
