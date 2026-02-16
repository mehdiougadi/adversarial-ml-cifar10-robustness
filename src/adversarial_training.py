import logging

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor):
    logger.info(f"Generating FGSM adversarial examples with epsilon={epsilon}")

    try:
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        return torch.clamp(perturbed_image, 0, 1)

    except Exception as e:
        logger.error(f"Error in FGSM attack: {e}")
        raise


def load_cifar10_data():
    logger.info("Loading CIFAR-10 dataset for adversarial training...")

    try:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

        total_size = len(train_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_data, val_data, test_data = random_split(
            train_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        logger.info(f"Train size: {len(train_data)}")
        logger.info(f"Validation size: {len(val_data)}")
        logger.info(f"Test size: {len(test_data)}")

        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Error loading CIFAR-10 data: {e}")
        raise


def adversarial_training():
    pass


def evaluate_on_clean_data():
    pass


def evaluate_on_adversarial_data():
    pass


def plot_defense_comparison():
    pass


def plot_training_curves():
    pass


def save_defense_results():
    pass


def run_adversarial_training_defense():
    pass
