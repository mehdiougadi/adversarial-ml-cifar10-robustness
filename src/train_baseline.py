"""
Baseline CNN Training on CIFAR-10
Practical Work 2 - Adversarial Machine Learning
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.model import get_model

logger = logging.getLogger(__name__)


def load_cifar10_data():
    logger.info("Loading CIFAR-10 dataset...")

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

        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        return train_dataset, test_dataset

    except Exception as e:
        logger.error(f"Error loading CIFAR-10 dataset: {e}")
        raise


def split_dataset(train_dataset):
    logger.info("Splitting dataset into train/val/test...")

    try:
        total_size = len(train_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_data, val_data, test_data = random_split(
            train_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        logger.info(f"Train size: {len(train_data)}")
        logger.info(f"Validation size: {len(val_data)}")
        logger.info(f"Test size: {len(test_data)}")

        return train_data, val_data, test_data

    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        raise


def create_data_loaders(train_data, val_data, test_data, batch_size=64):
    logger.info("Creating data loaders...")

    try:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        logger.info(f"Batch size: {batch_size}")

        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise


def train_model(model, train_loader, val_loader, device, epochs=5, lr=0.001):
    logger.info("Starting model training...")

    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            logger.info(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%"
            )

        logger.info("Model training completed!")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
        }

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


def evaluate_model(model, test_loader, device):
    logger.info("Evaluating model on test set...")

    try:
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "predictions": all_preds,
            "labels": all_labels,
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def save_results(metrics, training_history):
    logger.info("Saving results to file...")

    try:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "baseline_results.txt"
        with results_file.open("w") as f:
            f.write("=" * 60 + "\n")
            f.write("Section 1: Baseline CNN Model Results\n")
            f.write("=" * 60 + "\n\n")

            f.write("Dataset Split:\n")
            f.write("  Training: 70% (35,000 images)\n")
            f.write("  Validation: 15% (7,500 images)\n")
            f.write("  Test: 15% (7,500 images)\n\n")

            f.write("Training Results:\n")
            f.write(
                f"  Final Train Accuracy: "
                f"{training_history['train_accuracies'][-1]:.2f}%\n"
            )
            f.write(
                f"  Final Val Accuracy: "
                f"{training_history['val_accuracies'][-1]:.2f}%\n\n"
            )

            f.write("Test Performance Metrics:\n")
            f.write(
                f"  Accuracy: {metrics['accuracy']:.4f} "
                f"({metrics['accuracy']*100:.2f}%)\n"
            )
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n\n")

            f.write("Confusion Matrix:\n")
            f.write(str(metrics["confusion_matrix"]) + "\n\n")

            f.write("=" * 60 + "\n")

        logger.info("Results saved to results/baseline_results.txt")

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def save_model(model, device):
    logger.info("Saving trained model...")

    try:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        model_path = results_dir / "baseline_model.pth"
        torch.save(model.state_dict(), model_path)

        logger.info(f"Model saved to {model_path}")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def train_baseline_model(model=None, device="cpu", epochs=5, batch_size=64, lr=0.001):
    logger.info("Starting baseline training pipeline...")

    try:
        train_dataset, _ = load_cifar10_data()
        train_data, val_data, test_data = split_dataset(train_dataset)
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, val_data, test_data, batch_size
        )

        if model is None:
            model = get_model(device)

        training_history = train_model(
            model, train_loader, val_loader, device, epochs, lr
        )
        metrics = evaluate_model(model, test_loader, device)

        save_results(metrics, training_history)
        save_model(model, device)

        logger.info("Baseline training pipeline completed!")
        return model, metrics, training_history

    except Exception as e:
        logger.error(f"Error in baseline training pipeline: {e}")
        raise
