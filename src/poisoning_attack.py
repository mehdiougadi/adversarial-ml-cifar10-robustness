import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def flip_labels(dataset, flip_percentage: float, seed: int = 42):
    logger.info(f"Flipping {flip_percentage*100}% of labels...")

    try:
        np.random.seed(seed)
        torch.manual_seed(seed)

        total_samples = len(dataset)
        num_to_flip = int(total_samples * flip_percentage)

        flip_indices = np.random.choice(total_samples, num_to_flip, replace=False)

        poisoned_data = []
        flip_count = 0

        for idx in range(total_samples):
            image, label = dataset[idx]

            if idx in flip_indices:
                num_classes = 10
                wrong_labels = [c for c in range(num_classes) if c != label]
                flipped_label = np.random.choice(wrong_labels)
                poisoned_data.append((image, flipped_label))
                flip_count += 1
            else:
                poisoned_data.append((image, label))

        logger.info(f"Successfully flipped {flip_count} labels out of {total_samples}")

        return poisoned_data

    except Exception as e:
        logger.error(f"Error flipping labels: {e}")
        raise


class PoisonedDataset(torch.utils.data.Dataset):

    def __init__(self, poisoned_data):
        self.data = poisoned_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_and_poison_data(flip_percentage: float):
    logger.info(f"Loading CIFAR-10 with {flip_percentage*100}% poisoning...")

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

        train_list = [
            (train_dataset[idx][0], train_dataset[idx][1]) for idx in train_data.indices
        ]

        if flip_percentage > 0:
            poisoned_train_data = flip_labels(train_list, flip_percentage)
            train_data = PoisonedDataset(poisoned_train_data)

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        logger.info(f"Train size: {len(train_data)}")
        logger.info(f"Validation size: {len(val_data)}")
        logger.info(f"Test size: {len(test_data)}")

        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Error loading and poisoning data: {e}")
        raise


def train_poisoned_model(model, train_loader, val_loader, device, epochs=5, lr=0.001):
    logger.info("Training model on poisoned data...")

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
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

        logger.info("Model training on poisoned data completed!")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
        }

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


def evaluate_poisoned_model(model, test_loader, device):
    logger.info("Evaluating poisoned model on clean test set...")

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


def plot_poisoning_comparison(
    results_dict: dict,
    save_path: str = "results/poisoning_figures/poisoning_comparison.png",
):
    logger.info("Creating poisoning comparison plot...")

    try:
        poison_levels = list(results_dict.keys())
        accuracies = [results_dict[level]["accuracy"] for level in poison_levels]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            [f"{int(p*100)}%" for p in poison_levels],
            accuracies,
            color=["#2ecc71", "#f39c12", "#e74c3c"],
            alpha=0.8,
            edgecolor="black",
        )

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_xlabel("Poisoning Level (% of Training Labels Flipped)", fontsize=12)
        ax.set_ylabel("Test Accuracy", fontsize=12)
        ax.set_title(
            "Impact of Label-Flipping Poisoning on Model Performance",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Poisoning comparison plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
        raise


def plot_training_curves(
    results_dict: dict, save_path: str = "results/poisoning_figures/training_curves.png"
):
    logger.info("Creating training curves comparison...")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        for idx, (poison_level, results) in enumerate(results_dict.items()):
            epochs = range(1, len(results["train_accuracies"]) + 1)

            axes[0].plot(
                epochs,
                results["train_accuracies"],
                "o-",
                label=f"{int(poison_level*100)}% Poisoned (Train)",
                color=colors[idx],
                linewidth=2,
                markersize=6,
            )
            axes[0].plot(
                epochs,
                results["val_accuracies"],
                "s--",
                label=f"{int(poison_level*100)}% Poisoned (Val)",
                color=colors[idx],
                linewidth=2,
                markersize=6,
                alpha=0.7,
            )

            axes[1].plot(
                epochs,
                results["train_losses"],
                "o-",
                label=f"{int(poison_level*100)}% Poisoned (Train)",
                color=colors[idx],
                linewidth=2,
                markersize=6,
            )
            axes[1].plot(
                epochs,
                results["val_losses"],
                "s--",
                label=f"{int(poison_level*100)}% Poisoned (Val)",
                color=colors[idx],
                linewidth=2,
                markersize=6,
                alpha=0.7,
            )

        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Accuracy (%)", fontsize=11)
        axes[0].set_title(
            "Training & Validation Accuracy", fontsize=12, fontweight="bold"
        )
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Loss", fontsize=11)
        axes[1].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Training curves saved to {save_path}")

    except Exception as e:
        logger.error(f"Error creating training curves: {e}")
        raise


def save_poisoning_results(
    results_dict: dict,
    clean_accuracy: float,
    save_path: str = "results/poisoning_results.txt",
):
    logger.info("Saving poisoning results to file...")

    try:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        with Path.open(save_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Section 3: Data Poisoning Attack Results\n")
            f.write("=" * 80 + "\n\n")

            f.write("Label-Flipping Poisoning Experiment\n")
            f.write("-" * 80 + "\n\n")

            f.write(
                f"Baseline (Clean) Accuracy: {clean_accuracy:.4f} "
                f"({clean_accuracy*100:.2f}%)\n\n"
            )

            f.write(
                f"{'Poison Level':<15} {'Accuracy':<12} "
                f"{'Accuracy Drop':<15} {'Drop %':<10}\n"
            )
            f.write("-" * 80 + "\n")

            for poison_level, metrics in results_dict.items():
                acc = metrics["accuracy"]
                drop = clean_accuracy - acc
                drop_pct = (drop / clean_accuracy) * 100

                f.write(
                    f"{int(poison_level*100)}%{'':<12} "
                    f"{acc:<12.4f} "
                    f"{drop:<15.4f} "
                    f"{drop_pct:<10.2f}%\n"
                )

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Poisoning results saved to {save_path}")

    except Exception as e:
        logger.error(f"Error saving poisoning results: {e}")
        raise


def run_poisoning_experiment(
    model=None,
    device: str = "cpu",
    poison_levels: list = None,
    epochs: int = 5,
    clean_baseline_path: str = "results/baseline_results.txt",
):
    logger.info("Starting Data Poisoning Attack Pipeline...")

    if model is None:
        model = get_model(device)

    if poison_levels is None:
        poison_levels = [0.0, 0.05, 0.15]

    try:
        figures_dir = Path("results/poisoning_figures/")
        figures_dir.mkdir(parents=True, exist_ok=True)

        clean_accuracy = None
        try:
            with Path.open(clean_baseline_path) as f:
                for line in f:
                    if (
                        "Accuracy:" in line
                        and "Test Performance Metrics"
                        in Path.open(clean_baseline_path).read()
                    ):
                        clean_accuracy = float(line.split()[1])
                        break
        except Exception:
            logger.warning(
                "Could not load baseline accuracy, will use 0% poisoning as baseline"
            )

        results_dict = {}

        for poison_level in poison_levels:
            logger.info(f"Testing with {poison_level*100}% label poisoning")
            train_loader, val_loader, test_loader = load_and_poison_data(poison_level)

            training_history = train_poisoned_model(
                model, train_loader, val_loader, device, epochs=epochs
            )

            metrics = evaluate_poisoned_model(model, test_loader, device)

            results_dict[poison_level] = {**metrics, **training_history}

            model_path = Path(
                f"results/pth_files/poisoned_model_{int(poison_level*100)}pct.pth"
            )
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

            if poison_level == 0.0 and clean_accuracy is None:
                clean_accuracy = metrics["accuracy"]

        plot_poisoning_comparison(results_dict)
        plot_training_curves(results_dict)

        save_poisoning_results(results_dict, clean_accuracy)

        logger.info("Data Poisoning Attack Pipeline Completed!")

        return results_dict

    except Exception as e:
        logger.error(f"Error in poisoning experiment: {e}")
        raise
