import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.model import get_model

logger = logging.getLogger(__name__)


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor):
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


def adversarial_training(
    model, train_loader, val_loader, device, epsilon=0.03, epochs=5, lr=0.001
):
    logger.info(f"Starting adversarial training with epsilon={epsilon}...")

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
                images.requires_grad = True

                outputs = model(images)
                loss_clean = criterion(outputs, labels)

                model.zero_grad()
                loss_clean.backward()
                data_grad = images.grad.data

                perturbed_images = fgsm_attack(images, epsilon, data_grad)
                perturbed_images = perturbed_images.detach()

                optimizer.zero_grad()
                outputs_adv = model(perturbed_images)
                loss = criterion(outputs_adv, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs_adv, 1)
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

        logger.info("Adversarial training completed!")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
        }

    except Exception as e:
        logger.error(f"Error during adversarial training: {e}")
        raise


def evaluate_on_clean_data(model, test_loader, device):
    logger.info("Evaluating on clean test data...")

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

        logger.info(f"Clean Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    except Exception as e:
        logger.error(f"Error evaluating on clean data: {e}")
        raise


def evaluate_on_adversarial_data(model, test_loader, device, epsilon=0.03):
    logger.info(f"Evaluating on adversarial data (epsilon={epsilon})...")

    try:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        all_preds = []
        all_labels = []

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True

            outputs = model(images)
            loss = criterion(outputs, labels)

            model.zero_grad()
            loss.backward()
            data_grad = images.grad.data

            perturbed_images = fgsm_attack(images, epsilon, data_grad)

            adv_outputs = model(perturbed_images)
            _, predicted = torch.max(adv_outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        logger.info(f"Robust Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    except Exception as e:
        logger.error(f"Error evaluating on adversarial data: {e}")
        raise


def plot_defense_comparison(
    baseline_results: dict,
    defended_results: dict,
    save_path: str = "results/defense_figures/defense_comparison.png",
):
    logger.info("Creating defense comparison plot...")

    try:
        categories = ["Clean Accuracy", "Robust Accuracy\n(eps=0.03)"]

        baseline_values = [
            baseline_results["clean_accuracy"],
            baseline_results["robust_accuracy"],
        ]

        defended_values = [
            defended_results["clean_accuracy"],
            defended_results["robust_accuracy"],
        ]

        x = range(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(
            [i - width / 2 for i in x],
            baseline_values,
            width,
            label="Baseline Model",
            color="#e74c3c",
            alpha=0.8,
            edgecolor="black",
        )

        bars2 = ax.bar(
            [i + width / 2 for i in x],
            defended_values,
            width,
            label="Adversarially Trained",
            color="#2ecc71",
            alpha=0.8,
            edgecolor="black",
        )

        for bars in [bars1, bars2]:
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

        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            "Adversarial Training Defense: Before vs After",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Defense comparison plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Error creating defense comparison plot: {e}")
        raise


def plot_training_curves(
    training_history: dict,
    save_path: str = "results/defense_figures/adversarial_training_curves.png",
):
    logger.info("Creating adversarial training curves...")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(training_history["train_accuracies"]) + 1)

        axes[0].plot(
            epochs,
            training_history["train_accuracies"],
            "o-",
            label="Train Accuracy",
            linewidth=2,
            markersize=6,
            color="#3498db",
        )
        axes[0].plot(
            epochs,
            training_history["val_accuracies"],
            "s-",
            label="Val Accuracy",
            linewidth=2,
            markersize=6,
            color="#e74c3c",
        )
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Accuracy (%)", fontsize=11)
        axes[0].set_title(
            "Adversarial Training: Accuracy", fontsize=12, fontweight="bold"
        )
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            epochs,
            training_history["train_losses"],
            "o-",
            label="Train Loss",
            linewidth=2,
            markersize=6,
            color="#3498db",
        )
        axes[1].plot(
            epochs,
            training_history["val_losses"],
            "s-",
            label="Val Loss",
            linewidth=2,
            markersize=6,
            color="#e74c3c",
        )
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Loss", fontsize=11)
        axes[1].set_title("Adversarial Training: Loss", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=10)
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


def save_defense_results(
    baseline_results: dict,
    defended_results: dict,
    save_path: str = "results/defense_results.txt",
):
    logger.info("Saving defense results to file...")

    try:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        with Path.open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Section 4: Adversarial Training Defense Results\n")
            f.write("=" * 80 + "\n\n")

            f.write("Comparison: Baseline vs Adversarially Trained Model\n")
            f.write("-" * 80 + "\n\n")

            f.write(
                f"{'Metric':<25} {'Baseline':<15} {'Defended':<15} " f"{'Change':<15}\n"
            )
            f.write("-" * 80 + "\n")

            clean_base = baseline_results["clean_accuracy"]
            clean_def = defended_results["clean_accuracy"]
            clean_change = clean_def - clean_base

            f.write(
                f"{'Clean Accuracy':<25} {clean_base:<15.4f} "
                f"{clean_def:<15.4f} {clean_change:+.4f}\n"
            )

            robust_base = baseline_results["robust_accuracy"]
            robust_def = defended_results["robust_accuracy"]
            robust_change = robust_def - robust_base

            f.write(
                f"{'Robust Accuracy (eps=0.03)':<25} {robust_base:<15.4f} "
                f"{robust_def:<15.4f} {robust_change:+.4f}\n"
            )
            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Defense results saved to {save_path}")

    except Exception as e:
        logger.error(f"Error saving defense results: {e}")
        raise


def run_adversarial_training_defense(
    model=None,
    device: str = "cpu",
    epsilon: float = 0.03,
    epochs: int = 5,
    lr: float = 0.001,
):
    logger.info("Starting Adversarial Training Defense Pipeline...")

    if model is None:
        model = get_model(device)

    try:
        figures_dir = Path("results/defense_figures/")
        figures_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader, test_loader = load_cifar10_data()

        logger.info("Evaluating baseline model...")
        model.eval()

        baseline_clean = evaluate_on_clean_data(model, test_loader, device)
        baseline_robust = evaluate_on_adversarial_data(
            model, test_loader, device, epsilon
        )

        baseline_results = {
            "clean_accuracy": baseline_clean["accuracy"],
            "robust_accuracy": baseline_robust["accuracy"],
        }

        logger.info("Training adversarially robust model...")
        defended_model = get_model(device)
        defended_model.load_state_dict(model.state_dict())

        training_history = adversarial_training(
            defended_model, train_loader, val_loader, device, epsilon, epochs, lr
        )

        defended_clean = evaluate_on_clean_data(defended_model, test_loader, device)
        defended_robust = evaluate_on_adversarial_data(
            defended_model, test_loader, device, epsilon
        )

        defended_results = {
            "clean_accuracy": defended_clean["accuracy"],
            "robust_accuracy": defended_robust["accuracy"],
        }

        plot_defense_comparison(baseline_results, defended_results)
        plot_training_curves(training_history)
        save_defense_results(baseline_results, defended_results)

        model_path = Path("results/pth_files/adversarially_trained_model.pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(defended_model.state_dict(), model_path)
        logger.info(f"Adversarially trained model saved to {model_path}")

        logger.info("Adversarial Training Defense Pipeline Completed!")

        return baseline_results, defended_results, training_history

    except Exception as e:
        logger.error(f"Error in adversarial training defense pipeline: {e}")
        raise
