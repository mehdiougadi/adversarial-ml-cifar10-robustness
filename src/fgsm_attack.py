import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import get_model

logger = logging.getLogger(__name__)


def fgsm_attack(
    image: torch.Tensor, epsilon: float, data_grad: torch.Tensor
) -> torch.Tensor:
    try:
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        return torch.clamp(perturbed_image, 0, 1)
    except Exception as e:
        logger.error(f"Error in FGSM attack: {e}")
        raise


def generate_adversarial_examples(
    model: nn.Module,
    data_loader: DataLoader,
    epsilon: float,
    device: str,
    criterion: nn.Module,
) -> tuple[list, list, list, list]:
    logger.info(f"Generating adversarial examples with epsilon={epsilon}...")

    try:
        model.eval()
        clean_preds = []
        adv_preds = []
        all_labels = []
        adv_images_list = []

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True

            outputs = model(images)
            _, pred_clean = torch.max(outputs, 1)
            clean_preds.extend(pred_clean.cpu().numpy())

            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = images.grad.data

            perturbed_images = fgsm_attack(images, epsilon, data_grad)

            adv_outputs = model(perturbed_images)
            _, pred_adv = torch.max(adv_outputs, 1)
            adv_preds.extend(pred_adv.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())
            adv_images_list.append(perturbed_images.detach().cpu())

        logger.info(f"Generated {len(all_labels)} adversarial examples")
        return clean_preds, adv_preds, all_labels, adv_images_list

    except Exception as e:
        logger.error(f"Error generating adversarial examples: {e}")
        raise


def evaluate_attack(
    clean_preds: list, adv_preds: list, labels: list, epsilon: float
) -> dict:
    logger.info(f"Evaluating FGSM attack with epsilon={epsilon}...")

    try:
        clean_acc = accuracy_score(labels, clean_preds)
        adv_acc = accuracy_score(labels, adv_preds)
        attack_success_rate = 1 - np.mean(np.array(clean_preds) == np.array(adv_preds))

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, adv_preds, average="weighted", zero_division=0
        )

        metrics = {
            "epsilon": epsilon,
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "accuracy_drop": clean_acc - adv_acc,
            "attack_success_rate": attack_success_rate,
        }

        logger.info(f"Clean Accuracy: {clean_acc:.4f}")
        logger.info(f"Adversarial Accuracy: {adv_acc:.4f}")
        logger.info(f"Accuracy Drop: {metrics['accuracy_drop']:.4f}")

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating attack: {e}")
        raise


def visualize_adversarial_examples(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    clean_preds: list,
    adv_preds: list,
    labels: list,
    epsilon: float,
    num_examples: int = 5,
    save_path: str = "results/fgsm_figures/fgsm_examples.png",
) -> None:
    logger.info("Creating adversarial examples visualization...")

    try:
        classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])

        fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))

        for i in range(num_examples):
            orig_img = original_images[i].cpu().numpy().transpose(1, 2, 0)
            orig_img = std * orig_img + mean
            orig_img = np.clip(orig_img, 0, 1)

            adv_img = adversarial_images[i].cpu().numpy().transpose(1, 2, 0)
            adv_img = std * adv_img + mean
            adv_img = np.clip(adv_img, 0, 1)

            perturbation = adv_img - orig_img
            perturbation = (perturbation - perturbation.min()) / (
                perturbation.max() - perturbation.min() + 1e-8
            )

            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(
                f"Original\nTrue: {classes[labels[i]]}\n"
                f"Pred: {classes[clean_preds[i]]}"
            )
            axes[i, 0].axis("off")

            axes[i, 1].imshow(perturbation, cmap="hot")
            axes[i, 1].set_title(f"Perturbation\n(ε={epsilon})")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(adv_img)
            axes[i, 2].set_title(f"Adversarial\nPred: {classes[adv_preds[i]]}")
            axes[i, 2].axis("off")

        plt.tight_layout()
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualization saved to {save_path}")

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise


def plot_attack_strength_analysis(
    epsilon_results: list[dict],
    save_path: str = "results/fgsm_figures/fgsm_epsilon_analysis.png",
) -> None:
    logger.info("Creating epsilon analysis plot...")

    try:
        epsilons = [r["epsilon"] for r in epsilon_results]
        clean_accs = [r["clean_accuracy"] for r in epsilon_results]
        adv_accs = [r["adversarial_accuracy"] for r in epsilon_results]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            epsilons,
            clean_accs,
            "o-",
            label="Clean Accuracy",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            epsilons,
            adv_accs,
            "s-",
            label="Adversarial Accuracy",
            linewidth=2,
            markersize=8,
        )
        ax.set_xlabel("Epsilon (ε)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            "FGSM Attack: Accuracy vs. Epsilon", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Epsilon analysis plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Error creating epsilon analysis plot: {e}")
        raise


def save_fgsm_results(
    epsilon_results: list[dict], save_path: str = "results/fgsm_results.txt"
) -> None:
    logger.info("Saving FGSM results to file...")

    try:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        with Path.open(save_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Section 2.1: FGSM Attack Results\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"{'Epsilon':<12} {'Clean Acc':<12} {'Adv Acc':<12} "
                f"{'Acc Drop':<12}\n"
            )
            f.write("-" * 80 + "\n")

            for r in epsilon_results:
                line = (
                    f"{r['epsilon']:<12.3f} {r['clean_accuracy']:<12.4f} "
                    f"{r['adversarial_accuracy']:<12.4f} "
                    f"{r['accuracy_drop']:<12.4f}\n"
                )
                f.write(line)

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"FGSM results saved to {save_path}")

    except Exception as e:
        logger.error(f"Error saving FGSM results: {e}")
        raise


def run_fgsm_attack(
    model_path: str = "results/baseline_model.pth",
    device: str = "cpu",
    epsilon_values: list[float] | None = None,
    batch_size: int = 64,
) -> list[dict]:
    logger.info("Starting FGSM Attack Pipeline...")

    if epsilon_values is None:
        epsilon_values = [0.0, 0.01, 0.03, 0.05]

    try:
        figures_dir = Path("results/fgsm_figures/")
        figures_dir.mkdir(parents=True, exist_ok=True)

        model = get_model(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )

        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        epsilon_results = []
        first_batch_images, first_batch_labels = next(iter(test_loader))

        for epsilon in epsilon_values:
            logger.info(f"Testing FGSM with Epsilon = {epsilon}")

            clean_preds, adv_preds, labels, adv_images = generate_adversarial_examples(
                model, test_loader, epsilon, device, criterion
            )
            metrics = evaluate_attack(clean_preds, adv_preds, labels, epsilon)
            epsilon_results.append(metrics)

            if epsilon > 0:
                first_batch_images_device = first_batch_images.to(device)
                first_batch_images_device.requires_grad = True

                outputs = model(first_batch_images_device)
                loss = criterion(outputs, first_batch_labels.to(device))
                model.zero_grad()
                loss.backward()

                adv_batch = fgsm_attack(
                    first_batch_images_device,
                    epsilon,
                    first_batch_images_device.grad.data,
                )

                clean_outputs = model(first_batch_images_device)
                adv_outputs = model(adv_batch)

                _, clean_batch_preds = torch.max(clean_outputs, 1)
                _, adv_batch_preds = torch.max(adv_outputs, 1)

                visualize_adversarial_examples(
                    first_batch_images_device.detach(),
                    adv_batch.detach(),
                    clean_batch_preds.cpu().numpy(),
                    adv_batch_preds.cpu().numpy(),
                    first_batch_labels.numpy(),
                    epsilon,
                    num_examples=5,
                    save_path=(f"results/fgsm_figures/fgsm_examples_eps_{epsilon}.png"),
                )

        plot_attack_strength_analysis(epsilon_results)
        save_fgsm_results(epsilon_results)

        logger.info("FGSM Attack Pipeline Completed!")
        return epsilon_results

    except Exception as e:
        logger.error(f"Error in FGSM attack pipeline: {e}")
        raise
