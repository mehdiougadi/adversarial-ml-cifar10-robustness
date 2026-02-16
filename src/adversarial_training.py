import logging

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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


def plot_defense_comparison():
    pass


def plot_training_curves():
    pass


def save_defense_results():
    pass


def run_adversarial_training_defense():
    pass
