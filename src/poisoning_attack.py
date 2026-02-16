import logging

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

logger = logging.getLogger(__name__)


class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, poisoned_data):
        self.data = poisoned_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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


def plot_poisoning_comparison():
    pass


def plot_training_curves():
    pass


def save_poisoning_results():
    pass


def run_poisoning_experiment():
    pass
