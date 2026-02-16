import logging
import torch
import numpy as np

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
                wrong_labels = [l for l in range(num_classes) if l != label]
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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])

        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
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

        train_list = [(train_dataset[idx][0], train_dataset[idx][1])
                      for idx in train_data.indices]

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


def train_poisoned_model():
    pass


def evaluate_poisoned_model():
    pass


def plot_poisoning_comparison():
    pass


def plot_training_curves():
    pass


def save_poisoning_results():
    pass


def run_poisoning_experiment():
    pass
