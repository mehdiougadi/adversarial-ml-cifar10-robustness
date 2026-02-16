import logging
import torch

logger = logging.getLogger(__name__)


class PoisonedDataset(torch.utils.data.Dataset):

    def __init__(self, poisoned_data):
        self.data = poisoned_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def flip_labels():
    pass


def load_and_poison_data():
    pass


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
