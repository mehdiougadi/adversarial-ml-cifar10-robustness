import logging


logging.basicConfig(
    filename='train_baseline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def load_cifar10_data():
    pass


def split_dataset():
    pass


def create_data_loaders():
    pass


def train_model():
    pass


def evaluate_model():
    pass


def save_results():
    pass
