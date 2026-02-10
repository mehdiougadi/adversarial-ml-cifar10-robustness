import logging
from pathlib import Path

from src.model import create_and_summarize_model
from src.train_baseline import train_baseline_model


def setup_logging():
    log_dir = Path("./results")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "script.log"),
            logging.StreamHandler(),
        ],
    )


def main():
    setup_logging()
    model = create_and_summarize_model()
    model, metrics, history = train_baseline_model(
        model=model, epochs=5, batch_size=64, lr=0.001
    )


if __name__ == "__main__":
    main()
