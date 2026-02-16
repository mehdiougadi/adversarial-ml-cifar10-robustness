import logging
from pathlib import Path

from src.fgsm_attack import run_fgsm_attack
from src.model import create_and_summarize_model
from src.train_baseline import train_baseline_model
from src.pgd_attack import run_pgd_attack


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
    run_fgsm_attack(epsilon_values=[0.0, 0.01, 0.05, 0.1, 0.2])
    run_pgd_attack(epsilon_values=[0.0, 0.01, 0.03, 0.05])


if __name__ == "__main__":
    main()
