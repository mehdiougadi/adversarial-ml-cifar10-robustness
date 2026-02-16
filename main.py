import logging
from pathlib import Path

from src.adversarial_training import run_adversarial_training_defense
from src.fgsm_attack import run_fgsm_attack
from src.model import create_and_summarize_model
from src.pgd_attack import run_pgd_attack
from src.poisoning_attack import run_poisoning_experiment
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
    run_fgsm_attack(model=model, epsilon_values=[0.0, 0.01, 0.05, 0.1, 0.2])
    run_pgd_attack(model=model, epsilon_values=[0.0, 0.01, 0.03, 0.05])
    run_poisoning_experiment(model=model, poison_levels=[0.0, 0.05, 0.15])
    run_adversarial_training_defense(model=model, epsilon=0.03, epochs=5)


if __name__ == "__main__":
    main()
