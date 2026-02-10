from src.model import create_and_summarize_model
from src.train_baseline import train_baseline_model


def main():
    model = create_and_summarize_model()
    model, metrics, history = train_baseline_model(epochs=5, batch_size=64, lr=0.001)


if __name__ == "__main__":
    main()
