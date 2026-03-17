from pathlib import Path

from services.config import Config
from services.fine_tune import FineTune


def main():
    config = Config.load_config()
    check = start_fine_tune(config)
    if check:
        print("Fine-tuning done successfully")
    else:
        print("Fine-tune failed")


def start_fine_tune(config):
    print("Starting fine-tuning of base model")

    check = FineTune.fine_tune(config["model"], str(Path(__file__).resolve().parent / "data" / "dataset"))

    return check


if __name__ == "__main__":
    main()
