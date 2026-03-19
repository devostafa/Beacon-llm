from pathlib import Path

from services.config_service import ConfigService
from services.llm_service import LLMService


def main():
    config = ConfigService.load_config()
    start_fine_tune(config)


def start_fine_tune(config):
    print("Starting fine-tuning of base model")

    check = LLMService.fine_tune(config["model"], str(Path(__file__).resolve().parent / "data" / "dataset"))

    if check:
        print("Fine-tuning done successfully")
    else:
        print("Fine-tune failed")


if __name__ == "__main__":
    main()
