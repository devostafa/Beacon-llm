from pathlib import Path

from huggingface_hub import login

from beacon_llm.services.config_service import ConfigService
from beacon_llm.services.llm_service import LLMService


def main():
    config = ConfigService().load_config()

    login(config["hf_token"])

    start_fine_tune(config)


def start_fine_tune(config):
    print("Starting fine-tuning of base model")

    llm = LLMService()

    data_dir_path = str(Path(__file__).resolve().parent / "data" / "dataset")

    check = llm.fine_tune(config["model"], data_dir_path)

    if check:
        print("Fine-tuning done successfully")
    else:
        print("Fine-tune failed")


if __name__ == "__main__":
    main()
