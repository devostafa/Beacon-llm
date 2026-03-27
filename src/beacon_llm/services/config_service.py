import json
from pathlib import Path


class ConfigService:
    def create_config(self):
        config_path = Path.cwd() / "config.json"
        if not config_path.exists():
            default_config = {
                "hf_token": "0",
                "model": "none",
                "inference_api": "http://localhost:8000/"
            }

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)

    def load_config(self):
        config_path = Path.cwd() / "config.json"

        if not config_path.exists():
            self.create_config()

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
