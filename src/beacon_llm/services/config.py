import json
from pathlib import Path


class Config:
    def load_config(self):
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
