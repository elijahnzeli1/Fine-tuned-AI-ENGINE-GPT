import yaml
import json
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_text_data(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def save_text_data(data, file_path):
    with open(file_path, 'w') as f:
        f.write(data)

def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)