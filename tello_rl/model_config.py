"""Settings saved by train.py next to each model (env_config.json), loaded by evaluate/run scripts."""
import json, os


def load_config(model_path):
    p = os.path.join(os.path.dirname(os.path.abspath(model_path)), "env_config.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}
