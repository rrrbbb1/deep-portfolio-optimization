import yaml
from pathlib import Path

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config(exp_path):
    exp_cfg = load_yaml(exp_path)
    base_dir = Path(exp_path).parent.parent

    cfg = {}

    for entry in exp_cfg["defaults"]:
        for k, v in entry.items():
            path = base_dir / k / f"{v}.yaml"
            cfg[k] = load_yaml(path)

    # override with experiment-specific values
    for k, v in exp_cfg.items():
        if k != "defaults":
            cfg[k].update(v)

    return cfg
