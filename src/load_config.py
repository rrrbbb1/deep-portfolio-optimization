import yaml
from pathlib import Path


def _load_yaml(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(training_cfg, loss_cfg, data_cfg, inputs_cfg):
    cfg = {}

    cfg["training"] = _load_yaml(training_cfg)
    cfg["loss"] = _load_yaml(loss_cfg)
    cfg["data"] = _load_yaml(data_cfg)
    cfg["inputs"] = _load_yaml(inputs_cfg)

    return cfg
