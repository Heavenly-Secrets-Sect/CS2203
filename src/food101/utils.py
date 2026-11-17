import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return Path(os.environ.get("FOOD101_CONFIG_DIR", REPO_ROOT / "configs"))


def splits_dir() -> Path:
    return Path(os.environ.get("FOOD101_SPLITS_DIR", REPO_ROOT / "splits"))


def cache_root() -> Path:
    """Thu muc cache anh (shard tren dia). Mac dinh /kaggle/temp neu co."""
    env = os.environ.get("FOOD101_CACHE_DIR")
    if env:
        return Path(env)
    if Path("/kaggle/temp").exists():
        return Path("/kaggle/temp/food101_cache")
    return Path.home() / ".cache" / "food101"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_configs():
    d = config_dir()
    return load_yaml(d / "default.yaml"), load_yaml(d / "models.yaml")


def seed_everything(seed: int):
    """Dat seed ro rang. KHONG dung hash() (PYTHONHASHSEED chi co tac dung cho tien trinh con)."""
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_plain(o):
    if isinstance(o, dict):
        return {str(k): to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_write_text(path, json.dumps(to_plain(obj), indent=2))


def write_yaml(path, obj):
    atomic_write_text(path, yaml.safe_dump(to_plain(obj), sort_keys=False, allow_unicode=True))


def git_commit():
    for k in ("FOOD101_GIT_SHA", "GITHUB_SHA"):
        if os.environ.get(k):
            return os.environ[k]
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() != "HEAD" else None
    except Exception:
        return None


def lib_versions():
    import importlib
    out = {"python": sys.version.split()[0]}
    for m in ("torch", "torchvision", "timm", "numpy", "PIL", "datasets", "open_clip", "yaml"):
        try:
            mod = importlib.import_module(m) if m != "open_clip" else sys.modules.get("open_clip")
            if mod is not None:
                out[m] = str(getattr(mod, "__version__", "?"))
        except Exception:
            pass
    return out
