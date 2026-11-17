"""Split dev 10% phan tang, xac dinh. Id = chi so trong split HF 'train'."""
import hashlib
import json
from pathlib import Path

import numpy as np


def make_dev_split(labels, dev_frac=0.1, seed=1234):
    """Tra ve (train_idx, dev_idx) da sap xep. Moi lop: round(dev_frac * n_lop) anh vao dev."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    dev = []
    for c in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == c)          # da tang dan
        perm = rng.permutation(len(idx))
        k = int(round(dev_frac * len(idx)))
        dev.extend(idx[perm[:k]].tolist())
    dev = np.array(sorted(dev), dtype=np.int64)
    mask = np.ones(len(labels), bool)
    mask[dev] = False
    return np.flatnonzero(mask).astype(np.int64), dev


def ids_hash(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def subsample_stratified(indices, labels, frac, seed=1234):
    """Tap con phan tang, xac dinh (dung cho sweep 20%)."""
    indices = np.asarray(indices)
    if frac >= 1.0:
        return indices
    labels = np.asarray(labels)[indices]
    rng = np.random.RandomState(seed)
    out = []
    for c in sorted(np.unique(labels).tolist()):
        sel = indices[labels == c]
        k = max(1, int(round(frac * len(sel))))
        out.extend(sel[rng.permutation(len(sel))[:k]].tolist())
    return np.array(sorted(out), dtype=np.int64)


def load_or_create(labels, splits_dir, dev_frac=0.1, seed=1234, write=True):
    """Tinh split; neu splits/dev_ids.json da co thi PHAI khop (hash), neu khong -> loi.
    write=False (smoke) khong dong vao thu muc splits."""
    train_idx, dev_idx = make_dev_split(labels, dev_frac, seed)
    h = ids_hash(dev_idx)
    path = Path(splits_dir) / "dev_ids.json"
    if write:
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["hash"] != h:
                raise RuntimeError(f"dev_ids.json hash {saved['hash'][:12]} != tinh lai {h[:12]}; "
                                   "split bi doi hoac nhan khac. Dung ngay.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"seed": seed, "dev_frac": dev_frac, "n_dev": len(dev_idx),
                                        "n_train": len(train_idx), "hash": h,
                                        "dev_ids": dev_idx.tolist()}))
    return train_idx, dev_idx, h


def main():
    from . import data
    from .utils import load_configs, splits_dir
    cfg, _ = load_configs()
    meta = data.ensure_cache(cfg)
    labels = np.load(Path(meta["dir"]) / "train.labels.npy")
    tr, dv, h = load_or_create(labels, splits_dir(), cfg["split"]["dev_frac"], cfg["split"]["seed"])
    print(json.dumps({"n_train": len(tr), "n_dev": len(dv), "hash": h}))


if __name__ == "__main__":
    main()
