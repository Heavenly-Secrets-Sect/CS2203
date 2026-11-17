"""Du lieu: cache shard JPEG tren dia (memmap), KHONG nap het vao RAM.

Lua chon: moi split (train/validation) la 1 file .bin noi cac JPEG (canh ngan 256, q95) + offsets.npy
+ labels.npy. Dataset mo np.memmap lazily trong tung worker -> nhieu job/tien trinh dung chung page cache
cua HDH, RAM rieng moi job ~ khong doi. Dung khoa file (fcntl) nen 2 job song song chi build 1 lan.
"""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import cache_root

SPLIT_NAMES = ("train", "validation")  # validation == test (can doi chieu, xem WARNING)

TEST_WARNING = (
    "WARNING: HF dataset card (ethz/food101) KHONG khang dinh split 'validation' la test chinh thuc "
    "(meta/test.txt). Chi khop ve so luong (25,250 = 101 x 250). Doi chieu hash id/noi dung ben duoi "
    "voi danh sach goc truoc khi bao cao la 'test'."
)


# ---------- cache builder ----------
def _resize_short(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    if min(w, h) == short:
        return img
    if w <= h:
        nw, nh = short, max(short, int(round(h * short / w)))
    else:
        nw, nh = max(short, int(round(w * short / h))), short
    return img.resize((nw, nh), Image.BICUBIC)


def _encode(raw: bytes, short: int, quality: int) -> bytes:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = _resize_short(img, short)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _write_split(samples, out_dir: Path, name: str, short: int, quality: int):
    """samples: iterable (raw_bytes, label). Tra ve dict meta cua split."""
    offsets, labels = [0], []
    content = hashlib.sha256()
    with open(out_dir / f"{name}.bin", "wb") as f:
        for raw, lab in samples:
            content.update(hashlib.sha1(raw).digest())
            b = _encode(raw, short, quality)
            f.write(b)
            offsets.append(offsets[-1] + len(b))
            labels.append(int(lab))
    np.save(out_dir / f"{name}.offsets.npy", np.array(offsets, dtype=np.int64))
    labels = np.array(labels, dtype=np.int64)
    np.save(out_dir / f"{name}.labels.npy", labels)
    idh = hashlib.sha256("\n".join(f"{i}:{l}" for i, l in enumerate(labels.tolist())).encode()).hexdigest()
    return {"n": int(len(labels)), "per_class_min": int(np.bincount(labels).min()),
            "per_class_max": int(np.bincount(labels).max()), "ids_hash": idh,
            "content_hash": content.hexdigest()}


def _hf_samples(cfg, split):
    from datasets import Image as HFImage, load_dataset  # lazy
    d = cfg["dataset"]
    ds = load_dataset(d["name"], split=split, revision=d.get("revision"))
    names = ds.features["label"].names
    ds = ds.cast_column("image", HFImage(decode=False))

    def gen():
        for b in ds.iter(batch_size=256):
            for im, lab in zip(b["image"], b["label"]):
                yield im["bytes"], lab
    return gen(), names


def fake_samples(n_per_class, num_classes, seed, size=(40, 48)):
    """Anh gia sinh ngau nhien (xac dinh) cho --smoke: nen mau theo lop + nhieu."""
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 255, size=(num_classes, 3))
    out = []
    for c in range(num_classes):
        for _ in range(n_per_class):
            arr = np.clip(base[c] + rng.randint(-40, 40, size=(size[0], size[1], 3)), 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=90)
            out.append((buf.getvalue(), c))
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def ensure_cache(cfg, root=None):
    """Build (neu chua co) cache that tu HF. Tra ve meta (co 'dir')."""
    import fcntl
    root = Path(root) if root else cache_root()
    d = cfg["dataset"]
    key = f"{d['name'].replace('/', '_')}_{(d.get('revision') or 'main')[:10]}_s{d['short_side']}_q{d['jpeg_quality']}"
    final = root / key
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not (final / "meta.json").exists():
            tmp = Path(tempfile.mkdtemp(prefix=key + ".tmp", dir=root))
            print(f"[data] building cache -> {final}", file=sys.stderr)
            meta = {"splits": {}, "dataset": d}
            for sp in SPLIT_NAMES:
                gen, names = _hf_samples(cfg, sp)
                meta["splits"][sp] = _write_split(gen, tmp, sp, d["short_side"], d["jpeg_quality"])
                meta["class_names"] = names
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
            if final.exists():
                shutil.rmtree(final)
            os.replace(tmp, final)
    meta = json.loads((final / "meta.json").read_text())
    meta["dir"] = str(final)
    return meta


def build_smoke_cache(cfg, root, short, n_classes=10, n_train=20, n_test=4, quality=90):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    meta = {"splits": {}, "class_names": [f"class_{i}" for i in range(n_classes)], "smoke": True}
    meta["splits"]["train"] = _write_split(fake_samples(n_train, n_classes, 0), root, "train", short, quality)
    meta["splits"]["validation"] = _write_split(fake_samples(n_test, n_classes, 1), root, "validation", short, quality)
    meta["dir"] = str(root)
    return meta


def check_counts(meta, cfg):
    """Co kiem tra so luong; nem loi neu lech. In hash id de doi chieu."""
    d = cfg["dataset"]
    tr, te = meta["splits"]["train"], meta["splits"]["validation"]
    errs = []
    if tr["n"] != d["expected_train"]:
        errs.append(f"train {tr['n']} != {d['expected_train']}")
    if te["n"] != d["expected_test"]:
        errs.append(f"validation {te['n']} != {d['expected_test']}")
    if te["per_class_min"] != d["expected_test_per_class"] or te["per_class_max"] != d["expected_test_per_class"]:
        errs.append(f"validation/lop min={te['per_class_min']} max={te['per_class_max']} != {d['expected_test_per_class']}")
    if len(meta["class_names"]) != d["num_classes"]:
        errs.append(f"so lop {len(meta['class_names'])} != {d['num_classes']}")
    print(TEST_WARNING, file=sys.stderr)
    print(f"[data] test ids_hash={te['ids_hash']} content_hash={te['content_hash']} "
          f"train ids_hash={tr['ids_hash']}", file=sys.stderr)
    if errs:
        raise RuntimeError("Kiem tra so luong that bai: " + "; ".join(errs))


# ---------- dataset ----------
class ShardDataset(Dataset):
    def __init__(self, shard_dir, name, indices, transform):
        self.dir, self.name = str(shard_dir), name
        self.offsets = np.load(Path(shard_dir) / f"{name}.offsets.npy")
        self.all_labels = np.load(Path(shard_dir) / f"{name}.labels.npy")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self._mm = None

    def __len__(self):
        return len(self.indices)

    @property
    def labels(self):
        return self.all_labels[self.indices]

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_mm"] = None
        return s

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = np.memmap(Path(self.dir) / f"{self.name}.bin", dtype=np.uint8, mode="r")
        j = int(self.indices[i])
        raw = self._mm[self.offsets[j]:self.offsets[j + 1]].tobytes()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.transform(img), int(self.all_labels[j])


def make_transforms(mcfg, img_size, resize, rrc_scale):
    from torchvision import transforms as T
    interp = {"bicubic": T.InterpolationMode.BICUBIC, "bilinear": T.InterpolationMode.BILINEAR}[mcfg["interpolation"]]
    norm = T.Normalize(mcfg["mean"], mcfg["std"])
    train = T.Compose([T.RandomResizedCrop(img_size, scale=tuple(rrc_scale), interpolation=interp),
                       T.RandomHorizontalFlip(), T.ToTensor(), norm])
    test = T.Compose([T.Resize(resize, interpolation=interp), T.CenterCrop(img_size), T.ToTensor(), norm])
    return train, test
